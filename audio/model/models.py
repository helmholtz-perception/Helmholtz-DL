from __future__ import print_function

import keras
import tensorflow as tf
from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD, Adam

from os.path import isfile
from tensorflow.python.client import device_lib
import h5py
import numpy as np

def save_summary(model, header, suffix):
    assert(suffix.split(".")[0] == "")
    with open(header + suffix, 'w') as fh:
        # Pass the file handle in as a lambda functions to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

# I have not attempted much optimization, however it *is* fairly understandable
def build_model(X_shape, nb_classes, cnn_layers=3, mlps=[128]):
    # Inputs:
    #    X_shape = [ # spectrograms per batch, # audio channels, # spectrogram freq bins, # spectrogram time bins ]
    #    nb_classes = number of output n_classes
    #    nb_layers = number of conv-pooling sets in the CNN
    from keras import backend as K
    K.set_image_data_format('channels_last')                   # SHH changed on 3/1/2018 b/c tensorflow prefers channels_last

    nb_filters = 32  # number of convolutional filters = "feature maps"
    kernel_size = (3, 3)  # convolution kernel size
    pool_size = (2, 2)  # size of pooling area for max pooling
    cl_dropout = 0.5    # conv. layer dropout
    dl_dropout = 0.6    # dense layer dropout

    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    model = Sequential()
    model.add(Conv2D(nb_filters, kernel_size, padding='valid', input_shape=input_shape))
    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))        # Leave this relu & BN here.  ELU is not good here (my experience)

    for layer in range(cnn_layers):   # add more layers than just the first
        model.add(Conv2D(nb_filters, kernel_size))
        #model.add(BatchNormalization(axis=1))  # ELU authors reccommend no BatchNorm. I confirm.
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(cl_dropout))

    model.add(Flatten())
    for units in mlps:
        model.add(Dense(units))            # 128 is 'arbitrary' for now
            #model.add(Activation('relu'))   # relu (no BN) works ok here, however ELU works a bit better...
        model.add(Activation('elu'))
        model.add(Dropout(dl_dropout))
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model

# Used for when you want to use weights from a previously-trained model,
# with a different set/number of output classes
def attach_new_weights(model, new_nb_classes, n_pop = 2, n_p_dense = None, last_dropout = 0.6):

    # "penultimate" dense layer was originally 64 or 128. can change it here
    if (n_p_dense is not None):
        n_pop = 5

    # pop off the last n_pop layers. We definitely want the last 2: Activation() and Dense(nb_classes)
    for i in range(n_pop):
        model.pop()

    if (n_p_dense is not None):
        model.add(Dense(n_p_dense))
        model.add(Activation('elu'))
        model.add(Dropout(last_dropout))

    # attach final output layers
    model.add(Dense(new_nb_classes))  # new_nb_classes = new number of output classes
    model.add(Activation("softmax"))
    return model


# Next two routines are for attaching class names inside the saved model .hdf5 weights file
# From https://stackoverflow.com/questions/44310448/attaching-class-labels-to-a-keras-model
def load_model_ext(filepath, custom_objects=None):
    model = load_model(filepath, custom_objects=custom_objects)    # load the model normally

    #--- Now load it again and look for additional useful metadata
    f = h5py.File(filepath, mode='r')

    # initialize class_names with numbers (strings) in case hdf5 file doesn't have any
    output_length = model.layers[-1].output_shape[1]
    class_names = [str(x) for x in range(output_length)]
    if 'class_names' in f.attrs:
        class_names = f.attrs.get('class_names').tolist()
        class_names = [x.decode() for x in class_names]
    f.close()
    return model, class_names

def save_model_ext(model, filepath, overwrite=True, class_names=None):
    save_model(model, filepath, overwrite)
    if class_names is not None:
        f = h5py.File(filepath, mode='a')
        f.attrs['class_names'] = np.array(class_names, dtype='S')  # have to encode it somehow
        f.close()


# Freezing speeds up training by only declaring all but the last leave_last
# layers as non-trainable; but  likely results in lower accuracy
#  NOTE: In practice this achieves so little that I don't even use this:
#         Most of the model parameters are in the last few layers anyway
def freeze_layers(model, train_last=3):
    num_layers = len(model.layers)
    freeze_layers = min( num_layers - train_last, num_layers )  # any train_last too big, freezes whole model
    if (train_last < 0):                # special flag to disable freezing
        freeze_layers = 0
    print("Freezing ",freeze_layers,"/",num_layers," layers of model")
    for i in range(freeze_layers):
        model.layers[i].trainable = False
    return model


# This is the main routine for setting up a model
def setup_model(X, class_names, try_checkpoint=True,
    weights_file='weights.hdf5', quiet=True, missing_weights_fatal=False, multi_tag=False, setting=0):
    ''' In the following, the reason we hang on to & return serial_model,
         is because Keras can't save parallel models, but according to fchollet
         the serial & parallel versions will always share the same weights
         (Strange but true!)
    '''

    # Here's where one might 'swap out' different neural network 'model' choices
    if setting == 0:
        serial_model = build_model(X.shape, nb_classes=len(class_names), cnn_layers=3, mlps=[])
    elif setting == 1:
        serial_model = build_model(X.shape, nb_classes=len(class_names), cnn_layers=4)
    elif setting == 2:
        serial_model = build_model(X.shape, nb_classes=len(class_names), cnn_layers=3, mlps=[64, 128])
    elif setting == 3:
        serial_model = build_model(X.shape, nb_classes=len(class_names), cnn_layers=3)
    elif setting == 4:
        serial_model = build_model(X.shape, nb_classes=len(class_names), cnn_layers=3, mlps=[128, 128])
    elif setting == 5:
        serial_model = build_model(X.shape, nb_classes=len(class_names), cnn_layers=2)
    else:
        raise NotImplementedError("The architecture is not implemented!")

    #serial_model = old_model(X.shape, nb_classes=len(class_names), nb_layers=nb_layers)
    #serial_model = imageModels(X, nb_classes=len(class_names))

    # don't bother with freezing layers, at least with the hope of trianing on a laptop. doesn't speed up by more than a factor of 2.
    # serial_model = freeze_layers(serial_model, train_last = 3)

    # Initialize weights using checkpoint if it exists.
    if (try_checkpoint):
        print("Looking for previous weights...")
        if ( isfile(weights_file) ):
            print ('Weights file detected. Loading from ',weights_file)
            loaded_model = load_model(weights_file)   # strip any previous parallel part, to be added back in later
            serial_model.set_weights( loaded_model.get_weights() )   # assign weights based on checkpoint
        else:
            if (missing_weights_fatal):
                print("Need weights file to continue.  Aborting")
                assert(not missing_weights_fatal)
            else:
                print('No weights file detected, so starting from scratch.')


    opt = 'adadelta' # Adam(lr = 0.00001)  # So far, adadelta seems to work the best of things I've tried
    metrics = ['accuracy']

    if (multi_tag):     # multi_tag means more than one class can be 'chosen' at a time; default is 'only one'
        loss = 'binary_crossentropy'
    else:
        loss = 'categorical_crossentropy'

    serial_model.compile(loss=loss, optimizer=opt, metrics=metrics)

    if (not quiet):
        serial_model.summary()  # print out the model layers

    return serial_model   # fchollet says to hang on to the serial model for checkpointing

if __name__ == "__main__":
    model_list = ["cnn_x3_mlp_0", "cnn_x4_mlp_128", "cnn_x3_mlp_64_128", "cnn_x3_mlp_128", "cnn_x3_mlp_128x2", "cnn_x2_mlp_128"]
    parameters = []
    for setting in range(6):
        model = setup_model(np.zeros((1, 96, 173, 1)),[0]*12, try_checkpoint=False, setting=setting)
        save_summary(model, "parameters/model_" + str(model.count_params()) + "_" + model_list[setting], ".txt")
        print (model_list[setting] + ": " + str(model.count_params()))
        parameters.append(model.count_params())

