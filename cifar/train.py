from __future__ import print_function
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils, plot_model
from keras import backend as K
from keras.models import Model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
from models import build_model
import provider
import pandas
import cv2
import numpy as np
import argparse
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

K.set_image_dim_ordering('tf')

parser = argparse.ArgumentParser()
parser.add_argument('--quality', type=int, default=100, help='Image quality [default: 100]')
parser.add_argument('--setting', type=int, default=0, help='Model architecture (0-5) [default: 0]')
parser.add_argument('--model', type=str, default="", help='Model architecture description (0-5) [default: ""]')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training [default: 128]')
parser.add_argument('--num_epoch', type=int, default=200, help='Batch size during training [default: 200]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')
parser.add_argument('--decay_rate', type=float, default=1e-6, help='Decay rate [default: 1e-6]')
parser.add_argument('--log_dir', type=str, default="", help="The path of training log (saving directory)")
parser.add_argument('--train_dir', type=str, default="", help="The path of training data (loading directory)")

FLAGS = parser.parse_args()

QUALITY = FLAGS.quality
SETTING = FLAGS.setting
MODEL = FLAGS.model


model_list = ["all-cnns",
              "cnns-dense-64",
              "all-cnnsx2",
              "cnns-x2-dense-128",
              "cnns-dense-128",
              "cnns-dense-128-256",
             ]

if MODEL == "":
    SETTING = model_list[SETTING]
else:
    SETTING = MODEL

print ("model: " + SETTING)


nb_classes = 10
batch_size = FLAGS.batch_size
nb_epoch = FLAGS.num_epoch
learning_rate = FLAGS.learning_rate
decay_rate = FLAGS.decay_rate
rows, cols = 32, 32

channels = 3

train_dir = FLAGS.train_dir
log_dir = FLAGS.log_dir
if train_dir == "":
    train_dir = "data/quality_" + str(QUALITY) + "/"
if log_dir == "":
    log_dir = "logs/" + str(SETTING) + "/quality_" + str(QUALITY) + "/"

print ("train_dir: " + train_dir)
print ("log_dir: " + log_dir)

assert (os.path.exists(train_dir))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

os.system('cp %s %s' % ("train.py", log_dir)) # bkp of train procedure

FLAGS.train_dir = train_dir
FLAGS.log_dir = log_dir
FLAGS.model = SETTING
LOG_FOUT = open(os.path.join(log_dir, 'setting.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
LOG_FOUT.close()

def load_data(train_dir, test_dir):
    X_train, y_train = provider.load_data(train_dir, "train.h5")
    X_test, y_test = provider.load_data(test_dir, "test.h5")

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    return (X_train, Y_train), (X_test, Y_test)

def save_summary(model, header, suffix):
    assert(suffix.split(".")[0] == "")
    with open(header + suffix, 'w') as fh:
        # Pass the file handle in as a lambda functions to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def data_generator(wshift=0.1, hshift=0.1, horizontal_flip=True):
    data_gen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=wshift,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=hshift,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=horizontal_flip,  # randomly flip images
            vertical_flip=False)
    return data_gen

def train():
    (X_train, Y_train), (X_test, Y_test) = load_data(train_dir, train_dir)
    model = build_model(learning_rate, decay_rate, model_list.index(SETTING))
    # save_summary(model, "parameters/model", ".txt")
    # plot_model(model, to_file="parameters/model" + ".pdf", show_shapes=True)
    data_gen = data_generator()
    data_gen.fit(X_train)
    filepath = os.path.join(log_dir, "weights.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

    callbacks_list = [checkpoint]
        # Fit the model on the batches generated by datagen.flow().
    history_callback = model.fit_generator(data_gen.flow(X_train, Y_train,
                                           batch_size=batch_size),
                                           samples_per_epoch=X_train.shape[0],
                                           nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
                                           callbacks=callbacks_list, verbose=0)

    pandas.DataFrame(history_callback.history).to_csv(os.path.join(log_dir, "history.csv"))
    model.save(os.path.join(log_dir, 'model.h5'))


def predict(filepath):
    im = cv2.resize(cv2.imread(filepath), (224, 224)).astype(np.float32)
    out = model.predict(im)
    print (np.argmax(out))

if __name__ == "__main__":
    train()

