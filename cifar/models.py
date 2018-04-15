from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten, Activation, Convolution2D, GlobalAveragePooling2D
from keras.utils import np_utils, plot_model
from keras.optimizers import SGD

model_list = ["all-cnns",
              "cnns-dense-64",
              "all-cnnsx2",
              "cnns-x2-dense-128",
              "cnns-dense-128",
              "cnns-dense-128-256",
             ]

def save_summary(model, header, suffix):
    assert(suffix.split(".")[0] == "")
    with open(header + suffix, 'w') as fh:
        # Pass the file handle in as a lambda functions to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

def build_model(lr, decay, setting=0):
    model = Sequential()

    model.add(Convolution2D(32, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
    model.add(Dropout(0.5))

    if setting == 0:
        # all-cnns
        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, (1, 1), padding='valid'))

        model.add(GlobalAveragePooling2D())

    if setting == 1:
        # cnns-dense-64
        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))

    if setting == 2:
        # all-cnnsx2
        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (1, 1), padding='valid'))
        model.add(Activation('relu'))
        model.add(Convolution2D(10, (1, 1), padding='valid'))

        model.add(GlobalAveragePooling2D())

    if setting == 3:
        # cnns-x2-dense-128
        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))

    if setting == 4:
        # cnns-dense-128
        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))

    if setting == 5:
        # cnns-dense-128-256
        model.add(Convolution2D(128, (3, 3), padding = 'same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(128, (3, 3), padding='same', strides = (2,2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='relu'))

    model.add(Activation('softmax'))
    sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # print (model.count_params())
    # model.summary()
    return model

settings = [i for i in range(6)]
if __name__ == "__main__":
    models = []
    parameters = []
    for setting in settings:
        model = build_model(0.01, 1e-6, setting)
        save_summary(model, "parameters/model_" + str(model.count_params()) + "_" + model_list[setting], ".txt")
        print model_list[setting] + ": " + str(model.count_params())
        parameters.append(model.count_params())
        models.append(model_list[setting])
        # model.summary()
    print models
    print parameters
