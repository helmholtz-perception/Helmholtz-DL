#! /usr/bin/env python3

from __future__ import print_function
import numpy as np
import librosa
from model.models import *
from model.datautils import *
import os
from os.path import isfile
from timeit import default_timer as timer
from keras.callbacks import ModelCheckpoint
import pandas
import argparse

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description="trains network using training dataset")
parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
parser.add_argument('-c', '--classpath', #type=argparse.string,
        help='Train dataset directory with list of classes', default="")
parser.add_argument('--num_epoch', default=200, type=int, help="Number of iterations to train for")
parser.add_argument('--batch_size', default=50, type=int, help="Number of clips to send to GPU at once")
parser.add_argument('--val', default=0.25, type=float, help="Fraction of train to split off for validation")
parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",action="store_true")

parser.add_argument('--bitrate', type=int, default=128, help='Audio bitrate [default: 128kb/s]')
parser.add_argument('--setting', type=int, default=0, help='Model architecture (0-5) [default: 0]')
parser.add_argument('--model', type=str, default="", help='Model architecture description (0-5) [default: ""]')
parser.add_argument('--log_dir', type=str, default="", help="The path of training log (saving directory)")
parser.add_argument('--train_dir', type=str, default="", help="The path of training data (loading directory)")

FLAGS = parser.parse_args()

BITRATE = FLAGS.bitrate
SETTING = FLAGS.setting
MODEL = FLAGS.model
batch_size = FLAGS.batch_size
num_epoch = FLAGS.num_epoch
train_dir = FLAGS.train_dir
log_dir = FLAGS.log_dir
classpath = FLAGS.classpath
RATIO = FLAGS.val
TILE = FLAGS.tile

if classpath == "":
    classpath = "data/Preproc_" + str(BITRATE) + "/Train/"

model_list = ["cnn_x3_mlp_0",
              "cnn_x4_mlp_128",
              "cnn_x3_mlp_64_128",
              "cnn_x3_mlp_128",
              "cnn_x3_mlp_128x2",
              "cnn_x2_mlp_128"]

if MODEL == "":
    SETTING = model_list[SETTING]
else:
    SETTING = MODEL

print ("model: " + SETTING)

if train_dir == "":
    train_dir = "data/bitrate_" + str(BITRATE) + "/"
if log_dir == "":
    log_dir = "logs/" + str(SETTING) + "/bitrate_" + str(BITRATE) + "/"

print ("train_dir: " + train_dir)
print ("log_dir: " + log_dir)

assert (os.path.exists(train_dir))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

os.system('cp %s %s' % ("train.py", log_dir)) # bkp of train procedure

FLAGS.train_dir = train_dir
FLAGS.log_dir = log_dir
FLAGS.classpath = classpath
FLAGS.model = SETTING
print (FLAGS)

def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/", epochs=50, batch_size=20, val_split=0.25,tile=False, setting=0):
    np.random.seed(2337)

    # Get the data
    X_train, Y_train, paths_train, class_names = build_dataset(path=classpath, batch_size=batch_size, tile=tile)

    # Instantiate the model
    model = setup_model(X_train, class_names, weights_file=os.path.join(log_dir, weights_file), setting=setting)

    save_best_only = (val_split > 1e-6)
    checkpoint = ModelCheckpoint(filepath=os.path.join(log_dir, weights_file),
                                 monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='max')

    history_callback = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
          verbose=0, callbacks=[checkpoint], validation_split=val_split)  # validation_data=(X_val, Y_val),
    pandas.DataFrame(history_callback.history).to_csv(os.path.join(log_dir, "history.csv"))
    model.save(os.path.join(log_dir, 'model.h5'))

if __name__ == '__main__':
    train_network(weights_file=FLAGS.weights, classpath=classpath, epochs=num_epoch, batch_size=batch_size,
            val_split=RATIO, tile=TILE, setting=model_list.index(SETTING))
