import tensorflow as tf
import numpy as np
import os
import math
import glob
from scipy import ndimage, misc
from data.prepare_cifar import read_h5

# you need to change this to your data directory
train_dir = 'data/train/'

def load_data(img_dirs, h5_filename="data.h5"):
    f = os.path.join(img_dirs, h5_filename)
    data, label = read_h5(f)
    return data.value, label.value

def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
            data: B,... numpy array
            label: B, numpy array
        Return:
            shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

if __name__ == "__main__":
    f = os.path.join("data/quality_0", "train.h5")
    data, label = read_h5(f)
    print (data.value.shape, label.value.shape)

