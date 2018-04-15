import numpy as np
import h5py
from scipy import ndimage, misc
import glob
import os

quality_list = [1, 5, 10, 15, 20, 25, 50, 75, 100]

img_dirs = ["quality_" + str(i) for i in quality_list]

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_data(img_dirs, image_W=None, image_H=None):
    images = []
    labels = []

    files = glob.glob(os.path.join(img_dirs, "*.jpg"))

    for i, filepath in enumerate(files):
        name = filepath.split('/')[-1].split(".")[0]
        label = label_names.index(name.split('_')[0])
        labels.append(label)
        image = ndimage.imread(filepath, mode="RGB")
        if not image_W is None and image_H is None:
            image_resized = misc.imresize(image, (image_W, image_H)) / 255.0
        else:
            image_resized = image / 255.0
        image_float32 = image_resized.astype('float32')
        images.append(image_float32)
        # if i == 999:
        #     break

    images = np.stack(images, axis = 0)
    labels = np.asarray(labels)
    print images.shape, labels.shape
    return images, labels


    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    # Train
    with h5py.File(train_filename, 'w') as f:
        f.create_dataset('data', data=X, **comp_kwargs)
        f.create_dataset('label', data=y.astype(np.int_), **comp_kwargs)
    with open(os.path.join(cifar_caffe_directory, 'train.txt'), 'w') as f:
        f.write(train_filename + '\n')
    # Test
    with h5py.File(test_filename, 'w') as f:
        f.create_dataset('data', data=Xt, **comp_kwargs)
        f.create_dataset('label', data=yt.astype(np.int_), **comp_kwargs)
    with open(os.path.join(cifar_caffe_directory, 'test.txt'), 'w') as f:
        f.write(test_filename + '\n')


def save_h5(data, label, filename):
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    with h5py.File(filename, 'w') as f:
        f.create_dataset('data', data=data, **comp_kwargs)
        f.create_dataset('label', data=label, **comp_kwargs)

def read_h5(filename):
    f = h5py.File(filename,'r')
    data, label = f['data'], f['label']
    return data, label


if __name__ == "__main__":
    print img_dirs

    for img_dir in img_dirs:
        train_dir, test_dir = os.path.join(img_dir, "train"), os.path.join(img_dir, "test")
        train_data, train_label = load_data(train_dir)
        test_data, test_label = load_data(test_dir)
        save_h5(train_data, train_label, os.path.join(img_dir, "train.h5"))
        save_h5(test_data, test_label, os.path.join(img_dir, "test.h5"))
        # data, label = read_h5(os.path.join(img_dir, "data.h5"))
        # print label.value
        print img_dir
        # break
    # print data[0,0,:], label[0]
