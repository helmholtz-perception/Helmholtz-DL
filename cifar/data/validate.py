import h5py
from PIL import Image
import os
import numpy as np

quality_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 50, 75]

label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
num_counts = [0] * 10

def restore_images(h5_filename, out_dirs, quality = 15):
    if not os.path.exists(out_dirs):
        os.makedirs(out_dirs)
    h5_file = h5py.File(h5_filename, "r")
    images = h5_file['data'].value
    labels = h5_file['label'].value
    # print images.shape[0]
    for i in range(images.shape[0]):
        b = images[i, ...]
        img = Image.fromarray(images[i, ...])
        label = labels[i]
        num_counts[label] += 1
        outname = label_names[label] + "_" + str(num_counts[label]) + ".jpg"
        img.save(os.path.join(out_dirs, outname), quality=quality)
        new = Image.open(os.path.join(out_dirs, outname), mode='r')
        a = np.asarray(new)
        print (b == a).shape
        print np.sum(b == a)
        if i == 5:
            break

if __name__ == "__main__":
    print quality_list
    for quality in [1, 5, 10, 25, 50, 75, 100]:
        restore_images("cifar_10_h5/train.h5", "test/quality_" + str(quality) + "/train", quality)
        # restore_images("cifar_10_h5/test.h5", "quality_" + str(quality) + "/test")
        # print quality

    print num_counts
