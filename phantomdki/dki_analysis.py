#!/usr/bin/python3

"""Summarize voxel-based data files.

The idea here is to produce descriptive statistics for a set of related 3D data maps and plot those statistics. Especially when used as a command line script, each image should contain the data for one point, and all the images should need the same slicing.
"""

import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

class DerivedImage():
    def __init__(self, img):
        self.img = img

    def getImage(self):
        return self.img.get_data()

    def getFlatData(self):
        return self.img.get_data().flatten()

class MaskedDerivedImage(DerivedImage):
    def __init__(self, img, mask):
        DerivedImage.__init__(self, img)
        self.mask = mask
        img_data = self.img.get_data()
        self.data = np.ma.array(img_data, mask=~mask)

    def getImage(self):
        return self.data.data

    def getFlatData(self):
        return self.data.compressed()

def load_derived_image(image_path, mask_path=None):
    img = nib.load(image_path)

    if mask_path is not None:
        mask = nib.load(mask_path)
        return MaskedImage(img, mask.get_data())
    else:
        return DerivedImage(img)

def plot_single(ax, x_data, y_data, y_error, **param_dict):
    out = ax.errorbar(x_data, y_data, yerr=y_error, fmt='.-b', **param_dict)
    return out

def main(image_paths, mask_paths):
    """Compute statistics from a set of images and plot them.

    This is meant to facilitate running this module as a script.

    Parameters
    ----------
    image_paths
        collection of paths from which to load images.

    mask_paths
        collection of paths from which to load masks.
    """
    if mask_paths is not None:
        images = [load_derived_image(image_path) for image_path in image_paths]
    else:
        images = [load_derived_image(image_path, mask_path)
                for image_path, mask_path in zip(image_paths, mask_paths)]

    means = [np.mean(image.getFlatData()) for image in images]
    stds = [np.std(image.getFlatData()) for image in images]

    fig, ax = plt.subplots()
    x_data = list(range(len(means)))

    plot_single(ax, x_data, means, stds)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Load a nifti of voxel data and produce descriptive statistics.')
    parser.add_argument('images', nargs='+')
    parser.add_argument('--masks', nargs='*')
    parser.add_argument('--xlabel', default='')
    parser.add_argument('--ylabel', default='')
    parser.add_argument('--xvals', nargs="*", type=float)
    args = parser.parse_args()

    main(args.images, mask_paths=args.masks)

