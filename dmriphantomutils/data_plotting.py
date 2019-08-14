#!/usr/bin/python3

"""Summarize voxel-based data files.

The idea here is to produce descriptive statistics for a set of related 3D data maps and plot those statistics. Especially when used as a command line script, each image should contain the data for one point, and all the images should need the same slicing.
"""

import argparse

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

import image_io

def plot_single(ax, x_data, y_data, y_error, **param_dict):
    out = ax.errorbar(x_data, y_data, yerr=y_error, fmt='.-b', **param_dict)
    return out

def main(image_paths, mask_paths, xvals=None, xlabel="", ylabel=""):
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
        images = [image_io.load_derived_image(image_path)
                for image_path in image_paths]
    else:
        images = [image_io.load_derived_image(image_path, mask_path)
                for image_path, mask_path in zip(image_paths, mask_paths)]

    means = [np.mean(image.getFlatData()) for image in images]
    stds = [np.std(image.getFlatData()) for image in images]

    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xvals is not None:
        x_data = xvals
    else:
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

    main(args.images, mask_paths=args.masks, xlabel=args.xlabel,
            ylabel=args.ylabel, xvals=args.xvals)

