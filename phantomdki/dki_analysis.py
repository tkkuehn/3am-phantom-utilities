#!/usr/bin/python3

"""Summarize voxel-based data files.

The idea here is to produce descriptive statistics for a set of related 3D data maps and plot those statistics. Especially when used as a command line script, each image should contain the data for one point, and all the images should need the same slicing.
"""

import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def prepare_image(image_path, mask_path=None, x_slice=slice(None, None),
        y_slice=slice(None, None), z_slice=slice(None, None)):
    """Load, slice, and mask an image.

    Parameters
    ----------
    image_path
        path from which to load image.

    masks, optional
        path from which to load mask.

    x_slice, y_slice, z_slice : slice, optional
        slices of the image to consider.

    Returns
    -------
    prepared_image
        image ready for description.
    """
    image = nib.load(image_path)

    mask = []
    if mask_path is not None:
        mask = nib.load(mask_path).get_data()
    else:
        mask = np.ones(image.get_data().shape)

    sliced_image = image.get_data()[x_slice, y_slice, z_slice]
    sliced_mask = mask[x_slice, y_slice, z_slice]

    return sliced_image[sliced_mask.nonzero()]

def plot_single(ax, x_data, y_data, y_error, **param_dict):
    out = ax.errorbar(x_data, y_data, yerr=y_error, fmt='.-b', **param_dict)
    return out

def main(image_paths, mask_paths, x_slice, y_slice, z_slice):
    """Compute statistics from a set of images and plot them.

    This is meant to facilitate running this module as a script.

    Parameters
    ----------
    image_paths
        collection of paths from which to load images.

    mask_paths
        collection of paths from which to load masks.

    x_slice, y_slice, z_slice : slice
        slices of the images to consider.

    """
    prepared_images = [prepare_image(image_path, mask_path, x_slice, y_slice,
            z_slice) for image_path, mask_path in zip(image_paths, mask_paths)]
    means = [np.mean(prepared_image) for prepared_image in prepared_images]
    stds = [np.std(prepared_image) for prepared_image in prepared_images]

    fig, ax = plt.subplots()
    x_data = list(range(len(means)))

    plot_single(ax, x_data, means, stds)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Load a nifti of voxel data and produce descriptive statistics.')
    parser.add_argument('images', nargs='+')
    parser.add_argument('--masks', nargs='+')
    parser.add_argument('-x', nargs=2, type=int, default=[None, None])
    parser.add_argument('-y', nargs=2, type=int, default=[None, None])
    parser.add_argument('-z', nargs=2, type=int, default=[None, None])
    parser.add_argument('--xlabel', default='')
    parser.add_argument('--ylabel', default='')
    parser.add_argument('--xvals', nargs="*", type=float)
    args = parser.parse_args()

    main(args.images, mask_paths=args.masks,
            x_slice=slice(args.x[0], args.x[1]),
            y_slice=slice(args.y[0], args.y[1]),
            z_slice=slice(args.z[0], args.z[1]))

