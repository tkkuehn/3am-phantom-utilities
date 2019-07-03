#!/usr/bin/python3

"""Summarize voxel-based data files.

The idea here is to produce descriptive statistics for a set of related 3D data maps and plot those statistics. Especially when used as a command line script, each image should contain the data for one point, and all the images should need the same slicing.
"""

import argparse
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def prepare_images(image_paths, mask_paths=None, x_slice=slice(None, None),
        y_slice=slice(None, None), z_slice=slice(None, None)):
    """Load, slice, and mask a set of images.

    Parameters
    ----------
    image_paths
        collection of paths from which to load images.

    masks, optional
        collection of paths from which to load masks.

    x_slice, y_slice, z_slice : slice, optional
        slices of the image to consider.

    Returns
    -------
    prepared_images
        collection of images ready for description.
    """
    images = [nib.load(image_path) for image_path in image_paths]

    masks = []
    if mask_paths is not None:
        masks = [nib.load(mask_path).get_data() for mask_path in mask_paths]
    else:
        masks = [np.ones(image.get_data().shape) for image in images]

    sliced_images = [image.get_data()[x_slice, y_slice, z_slice]
            for image in images]
    sliced_masks = [mask[x_slice, y_slice, z_slice]
            for mask in masks]

    return [image[mask.nonzero()] 
            for image, mask in zip(sliced_images, sliced_masks)]

def compute_statistics(prepared_images):
    """Compute descriptive statistics for a set of images.

    Parameters
    ----------
    prepared_images
        Collection of images to be described.

    Returns
    -------
    means
        List of mean values.

    stds
        List of standard deviation values.
    """
    means = [np.mean(prepared_image) for prepared_image in prepared_images]
    stds = [np.std(prepared_image) for prepared_image in prepared_images]
    return (means, stds)

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
        slices of the image to consider.

    """
    prepared_images = prepare_images(image_paths, mask_paths, x_slice, y_slice,
            z_slice)
    means, stds = compute_statistics(prepared_images)

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

