#!/usr/bin/python3

"""Automatically generate masks for a phantom in a scan."""

import argparse

import nibabel as nib
import numpy as np
import skimage.filters as filters
import skimage.morphology as mm

def mask_phantom(slice_b0):
    """Generate a mask for the phantom material in a slice.

    Assuming the phantom's diameter is similar to the test tube's,
    Otsu's method segments the phantom well. The initial results are
    eroded to avoid outliers near the edges of the phantom.

    Big enough air bubbles may also be masked out by this function.

    Parameters
    ----------
    slice_b0 : array_like
        2D array of image data from a single slice (containing a single
        phantom).

    Returns
    -------
    array_like
        2D boolean array, where True values indicate phantom voxels.
    """
    slice_b0_threshold = filters.threshold_otsu(slice_b0)
    slice_phantom_mask = slice_b0 > slice_b0_threshold
    return mm.binary_erosion(slice_phantom_mask, mm.disk(3))

def main(nifti_path, slice_idx, mask_output):
    """Load an image and mask one z-slice.

    Parameters
    ----------
    nifti_path : str
        Path to the image to be masked.
    slice_idx : int
        Index of the slice to be masked.
    mask_output : str
        Path to which the mask image should be saved.
    """
    img = nib.load(nifti_path)
    img_data = img.get_data()
    img_b0 = img_data[:, :, :, 0]

    slice_b0 = img_b0[:, :, slice_idx]
    slice_phantom_mask = mask_phantom(slice_b0)

    out_data = np.zeros(img_b0.shape)
    out_data[:, :, slice_idx] = slice_phantom_mask

    mask_img = nib.nifti1.Nifti1Image(out_data, img.affine, header=img.header)
    nib.save(mask_img, mask_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Load a nifti image and mask the phantom from one slice.')
    parser.add_argument('nifti')
    parser.add_argument('z_slice', type=int)
    parser.add_argument('output')

    args = parser.parse_args()

    main(args.nifti, args.z_slice, args.output)

