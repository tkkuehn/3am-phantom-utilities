#!/usr/bin/python3

import argparse
import nibabel as nib
import numpy as np
import skimage.filters as filters
import skimage.morphology as mm

img = nib.load('../resources/paper_phantoms/20190308/dwi_bottom_4.nii.gz')

def mask_bubbles(slice_b0):
    slice_b0_threshold = filters.threshold_otsu(slice_b0)
    slice_phantom_mask = slice_b0 > slice_b0_threshold
    slice_phantom_data = slice_b0[mm.binary_erosion(
        slice_phantom_mask, mm.disk(3)).nonzero()]
    slice_bubble_threshold = filters.threshold_otsu(slice_phantom_data)
    return slice_b0 > slice_bubble_threshold

def main(nifti_path, slice_idx, mask_output):
    img = nib.load(nifti_path)
    img_data = img.get_data()
    img_b0 = img_data[:, :, :, 0]

    slice_b0 = img_b0[:, :, slice_idx]
    slice_bubble_mask = mask_bubbles(phantom_b0)

    out_data = np.zeros(img_b0.shape)
    out_data[:, :, slice_idx] = slice_bubble_mask

    mask_img = nib.nifti1.Nifti1Image(out_data, img.affine, header=img.header)
    nib.save(mask_img, mask_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Load a nifti image and mask the bubbles from one slice.')
    parser.add_argument('nifti')
    parser.add_argument('z_slice', type=int)
    parser.add_argument('output')

    args = parser.parse_args()

    main(args.nifti, args.z_slice, args.output)

