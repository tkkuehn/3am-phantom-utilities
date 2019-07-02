#!/usr/bin/python3

import argparse
import sys
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dki as dki

def load_data(nifti_path, bval_path, bvec_path, mask_path=None,
        b0_threshold=250):
    """Load the data needed to process a diffusion-weighted image.

    Parameters
    ----------
    nifti_path : string
        Path to the nifti DWI
    bval_path : string
        Path to the .bval file
    bvec_path : string
        Path to the .bvec file
    mask_path : string, optional
        Path to the nifti mask, if one exists
    b0_threshold
        Threshold below which a b-value is considered zero

    Returns
    -------
    img
        image data
    gtab
        diffusion gradient information
    mask or None
        mask data, if a mask path was provided
    """

    mask = None
    if mask_path is not None:
        mask = nib.load(mask_path)

    img = nib.load(nifti_path)
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    return (img, gtab, mask)

def fit_dki(dkimodel, img, mask = None):
    """Fit a DKI model to a DWI, applying a mask if provided.

    Parameters
    ---------
    dkimodel
        A dki model derived from the scan parameters
    img
        DWI data to fit to the model
    mask, optional
        A mask isolating the data of interest

    Returns
    -------
    dwifit
        A fit from which parameter maps can be generated
    """

    mask_data = None
    if mask is not None:
        mask_data = mask.get_data()

    return dkimodel.fit(img.get_data(), mask_data)

def save_image(data, affine, header, output_path):
    """Save some data to a nifti file

    Parameters
    ----------
    data
        The image data to be saved
    affine
        The affine transform to be used
    header
        The nifti header to be used
    output_path : string
        Path to the file to be saved 
    """
    new_img = nib.nifti1.Nifti1Image(data, affine,
            header=header)
    nib.save(new_img, output_path)

def main(nifti_path, bval_path, bvec_path, mask_path=None, mk_path=None,
        ak_path=None, rk_path=None):
    """Load and fit an image to a DKI model, then save its parameters.

    This is meant to deal with the functionality of this module being called as
    a script.

    Parameters
    ----------
    nifti_path : string
        Path to the nifti DWI
    bval_path : string
        Path to the .bval file
    bvec_path : string
        Path to the .bvec file
    mask_path : string, optional
        Path to the nifti mask, if one exists
    mk_path : string, optional
        Path to which the mean kurtosis image should be saved
    ak_path : string, optional
        Path to which the axial kurtosis image should be saved
    rk_path : string, optional
        Path to which the radial kurtosis image should be saved
    """

    img, gtab, mask = load_data(nifti_path, bval_path, bvec_path, mask_path)

    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = fit_dki(dkimodel, img, mask)

    # Should think about theoretical min and max kurtosis values for us
    mk = dkifit.mk()
    ak = dkifit.ak()
    rk = dkifit.rk()

    source_affine = img.affine
    source_header = img.header
    if mk_path is not None:
        save_image(mk, source_affine, source_header, mk_path)

    if ak_path is not None:
        save_image(ak, source_affine, source_header, ak_path)

    if rk_path is not None:
        save_image(rk, source_affine, source_header, rk_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Load a nifti image and fit the data to a DKI model.')
    parser.add_argument('nifti')
    parser.add_argument('bval')
    parser.add_argument('bvec')
    parser.add_argument('--mask')
    parser.add_argument('--mk')
    parser.add_argument('--ak')
    parser.add_argument('--rk')
    args = parser.parse_args()
    main(args.nifti, args.bval, args.bvec, args.mask, args.mk, args.ak, args.rk)

