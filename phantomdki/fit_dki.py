#!/usr/bin/python3

import argparse
import sys
import nibabel as nib
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import dipy.reconst.dki as dki
import numpy as np

class DiffusionWeightedImage:
    def __init__(self, img, gtab):
        self.img = img
        self.gtab = gtab

    def getImage(self):
        return self.img.get_data()

    def getFlatData(self):
        return self.img.get_data().flatten()

class MaskedDiffusionWeightedImage(DiffusionWeightedImage):
    def __init__(self, img, gtab, mask):
        DiffusionWeightedImage.__init__(self, img, gtab)
        self.mask = mask
        img_data = self.img.get_data()
        self.data = np.ma.array(img_data, mask=np.repeat(
            ~mask[:, :, :, np.newaxis], img_data.shape[3], axis=3))

    def getImage(self):
        return self.data.data

    def getFlatData(self):
        return self.data.compressed()

def load_dwi(nifti_path, bval_path, bvec_path, mask_path=None,
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
    img = nib.load(nifti_path)
    bvals, bvecs = read_bvals_bvecs(bval_path, bvec_path)
    gtab = gradient_table(bvals, bvecs, b0_threshold=b0_threshold)

    if mask_path is not None:
        mask = nib.load(mask_path)
        return MaskedDiffusionWeightedImage(img, gtab, mask.get_data())
    else:
        return DiffusionWeightedImage(img, gtab)

def fit_dki(dkimodel, dwi):
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

    data = dwi.getImage()

    try:
        mask = dwi.mask
    except AttributeError:
        mask = np.ones(data.shape[:3])

    return dkimodel.fit(data, mask)

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

def main(nifti_path, bval_path, bvec_path, mask_path=None,
        fa_path=None, md_path=None, ad_path=None, rd_path=None, mk_path=None,
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
    fa_path : string, optional
        Path to which the fractional anisotropy image should be saved
    md_path : string, optional
        Path to which the mean diffusivity image should be saved
    ad_path : string, optional
        Path to which the axial diffusivity image should be saved
    rd_path : string, optional
        Path to which the radial diffusivity image should be saved
    mk_path : string, optional
        Path to which the mean kurtosis image should be saved
    ak_path : string, optional
        Path to which the axial kurtosis image should be saved
    rk_path : string, optional
        Path to which the radial kurtosis image should be saved
    """

    dwi = load_dwi(nifti_path, bval_path, bvec_path, mask_path)

    dkimodel = dki.DiffusionKurtosisModel(dwi.gtab)
    dkifit = fit_dki(dkimodel, dwi)

    source_affine = dwi.img.affine
    source_header = dwi.img.header

    # Should think about theoretical min and max kurtosis values for us
    if fa_path is not None:
        save_image(dkifit.fa, source_affine, source_header, fa_path)

    if md_path is not None:
        save_image(dkifit.md, source_affine, source_header, md_path)

    if ad_path is not None:
        save_image(dkifit.ad, source_affine, source_header, ad_path)

    if rd_path is not None:
        save_image(dkifit.rd, source_affine, source_header, rd_path)

    if mk_path is not None:
        save_image(dkifit.mk(), source_affine, source_header, mk_path)

    if ak_path is not None:
        save_image(dkifit.ak(), source_affine, source_header, ak_path)

    if rk_path is not None:
        save_image(dkifit.rk(), source_affine, source_header, rk_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
            'Load a nifti image and fit the data to a DKI model.')
    parser.add_argument('nifti')
    parser.add_argument('bval')
    parser.add_argument('bvec')
    parser.add_argument('--mask')
    parser.add_argument('--fa')
    parser.add_argument('--md')
    parser.add_argument('--ad')
    parser.add_argument('--rd')
    parser.add_argument('--mk')
    parser.add_argument('--ak')
    parser.add_argument('--rk')
    args = parser.parse_args()
    main(args.nifti, args.bval, args.bvec, args.mask,
            fa_path=args.fa, md_path=args.md, ad_path=args.ad, rd_path=args.rd,
            mk_path=args.mk, ak_path=args.ak, rk_path=args.rk)

