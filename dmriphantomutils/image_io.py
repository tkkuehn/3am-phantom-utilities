from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import nibabel as nib
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

        img_data = self.img.get_data()
        if len(mask.shape) >= len(img_data.shape):
            mask = mask[..., 0]

        self.mask = mask
        self.data = np.ma.array(img_data, mask=np.repeat(
            np.logical_not(mask[:, :, :, np.newaxis]),
            img_data.shape[3], axis=3))

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
        img_data = self.img.get_data()

        if len(img_data.shape) > 3:
            img_data = img_data[..., 0]

        # mask is often 4D for raw data
        if len(mask.shape) > len(img_data.shape):
            mask = mask[..., 0]

        self.mask = mask
        self.data = np.ma.array(img_data, mask=~mask)

    def getImage(self):
        return self.data.data

    def getFlatData(self):
        return self.data.compressed()

def load_derived_image(image_path, mask_path=None):
    img = nib.load(image_path)

    if mask_path is not None:
        mask = nib.load(mask_path)
        return MaskedDerivedImage(img, mask.get_data())
    else:
        return DerivedImage(img)

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

