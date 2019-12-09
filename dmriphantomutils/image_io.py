"""Wrappers for saving and loading DWIs of 3D printed phantoms.

In this module, images are classified as DWIs or derived images. A DWI
should be the raw 4D data from a diffusion MRI scan, and have associated
information about the diffusion gradients. A derived image should be
(usually) 3D data from analysis of a DWI.

Either of the two may have a mask associated with them.
"""

from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import nibabel as nib
import numpy as np

class DiffusionWeightedImage:
    """Wrapper class including image and gradient data.

    Parameters
    ----------
    img : SpatialImage
        The NiBabel image of the DWI
    gtab : GradientTable
        The DIPY gradient table associated with the scan
    """

    def __init__(self, img, gtab):
        self.img = img
        self.gtab = gtab

    def get_image(self):
        """A 3D numpy array with the image data."""
        return self.img.get_data()

    def get_flat_data(self):
        """A 1D numpy array with the image data."""
        return self.img.get_data().flatten()

class MaskedDiffusionWeightedImage(DiffusionWeightedImage):
    """Wrapper class including an image, mask, and gradient data.

    Parameters
    ----------
    img : SpatialImage
        The NiBabel image of the DWI
    gtab : GradientTable
        The DIPY gradient table associated with the scan
    mask : array_like
        A binary numpy array, where 1s indicate voxels to be included.
    """

    def __init__(self, img, gtab, mask):
        DiffusionWeightedImage.__init__(self, img, gtab)

        img_data = self.img.get_data()
        if len(mask.shape) >= len(img_data.shape):
            mask = mask[..., 0]

        self.mask = mask
        self.data = np.ma.array(img_data, mask=np.repeat(
            np.logical_not(mask[:, :, :, np.newaxis]),
            img_data.shape[3], axis=3))

    def get_image(self):
        """A 3D numpy array with the image data, ignoring the mask."""

        return self.data.data

    def get_flat_data(self):
        """A 1D numpy array with only the masked data."""
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
    img : DiffusionWeightedImage
        The DWI data with a mask, if applicable.
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
    """Wrapper class including an image of data derived from a DWI.

    Parameters
    ----------
    img : SpatialImage
        The NiBabel image of the derived data.
    """

    def __init__(self, img):
        self.img = img

    def get_image(self):
        """A 3D numpy array with the image data."""

        return self.img.get_data()

    def get_flat_data(self):
        """A 1D numpy array with the image data."""

        return self.img.get_data().flatten()

class MaskedDerivedImage(DerivedImage):
    """Wraps an image of data derived from a DWI with a mask.

    Parameters
    ----------
    img : SpatialImage
        The NiBabel image of the derived data.
    mask : array_like
        A 3D binary numpy array, where 1s indicate voxels to be included.
    """

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

    def get_image(self):
        """A 3D numpy array with the derived data, ignoring the mask."""

        return self.data.data

    def get_flat_data(self):
        """A 1D numpy array with the masked derived data."""

        return self.data.compressed()

def load_derived_image(image_path, mask_path=None):
    """Load the data from a derived image.

    Parameters
    ----------
    image_path : string
        Path to the nifti derived data volume.
    mask_path : string, optional
        Path to the nifti mask, if one exists

    Returns
    -------
    img : DerivedImage
        The derived data with a mask, if applicable.
    """

    img = nib.load(image_path)

    if mask_path is not None:
        mask = nib.load(mask_path)
        return MaskedDerivedImage(img, mask.get_data())
    else:
        return DerivedImage(img)

def save_image(data, affine, header, output_path):
    """Save some data to a nifti file.

    Parameters
    ----------
    data : array_like
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

