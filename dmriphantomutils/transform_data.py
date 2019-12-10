"""Enable transformation of data for comparison to ground truth.

If we know the infill pattern of a given phantom, we know a few things
about the geometry of diffusion in that phantom. This module provides a
way to compare scan data to that known information.

Specifically, we define a "ground truth space," where the centre of the
phantom is at the origin, and a fiducial visible from the image is on the
negative y-axis. Then a translation and rotation can move each voxel's
coordinates from image space to ground truth space.
"""

import warnings

import numpy as np
from skimage.morphology import binary_closing, disk
from skimage.measure import label, regionprops

def find_centroid(mask):
    """Find the centroid of a phantom's mask.

    Parameters
    ----------
    mask : array_like
        A 2D binary array, where 1s indicate voxels containing a
        phantom.

    Returns
    -------
    array_like
        A 1D array containing the coordinates of the mask's centroid.
    """

    # Consider voxels that were masked out due to air bubbles etc.
    closed_mask = label(binary_closing(mask, disk(6)))
    return regionprops(closed_mask)[0].centroid

def transform_image_point(point, centroid, angle=None, fiducial=None):
    """Perform a rigid transform of a given point.

    The infill pattern definitions assume the origin is at the centroid
    of the phantom. This is never the case for scan data, so to compare
    scan data to a ground truth, we need to translate the image data to
    move the phantom's centroid to the origin, and rotate it to align a
    fiducial to the known ground truth.

    Parameters
    ----------
    point : tuple of int
        The indices of the point to be transformed in image space
    centroid : tuple of float
        The indices of the phantom's centroid in image space
    angle : float, optional
        The angle by which the phantom would need to be rotated to have
        the fiducial at the bottom in the x-y plane. Exactly one of
        angle or fiducial must be included as an argument.
    fiducial : tuple of float, optional
        The location of the fiducial in image space. Exactly one of
        fiducial or angle must be included as an argument.

    Returns
    -------
    tuple of float
        The corresponding indices of the original point in ground truth
        space.
    """

    if angle is not None and fiducial is not None:
        # Should only provide one. Default to using angle...
        warnings.warn('Both angle ({}) and fiducial ({}) were provided to'
            + 'transform_image_point. Defaulting to angle.'.format(
                angle, fiducial))
    elif fiducial is not None:
        translated_fiducial = np.array(fiducial) - np.array(centroid)
        angle = -90 - np.degrees(
            np.arctan2(translated_fiducial[1], translated_fiducial[0]))
    elif angle is None:
        raise TypeError("Exactly one of angle or fiducial is required.")

    translated_point = np.array(point) - np.array(centroid)

    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -1 * np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    rotated_point = rotation @ np.array([translated_point]).T

    return (rotated_point[0, 0], rotated_point[1, 0])

def gen_geometry_data(mask_data, geometry_generator, centroid, scaling,
        angle=None, fiducial=None):
    """Generate geometric ground truth data from a mask.

    Parameters
    ----------
    mask_data : array_like 
        3D mask of points to be analyzed.
    geometry_generator : function(tuple of float)
        Function to get the quantity of interest given a point.
    centroid : tuple of int
        Centroid of the phantom in image space.
    scaling : float
        Isotropic scale factor from coords to image space.
    angle : float, optional
        The angle by which the phantom would need to be rotated to have
        the fiducial at the bottom in the x-y plane. Exactly one of
        angle or fiducial must be included as an argument.
    fiducial : tuple of float, optional
        The location of the fiducial in image space. Exactly one of
        fiducial or angle must be included as an argument.

    Returns
    -------
    array_like
        An image of the calculated geometry data
    """

    use_fiducial = False
    if angle is not None and fiducial is not None:
        # Should only provide one. Default to using angle...
        warnings.warn('Both angle ({}) and fiducial ({}) were provided to'
            + 'gen_geometry_data. Defaulting to angle.'.format(
                angle, fiducial))
        fiducial = None
    elif fiducial is not None:
        use_fiducial = True
    elif angle is None:
        raise TypeError("Exactly one of angle or fiducial is required.")

    geometry_data = np.zeros(mask_data.shape)

    for z in range(mask_data.shape[2]):
        for m_idx, m_val in np.ndenumerate(mask_data[..., z]):
            if m_val:
                if use_fiducial:
                    transformed = transform_image_point(
                        m_idx, centroid, fiducial=fiducial)
                else:
                    transformed = transform_image_point(
                        m_idx, centroid, angle=angle)
                transformed = (transformed[0] * scaling,
                               transformed[1] * scaling)
                ground_truth = geometry_generator(transformed)
                geometry_data[m_idx[0], m_idx[1], z] = ground_truth

    return geometry_data

