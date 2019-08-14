"""Enable transformation of data for comparison to ground truth.

If we know the infill pattern of a given phantom, we know a few things
about the geometry of diffusion in that phantom. This module provides a
way to compare scan data to that known information
"""

import numpy as np

def transform_image_point(point, centroid, angle):
    """Perform a rigid transform of a given point.

    The infill pattern definitions assume the origin is at the centroid
    of the phantom. This is never the case for scan data, so to compare
    scan data to a ground truth, we need to perform a rigid
    transformation to that ground truth space.

    Parameters
    ----------
    point : tuple of int
        The indices of the point to be transformed in image space
    centroid : tuple of int
        The indices of the phantom's centroid in image space
    angle : float
        The angle by which the phantom would need to be rotated to have
        the fiducial at the bottom in the x-y plane

    Returns
    -------
    tuple of float
        The corresponding indices of the original point in ground truth
        space.
    """
    translated_point = np.array(point) - np.array(centroid)

    theta = np.radians(angle)
    rotation = np.array([[np.cos(theta), -1 * np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])

    rotated_point = rotation @ np.array([translated_point]).T

    return (rotated_point[0, 0], rotated_point[1, 0])

def compare_to_pattern(img, pattern, truth_from_pattern, centroid, angle):
    """Compare scan data to some ground truth.

    Parameters
    ----------
    img : Image
        3D image (from image_io) to be analyzed.
    pattern : Pattern
        Pattern (from scan_info) to provide ground truth.
    truth_from_pattern : function(pattern, tuple of float)
        Function to get the quantity of interest from a pattern.
    centroid : tuple of int
        Centroid of the phantom in image space.
    angle : float
        The angle by which the phantom would need to be rotated to have
        the fiducial at the bottom in the x-y plane.

    Returns
    -------
    tuple of array_like
        An array of ground truth values and an array of the
        corresponding values from the scan data
    """
    independent = []
    dependent = []

    data = img.getImage()
    try:
        mask = img.mask
    except AttributeError:
        mask = np.ones(data.shape)

    for z in range(data.shape[2]):
        for (d_idx, d_val), (m_idx, m_val)  in zip(
                np.ndenumerate(data[..., z]), np.ndenumerate(mask[..., z])):
            if m_val:
                transformed = transform_image_point(
                        d_idx, centroid, angle)
                ground_truth = truth_from_pattern(pattern, transformed)
                independent.append(ground_truth)
                dependent.append(d_val)

    return np.array(independent), np.array(dependent)

