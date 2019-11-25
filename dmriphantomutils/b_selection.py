"""Filter a set of gradients by direction and b-val."""

import numpy as np

def get_acquisitions_by_bval(bvals, lower, upper):
    """Get b-vals within a certain range.

    Parameters
    ----------
    bvals : array_like
        A 1D array of b-values to be filtered.
    lower : float
        The minimum b-value to include.
    upper : float
        The maximum b-value to include.

    Returns
    -------
    array_like
        Logical index array for b-values to include.
    """

    return np.logical_and(bvals >= lower, bvals < upper)

def spherical_distance(theta1, phi1, theta2, phi2):
    """Get the spherical distance between two points on the unit sphere.

    Parameters
    ----------
    theta1, theta2 : float
        Polar angles of the two points, between 0 and pi.
    phi1, phi1 : float
        Azimuthal angles of the two points, between -pi and pi.

    Returns
    -------
    float
        Spherical distance between the given points.
    """

    return np.arccos((np.cos(theta1) * np.cos(theta2))
            + (np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)))

def get_acquisitions_by_dir(bvecs, phi, theta, tolerance):
    """Get b-vectors close to a given direction.

    Parameters
    ----------
    bvecs : array_like
        2D array of b-vectors to be filtered.
    phi : float
        Azimuthal angle of the direction, between 0 and 180.
    theta : float
        Polar angle of the point, between -180 and 180.
    tolerance : float
        b-vecs with spherical distance less than this tolerance will be
        included.

    Returns
    -------
    array_like
        Index array of the b-vectors to be included.
    """

    phi = np.radians(phi)
    theta = np.radians(theta)
    tolerance = np.radians(tolerance)

    antipodal_theta = (-1 * (theta - (np.pi / 2))) + (np.pi / 2)
    if phi > 0:
        antipodal_phi = phi - np.pi
    else:
        antipodal_phi = phi + np.pi

    bvec_phi = np.arctan2(bvecs[:, 1], bvecs[:, 0])
    bvec_r = np.linalg.norm(bvecs, axis=1)
    bvec_theta = np.arccos(bvecs[:, 2] / bvec_r)

    target_dist = spherical_distance(theta, phi, bvec_theta, bvec_phi)
    antipodal_dist = spherical_distance(antipodal_theta, antipodal_phi,
            bvec_theta, bvec_phi)

    dist = np.minimum(target_dist, antipodal_dist)

    return dist < tolerance

