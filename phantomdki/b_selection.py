import numpy as np

def get_acquisitions_by_bval(bvals, lower, upper):
    return np.logical_and(bvals >= lower, bvals < upper)

def spherical_distance(theta1, phi1, theta2, phi2):
    return np.arccos((np.cos(theta1) * np.cos(theta2))
            + (np.sin(theta1) * np.sin(theta2) * np.cos(phi1 - phi2)))

def get_acquisitions_by_dir(bvecs, phi, theta, tolerance):
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

