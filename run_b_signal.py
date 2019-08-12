import numpy as np
import phantomdki.fit_dki as dki
from phantomdki.b_selection import get_acquisitions_by_bval, spherical_distance
from phantomdki.b_selection import get_acquisitions_by_dir
import dipy.reconst.dki
import matplotlib.pyplot as plt

B0_THRESHOLD = 250

img = dki.load_dwi(
        'resources/paper_phantoms/20190308/dwi_bottom_4.nii.gz',
        'resources/paper_phantoms/20190308/dwi_bottom_4.bval',
        'resources/paper_phantoms/20190308/dwi_bottom_4.bvec',
        mask_path='resources/paper_phantoms/20190308/mask_bottom_4_slice_2.nii.gz',
        b0_threshold=B0_THRESHOLD)

# set up data
slice_2_data = img.getImage()[:, :, 2, :]
slice_2_bvals = img.gtab.bvals
slice_2_bvecs = img.gtab.bvecs
mask = img.mask[:, :, 2] > 0

# get acquisitions by bval
b0_acquisitions = get_acquisitions_by_bval(slice_2_bvals, 0, 250)
b1000_acquisitions = get_acquisitions_by_bval(slice_2_bvals, 250, 1500)
b2000_acquisitions = get_acquisitions_by_bval(slice_2_bvals, 1500, 2500)

# get acquisitions by direction
x_acquisitions = get_acquisitions_by_dir(slice_2_bvecs, 0, 90, 22.5)
y_acquisitions = get_acquisitions_by_dir(slice_2_bvecs, 90, 90, 22.5)
z_acquisitions = get_acquisitions_by_dir(slice_2_bvecs, 0, 0, 22.5)

dkimodel = dipy.reconst.dki.DiffusionKurtosisModel(img.gtab)
dkifit = dki.fit_dki(dkimodel, img)
principal_dir = np.mean(dkifit.directions[mask, 2, :, :], axis=(0, 1))

dir_phi = np.degrees(np.arctan2(principal_dir[1], principal_dir[0]))
dir_r = np.linalg.norm(principal_dir)
dir_theta = np.degrees(np.arccos(principal_dir[2] / dir_r))
dir_acquisitions = get_acquisitions_by_dir(slice_2_bvecs, dir_phi, dir_theta, 22.5)

# combine bvals and directions we want
dir_b1000_acquisitions = np.logical_and(
        dir_acquisitions,
        b1000_acquisitions)
dir_b2000_acquisitions = np.logical_and(
        dir_acquisitions,
        b2000_acquisitions)
z_b1000_acquisitions = np.logical_and(
        z_acquisitions,
        b1000_acquisitions)
z_b2000_acquisitions = np.logical_and(
        z_acquisitions,
        b2000_acquisitions)

# access the data of interest
slice_2_b0 = slice_2_data[:, :, b0_acquisitions]
slice_2_dir_b1000 = slice_2_data[:, :, dir_b1000_acquisitions]
slice_2_dir_b2000 = slice_2_data[:, :, dir_b2000_acquisitions]
slice_2_z_b1000 = slice_2_data[:, :, z_b1000_acquisitions]
slice_2_z_b2000 = slice_2_data[:, :, z_b2000_acquisitions]
b = np.array([0, 1000, 2000])
isotropic = np.mean(slice_2_b0[mask, :]) * np.exp(-1 * b * 1.69 * 1000 / 1000000)
low_diffusivity = np.mean(slice_2_b0[mask, :]) * np.exp(-1 * b * 0.6 * 1000 / 1000000)

dir_signal = np.array([np.mean(slice_2_b0[mask, :]),
        np.mean(slice_2_dir_b1000[mask, :]),
        np.mean(slice_2_dir_b2000[mask, :])])
dir_std = np.array([np.std(slice_2_b0[mask, :]),
        np.std(slice_2_dir_b1000[mask, :]),
        np.std(slice_2_dir_b2000[mask, :])])

z_signal = np.array([np.mean(slice_2_b0[mask, :]),
    np.mean(slice_2_z_b1000[mask, :]),
    np.mean(slice_2_z_b2000[mask, :])])
z_std = np.array([np.std(slice_2_b0[mask, :]),
    np.std(slice_2_z_b1000[mask, :]),
    np.std(slice_2_z_b2000[mask, :])])

plt.errorbar(b, dir_signal, yerr=dir_std, fmt='bo-', label='Axial')
plt.errorbar(b, z_signal, yerr=z_std, fmt='ro-', label='Radial')
plt.plot(b, isotropic, 'ko-', label='S0 * e^-bD (Axial)')
plt.plot(b, low_diffusivity, 'go-', label='S0 * e^-bD (Radial)')
plt.xlabel('b-value')
plt.ylabel('signal')
plt.yscale('log')
plt.legend()
plt.show()

