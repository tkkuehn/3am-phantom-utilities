import numpy as np
import phantomdki.fit_dki as dki
import matplotlib.pyplot as plt

B0_THRESHOLD = 250

img = dki.load_dwi(
        'resources/paper_phantoms/20190308/dwi_bottom_4.nii.gz',
        'resources/paper_phantoms/20190308/dwi_bottom_4.bval',
        'resources/paper_phantoms/20190308/dwi_bottom_4.bvec',
        mask_path='resources/paper_phantoms/20190308/mask_bottom_4_slice_2.nii.gz',
        b0_threshold=B0_THRESHOLD)

slice_2_data = img.getImage()[:, :, 2, :]
slice_2_bvals = img.gtab.bvals
slice_2_bvecs = img.gtab.bvecs

b0_acquisitions = slice_2_bvals < 250
b1000_acquisitions = np.logical_and(
        slice_2_bvals < 1500,
        np.logical_not(b0_acquisitions))
b2000_acquisitions = np.logical_and(
        np.logical_not(b0_acquisitions),
        np.logical_not(b1000_acquisitions))

x_acquisitions = np.abs(slice_2_bvecs[:, 0]) > 0.85
x_b1000_acquisitions = np.logical_and(
        x_acquisitions,
        b1000_acquisitions)
x_b2000_acquisitions = np.logical_and(
        x_acquisitions,
        b2000_acquisitions)

mask = slice_2_data[:, :, 0] > 0

slice_2_b0 = slice_2_data[:, :, b0_acquisitions]
slice_2_x_b1000 = slice_2_data[:, :, x_b1000_acquisitions]
slice_2_x_b2000 = slice_2_data[:, :, x_b2000_acquisitions]
b = np.array([0, 1000, 2000])

signal = np.array([np.mean(slice_2_b0[mask, :]),
        np.mean(slice_2_x_b1000[mask, :]),
        np.mean(slice_2_x_b2000[mask, :])])

plt.plot(b, signal, 'bo-')
plt.yscale('log')
plt.show()

