import os.path
import tempfile

from dipy.core.gradients import GradientTable
import numpy as np

class MockDiffusionWeightedImage():
    def __init__(self, data=None, gtab=None, mask=None):
        if data is None:
            data = np.random.uniform(100, 300, size=(50, 50, 3, 100))
        if gtab is None:
            b0s = 20 * np.ones([20, 1])
            b1000s = 1020 * np.ones([80, 1])
            bvals = np.append(b0s, b1000s, 0)
            np.random.shuffle(bvals)

            phis = np.random.uniform(0, 2 * np.pi, (100, 1))
            thetas = np.arccos(np.random.uniform(-1, 1, (100, 1)))
            x = bvals * np.sin(thetas) * np.cos(phis)
            y = bvals * np.sin(thetas) * np.cos(phis)
            z = bvals * np.cos(thetas)
            gradients = np.append(x, np.append(y, z, 1), 1)

            gtab = GradientTable(gradients)
        if mask is None:
            mask = np.ones([50, 50, 3])

        self.data = data
        self.mask = mask
        self.gtab = gtab

    def get_image(self):
        return self.data

    def get_flat_data(self):
        return self.data.flatten()

class MockDerivedImage():
    def __init__(self, data=None, mask=None):
        if data is None:
            data = np.zeros([30, 30, 4])
            mask = np.zeros([30, 30, 4])
            centroid = np.array([15, 15])
            z_slice = 2

            for idx, val in np.ndenumerate(data[..., 2]):
                disp = idx - centroid
                r = np.linalg.norm(disp)
                if r <= 10:
                    data[idx[0], idx[1], z_slice] = r
                    mask[idx[0], idx[1], z_slice] = 1

        self.data = data
        self.mask = mask

    def get_image(self):
        return self.data

    def get_flat_data(self):
        return self.data[mask == 1]

