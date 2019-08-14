import os.path
import tempfile

import numpy as np

TEST_RESOURCES = os.path.join('test', 'resources')

def get_data_nifti():
    return os.path.join(TEST_RESOURCES, 'test.nii')

def get_data_bval():
    return os.path.join(TEST_RESOURCES, 'test.bval')

def get_data_bvec():
    return os.path.join(TEST_RESOURCES, 'test.bvec')

def get_data_mask_nifti():
    return os.path.join(TEST_RESOURCES, 'test_mask.nii.gz')

def get_temp_output_nifti():
    return os.path.join(tempfile.gettempdir(), 'test.nii.gz')

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

    def getImage(self):
        return self.data

    def getFlatData(self):
        return self.data[mask == 1]

