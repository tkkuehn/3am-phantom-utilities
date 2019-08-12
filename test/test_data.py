import os.path
import tempfile

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

