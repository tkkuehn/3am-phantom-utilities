import os.path
import tempfile
import unittest

from dmriphantomutils import dipy_fit
from test import test_data

class TestDipyFit(unittest.TestCase):
    def testFitDki(self):
        dwi = test_data.MockDiffusionWeightedImage()

        dkifit = dipy_fit.fit_dki(dwi)

