import unittest

import numpy as np

from dmriphantomutils import scan_info, transform_data
from test import test_data

class TestTransformData(unittest.TestCase):
    def test_transform_image_point(self):
        self.assertAlmostEqual(
                transform_data.transform_image_point((3, 3), (2, 2), 0)[0], 1)
        self.assertAlmostEqual(
                transform_data.transform_image_point((3, 3), (2, 2), 0)[1], 1)
        self.assertAlmostEqual(
                transform_data.transform_image_point((3, 0), (2, 0), 90)[0], 0)
        self.assertAlmostEqual(
                transform_data.transform_image_point((3, 0), (2, 0), 90)[1], 1)

    def test_compare_to_pattern(self):
        img = test_data.MockDerivedImage()
        pattern = scan_info.ConcentricArcPattern((0, 0))
        truth_from_pattern = (lambda pattern, point:
                np.linalg.norm(np.array(point) - np.array(pattern.origin)))
        centroid = (15, 15)
        angle = 45

        indep, dep = transform_data.compare_to_pattern(
                img, pattern, truth_from_pattern, centroid, angle)

        self.assertAlmostEqual(np.mean(indep), np.mean(dep))
        self.assertAlmostEqual(np.std(indep), np.std(dep))

