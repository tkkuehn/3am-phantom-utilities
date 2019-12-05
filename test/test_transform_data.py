import unittest

import numpy as np

from dmriphantomutils import scan_info, transform_data
from test import test_data

class TestTransformData(unittest.TestCase):
    def test_find_centroid(self):
        mask_data = np.zeros([50, 50, 3], dtype=np.int)
        mask_data[30, 30, 1] = 1
        mask_data[29, 30, 1] = 1
        mask_data[31, 30, 1] = 1
        mask_data[30, 29, 1] = 1
        mask_data[30, 31, 1] = 1

        self.assertAlmostEqual(
                transform_data.find_centroid(mask_data[..., 1])[0], 30)
        self.assertAlmostEqual(
                transform_data.find_centroid(mask_data[..., 1])[1], 30)

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
        mask = img.mask
        pattern = scan_info.ConcentricArcPattern((0, 0))
        get_arc_radius = pattern.get_geometry_generators()['arc_radius']
        centroid = (15, 15)
        angle = 45

        pattern_r = transform_data.gen_geometry_data(
                mask, get_arc_radius, centroid, angle, 1)

        self.assertAlmostEqual(pattern_r[15, 15, 2], 0)

