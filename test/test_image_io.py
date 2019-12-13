import unittest

from dmriphantomutils import image_io
from test import test_data

class TestImageIo(unittest.TestCase):
    def test_gen_table(self):
        self.assertEqual(image_io.gen_table([]).shape, (0,))

        image_1 = test_data.MockDerivedImage()
        image_2 = test_data.MockDerivedImage()

        table_1 = image_io.gen_table([image_1, image_2])
        self.assertEqual(table_1.shape, (317, 2))

        image_2.mask[0, 0, 1] = 1

        with self.assertRaises(ValueError) as e:
            image_io.gen_table([image_1, image_2])

            assertEqual(e.msg[:5], 'Image')
