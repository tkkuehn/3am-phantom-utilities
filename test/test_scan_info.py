import unittest

from dmriphantomutils import scan_info

class TestConcentricArcPattern(unittest.TestCase):
    def setUp(self):
        self.pattern = scan_info.ConcentricArcPattern((0, 0))

    def test_get_direction(self):
        self.assertEqual(self.pattern.get_direction((0, 0)), 0)
        self.assertEqual(self.pattern.get_direction((1, 0)), 90)
        self.assertEqual(self.pattern.get_direction((0, 1)), 0)
        self.assertEqual(self.pattern.get_direction((-1, 0)), 90)
        self.assertEqual(self.pattern.get_direction((0, -1)), 0)
        self.assertEqual(self.pattern.get_direction((1, 1)), -45)
        self.assertEqual(self.pattern.get_direction((1, -1)), 45)
        self.assertEqual(self.pattern.get_direction((-1, 1)), 45)
        self.assertEqual(self.pattern.get_direction((-1, -1)), -45)

class TestParallelLinePattern(unittest.TestCase):
    def setUp(self):
        self.pattern = scan_info.ParallelLinePattern(0)

    def test_get_direction(self):
        self.assertEqual(self.pattern.get_direction((1, 1)), 90)

class TestAlternatingPattern(unittest.TestCase):
    def setUp(self):
        self.pattern = scan_info.AlternatingPattern(
                scan_info.ParallelLinePattern(0),
                scan_info.ConcentricArcPattern((0, 0)))

    def test_get_crossing_angle(self):
        self.assertEqual(self.pattern.get_crossing_angle((0, 0)), 90)
        self.assertEqual(self.pattern.get_crossing_angle((1, 0)), 0)

