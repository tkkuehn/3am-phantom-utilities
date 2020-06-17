import unittest

from dmriphantomutils import scan_info

class TestConcentricArcPattern(unittest.TestCase):
    def setUp(self):
        self.pattern = scan_info.ConcentricArcPattern((0, 0))

    def test_get_direction(self):
        get_direction = self.pattern.get_geometry_generators()['direction']
        self.assertEqual(get_direction((0, 0)), 0)
        self.assertEqual(get_direction((1, 0)), 90)
        self.assertEqual(get_direction((0, 1)), 0)
        self.assertEqual(get_direction((-1, 0)), 90)
        self.assertEqual(get_direction((0, -1)), 0)
        self.assertEqual(get_direction((1, 1)), -45)
        self.assertEqual(get_direction((1, -1)), 45)
        self.assertEqual(get_direction((-1, 1)), 45)
        self.assertEqual(get_direction((-1, -1)), -45)

    def test_get_arc_radius(self):
        get_arc_radius = self.pattern.get_geometry_generators()['arc_radius']
        self.assertEqual(get_arc_radius((3, 4)), 5)

class TestParallelLinePattern(unittest.TestCase):
    def setUp(self):
        self.pattern = scan_info.ParallelLinePattern(0)

    def test_get_direction(self):
        get_direction = self.pattern.get_geometry_generators()['direction']
        self.assertEqual(get_direction((1, 1)), 90)

class TestAlternatingPattern(unittest.TestCase):
    def setUp(self):
        self.pattern = scan_info.AlternatingPattern(
                scan_info.ParallelLinePattern(0),
                scan_info.ConcentricArcPattern((0, 0)))

    def test_get_crossing_angle(self):
        get_crossing_angle = (
            self.pattern.get_geometry_generators()['crossing_angle'])
        self.assertEqual(get_crossing_angle((0, 0)), 90)
        self.assertEqual(get_crossing_angle((1, 0)), 0)

    def test_get_arc_radius(self):
        get_arc_radius = (
            self.pattern.get_geometry_generators()['arc_radius'])
        self.assertEqual(get_arc_radius((0, 1)), 1)
        self.assertEqual(get_arc_radius((3, 4)), 5)

