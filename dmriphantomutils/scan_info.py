"""A set of classes used to describe a phantom dataset."""

import math

import numpy as np

class Phantom:
    def __init__(self, hotend_temp, print_speed, layer_thickness,
                 infill_density, infill_pattern):
        self.hotend_temp = hotend_temp
        self.print_speed = print_speed
        self.layer_thickness = layer_thickness
        self.infill_density = infill_density
        self.infill_pattern = infill_pattern

class WaterSlice:
    def __init__(self):
        self.infill_pattern = EmptyPattern() 

class Study:
    def __init__(self, name, tube, sessions):
        self.name = name
        self.tube = tube
        self.sessions = sessions

class ScanSession:
    def __init__(self, date, scans):
        self.date = date
        self.scans = scans

class SingleScan:
    def __init__(self, tube_slice):
        self.tube_slice = tube_slice

class EmptyPattern:
    def __init__(self):
        pass

    def get_geometry_generators(self):
        return {}

class ParallelLinePattern:
    def __init__(self, cura_angle):
        self.cura_angle = cura_angle

    def get_geometry_generators(self):
        return {'direction': self._get_direction,
                'crossing_angle': lambda point: 0}

    def _get_direction(self, point):
        return 90 - self.cura_angle

class ConcentricArcPattern:
    def __init__(self, origin):
        self.origin = origin

    def get_geometry_generators(self):
        return {'direction': self._get_direction,
                'arc_radius': self._get_arc_radius,
                'crossing_angle': lambda point: 0}

    def _get_direction(self, point):
        displacement = (self.origin[0] - point[0], self.origin[1] - point[1])
        if displacement[0] == 0:
            return 0

        disp_angle = math.degrees(math.atan(displacement[1] / displacement[0]))

        if disp_angle > 0:
            return disp_angle - 90
        else:
            return disp_angle + 90

    def _get_arc_radius(self, point):
        displacement = np.array(
            [self.origin[0] - point[0], self.origin[1] - point[1]])

        return np.linalg.norm(displacement)

class AlternatingPattern:
    def __init__(self, pattern_0, pattern_1):
        self.pattern_0 = pattern_0
        self.pattern_1 = pattern_1

    def get_geometry_generators(self):
        return {'crossing_angle': self._get_crossing_angle}

    def _get_crossing_angle(self, point):
        patterns = [self.pattern_0, self.pattern_1]
        directions = [pattern.get_geometry_generators()['direction'](point)
            for pattern in patterns]

        crossing_angle = max(directions) - min(directions)

        if crossing_angle > 90:
            return 180 - crossing_angle
        else:
            return crossing_angle

