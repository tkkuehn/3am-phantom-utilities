"""A set of classes used to describe a phantom dataset.

Broadly, the hierarchy of classes here is as follows:

- A Study is comprised of a set of ScanSessions, each examining the same
  sample test tube.
- A ScanSession is comprised of a set of SingleScans of (different regions
  of) the sample.
- A sample test tube contains a set of Phantoms.
- Each phantom has an infill Pattern.
"""

import math

import numpy as np

class Phantom:
    """Description of a phantom's design and print parameters.

    This class has no intrinsic functionality, but contains a full
    description of a phantom's design and print process.

    Parameters
    ----------
    hotend_temp : float
        The temperature, in degrees Celsius, at which the phantom was 3D
        printed.
    print_speed : float
        The speed, in mm/s, at which the phantom was 3D printed.
    layer_thickness : float
        The thickness, in mm, of each of the phantom's layers.
    infill_density : float
        The infill density, as a percentage, with which the phantom was
        printed.
    infill_pattern : Pattern
        The infill pattern with which the phantom was printed.
    """

    def __init__(self, hotend_temp, print_speed, layer_thickness,
                 infill_density, infill_pattern):
        self.hotend_temp = hotend_temp
        self.print_speed = print_speed
        self.layer_thickness = layer_thickness
        self.infill_density = infill_density
        self.infill_pattern = infill_pattern

class WaterSlice:
    """A placeholder for a scan slice containing only water."""

    def __init__(self):
        self.infill_pattern = EmptyPattern() 

class Study:
    """A data class fully documenting a study using one set of phantoms.

    A "study," in this context, refers to a series of scans of the same
    set of phantoms.

    Parameters
    ----------
    name : str
        A name for the study.
    tube : list of Phantom
        A list containing each phantom in the set that was scanned.
    sessions : list of ScanSession
        A list containing each session of phantom scans.
    """

    def __init__(self, name, tube, sessions):
        self.name = name
        self.tube = tube
        self.sessions = sessions

class ScanSession:
    """A description of a single day of scans covering a set of phantoms.

    Parameters
    ---------
    date : datetime.date
        The date of the scan session.
    scans : list of SingleScan
        The set of scans that were conducted.
    """

    def __init__(self, date, scans):
        self.date = date
        self.scans = scans

class SingleScan:
    """A single scan covering some subset of a set of phantoms.

    There should be one DWI that corresponds to each scan.

    Parameters
    ----------
    tube_slice : slice
        The slice of a list of phantoms that was covered in the scan.
    """

    def __init__(self, tube_slice):
        self.tube_slice = tube_slice

class EmptyPattern:
    """A placeholder infill pattern for empty slices."""

    def __init__(self):
        pass

    def get_geometry_generators(self):
        return {}

class ParallelLinePattern:
    """An infill pattern composed only of parallel lines.

    Parameters
    ----------
    cura_angle : float
        The "direction" of the lines, as specified in cura, in degrees.
    """

    def __init__(self, cura_angle):
        self.cura_angle = cura_angle

    def get_geometry_generators(self):
        """A dictionary of functions that describe the infill geometry.

        For parallel line infill, the fibre direction and crossing angle
        are both constant because there is no orientation dispersion of
        any kind.
        """

        return {'direction': self._get_direction,
                'crossing_angle': lambda point: 0}

    def _get_direction(self, point):
        return 90 - self.cura_angle

class ConcentricArcPattern:
    """An infill pattern composed of concentric arcs.

    This pattern represents bending fibres in the brain.

    Parameters
    ----------
    origin : tuple of float
        The x and y coordinates of the arcs' common centre, relative to the
        centroid of the phantom's cross section.
    """

    def __init__(self, origin):
        self.origin = origin

    def get_geometry_generators(self):
        """A dictionary of functions that describe the infill geometry.

        For concentric arc infill, there is a constant crossing angle (0),
        but the fibre direction (tangent to each arc) changes and there is
        an arc radius that changes in the phantom.
        """

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
    """An infill pattern that changes from layer to layer.

    Parameters
    ----------
    pattern_0, pattern_1 : Pattern
        The two patterns that alternate.
    """

    def __init__(self, pattern_0, pattern_1):
        self.pattern_0 = pattern_0
        self.pattern_1 = pattern_1

    def get_geometry_generators(self):
        """A dictionary of functions that describe the infill geometry.

        For alternating patterns, only the crossing angle has a consistent
        definition, which is based on the directions of the underlying
        infill patterns. That said, this method will also return
        'arc_radius', which returns the minimum radius of fibre curvature in
        a voxel, unless there is no fibre curvature, in which case it will
        default to zero.
        """

        return {'crossing_angle': self._get_crossing_angle,
                'arc_radius': self._get_arc_radius}

    def _get_crossing_angle(self, point):
        patterns = [self.pattern_0, self.pattern_1]
        directions = [pattern.get_geometry_generators()['direction'](point)
            for pattern in patterns]

        crossing_angle = max(directions) - min(directions)

        if crossing_angle > 90:
            return 180 - crossing_angle
        else:
            return crossing_angle

    def _get_arc_radius(self, point):
        patterns = [self.pattern_0, self.pattern_1]
        radii = []
        for pattern in patterns:
            generators = pattern.get_geometry_generators()
            if 'arc_radius' in generators.keys():
                radii.append(generators['arc_radius'](point))

        if len(radii) == 0:
            # Zero will mean no fibre curvature in this voxel
            return 0
        
        return min(radii)

