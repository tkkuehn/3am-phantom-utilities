import math

class LinearInfillPattern:
    def __init__(self):
        pass

class CrossingLinesInfillPattern:
    def __init__(self, crossing_angle):
        self.crossing_angle = crossing_angle

class ConcentricArcPattern:
    def __init__(self, origin):
        self.origin = origin

class TwoArcPattern:
    def __init__(self, origin1, origin2):
        self.origin1 = origin1
        self.origin2 = origin2

class ArcLinePattern:
    """
    A pattern consisting of alternating arcs and lines.

    Parameters
    ----------
    origin : tuple of int
        x, y coordinates of concentric arc origin
    cura_angle : float
        angle of straight lines as input to Cura (i.e. from positive y-axis)

    Methods
    -------
    get_crossing_angle(point)
        compute the crossing angle at the given point
    """

    def __init__(self, origin, cura_angle):
        self.origin = origin
        self.cura_angle = cura_angle

    def get_crossing_angle(self, point):
        displacement = (self.origin[0] - point[0], self.origin[1] - point[1])
        if displacement[0] == 0:
            tangent_angle = 0
        else:
            disp_angle = math.degrees(math.atan(displacement[1] / displacement[0]))
            if disp_angle > 0:
                tangent_angle = disp_angle - 90
            else:
                tangent_angle = disp_angle + 90

        line_angle = 90 - self.cura_angle
        crossing_angle = max(tangent_angle, line_angle) - min(
                tangent_angle, line_angle)
        if crossing_angle > 90:
            return 180 - crossing_angle
        else:
            return crossing_angle

