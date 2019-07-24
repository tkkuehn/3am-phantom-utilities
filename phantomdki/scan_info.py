import math

DKI_METRICS = ['fa', 'md', 'ad', 'rd', 'mk', 'ak', 'rk']

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
    def __init__(self, origin, cura_angle):
        self.origin = origin
        self.cura_angle = cura_angle

    def getCrossingAngle(self, point):
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

class Phantom:
    def __init__(self, hotend_temp, print_speed, layer_thickness,
            infill_density, infill_pattern):
        self.hotend_temp = hotend_temp
        self.print_speed = print_speed
        self.layer_thickness = layer_thickness
        self.infill_density = infill_density
        self.infill_pattern = infill_pattern

class ScanSession:
    def __init__(self, name, date, tube, scans):
        self.name = name
        self.date = date
        self.tube = tube
        self.scans = scans

class SingleScan:
    def __init__(self, name, tube_slice):
        self.name = name
        self.tube_slice = tube_slice

