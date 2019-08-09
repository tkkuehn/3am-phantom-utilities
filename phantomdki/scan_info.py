import math

DKI_METRICS = ['fa', 'md', 'ad', 'rd', 'mk', 'ak', 'rk']

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

