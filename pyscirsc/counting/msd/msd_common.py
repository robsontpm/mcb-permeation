__author__ = 'robson'


# convert D in nm^2/ps -> 10^-5 cm^2/s for 3D/2D/1D diffusion - same as GROMACS!
FACTOR = {
    'msd': 1000.0 / 6.0,
    'msd_xy': 1000.0 / 4.0,
    'msd_yz': 1000.0 / 4.0,
    'msd_xz': 1000.0 / 4.0,
    'msd_x': 1000.0 / 2.0,
    'msd_y': 1000.0 / 2.0,
    'msd_z': 1000.0 / 2.0,
}


class MSDException(Exception):
    def __init__(self, *args, **kwargs):
        super(MSDException, self).__init__(*args, **kwargs)
