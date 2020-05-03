import numpy as np

from XRaySimulation import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi


class TelescopeForCPA:
    """
    This is a class designed specially for the simulation of
    chirped pulse amplification.

    Because it's difficult to extend my program to include a general
    analysis, therefore, I make no effort to make my simulation
    extensive and flexible.

    In this class, I assume that the two lens are of exactly the
    same focal length and aligned perfectly.

    Also, later, in the corresponding function in the groutine
    module, I will assume that the effects of the telescope is as simple
    as depicted my a simple geometric optics calculation.
    """

    def __init__(self):
        self.lens_axis = np.array([0, 0, 1], dtype=np.float64)
        self.lens_position = np.array([0, 0, 0], dtype=np.float64)

        self.focal_length = 1.5e6

        # Total transmission efficiency
        self.efficiency = 1.

        # Type
        self.type = "Transmission Telescope for CPA"

    def shift(self, displacement):
        """
        :param displacement:
        :return:
        """
        self.lens_position += displacement

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.lens_position = np.ascontiguousarray(rot_mat.dot(self.lens_position))
        # Change optical axis
        self.lens_axis = np.ascontiguousarray(rot_mat.dot(self.lens_axis))

    def rotate_wrt_point(self, rot_mat, ref_point):
        """
        This is a function designed
        :param rot_mat:
        :param ref_point:
        :return:
        """
        # Step 1: shift with respect to that point
        self.shift(displacement=-ref_point)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=ref_point)
