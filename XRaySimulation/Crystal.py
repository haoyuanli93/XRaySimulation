"""
This module is used to create classes describing crystals

fs, um are the units
"""

import numpy as np

from XRaySimulation import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi

# Default incident photon energy
bragg_energy = 6.95161 * 2  # kev
wavenumber = util.kev_to_wavevec_length(bragg_energy)

# Some numerical values
cot_pi_8 = 1. + np.sqrt(2)


class CrystalBlock3D:
    def __init__(self,
                 h=np.array([0, wavenumber, 0], dtype=np.float64),
                 normal=np.array([0., -1., 0.]),
                 surface_point=np.zeros(3, dtype=np.float64),
                 thickness=1e6,
                 chi_dict=None):
        """

        :param h:
        :param normal:
        :param surface_point:
        :param thickness:
        :param chi_dict:
        """
        # Add a type to help functions to choose how to treat this object
        self.type = "Crystal: Bragg Reflection"

        #############################
        # First level of parameters
        ##############################
        # This is just a default value

        # Reciprocal lattice in um^-1
        self.h = np.copy(h)
        self.normal = np.copy(normal)
        self.surface_point = np.copy(surface_point)
        self.thickness = thickness

        if chi_dict is None:
            chi_dict = {"chi0": complex(-0.15124e-4, 0.13222E-07),
                        "chih_sigma": complex(0.37824E-05, -0.12060E-07),
                        "chihbar_sigma": complex(0.37824E-05, -0.12060E-07),
                        "chih_pi": complex(0.37824E-05, -0.12060E-07),
                        "chihbar_pi": complex(0.37824E-05, -0.12060E-07)}

        # zero component of electric susceptibility's fourier transform
        self.chi0 = chi_dict["chi0"]

        # h component of electric susceptibility's fourier transform
        self.chih_sigma = chi_dict["chih_sigma"]

        # hbar component of electric susceptibility's fourier transform
        self.chihbar_sigma = chi_dict["chihbar_sigma"]

        # h component of electric susceptibility's fourier transform
        self.chih_pi = chi_dict["chih_pi"]

        # hbar component of electric susceptibility's fourier transform
        self.chihbar_pi = chi_dict["chihbar_pi"]

        #############################
        # Second level of parameters. These parameters can be handy in the simulation_2019_11_5_2
        #############################
        self.dot_hn = np.dot(self.h, self.normal)
        self.h_square = self.h[0] ** 2 + self.h[1] ** 2 + self.h[2] ** 2
        self.h_len = np.sqrt(self.h_square)

        #############################
        # These parameters are designed for the light path simulation for the experiment
        #############################
        # The boundary_2d is defined in such a way that
        # boundary_2d[0] is the first point on the boundary.
        # boundary_2d[1] is the second ...
        # If one connect all the point in sequence, then one get the whole boundary.
        self.boundary = np.zeros((4, 3))

    def set_h(self, reciprocal_lattice):
        self.h = np.array(reciprocal_lattice)
        self._update_dot_nh()
        self._update_h_square()

    def set_surface_normal(self, normal):
        """
        Define the normal direction of the incident surface. Notice that, this algorithm assumes that
        the normal vector points towards the interior of the crystal.

        :param normal:
        :return:
        """
        self.normal = normal
        self._update_dot_nh()

    def set_surface_point(self, surface_point):
        """

        :param surface_point:
        :return:
        """
        self.surface_point = surface_point

    def set_thickness(self, d):
        """
        Set the lattice thickness
        :param d:
        :return:
        """
        self.thickness = d

    def set_chi0(self, chi0):
        self.chi0 = chi0

    def set_chih_sigma(self, chih):
        self.chih_sigma = chih

    def set_chihbar_sigma(self, chihb):
        self.chihbar_sigma = chihb

    def set_chih_pi(self, chih):
        self.chih_pi = chih

    def set_chihbar_pi(self, chihb):
        self.chihbar_pi = chihb

    def _update_dot_nh(self):
        self.dot_hn = np.dot(self.normal, self.h)

    def _update_h_square(self):
        self.h_square = self.h[0] ** 2 + self.h[1] ** 2 + self.h[2] ** 2
        self.h_len = np.sqrt(self.h_square)

    def shift(self, displacement, include_boundary=True):
        """

        :param displacement:
        :param include_boundary: Whether to shift the boundary or not.
        :return:
        """
        self.surface_point += displacement

        if include_boundary:
            self.boundary += displacement[np.newaxis, :]

    def rotate(self, rot_mat, include_boundary=True):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.h = np.ascontiguousarray(rot_mat.dot(self.h))
        self.normal = np.ascontiguousarray(rot_mat.dot(self.normal))
        self.surface_point = np.asanyarray(np.dot(rot_mat, self.surface_point))

        if include_boundary:
            self.boundary = np.asanyarray(np.dot(self.boundary, rot_mat.T))

    ##############################################
    #   This is a methods designed for the simulation of the light path
    #   to investigate whether the crystal will block hte light or not.
    ##############################################
    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        """
        This is a function designed
        :param rot_mat:
        :param ref_point:
        :param include_boundary:
        :return:
        """
        # Step 1: shift with respect to that point
        self.shift(displacement=-ref_point, include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=ref_point, include_boundary=include_boundary)


class RectangleGrating:
    def __init__(self,
                 a=1.,
                 b=1.,
                 n=1. - 0.73031 * 1e-5 + 1.j * 0.61521 * 1e-8,
                 height=8.488,
                 base_thickness=10.,
                 direction=np.array([0., 1., 0.], dtype=np.float64),
                 surface_point=np.zeros(3),
                 normal=np.array([0., 0., 1.], dtype=np.float64),
                 order=1.,
                 ):
        """

        :param a: The width of the groove
        :param b: The width of the tooth
        :param n: The refraction index
        :param height: The height of the tooth
        :param base_thickness: The thickness of the base plate
        :param direction: The direction of the wave vector transfer.
        :param surface_point: One point through which the surface goes through.
        :param normal: The normal direction of the surface
        """
        self.type = "Transmissive Grating"

        # Structure info
        self.a = a  # (um)
        self.b = b  # (um)
        self.n = n  # The default value is for diamond
        self.height = height

        # The thickness of the base
        self.base_thickness = base_thickness  # (um)

        # Geometry info
        self.direction = direction
        self.surface_point = surface_point
        self.normal = normal

        # Derived parameter to calculate effects
        self.ab_ratio = self.b / (self.a + self.b)
        self.h = self.height * self.normal
        self.thick_vec = self.base_thickness * self.normal
        self.period = self.a + self.b  # (um)
        self.base_wave_vector = self.direction * np.pi * 2. / self.period

        # Momentum transfer
        self.order = order
        self.momentum_transfer = self.order * self.base_wave_vector

    def __update_period_wave_vector(self):
        self.period = self.a + self.b  # (um)
        self.base_wave_vector = self.direction * np.pi * 2. / self.period
        self.ab_ratio = self.b / (self.a + self.b)
        self.momentum_transfer = self.order * self.base_wave_vector

    def __update_h(self):
        self.h = self.height * self.normal
        self.thick_vec = self.base_thickness * self.normal

    def set_a(self, a):
        self.a = a
        self.__update_period_wave_vector()

    def set_b(self, b):
        self.b = b
        self.__update_period_wave_vector()

    def set_height(self, height):
        self.height = height
        self.__update_h()

    def set_surface_point(self, surface_point):
        self.surface_point = surface_point

    def set_normal(self, normal):
        self.normal = normal / util.l2_norm(normal)
        self.__update_h()

    def set_diffraction_order(self, order):
        self.order = order
        self.momentum_transfer = self.order * self.base_wave_vector

    def shift(self, displacement):
        self.surface_point += displacement

    def rotate(self, rot_mat):
        # The shift of the space does not change the reciprocal lattice and the normal direction
        self.direction = np.ascontiguousarray(rot_mat.dot(self.direction))
        self.normal = np.ascontiguousarray(rot_mat.dot(self.normal))

        # Update h and wave vector
        self.__update_h()
        self.__update_period_wave_vector()

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        """
        This is a function designed
        :param rot_mat:
        :param ref_point:
        :param include_boundary:
        :return:
        """
        # Step 1: shift with respect to that point
        self.shift(displacement=-ref_point)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat)

        # Step 3: shift it back to the reference point
        self.shift(displacement=ref_point)


class Prism:
    def __init__(self,
                 wavevec_delta=np.array([0, 50677. * 5e-6, 0]),
                 ):
        """

        :param wavevec_delta: The width of the groove
        """
        self.type = "Prism"

        # Structure info
        self.wavevec_delta = wavevec_delta  # (um)
        self.surface_point = np.zeros(3, dtype=np.float64)
        self.normal = np.array([0, 0, 1], dtype=np.float64)
