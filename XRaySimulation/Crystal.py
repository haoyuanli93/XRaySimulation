"""
This module is used to create classes describing crystals

fs, um are the units
"""

import numpy as np
import requests

from XRaySimulation import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi

# Default incident photon energy
bragg_energy = 6.95161 * 2  # kev
wavenumber = util.kev_to_wavevec_length(bragg_energy)

# Some numerical values
cot_pi_8 = 1. + np.sqrt(2)


class ChannelCut():
    def __int__(self,
                crystal_type="Silicon",
                miller_index="220",
                energy_keV=10.0,
                thickness_list=np.array([1e4, 1e4]),
                gap=1e4,
                edge_length_list=np.array([5e4, 5e4]),
                asymmetry_angle_list=np.deg2rad(np.array([0., 0.])),
                first_surface_loc="lower left",
                ):
        """
        Calculate the geometry

                 X-ray goes from the top left to bottom right incident on the first crystal.

                                                                        edge length
                                                                 --------------------------
                                                                 |       Crystal 2        |
                                                                 --------------------------
                                                         ---
                                                          |
                                                          |  Gap, distance between the two surface center is
                                 edge length              |             gap / tan(Bragg)
                            --------------------------   ---
                            |       Crystal 1        |
                            --------------------------

                If the mirror symmetric flag is true, then the first crystal is on the top.

        :param crystal_type:
        :param miller_index:
        :param energy_keV:
        :param thickness_list:
        :param gap:
        :param edge_length_list:
        :param asymmetry_angle_list:
        :param first_surface_loc:
        :return:
        """

        # TODO: Explain to Selene why I am doing this and ask her to write down the comment

        # Add a type to help functions to choose how to treat this object
        self.type = "Channel cut with two surfaces"

        # The location of the first crystal determines which direction should the channel-cut rotate
        self.first_crystal_loc = first_surface_loc

        # Get the atomic plane distance.
        crystal_property = get_crystal_param(crystal_type=crystal_type,
                                             miller_index=miller_index,
                                             energy_kev=energy_keV)

        # Get wave-length
        wave_length = 2 * np.pi / util.kev_to_wavevec_length(energy=energy_keV)

        # Get geometric bragg angle
        bragg_theta = util.get_bragg_angle(wave_length=wave_length, plane_distance=crystal_property['d'])

        # Create 2 crystals
        self.crystal_list = [
            CrystalBlock3D(h=np.array([0., 2. * np.pi / crystal_property['d'], 0.]),
                           normal=np.array([0., -np.cos(asymmetry_angle_list[x]), np.sin(asymmetry_angle_list[x])]),
                           surface_point=np.zeros(3, dtype=np.float64),
                           thickness=thickness_list[x],
                           chi_dict=crystal_property,
                           edge_length=edge_length_list[x]) for x in range(2)]

        # Shift and rotate crystals to the correct position
        if first_surface_loc == "lower left":
            displacement = np.array([0, gap, gap / np.tan(bragg_theta)])

            # Shift the surface center
            self.crystal_list[1].shift(displacement=displacement, include_boundary=True)

            # Rotate the crystal
            rot_mat = np.array([[1., 0., 0., ],
                                [0., 0., 1., ],
                                [0., -1., 0., ],
                                ], dtype=np.float64)
            self.crystal_list[1].rotate_wrt_point(rot_mat=rot_mat,
                                                  ref_point=np.copy(self.crystal_list[1].surface_point),
                                                  include_boundary=True)
        elif first_surface_loc == "upper left":
            # Rotate the first crystal
            rot_mat = np.array([[1., 0., 0., ],
                                [0., 0., 1., ],
                                [0., -1., 0., ],
                                ], dtype=np.float64)
            self.crystal_list[0].rotate_wrt_point(rot_mat=rot_mat,
                                                  ref_point=np.copy(self.crystal_list[0].surface_point),
                                                  include_boundary=True)

            # Shift the second crystal
            displacement = np.array([0, -gap, gap / np.tan(bragg_theta)])

            # Shift the surface center
            self.crystal_list[1].shift(displacement=displacement, include_boundary=True)
        else:
            print("The first_surface_loc has to be lower left or upper left")

    def shift(self, displacement, include_boundary=True):
        for x in range(2):
            self.crystal_list[x].shift(displacement=displacement,
                                       include_boundary=include_boundary)

    def rotate(self, rot_mat, include_boundary=True):
        for x in range(2):
            self.crystal_list[x].rotate(rot_mat=rot_mat,
                                        include_boundary=include_boundary)

    def rotate_wrt_point(self, rot_mat, ref_point, include_boundary=True):
        """
        This is a function designed
        :param rot_mat:
        :param ref_point:
        :param include_boundary:
        :return:
        """
        for x in range(2):
            self.crystal_list[x].rotate_wrt_point(rot_mat=rot_mat,
                                                  ref_point=np.copy(ref_point),
                                                  include_boundary=include_boundary)


class CrystalBlock3D:
    def __init__(self,
                 h=np.array([0, wavenumber, 0], dtype=np.float64),
                 normal=np.array([0., -1., 0.]),
                 surface_point=np.zeros(3, dtype=np.float64),
                 thickness=1e4,
                 chi_dict=None,
                 edge_length=5e4):
        """

        :param h:
        :param normal:
        :param surface_point:
        :param thickness:
        :param chi_dict:
        :param edge_length: The length of the surface edge
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

        self.chi_dict = chi_dict.copy()

        #############################
        # Second level of parameters. These parameters can be handy in the simulation_2019_11_5_2
        #############################
        self.dot_hn = np.dot(self.h, self.normal)
        self.h_square = self.h[0] ** 2 + self.h[1] ** 2 + self.h[2] ** 2
        self.h_len = np.sqrt(self.h_square)

        #############################
        # These parameters are designed for the light path simulation for the experiment
        #############################
        #      The boundary is defined in the following way
        #
        #    (top, left) point 0        (middle perpendicular to self.normal) self.surface_point      point 1
        #      parallel to h
        #       point 3                                                                               point 2
        #

        # direction perpendicular to the normal direction
        direction1 = np.outer(np.array([1, 0, 0]), self.normal)
        direction1 /= np.linalg.norm(direction1)

        # direction parallel to the reciprocal lattice
        direction2 = - self.h / np.linalg.norm(self.h)

        point0 = self.surface_point - direction1 * edge_length / 2.
        point1 = self.surface_point + direction1 * edge_length / 2.
        point2 = point1 + thickness * direction2
        point3 = point2 - direction1 * edge_length
        point4 = np.copy(point0)

        # Assemble
        self.boundary = np.vstack([point0, point1, point2, point3, point4])

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
        tmp = np.copy(ref_point)
        # Step 1: shift with respect to that point
        self.shift(displacement=-np.copy(tmp), include_boundary=include_boundary)

        # Step 2: rotate the quantities
        self.rotate(rot_mat=rot_mat, include_boundary=include_boundary)

        # Step 3: shift it back to the reference point
        self.shift(displacement=np.copy(tmp), include_boundary=include_boundary)


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
        self.normal = normal / np.linalg.norm(normal)
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


def get_crystal_param(crystal_type, miller_index, energy_kev):
    """

    :param crystal_type:
    :param miller_index:
    :param energy_kev:
    :return:
    """

    ###########################################################
    #    Get response from the website
    ###########################################################
    if crystal_type in ("Silicon", "Germanium", "Diamond", "GaAs"):
        pass
    else:
        print("The requested crystal type is not recognized. Please check the source code.")
        return

    df1df2 = -1
    modeout = 1  # 0 - html out, 1 - quasy-text out with keywords
    detail = 0  # 0 - don't print coords, 1 = print coords

    commandline = str(r"https://x-server.gmca.aps.anl.gov/cgi/x0h_form.exe?"
                      + r"xway={}".format(2)
                      + r'&wave={}'.format(energy_kev)
                      + r'&line='
                      + r'&coway={}'.format(0)
                      + r'&code={}'.format(crystal_type)
                      + r'&amor='
                      + r'&chem='
                      + r'&rho='
                      + r'&i1={}'.format(miller_index[0])
                      + r'&i2={}'.format(miller_index[1])
                      + r'&i3={}'.format(miller_index[2])
                      + r'&df1df2={}'.format(df1df2)
                      + r'&modeout={}'.format(modeout)
                      + r'&detail={}'.format(detail))

    data = requests.get(commandline).text

    #############################################################
    #   Parse the parameters
    #############################################################
    lines = data.split("\n")
    info_holder = {}

    line_idx = 0
    total_line_num = len(lines)

    while line_idx < total_line_num:
        line = lines[line_idx]
        words = line.split()
        # print(words)
        if words:
            if words[0] == "Density":
                info_holder.update({"Density (g/cm^3)": float(words[-1])})

            elif words[0] == "Unit":
                info_holder.update({"Unit cell a1 (A)": float(words[-1])})

                # Get a2
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a2 (A)": float(words[-1])})

                # Get a3
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a3 (A)": float(words[-1])})

                # Get a4
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a4 (deg)": float(words[-1])})

                # Get a5
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a5 (deg)": float(words[-1])})

                # Get a6
                line_idx += 1
                line = lines[line_idx]
                words = line.split()
                info_holder.update({"Unit cell a6 (deg)": float(words[-1])})

            elif words[0] == "Poisson":
                info_holder.update({"Poisson ratio": float(words[-1])})

            elif words[0] == '<pre>':
                if words[1] == '<i>n':

                    # Get the real part of chi0
                    a = float(words[-1][4:])

                    # Get the imagniary part of chi0
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    b = float(words[-1])

                    # Add an entry
                    info_holder.update({"chi0": complex(a, b)})

                    # Skip the following two lines
                    line_idx += 2

                elif words[-1] == 'pol=Sigma':
                    # Get the real part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    a = float(words[-1])

                    # Get the imaginary part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    b = float(words[-1])

                    info_holder.update({"chih_sigma": complex(a, -b)})

                elif words[-1] == 'pol=Pi':
                    # Get the real part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    a = float(words[-1])

                    # Get the imaginary part
                    line_idx += 1
                    line = lines[line_idx]
                    words = line.split()
                    b = float(words[-1])

                    info_holder.update({"chih_pi": complex(a, -b)})
                elif words[1] == "Bragg":
                    if words[2] == "angle":
                        info_holder.update({"Bragg angle (deg)": float(words[-1])})
                else:
                    pass

            elif words[0] == 'Interplanar':
                info_holder.update({"d": float(words[-1]) * 1e-4})

            else:
                pass

        line_idx += 1

    return info_holder
