import numpy as np
from XRaySimulation import util

hbar = util.hbar  # This is the reduced planck constant in keV/fs
c = util.c  # The speed of light in um / fs
pi = util.pi
two_pi = 2 * pi


class GaussianPulse3D:
    def __init__(self):
        self.klen0 = 100.
        self.x0 = np.zeros(3, dtype=np.float64)
        self.k0 = np.zeros(3, dtype=np.float64)
        self.n = np.zeros(3, dtype=np.float64)

        self.omega0 = 20000.  # PHz

        # Basically, this mean that initially, we are in the frame where
        # different components of the pulse decouple. Then we rotate
        # back in to the lab frame.
        self.sigma_x = 0.1  # fs
        self.sigma_y = 33.  # fs
        self.sigma_z = 33.  # fs

        self.sigma_mat = np.diag(np.array([self.sigma_x ** 2,
                                           self.sigma_y ** 2,
                                           self.sigma_z ** 2], dtype=np.float64))

        # Intensity. Add a coefficient for overall intensity.
        self.scaling = 1.

        # Polarization
        self.polar = np.array([1., 0., 0.], dtype=np.complex128)

    def set_pulse_properties(self, central_energy, polar, sigma_x, sigma_y, sigma_z, x0):
        """
        Set the pulse properties assuming that the pulse is propagating along
        the positive z direction.
        :param central_energy:
        :param polar:
        :param sigma_x: The unit is fs. However, in the function, it's converted into um.
        :param sigma_y: The unit is fs. However, in the function, it's converted into um.
        :param sigma_z: The unit is fs. However, in the function, it's converted into um.
        :param x0:
        :return:
        """
        # Get the corresponding wave vector
        self.klen0 = util.kev_to_wavevec_length(energy=central_energy)

        self.polar = np.array(np.reshape(polar, (3,)),
                              dtype=np.complex128)

        self.k0 = np.array([0., 0., self.klen0])
        self.n = self.k0 / util.l2_norm(self.k0)
        self.omega0 = self.klen0 * util.c
        self.x0 = x0

        # Initialize the sigma matrix
        self.set_sigma_mat(sigma_x=sigma_x,
                           sigma_y=sigma_y,
                           sigma_z=sigma_z)

        # Normalize the pulse such that the incident total energy is 1 au
        # Then in this case, if one instead calculate the square L2 norm of the spectrum, then
        # the value is 8 * pi ** 3
        self.scaling = 2. * np.sqrt(2) * np.power(np.pi, 0.75) * np.sqrt(sigma_x *
                                                                         sigma_y *
                                                                         sigma_z *
                                                                         (util.c ** 3))

    def set_sigma_mat(self, sigma_x, sigma_y, sigma_z):
        """
        Notice that this function assumes that the pulse propagates long the z direction.

        :param sigma_x:
        :param sigma_y:
        :param sigma_z:
        :return:
        """

        self.sigma_x = sigma_x  # sigma_t
        self.sigma_y = sigma_y  # sigma_t  # fs
        self.sigma_z = sigma_z  # fs
        self.sigma_mat = np.diag(np.array([self.sigma_x ** 2,
                                           self.sigma_y ** 2,
                                           self.sigma_z ** 2], dtype=np.float64))
        self.sigma_mat *= util.c ** 2

    def shift(self, displacement):
        """

        :param displacement:
        :return:
        """
        self.x0 += displacement

    def rotate(self, rot_mat):
        """
        Rotate the pulse with respect to the origin

        :param rot_mat:
        :return:
        """
        self.x0 = np.dot(rot_mat, self.x0)
        self.k0 = np.dot(rot_mat, self.k0)
        self.polar = np.dot(rot_mat, self.polar)
    
        self.sigma_mat = np.dot(np.dot(rot_mat, self.sigma_mat), rot_mat.T)

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


# --------------------------------------------------------------
#          Get spectrum
# --------------------------------------------------------------
def get_gaussian_pulse_spectrum(k_grid, sigma_mat, scaling, k0):
    # Get the momentum difference
    dk = k0[np.newaxis, :] - k_grid

    # Get the quadratic term
    quad_term = - (dk[:, 0] * sigma_mat[0, 0] * dk[:, 0] + dk[:, 0] * sigma_mat[0, 1] * dk[:, 1] +
                   dk[:, 0] * sigma_mat[0, 2] * dk[:, 2] +
                   dk[:, 1] * sigma_mat[1, 0] * dk[:, 0] + dk[:, 1] * sigma_mat[1, 1] * dk[:, 1] +
                   dk[:, 1] * sigma_mat[1, 2] * dk[:, 2] +
                   dk[:, 2] * sigma_mat[2, 0] * dk[:, 0] + dk[:, 2] * sigma_mat[2, 1] * dk[:, 1] +
                   dk[:, 2] * sigma_mat[2, 2] * dk[:, 2]) / 2.

    # if quad_term >= -200:
    magnitude = scaling * (np.exp(quad_term) + 0.j)
    return magnitude


def get_square_pulse_spectrum(k_grid, k0, a_val, b_val, c_val, scaling):
    dk = k_grid - k0[np.newaxis, :]
    spectrum = np.multiply(np.multiply(
        np.sinc((a_val / 2. / np.pi) * dk[:, 0]),
        np.sinc((b_val / 2. / np.pi) * dk[:, 1])),
        np.sinc((c_val / 2. / np.pi) * dk[:, 2])) + 0.j
    spectrum *= scaling

    return spectrum


def get_square_pulse_spectrum_smooth(k_grid, k0, a_val, b_val, c_val, scaling, sigma):
    dk = k_grid - k0[np.newaxis, :]
    spectrum = np.multiply(np.multiply(
        np.sinc((a_val / 2. / np.pi) * dk[:, 0]),
        np.sinc((b_val / 2. / np.pi) * dk[:, 1])),
        np.sinc((c_val / 2. / np.pi) * dk[:, 2])) + 0.j

    spectrum *= scaling

    # Add the Gaussian filter
    tmp = - (dk[:, 0] ** 2 + dk[:, 1] ** 2 + dk[:, 2] ** 2) * sigma ** 2 / 2.
    gaussian = np.exp(tmp)

    return np.multiply(spectrum, gaussian)
