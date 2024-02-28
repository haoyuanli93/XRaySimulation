import datetime
import time

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np

# from XRaySimulation.Crystal import get_bragg_rocking_curve

"""
This module is the lowest-level module. It does not depend on another modules.
"""
pi = np.pi
two_pi = 2. * np.pi

hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


# --------------------------------------------------------------
#               Unit conversion
# --------------------------------------------------------------
def kev_to_petahertz_frequency(energy):
    return energy / (hbar * 2 * pi)


def kev_to_petahertz_angular_frequency(energy):
    return energy / hbar


def kev_to_wavevec_length(energy):
    return energy / hbar / c


def petahertz_frequency_to_kev(frequency):
    return hbar * 2 * pi * frequency


def petahertz_angular_frequency_to_kev(angular_frequency):
    return hbar * angular_frequency


def petahertz_angular_frequency_to_wavevec(angular_frequency):
    return angular_frequency / c


def wavevec_to_kev(wavevec):
    """
    Convert wavevector
    wavevector = 2 pi / wavelength
    :param wavevec:
    :return:
    """
    return wavevec * hbar * c


def sigma_to_fwhm(sigma):
    return 2. * np.sqrt(2 * np.log(2)) * sigma


def fwhm_to_sigma(fwhm):
    return fwhm / (2. * np.sqrt(2 * np.log(2)))


def intensity_fwhm_to_field_sigma(fwhm):
    return fwhm / (2. * np.sqrt(2 * np.log(2))) * np.sqrt(2)


def field_sigma_to_intensity_fwhm(sigma):
    return sigma * (2. * np.sqrt(2 * np.log(2))) / np.sqrt(2)


# --------------------------------------------------------------
#          Uncertainty Principle
# --------------------------------------------------------------
def bandwidth_sigma_kev_to_duration_sigma_fs(bandwidth_kev):
    return hbar / 2. / bandwidth_kev


def get_intensity_fwhm_duration_from_intensity_bandwidth(bandwidth_kev):
    # Convert intensity bandwidth to field bandwidth
    field_bandwidth = bandwidth_kev * np.sqrt(2)
    field_bandwidth_sigma = fwhm_to_sigma(field_bandwidth)

    # Calcualte the pulse duration
    field_duration_sigma = bandwidth_sigma_kev_to_duration_sigma_fs(field_bandwidth_sigma)
    field_duration_fwhm = sigma_to_fwhm(field_duration_sigma)

    # Convert the field duration fwhm to intensity duration fwhm
    intensity_duration_fwhm = field_duration_fwhm / np.sqrt(2)
    return intensity_duration_fwhm


# --------------------------------------------------------------
#          Rotation
# --------------------------------------------------------------
def rot_mat_in_yz_plane(theta):
    """
    Get a rotation matrix 3x3 for rotation around x axis
    in the yz plane

    :param theta:
    :return:
    """
    rotmat = np.zeros((3, 3))
    rotmat[0, 0] = 1.
    rotmat[1, 1] = np.cos(theta)
    rotmat[1, 2] = - np.sin(theta)
    rotmat[2, 1] = np.sin(theta)
    rotmat[2, 2] = np.cos(theta)

    return rotmat


def get_rotmat_around_axis(angleRadian, axis):
    """
    Get a rotation matrix that rotate a vector
    with respect to an axis by some angle in radian.

    According to the right hand rule,
    if one aligns the thumb with the positive direction of the axis,
    then a positive angle is direction of your four fingers with
    a hollow fist.

    :param angleRadian:
    :param axis:
    :return:
    """

    # Check the axis length and normalize it
    if np.linalg.norm(axis) < 1e-6:
        print("The axis has to be a vector of unit length.")
        return False
    axis /= np.linalg.norm(axis)

    # Step 1: get a vector that is not parallel with the axis
    newAxis = np.zeros(3, dtype=np.float64)
    newAxis[0] = 1.0

    if np.linalg.norm(newAxis - axis) < 1e-12:
        # If this relative is valid, then axis[0] ~ 1 while  axis[1] = axis[2] = 0
        newAxis[0] = 0.0
        newAxis[1] = 1.0

    # Step 2: remove the projection of the newAxis on the axis direction
    newAxis -= axis * np.dot(axis, newAxis)
    newAxis /= np.linalg.norm(newAxis)

    # Step 2: get the other vector though cross project
    newAxis2 = np.cross(axis, newAxis)

    # Construct the matrix
    rotMat = np.zeros((3, 3))
    rotMat += np.outer(axis, axis) + np.cos(angleRadian) * (np.outer(newAxis, newAxis) + np.outer(newAxis2, newAxis2))
    rotMat += np.sin(angleRadian) * (np.outer(newAxis2, newAxis) - np.outer(newAxis, newAxis2))

    return rotMat


# --------------------------------------------------------------
#          For Bragg Reflection
# --------------------------------------------------------------
def get_bragg_angle(wave_length, plane_distance):
    """
    Return the estimated bragg angle according to the geometric Bragg law.
    :param wave_length:
    :param plane_distance:
    :return:
    """
    return np.arcsin(wave_length / (2. * plane_distance))


def get_bragg_kout(kin, h, normal):
    """
    This function produce the output wave vector from a Bragg reflection.

    :param kin: (3,) numpy array. The incident wave vector
    :param h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.

    :return: kout: (3,) numpy array. The diffraction wave vector.
    """

    # kout holder
    kout = kin + h

    # Incident wave number
    klen = np.sqrt(np.dot(kin, kin))

    # Get gamma and alpha
    gammah = np.dot(kin + h, normal) / klen
    alpha = (2 * np.dot(kin, h) + np.dot(h, h)) / np.square(klen)

    # if np.abs(-gammah - np.sqrt(gammah ** 2 - alpha)) > np.abs(-gammah + np.sqrt(gammah ** 2 - alpha)):
    #    momentum = klen * (-gammah + np.sqrt(gammah ** 2 - alpha))
    # else:
    #    #momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))
    #    pass

    momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))
    # Add momentum transfer
    kout += normal * momentum

    return kout


def get_bragg_kout_array(kin, crystal_h, normal):
    """
    This function produce the output wave vector from a Bragg reflection.

    :param kin: (n, 3) numpy array. The incident wave vector
    :param crystal_h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.

    :return: kout: (n, 3) numpy array. The diffraction wave vector.
    """

    # kout holder
    kout = kin + crystal_h[np.newaxis, :]

    # Incident wave number
    klen = np.sqrt(np.sum(np.square(kin), axis=-1))

    # Get gamma and alpha
    gammah = np.dot(kout, normal) / klen
    alpha = (2 * np.dot(kin, crystal_h) + np.dot(crystal_h, crystal_h)) / np.square(klen)

    # Get the momentum transfer
    momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))

    # Add momentum transfer
    kout += np.outer(momentum, normal)

    return kout


def get_laue_kout_array(kin, crystal_h, normal):
    """
    This function produce the output wave vector from a laue reflection.

    :param kin: (n, 3) numpy array. The incident wave vector
    :param crystal_h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.

    :return: kout: (n, 3) numpy array. The diffraction wave vector.
    """

    # kout holder
    kout = kin + crystal_h[np.newaxis, :]

    # Incident wave number
    klen = np.sqrt(np.sum(np.square(kin), axis=-1))

    # Get gamma and alpha
    gammah = np.dot(kout, normal) / klen
    alpha = (2 * np.dot(kin, crystal_h) + np.dot(crystal_h, crystal_h)) / np.square(klen)

    # Get the momentum transfer
    momentum = klen * (-gammah + np.sqrt(gammah ** 2 - alpha))

    # Add momentum transfer
    kout += np.outer(momentum, normal)

    return kout


def get_bragg_reflection_array(kin_grid, d, h, n,
                               chi0, chih_sigma, chihbar_sigma,
                               chih_pi, chihbar_pi):
    """
    This function aims to get the info quickly with cpu.

    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """
    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(kin_grid, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = np.linalg.norm(kin_grid, axis=-1)
    dot_hn = np.dot(h, n)
    h_square = np.sum(np.square(h))

    # Get gamma and alpha and b
    dot_kn = np.dot(kin_grid, n)
    dot_kh = np.dot(kin_grid, h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + dot_hn, klen_grid)
    # print(gamma_h)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)

    # mask = np.zeros_like(sqrt_gamma_alpha, dtype=bool)
    # mask[np.abs(-gamma_h - sqrt_gamma_alpha) > np.abs(-gamma_h + sqrt_gamma_alpha)] = True

    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)
    # m_trans[mask] = np.multiply(klen_grid[mask], -gamma_h[mask] + sqrt_gamma_alpha[mask])

    # Update the kout_grid
    kout_grid[:, 0] = kin_grid[:, 0] + h[0] + m_trans * n[0]
    kout_grid[:, 1] = kin_grid[:, 1] + h[1] + m_trans * n[1]
    kout_grid[:, 2] = kin_grid[:, 2] + h[2] + m_trans * n[2]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    # mask = np.zeros_like(im, dtype=bool)
    # mask[im <= 100] = True

    reflect_s = chih_sigma * b_cplx / denominator
    # reflect_s[mask] = chih_sigma * b_cplx[mask] * numerator[mask] / denominator[mask]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = complex(1.)  # np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    # mask = np.zeros_like(im, dtype=bool)
    # mask[im <= 400] = True

    reflect_p = bp * chih_pi / denominator
    # reflect_p[mask] = bp[mask] * chih_pi * numerator[mask] / denominator[mask]

    return reflect_s, reflect_p, b, kout_grid


def _bragg_r1r2_kappa1d_kapp2d(lambdaH, scriptG, chi0, b, alpha, gamma0, klen, d):
    """
    Abbreviations for some calculation of the
    reflectivity

    Notice that, for numerical stability, we are here forcing
    the imaginary part of kappa1 * d to be positive so that the
    phase term in the exponential does not explode

    :param lambdaH:
    :param scriptG:
    :param chi0:
    :param b:
    :param d:
    :param alpha:
    :param gamma0:
    :param klen:
    :return:
    """
    y = klen * lambdaH / 2 / gamma0
    y *= (b * alpha + chi0 * (1 - b))

    # Get kappa1 * d and kappa2 * d
    kappa1d = chi0 * klen * d / 2 / gamma0
    kappa1d += d / lambdaH * (-y + np.sqrt(y ** 2 + b / np.abs(b)))

    kappa2d = chi0 * klen * d / 2 / gamma0
    kappa2d -= d / lambdaH * (-y + np.sqrt(y ** 2 + b / np.abs(b)))

    # Get the R1 and R2 defined in the note
    r1, r2 = (scriptG * (-y + np.sqrt(y ** 2 + b / np.abs(b))),
              scriptG * (-y - np.sqrt(y ** 2 + b / np.abs(b))),)

    # Get the positiveness of the imaginary part of kappa1 * d
    mask = np.zeros_like(kappa1d, dtype=bool)
    mask[kappa1d.imag < 0] = True

    # Get a tmp array
    tmp = np.copy(kappa1d)
    kappa1d[mask] = kappa2d[mask]
    kappa2d[mask] = tmp[mask]

    tmp = np.copy(r1)
    r1[mask] = r2[mask]
    r2[mask] = tmp[mask]

    return r1, r2, kappa1d, kappa2d


def get_ROO_ROH_for_Bragg(kin_grid, d, h, n,
                          chi0, chih_sigma, chihbar_sigma,
                          chih_pi, chihbar_pi):
    """
    The derivation follows that from Yuri's paper:


    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """

    # -----------------------------------------
    # Get the Output wave-vector
    # -----------------------------------------
    # Create holder to save the reflectivity and output momentum
    klen_grid = np.linalg.norm(kin_grid, axis=-1)

    gamma_0 = np.divide(np.dot(kin_grid, n), klen_grid)
    gamma_h = np.divide(np.dot(kin_grid + h[np.newaxis, :], n), klen_grid)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * np.dot(kin_grid, h) + np.linalg.norm(h) ** 2, np.square(klen_grid))

    # Get the surface momentum transfer from phase matching condition
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)
    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)

    # Get the output wave-vector
    kout_grid = kin_grid + h[np.newaxis, :] + m_trans[:, np.newaxis] * n[np.newaxis, :]

    # ------------------------------------------------------------
    # Get the parameters for the reflectivity
    # ------------------------------------------------------------
    lambdaH_sigma = np.sqrt(gamma_0 * np.abs(gamma_h) / chih_sigma / chihbar_sigma) / klen_grid
    lambdaH_pi = np.sqrt(gamma_0 * np.abs(gamma_h) / chih_pi / chihbar_pi) / klen_grid
    scriptG_sigma = np.sqrt(np.abs(b)) * np.sqrt(chih_sigma * chihbar_sigma) / chihbar_sigma
    scriptG_pi = np.sqrt(np.abs(b)) * np.sqrt(chih_pi * chihbar_pi) / chihbar_pi

    # ------------------------------------------------------------
    # Get the reflectivity for the sigma polarization
    # ------------------------------------------------------------

    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    # mask = np.zeros_like(im, dtype=bool)
    # mask[im <= 100] = True

    reflect_s = chih_sigma * b_cplx / denominator
    # reflect_s[mask] = chih_sigma * b_cplx[mask] * numerator[mask] / denominator[mask]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = complex(1.)  # np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    # mask = np.zeros_like(im, dtype=bool)
    # mask[im <= 400] = True

    reflect_p = bp * chih_pi / denominator
    # reflect_p[mask] = bp[mask] * chih_pi * numerator[mask] / denominator[mask]

    return reflect_s, reflect_p, b, kout_grid


def get_R00_R0H_for_Laue(kin_grid, d, h, n,
                         chi0, chih_sigma, chihbar_sigma,
                         chih_pi, chihbar_pi):
    """
    This function aims to get the info quickly with cpu.

    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """
    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(kin_grid, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = np.linalg.norm(kin_grid, axis=-1)
    dot_hn = np.dot(h, n)
    h_square = np.sum(np.square(h))

    # Get gamma and alpha and b
    dot_kn = np.dot(kin_grid, n)
    dot_kh = np.dot(kin_grid, h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + dot_hn, klen_grid)
    # print(gamma_h)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)

    # mask = np.zeros_like(sqrt_gamma_alpha, dtype=bool)
    # mask[np.abs(-gamma_h - sqrt_gamma_alpha) > np.abs(-gamma_h + sqrt_gamma_alpha)] = True

    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)
    # m_trans[mask] = np.multiply(klen_grid[mask], -gamma_h[mask] + sqrt_gamma_alpha[mask])

    # Update the kout_grid
    kout_grid[:, 0] = kin_grid[:, 0] + h[0] + m_trans * n[0]
    kout_grid[:, 1] = kin_grid[:, 1] + h[1] + m_trans * n[1]
    kout_grid[:, 2] = kin_grid[:, 2] + h[2] + m_trans * n[2]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get alpha tidle
    alpha_tidle = (alpha * b + chi0 * (1. - b)) / 2.

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + np.multiply(b_cplx, chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    # mask = np.zeros_like(im, dtype=bool)
    # mask[im <= 100] = True

    reflect_s = chih_sigma * b_cplx / denominator
    # reflect_s[mask] = chih_sigma * b_cplx[mask] * numerator[mask] / denominator[mask]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = complex(1.)  # np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * d / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * d / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    # mask = np.zeros_like(im, dtype=bool)
    # mask[im <= 400] = True

    reflect_p = bp * chih_pi / denominator
    # reflect_p[mask] = bp[mask] * chih_pi * numerator[mask] / denominator[mask]

    return reflect_s, reflect_p, b, kout_grid


# --------------------------------------------------------------
#          Geometry functions
# --------------------------------------------------------------
def get_intersection(initial_position, k, normal, surface_point):
    """
    Assume that a line starts from point s along the direction k. It will intersect with
    the plane that passes through point x0 and has normal direction n. The function find the
    resulted intersection point.

    This function assumes that the arguments are arrays of points.

    :param initial_position: array of shape [3], starting points for each array
    :param k: array of shape [3], the direction for each array
    :param normal: array of shape [3], the normal direction of the surface
    :param surface_point: array of shape [3], one point on this surface
    :return:
    """
    # The intersection points for each array
    x = np.copy(initial_position)

    # Do the math
    tmp = np.divide(np.dot(surface_point - initial_position, normal), np.dot(k, normal))
    x += tmp * k
    return x


# --------------------------------------------------------------
#          Geometric operation
# --------------------------------------------------------------
def get_total_path_length(point_list):
    """
    Get the path length of a series of points

    :param point_list:
    :return:
    """
    number = len(point_list)
    total_path = 0.
    for idx in range(number - 1):
        total_path += np.linalg.norm(point_list[idx + 1] - point_list[idx])

    return total_path


# ---------------------------------------------------------------------------
#                     Grating
# ---------------------------------------------------------------------------
def get_grating_output_momentum(grating_wavenum, k_vec):
    """
    Calculate output momentum of the grating with the specified wave number and
    the corresponding incident k_vec

    :param grating_wavenum:
    :param k_vec:
    :return:
    """
    wavenum_reshape = np.reshape(grating_wavenum, (1, 3))
    return k_vec + wavenum_reshape


def get_grating_wavenumber_1d(direction, period, order):
    """

    :param direction:
    :param period:
    :param order:
    :return:
    """
    return order * direction * 2. * np.pi / period


def get_grating_period(dtheta, klen_in):
    """
    Derive the grating period based on the deviation angle and the incident wave number.
    Here, one assume that the incident wave vector is perpendicular to the the grating surface.

    :param dtheta:
    :param klen_in:
    :return:
    """
    period = 2 * np.pi / klen_in / np.tan(dtheta)
    return period


def get_square_grating_transmission(kin, height_vec, ab_ratio, base, refractive_index, order, grating_k):
    # The argument for exp(ik(n-1)h)
    nhk = np.dot(height_vec, kin).astype(np.complex128) * (refractive_index - complex(1.))

    # The argument for exp(ik(n-1)t) for the phase different and absorption from
    # the base of the grating
    thick_k_n = np.dot(base, kin).astype(np.complex128) * (refractive_index - complex(1.))

    first_factor = complex(1.
                           - np.cos(two_pi * order * ab_ratio),
                           - np.sin(two_pi * order * ab_ratio))
    second_factor = complex(1.) - complex(np.exp(-nhk.imag) * np.cos(nhk.real),
                                          np.exp(-nhk.imag) * np.sin(nhk.real))

    # Factor from the base
    factor_base = complex(np.cos(thick_k_n.real) * np.exp(-thick_k_n.imag),
                          np.sin(thick_k_n.real) * np.exp(-thick_k_n.imag))

    factor = 1.j / complex(2. * np.pi * order) * first_factor * second_factor * factor_base

    # Step 3: Update the momentum and the length of the momentum
    kout = kin + order * grating_k
    klen = np.linalg.norm(kout)

    return factor, kout, klen


def get_square_grating_0th_transmission(kin, height_vec, refractive_index, ab_ratio, base):
    # The argument for exp(ik(n-1)h)
    nhk = np.dot(height_vec, kin).astype(np.complex128) * (refractive_index - complex(1.))

    # The argument for exp(ik(n-1)t) for the phase different and absorption from
    # the base of the grating
    thick_k_n = np.dot(base, kin).astype(np.complex128) * (refractive_index - complex(1.))

    # Factor from the base
    factor_base = complex(np.cos(thick_k_n.real) * np.exp(-thick_k_n.imag),
                          np.sin(thick_k_n.real) * np.exp(-thick_k_n.imag))

    pre_factor = complex(1.) - complex(np.exp(-nhk.imag) * np.cos(nhk.real),
                                       np.exp(-nhk.imag) * np.sin(nhk.real))

    factor = (complex(1.) - complex(ab_ratio) * pre_factor) * factor_base

    return factor


# ---------------------------------------------------------------------------
#                  Get k mesh
# ---------------------------------------------------------------------------

def get_k_mesh_3d(number_x, number_y, number_z, delta_e_x, delta_e_y, delta_e_z):
    # Get the corresponding energy mesh
    energy_grid_x = np.linspace(start=- delta_e_x,
                                stop=+ delta_e_x,
                                num=number_x)
    energy_grid_y = np.linspace(start=- delta_e_y,
                                stop=+ delta_e_y,
                                num=number_y)
    energy_grid_z = np.linspace(start=- delta_e_z,
                                stop=+ delta_e_z,
                                num=number_z)

    # Get the k grid
    kx_grid = np.ascontiguousarray(kev_to_wavevec_length(energy=energy_grid_x))
    ky_grid = np.ascontiguousarray(kev_to_wavevec_length(energy=energy_grid_y))
    kz_grid = np.ascontiguousarray(kev_to_wavevec_length(energy=energy_grid_z))

    # Get the spatial mesh along x axis
    dkx = kev_to_wavevec_length(energy=energy_grid_x[1] - energy_grid_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=-x_range / 2., stop=x_range / 2., num=number_x)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = kev_to_wavevec_length(energy=energy_grid_y[1] - energy_grid_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_y)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = kev_to_wavevec_length(energy=energy_grid_z[1] - energy_grid_z[0])
    z_range = np.pi * 2 / dkz

    z_idx = np.linspace(start=-z_range / 2., stop=z_range / 2., num=number_z)
    z_idx_tick = ["{:.2f}".format(x) for x in z_idx]

    # Assemble the indexes and labels
    axis_info = {"x_range": x_range,
                 "x_idx": x_idx,
                 "x_idx_tick": x_idx_tick,
                 "dkx": dkx,
                 "energy_grid_x": energy_grid_x,

                 "y_range": y_range,
                 "y_idx": y_idx,
                 "y_idx_tick": y_idx_tick,
                 "dky": dky,
                 "energy_grid_y": energy_grid_y,

                 "z_range": z_range,
                 "z_idx": z_idx,
                 "z_idx_tick": z_idx_tick,
                 "dkz": dkz,
                 "energy_grid_z": energy_grid_z,
                 "z_time_idx": np.divide(z_idx, c),
                 "z_time_tick": ["{:.2f}".format(x) for x in np.divide(z_idx, c)],

                 "de_x_in_meV": np.linspace(start=- delta_e_x * 1e6,
                                            stop=+ delta_e_x * 1e6,
                                            num=number_x)}
    return kx_grid, ky_grid, kz_grid, axis_info


def get_k_mesh_1d(number, energy_range):
    """
    Get a (n,3) numpy array as the wave vector array.

    Here, the output[:,2] contains non-zero values.
    I.e. I assume that the propagation direction is along z direction.

    :param number:
    :param energy_range:
    :return:
    """
    # Get the corresponding energy mesh
    energy_grid_z = np.linspace(start=- energy_range,
                                stop=+ energy_range,
                                num=number)

    # Get the k grid
    kz_grid = np.ascontiguousarray(kev_to_wavevec_length(energy=energy_grid_z))

    # Get a wave vector array
    k_grid = np.zeros((kz_grid.shape[0], 3), dtype=np.float64)
    k_grid[:, 2] = kz_grid[:]

    # Get the spatial mesh along z axis
    dkz = kev_to_wavevec_length(energy=energy_grid_z[1] - energy_grid_z[0])
    z_range = np.pi * 2 / dkz

    z_idx = np.linspace(start=-z_range / 2., stop=z_range / 2., num=number)
    z_idx_tick = ["{:.2f}".format(x) for x in z_idx]

    # Assemble the indexes and labels
    axis_info = {"spatial_range": z_range,
                 "spatial_grid": z_idx,
                 "spatial_grid_tick": z_idx_tick,
                 "dkz": dkz,
                 "energy_grid": energy_grid_z,
                 "time_grid": np.divide(z_idx, c),
                 "time_grid_tick": ["{:.2f}".format(x) for x in np.divide(z_idx, c)],
                 }

    return k_grid, axis_info


def get_coordinate(nx, ny, nz, dx, dy, dz, k0=0):
    """

    :param nx:
    :param ny:
    :param nz:
    :param dx:
    :param dy:
    :param dz:
    :param k0:
    :return:
    """
    xCoor = np.arange(nx) * dx - nx * dx / 2.
    yCoor = np.arange(ny) * dy - ny * dy / 2.
    zCoor = np.arange(nz) * dz - nz * dz / 2.
    tCoor = zCoor / c

    # Get k mesh
    kxCoor = np.fft.fftshift(np.fft.fftfreq(nx, d=dx) * 2 * np.pi)
    kyCoor = np.fft.fftshift(np.fft.fftfreq(ny, d=dy) * 2 * np.pi)
    kzCoor = np.fft.fftshift(np.fft.fftfreq(nz, d=dz) * 2 * np.pi)
    kzCoor += k0

    # Convert wavevector to photon energy for illustration
    ExCoor = wavevec_to_kev(kxCoor)
    EyCoor = wavevec_to_kev(kyCoor)
    EzCoor = wavevec_to_kev(kzCoor - k0)

    return xCoor, yCoor, zCoor, tCoor, kxCoor, kyCoor, kzCoor, ExCoor, EyCoor, EzCoor


# ----------------------------------------------------------------------------
#               For telescope
# ----------------------------------------------------------------------------

def get_telescope_kout(optical_axis, kin):
    """

    :param optical_axis:
    :param kin:
    :return:
    """
    # Get k parallel
    k_parallel = np.dot(optical_axis, kin)
    kvec_parallel = optical_axis * k_parallel

    # Get kout
    kout = 2 * kvec_parallel - kin

    return kout


def get_mirror_kout(kin,
                    normal,
                    ):
    # Get the projection of the kin along the direction of normal
    # we always assume that the normal direction of the mirror is point towards the outer direction of the mirror
    # Therefore the reflection is
    # kout = kin - (kin . normal) normal
    proj_len = np.dot(kin, normal)
    kout = kin - normal * (2 * proj_len)
    return kout


def get_telescope_kout_list(optical_axis, kin):
    """

    :param optical_axis:
    :param kin:
    :return:
    """
    k_parallel = np.dot(kin, optical_axis)
    kvec_parallel = k_parallel[:, np.newaxis] * optical_axis[np.newaxis, :]

    # Get kout
    kout = 2 * kvec_parallel - kin
    return kout


def get_image_from_telescope_for_cpa(object_point, lens_axis, lens_position, focal_length):
    """
    Get the image point after the telescope.

    :param object_point:
    :param lens_axis:
    :param lens_position:
    :param focal_length:
    :return:
    """

    # Object position with respect to the lens
    object_position = lens_position - object_point
    object_distance = np.dot(lens_axis, object_position)
    image_vector = object_position - object_distance * lens_axis

    # Image position
    tmp_length = 4 * focal_length - object_distance
    image_position = lens_position + tmp_length * lens_axis

    # This is the image point of the source point.
    image_position -= image_vector

    return image_position


#####################################################################
#     New functions
#####################################################################
def get_axis(number, resolution):
    """
    Generate real space and reciprocal space coordinate with specified numbers and resolution

    :param number:
    :param resolution:
    :return:
    """
    left_end = -int(number) // 2
    right_end = int(number) + left_end

    # Create the real space axis
    real_axis = np.arange(left_end, right_end) * resolution

    # Find wave number range and resolution
    wavevec_range = np.pi * 2. / resolution
    wavevec_reso = wavevec_range / number

    # Create wave number axis
    wavevec_axis = np.arange(left_end, right_end) * wavevec_reso

    # Get the corresponding energy range
    energy_range = wavevec_to_kev(wavevec=wavevec_range)

    return energy_range, real_axis, wavevec_axis


def get_axes_3d(numbers, resolutions):
    holder = {"energy range": {},
              "real axis": {},
              'wavevec axis': {}}

    axis_name = ['x', 'y', 'z']

    for idx in range(3):
        tmp_energy, tmp_real, tmp_wavenum = get_axis(number=numbers[idx],
                                                     resolution=resolutions[idx])
        holder['energy range'].update({axis_name[idx]: np.copy(tmp_energy)})
        holder['real axis'].update({axis_name[idx]: np.copy(tmp_real)})
        holder['wavevec axis'].update({axis_name[idx]: np.copy(tmp_wavenum)})

    return holder


def get_fft_mesh_2d(dy, ny, yc, dz, nz, zc):
    # get the y axis
    ky_list = np.fft.fftshift(np.fft.fftfreq(ny, d=dy) * 2 * np.pi)
    ky_list += yc

    # Get the z axis
    kz_list = np.fft.fftshift(np.fft.fftfreq(nz, d=dz) * 2 * np.pi)
    kz_list += zc

    k_grid = np.zeros((ny, nz, 3), dtype=np.float64)
    k_grid[:, :, 1] = ky_list[:, np.newaxis]
    k_grid[:, :, 2] = kz_list[np.newaxis, :]

    return k_grid


########################################################################################################################
#                    For I/O operation
########################################################################################################################
def save_branch_result_to_h5file(file_name, io_type, branch_name,
                                 result_3d_dict, result_2d_dict, check_dict):
    with h5.File(file_name, io_type) as h5file:
        group = h5file.create_group(branch_name)
        # Save the meta data
        group_check = group.create_group('check')
        for entry in list(check_dict.keys()):
            group_check.create_dataset(entry, data=check_dict[entry])

        group_2d = group.create_group('result_2d')
        for entry in list(result_2d_dict.keys()):
            group_2d.create_dataset(entry, data=result_2d_dict[entry])

        group_3d = group.create_group('result_3d')
        for entry in list(result_3d_dict.keys()):
            group_3d.create_dataset(entry, data=result_3d_dict[entry])


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


########################################################################################################################
#                     Curve analysis
########################################################################################################################
def get_fwhm(coordinate, curve_values, center=False):
    """
    Get the FWHM in the straightforward way.
    However, notice that, when one calculate the FWHM in this way, the result
    is sensitive to small perturbations of the curve's shape.

    :param coordinate:
    :param curve_values:
    :param center: Whether return the coordinate of the center of the region within FWHM
    :return:
    """
    # Get the half max value
    half_max = np.max(curve_values) / 2.

    # Get the indexes for the range.
    indexes = np.arange(len(coordinate), dtype=np.int64)
    mask = np.zeros_like(indexes, dtype=bool)
    mask[curve_values >= half_max] = True

    indexes_above = indexes[mask]

    # Get the ends of the region
    left_idx = np.min(indexes_above)
    right_idx = np.max(indexes_above)

    # Convert the indexes into coordinates
    fwhm = coordinate[right_idx] - coordinate[left_idx]

    if center:
        distribution = curve_values[mask]
        distribution /= np.sum(distribution)

        coordinate_roi = coordinate[mask]

        mean = np.sum(np.multiply(distribution, coordinate_roi))

        return fwhm, mean
    else:
        return fwhm


def get_statistics(distribution, coor=None):
    # Get a holder for the analysis result
    holder = {"2d slice": {},
              "2d projection": {},
              "1d slice": {},
              "1d projection": {},
              }

    # Get distribution shape
    dist_shape = np.array(distribution.shape, dtype=np.int64)
    center_position = dist_shape // 2
    x_c = center_position[0]
    y_c = center_position[1]
    z_c = center_position[2]

    # Get the 2d slice
    tmp_xy = distribution[:, :, z_c]
    tmp_xz = distribution[:, y_c, :]
    tmp_yz = distribution[x_c, :, :]
    holder['2d slice'].update({"xy": np.copy(tmp_xy),
                               "xz": np.copy(tmp_xz),
                               "yz": np.copy(tmp_yz),
                               })

    # Get 2d projection
    tmp_xy = np.sum(distribution, axis=2)
    tmp_xz = np.sum(distribution, axis=1)
    tmp_yz = np.sum(distribution, axis=0)
    holder['2d projection'].update({"xy": np.copy(tmp_xy),
                                    "xz": np.copy(tmp_xz),
                                    "yz": np.copy(tmp_yz),
                                    })

    # Get 1d slice
    holder['1d slice'].update({"x": np.copy(distribution[:, y_c, z_c]),
                               "y": np.copy(distribution[x_c, :, z_c]),
                               "z": np.copy(distribution[x_c, y_c, :]),
                               })

    # Get 1d projection
    holder['1d projection'].update({"x": np.copy(np.sum(tmp_xy, axis=1)),
                                    "y": np.copy(np.sum(tmp_xy, axis=0)),
                                    "z": np.copy(np.sum(tmp_xz, axis=0)),
                                    })

    if coor is not None:
        # Create an entry called sigma to get the sigma and FWHM
        holder.update({"sigma": {},
                       "fwhm": {}})

        for axis in ['x', 'y', 'z']:
            # Normalize to get the distribution
            tmp = np.copy(holder['1d projection'][axis])
            prob_dist = tmp / np.sum(tmp)

            # Get sigma
            mean = np.sum(np.multiply(prob_dist, coor[axis]))
            std = np.sum(np.multiply(np.square(coor[axis]), prob_dist)) - np.square(mean)

            holder["sigma"].update({axis: np.copy(std)})

            # Get fwhm
            holder["fwhm"].update({axis: get_fwhm(coordinate=coor[axis], curve_values=prob_dist)})

    return holder


def get_gaussian_fit(curve, coordinate):
    """
    Fit the target curve with a Gaussian function
    :param curve:
    :param coordinate:
    :return:
    """
    total = np.sum(curve)
    distribution = curve / total

    mean = np.sum(np.multiply(distribution, coordinate))
    std = np.sum(np.multiply(distribution, np.square(coordinate))) - mean ** 2
    std = np.sqrt(std)

    gaussian_fit = np.exp(- np.square(coordinate - mean) / 2. / std ** 2)
    gaussian_fit /= np.sum(gaussian_fit)

    gaussian_fit *= total
    return gaussian_fit


############################################################
#     Show stats
############################################################
def show_stats_2d(stats_holder, fig_height, fig_width):
    fig, axes = plt.subplots(nrows=3, ncols=2)

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    #########################################
    #    xy slice and projection
    #########################################
    im00 = axes[0, 0].imshow(stats_holder['2d slice']['xy'], cmap='jet')
    axes[0, 0].set_title("XY Slice")
    axes[0, 0].set_axis_off()
    fig.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(stats_holder['2d projection']['xy'], cmap='jet')
    axes[0, 1].set_title("XY Projection")
    axes[0, 1].set_axis_off()
    fig.colorbar(im01, ax=axes[0, 1])

    #########################################
    #    xz slice and projection
    #########################################
    im10 = axes[1, 0].imshow(stats_holder['2d slice']['xz'], cmap='jet', aspect="auto")
    axes[1, 0].set_title("XZ Slice")
    axes[1, 0].set_axis_off()
    fig.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(stats_holder['2d projection']['xz'], cmap='jet', aspect="auto")
    axes[1, 1].set_title("XZ Projection")
    axes[1, 1].set_axis_off()
    fig.colorbar(im11, ax=axes[1, 1])

    #########################################
    #    yz slice and projection
    #########################################
    im20 = axes[2, 0].imshow(stats_holder['2d slice']['yz'], cmap='jet', aspect="auto")
    axes[2, 0].set_title("YZ Slice")
    axes[2, 0].set_axis_off()
    fig.colorbar(im20, ax=axes[2, 0])

    im21 = axes[2, 1].imshow(stats_holder['2d projection']['yz'], cmap='jet', aspect="auto")
    axes[2, 1].set_title("YZ Projection")
    axes[2, 1].set_axis_off()
    fig.colorbar(im21, ax=axes[2, 1])

    plt.show()


def show_stats_1d(stats_holder, coor, fig_height, fig_width):
    fig, axes = plt.subplots(nrows=3, ncols=2)

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    #########################################
    #    x slice and projection
    #########################################
    axes[0, 0].plot(coor['x'], stats_holder['1d slice']['x'])
    axes[0, 0].set_title("X Slice")

    axes[0, 1].plot(coor['x'], stats_holder['1d projection']['x'])
    axes[0, 1].set_title("X Projection")

    #########################################
    #    y slice and projection
    #########################################
    axes[1, 0].plot(coor['y'], stats_holder['1d slice']['y'])
    axes[1, 0].set_title("Y Slice")

    axes[1, 1].plot(coor['y'], stats_holder['1d projection']['y'])
    axes[1, 1].set_title("Y Projection")

    #########################################
    #    yz slice and projection
    #########################################
    axes[2, 0].plot(coor['z'], stats_holder['1d slice']['z'])
    axes[2, 0].set_title("Z Slice")

    axes[2, 1].plot(coor['z'], stats_holder['1d projection']['z'])
    axes[2, 1].set_title("Z Projection")

    plt.show()
