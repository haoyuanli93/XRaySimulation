import numpy as np
from XRaySimulation import misc

"""
This module is the lowest-level module. It does not depend on another modules.
"""
pi = np.pi
two_pi = 2. * np.pi

hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


# --------------------------------------------------------------
#               Simple functions
# --------------------------------------------------------------
def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def l2_square(x):
    return np.sum(np.square(x))


def l2_norm_batch(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


def l2_square_batch(x):
    return np.sum(np.square(x), axis=-1)


# --------------------------------------------------------------
#               Unit conversion
# --------------------------------------------------------------
def kev_to_petahertz_frequency(energy):
    return energy / hbar * 2 * pi


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

    if np.abs(-gammah - np.sqrt(gammah ** 2 - alpha)) > np.abs(-gammah + np.sqrt(gammah ** 2 - alpha)):
        momentum = klen * (-gammah + np.sqrt(gammah ** 2 - alpha))
    else:
        momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))

    # Add momentum transfer
    kout += normal * momentum

    return kout


def get_bragg_kout_array(kin, h, normal):
    """
    This function produce the output wave vector from a Bragg reflection.

    :param kin: (n, 3) numpy array. The incident wave vector
    :param h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.

    :return: kout: (n, 3) numpy array. The diffraction wave vector.
    """

    # kout holder
    kout = kin + h[np.newaxis, :]

    # Incident wave number
    klen = np.sqrt(np.sum(np.square(kin), axis=-1))

    # Get gamma and alpha
    gammah = np.dot(kout, normal) / klen
    alpha = (2 * np.dot(kin, h) + np.dot(h, h)) / np.square(klen)

    # Get the momentum transfer
    momentum = klen * (-gammah - np.sqrt(gammah ** 2 - alpha))

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
    klen_grid = l2_norm_batch(kin_grid)
    dot_hn = np.dot(h, n)
    h_square = l2_square(h)

    # Get gamma and alpha and b
    dot_kn = np.dot(kin_grid, n)
    dot_kh = np.dot(kin_grid, h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + dot_hn, klen_grid)
    #print(gamma_h)

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)

    # mask = np.zeros_like(sqrt_gamma_alpha, dtype=np.bool)
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
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
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
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im <= 400] = True

    reflect_s = chih_sigma * b_cplx / denominator
    reflect_s[mask] = chih_sigma * b_cplx[mask] * numerator[mask] / denominator[mask]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = complex(1.)  # np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
    bp = b_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
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
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im <= 400] = True

    reflect_p = bp * chih_pi / denominator
    reflect_p[mask] = bp[mask] * chih_pi * numerator[mask] / denominator[mask]

    return reflect_s, reflect_p, b, kout_grid


def get_bragg_rocking_curve(kin, scan_range, scan_number, h_initial, normal_initial, thickness,
                            chi0, chih_sigma, chihbar_sigma,
                            chih_pi, chihbar_pi):
    """

    :param kin:
    :param scan_range:
    :param scan_number:
    :param h_initial:
    :param normal_initial:
    :param thickness:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :return:
    """

    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning angle
    angles = np.linspace(start=-scan_range / 2, stop=scan_range / 2, num=scan_number)

    for idx in range(scan_number):
        rot_mat = rot_mat_in_yz_plane(theta=angles[idx])
        h_array[idx] = rot_mat.dot(h_initial)
        normal_array[idx] = rot_mat.dot(normal_initial)

    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(h_array, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen = l2_norm(kin)
    dot_hn = np.dot(h_initial, normal_initial)
    h_square = l2_square(h_initial)

    # Get gamma and alpha and b
    dot_kn_grid = np.dot(normal_array, kin)
    dot_kh_grid = np.dot(h_array, kin)

    gamma_0 = dot_kn_grid / klen
    gamma_h = (dot_kn_grid + dot_hn) / klen

    b_array = np.divide(gamma_0, gamma_h)
    b_list_cplx = b_array.astype(np.complex128)
    alpha_array = (2 * dot_kh_grid + h_square) / np.square(klen)

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha_array)

    mask = np.zeros_like(sqrt_gamma_alpha, dtype=np.bool)
    mask[np.abs(-gamma_h - sqrt_gamma_alpha) > np.abs(-gamma_h + sqrt_gamma_alpha)] = True

    m_trans = klen * (-gamma_h - sqrt_gamma_alpha)
    m_trans[mask] = klen * (-gamma_h[mask] + sqrt_gamma_alpha[mask])

    # Update the kout_grid
    kout_grid[:, 0] = kin[0] + h_array[:, 0] + m_trans * normal_array[:, 0]
    kout_grid[:, 1] = kin[1] + h_array[:, 1] + m_trans * normal_array[:, 1]
    kout_grid[:, 2] = kin[2] + h_array[:, 2] + m_trans * normal_array[:, 2]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    # Get alpha tidle
    alpha_tidle = (alpha_array * b_array + chi0 * (1. - b_array)) / 2.

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + chih_sigma * chihbar_sigma * b_list_cplx)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen * thickness / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im <= 400] = True

    reflect_s = chih_sigma * b_list_cplx / denominator
    reflect_s[mask] = chih_sigma * b_list_cplx[mask] * numerator[mask] / denominator[mask]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for pi polarization
    # ------------------------------------------------------------

    # Get the polarization factor with the asymmetric factor b.
    p_value = complex(1.)  # np.dot(kout_grid, kin) / np.square(klen)
    bp_array = b_list_cplx * p_value

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + bp_array * p_value * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=np.bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen * thickness / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

    # Take care of the exponential
    mask = np.zeros_like(im, dtype=np.bool)
    mask[im <= 400] = True

    reflect_p = bp_array * chih_pi / denominator
    reflect_p[mask] = bp_array[mask] * chih_pi * numerator[mask] / denominator[mask]

    return angles, reflect_s, reflect_p, b_array, kout_grid


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
        total_path += l2_norm(point_list[idx + 1] -
                              point_list[idx])

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
    klen = l2_norm(kout)

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


# -------------------------------------------------------------
#               Alignment
# -------------------------------------------------------------
def align_crystal_reciprocal_lattice(crystal, axis, rot_center=None):
    """

    :param crystal: The crystal to align
    :param axis: The direction along which the reciprocal lattice will be aligned.
    :param rot_center:
    :return:
    """
    if rot_center is None:
        rot_center = crystal.surface_point

    # 1 Get the angle
    cos_val = np.dot(axis, crystal.h) / l2_norm(axis) / l2_norm(crystal.h)
    rot_angle = np.arccos(np.clip(cos_val, -1, 1))

    # print("rot_angle:{:.2e}".format(np.rad2deg(rot_angle)))

    # 2 Try the rotation
    rot_mat = rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, crystal.h)
    # print(new_h)

    if np.dot(new_h, axis) / l2_norm(new_h) / l2_norm(axis) < 0.999:
        # print("aaa")
        rot_mat = rot_mat_in_yz_plane(theta=-rot_angle)

    crystal.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=rot_center)


def align_crystal_geometric_bragg_reflection(crystal, kin, rot_direction=1, rot_center=None):
    if rot_center is None:
        rot_center = crystal.surface_point

    ###########################
    #   Align the recirpocal lattice with kin
    ###########################
    align_crystal_reciprocal_lattice(crystal=crystal, axis=kin, rot_center=rot_center)
    # print(crystal.h)

    ###########################
    #   Alignment based on geometric theory of bragg diffraction
    ###########################
    # Estimate the Bragg angle
    bragg_estimation = get_bragg_angle(wave_length=two_pi / l2_norm(kin),
                                       plane_distance=two_pi / l2_norm(crystal.h))

    # print("Bragg angle:{:.2e}".format(np.rad2deg(bragg_estimation)))

    # Align the crystal to the estimated Bragg angle
    rot_mat = rot_mat_in_yz_plane(theta=(bragg_estimation + np.pi / 2) * rot_direction)

    crystal.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=rot_center)


def align_crystal_dynamical_bragg_reflection(crystal, kin, rot_direction=1,
                                             scan_range=0.0005, scan_number=10000,
                                             rot_center=None,
                                             get_curve=False):
    """
    Align the crystal such that the incident wave vector is at the center of the
    reflectivity curve

    :param crystal:
    :param kin:
    :param rot_direction:
    :param scan_range:
    :param scan_number:
    :param rot_center:
    :param get_curve:
    :return:
    """
    if rot_center is None:
        rot_center = crystal.surface_point

    # Align the crystal with geometric bragg reflection theory
    align_crystal_geometric_bragg_reflection(crystal=crystal,
                                             kin=kin,
                                             rot_direction=rot_direction,
                                             rot_center=rot_center)

    # Align the crystal with dynamical diffraction theory
    (angles,
     reflect_s,
     reflect_p,
     b_array,
     kout_grid) = get_bragg_rocking_curve(kin=kin,
                                          scan_range=scan_range,
                                          scan_number=scan_number,
                                          h_initial=crystal.h,
                                          normal_initial=crystal.normal,
                                          thickness=crystal.thickness,
                                          chi0=crystal.chi0,
                                          chih_sigma=crystal.chih_sigma,
                                          chihbar_sigma=crystal.chihbar_sigma,
                                          chih_pi=crystal.chih_pi,
                                          chihbar_pi=crystal.chihbar_pi)

    # rocking_curve = np.square(np.abs(reflect_s)) / np.abs(b_array)

    # Third: find bandwidth of the rocking curve and the center of the rocking curve
    fwhm, angle_adjust = misc.get_fwhm(coordinate=angles,
                                       curve_values=np.square(np.abs(reflect_s)),
                                       center=True)

    # Fourth: Align the crystal along that direction.
    rot_mat = rot_mat_in_yz_plane(theta=angle_adjust)
    crystal.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=rot_center)
    if get_curve:
        return angles, np.square(np.abs(reflect_s))


def align_grating_normal_direction(grating, axis):
    # 1 Get the angle
    cos_val = np.dot(axis, grating.normal) / l2_norm(axis) / l2_norm(grating.normal)
    rot_angle = np.arccos(cos_val)

    # 2 Try the rotation
    rot_mat = rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, grating.normal)

    if np.dot(new_h, axis) < 0:
        rot_mat = rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    grating.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=grating.surface_point)


def align_telescope_optical_axis(telescope, axis):
    # 1 Get the angle
    cos_val = np.dot(axis, telescope.lens_axis) / l2_norm(axis) / l2_norm(telescope.lens_axis)
    rot_angle = np.arccos(cos_val)

    # 2 Try the rotation
    rot_mat = rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, telescope.lens_axis)

    if np.dot(new_h, axis) < 0:
        rot_mat = rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    telescope.rotate_wrt_point(rot_mat=rot_mat,
                               ref_point=telescope.lens_point)


# --------------------------------------------------------------------------------------------------------------
#       Wrapper functions for different devices
# --------------------------------------------------------------------------------------------------------------
def get_kout(device, kin):
    """
    Get the output wave vector given the incident wave vector

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        kout = get_bragg_kout(kin=kin,
                              h=device.h,
                              normal=device.normal)
        return kout

    if device.type == "Transmissive Grating":
        kout = kin + device.momentum_transfer
        return kout

    if device.type == "Transmission Telescope for CPA":
        kout = get_telescope_kout(optical_axis=device.lens_axis,
                                  kin=kin)
        return kout


def get_intensity_efficiency_sigma_polarization(device, kin):
    """
    Get the output intensity efficiency for the given wave vector
    assuming a monochromatic plane incident wave.

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        tmp = np.zeros((1, 3))
        tmp[0, :] = kin

        (reflect_s,
         reflect_p,
         b,
         kout_grid) = get_bragg_reflection_array(kin_grid=tmp,
                                                 d=device.thickness,
                                                 h=device.h,
                                                 n=device.normal,
                                                 chi0=device.chi0,
                                                 chih_sigma=device.chih_sigma,
                                                 chihbar_sigma=device.chihbar_sigma,
                                                 chih_pi=device.chih_pi,
                                                 chihbar_pi=device.chihbar_pi)

        efficiency = np.square(np.abs(reflect_s)) / np.abs(b)
        return efficiency

    if device.type == "Transmissive Grating":

        # Determine the grating order
        if device.order == 0:
            efficiency = get_square_grating_0th_transmission(kin=kin,
                                                             height_vec=device.h,
                                                             refractive_index=device.n,
                                                             ab_ratio=device.ab_ratio,
                                                             base=device.thick_vec)
        else:
            efficiency, _, _ = get_square_grating_transmission(kin=kin,
                                                               height_vec=device.h,
                                                               ab_ratio=device.ab_ratio,
                                                               base=device.thick_vec,
                                                               refractive_index=device.n,
                                                               order=device.order,
                                                               grating_k=device.momentum_transfer)
        # Translate to the intensity efficiency
        return np.square(np.abs(efficiency))

    if device.type == "Transmission Telescope for CPA":
        return np.square(np.abs(device.efficiency))


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
