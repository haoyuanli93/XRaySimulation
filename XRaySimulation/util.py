import numpy as np

"""
This module is the lowest-level module. It does not depend on another modules.
"""
pi = np.pi
hbar = 0.0006582119514  # This is the reduced planck constant in keV/fs

c = 299792458. * 1e-9  # The speed of light in um / fs


# --------------------------------------------------------------
#               Simple functions
# --------------------------------------------------------------
def exp_stable(x):
    """
    This function calculate the exponential of a complex variable in a stable way.
    :param x:
    :return:
    """
    re = x.real
    im = x.imag

    im = np.mod(im, 2 * pi)
    phase = np.cos(im) + 1.j * np.sin(im)

    # Build a mask to find too small values
    # Assume that when re is less than -100, define the value to be 0
    magnitude = np.zeros_like(re, dtype=np.complex128)
    magnitude[re >= -100] = np.exp(re[re >= -100]) + 0.j
    return np.multiply(magnitude, phase)


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


def kev_to_wave_number(energy):
    return energy / hbar / c


def petahertz_frequency_to_kev(frequency):
    return hbar * 2 * pi * frequency


def petahertz_angular_frequency_to_kev(angular_frequency):
    return hbar * angular_frequency


def petahertz_angular_frequency_to_wave_number(angular_frequency):
    return angular_frequency / c


def wave_number_to_kev(wavevec):
    return wavevec * hbar * c


# --------------------------------------------------------------
#          Get output wave vectors
# --------------------------------------------------------------
def get_bragg_kout(kin, h, normal, compare_length=False):
    """
    This function produce the output wave vector from a Bragg reflection.

    :param kin: (3,) numpy array. The incident wave vector
    :param h: The reciprocal lattice of the crystal
    :param normal: The normal direction of the reflection surface.
                    For a bragg reflection, n is pointing to the inside of the crystal.
    :param compare_length: Whether compare the length of the incident wave vector and the output wave vector

    :return: kout: (3,) numpy array. The diffraction wave vector.
            ratio: When compare_length=True, the second output is the ratio between the incident wave number
                                        and the output wave number.
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

    if compare_length:
        return kout, klen / l2_norm(kout)
    else:
        return kout


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
def get_total_path_length(intersect_list):
    """
    Get the path length of a series of points

    :param intersect_list:
    :return:
    """
    number = len(intersect_list)
    total_path = 0.
    for l in range(number - 1):
        total_path += l2_norm(intersect_list[l + 1] -
                              intersect_list[l])

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


# ---------------------------------------------------------------------------
#                     IO
# ---------------------------------------------------------------------------


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
    kx_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_x))
    ky_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_y))
    kz_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid_z))

    # Get the spatial mesh along x axis
    dkx = kev_to_wave_number(energy=energy_grid_x[1] - energy_grid_x[0])
    x_range = np.pi * 2 / dkx

    x_idx = np.linspace(start=-x_range / 2., stop=x_range / 2., num=number_x)
    x_idx_tick = ["{:.2f}".format(x) for x in x_idx]

    # Get the spatial mesh along y axis
    dky = kev_to_wave_number(energy=energy_grid_y[1] - energy_grid_y[0])
    y_range = np.pi * 2 / dky

    y_idx = np.linspace(start=-y_range / 2., stop=y_range / 2., num=number_y)
    y_idx_tick = ["{:.2f}".format(x) for x in y_idx]

    # Get the spatial mesh along z axis
    dkz = kev_to_wave_number(energy=energy_grid_z[1] - energy_grid_z[0])
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


# ---------------------------------------------------
#              For DuMond Diagram
# ---------------------------------------------------
def get_klen_and_angular_mesh(k_num, theta_num, phi_num, energy_range, theta_range, phi_range):
    # Get the corresponding energy mesh
    energy_grid = np.linspace(start=energy_range[0], stop=energy_range[1], num=k_num)
    # Get the k grid
    klen_grid = np.ascontiguousarray(kev_to_wave_number(energy=energy_grid))

    # Get theta grid
    theta_grid = np.linspace(start=theta_range[0], stop=theta_range[1], num=theta_num)

    # Get phi grid
    phi_grid = np.linspace(start=phi_range[0], stop=phi_range[1], num=phi_num)

    info_dict = {"energy_grid": energy_grid,
                 "klen_grid": klen_grid,
                 "theta_grid": theta_grid,
                 "phi_grid": phi_grid}
    return info_dict


###############################################################################################
###############################################################################################
#
#    The following code handle bragg reflectivity with cpu in details
#
###############################################################################################
###############################################################################################
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

    b = np.divide(gamma_0, gamma_h)
    b_cplx = b.astype(np.complex128)
    alpha = np.divide(2 * dot_kh + h_square, np.square(klen_grid))

    # Get momentum tranfer
    sqrt_gamma_alpha = np.sqrt(gamma_h ** 2 - alpha)

    mask = np.zeros_like(sqrt_gamma_alpha, dtype=np.bool)
    mask[np.abs(-gamma_h - sqrt_gamma_alpha) > np.abs(-gamma_h + sqrt_gamma_alpha)] = True

    m_trans = np.multiply(klen_grid, -gamma_h - sqrt_gamma_alpha)
    m_trans[mask] = np.multiply(klen_grid[mask], -gamma_h[mask] + sqrt_gamma_alpha[mask])

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
    p_value = np.sum(np.multiply(kout_grid, kin_grid), axis=-1) / np.square(klen_grid)
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


def get_rocking_curve(kin_list, crystal_list):
    """
    Get the reflectivity for each kin.

    :param kin_list:
    :param crystal_list:
    :return:
    """
    k_num = kin_list.shape[0]
    x_num = len(crystal_list)

    # Define some holder to save data
    kout_list = []
    reflect_p_list = []
    reflect_s_list = []
    reflect_p_total = np.ones(k_num, dtype=np.complex128)
    reflect_s_total = np.ones(k_num, dtype=np.complex128)
    b_total = np.ones(k_num, dtype=np.float64)

    kout_tmp = np.copy(kin_list)
    for x in range(x_num):
        # Get info
        (reflect_s_tmp,
         reflect_p_tmp,
         b_tmp,
         kout_tmp) = get_bragg_reflection_array(kin_grid=kout_tmp,
                                                d=crystal_list[x].d,
                                                h=crystal_list[x].h,
                                                n=crystal_list[x].normal,
                                                chi0=crystal_list[x].chi0,
                                                chih_sigma=crystal_list[x].chih_sigma,
                                                chihbar_sigma=crystal_list[x].chihbar_sigma,
                                                chih_pi=crystal_list[x].chih_pi,
                                                chihbar_pi=crystal_list[x].chihbar_pi)
        b_tmp = np.abs(b_tmp)

        # Save info to holders
        kout_list.append(np.copy(kout_tmp))
        reflect_p_list.append(np.square(np.abs(reflect_p_tmp)) / b_tmp)
        reflect_s_list.append(np.square(np.abs(reflect_s_tmp)) / b_tmp)

        # Update the total reflectivity
        reflect_s_total = np.multiply(reflect_s_total, reflect_s_tmp)
        reflect_p_total = np.multiply(reflect_p_total, reflect_p_tmp)
        b_total = np.multiply(b_total, b_tmp)

    reflect_s_total = np.square(np.abs(reflect_s_total)) / b_total
    reflect_p_total = np.square(np.abs(reflect_p_total)) / b_total

    return reflect_s_total, reflect_p_total, reflect_s_list, reflect_p_list, kout_list


###############################################################################################
###############################################################################################
#
#    The following code handle grating transmission with cpu
#
###############################################################################################
###############################################################################################
def get_square_grating_transmission(k, m, n, h, a, b, base=0.):
    """
    k: Wave number of the photon = 2 * pi * c / lambda
    m: The order of diffraction
    n: The complex refraction index.
    h: The height of the tooth.
    a: The width of the groove.
    b: The width of the tooth.
    """

    if not isinstance(m, int):
        raise Exception("m is the order of diffraction. This value has to be an integer.")
    if not isinstance(n, complex):
        raise Exception("n is the complex refraction index. This value has to be a complex number.")

    # First consider diffractions that are not the zeroth order.
    if m != 0:
        # Get the real part and the imaginary part of the refraction coefficient
        n_re = n.real - 1
        n_im = n.imag

        term_1 = 1 - np.exp(-k * h * n_im) * (np.cos(k * h * n_re) + 1.j * np.sin(k * h * n_re))
        term_2 = 1 - np.cos(2 * np.pi * b * m / (a + b)) + np.sin(2 * np.pi * b * m / (a + b))

        transmission = np.square(np.abs(term_1 * term_2)) / (4 * (np.pi * m) ** 2)
        transmission *= np.exp(- 2 * k * base * n_im)

        # Then consider the zeroth order
    else:
        # Get the real part and the imaginary part of the refraction coefficicent
        n_re = n.real - 1
        n_im = n.imag

        term_1 = (b + a * np.exp(-k * h * n_im) * (np.cos(k * h * n_re) +
                                                   1.j * np.sin(k * h * n_re))) / (a + b)

        transmission = np.square(np.abs(term_1))
        transmission *= np.exp(- 2 * k * base * n_im)

    return transmission


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


def get_image_from_telescope_for_cpa(object_point, lens_axis, lens_point, focal_length,
                                     get_object_distance=False):
    """
    Get the image point after the telescope.

    :param object_point:
    :param lens_axis:
    :param lens_point:
    :param focal_length:
    :param get_object_distance: If this value is set to True, then also return the object distance
    :return:
    """

    # Object position with respect to the lens
    object_position = lens_point - object_point
    object_distance = np.dot(lens_axis, object_position)
    image_vector = object_position - object_distance * lens_axis

    # Image position
    tmp_length = 4 * focal_length - object_distance
    image_position = lens_point + tmp_length * lens_axis

    # This is the image point of the source point.
    image_position -= image_vector

    return image_position
