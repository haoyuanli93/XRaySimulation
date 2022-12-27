####################################################
#       Align all the devices
####################################################
import XRaySimulation.Crystal


def align_devices(device_list,
                  config_list,
                  requirements,
                  kin,
                  searching_range=0.01,
                  searching_number=10000):
    """
    This function aims to make the alignment procedure easier.

    :param device_list: This is the list of devices.
    :param config_list: This contains the geometry of the crystals.
    :param requirements: Special requirements for the device. For example, if device_list[1] and
                            device_list[2] should form a CC, then this function would make sure
                            that the device_list[2] should have negative reciprocal lattice compared
                            with device_list[1]
    :param kin:
    :param searching_range: This is the searching range for the Bragg reflection around the geometric estimation.
                            The default value is 0.1 rad
    :param searching_number: The searching number in the searching range.
    :return:
    """
    kout_list = [np.copy(kin)]
    device_num = len(device_list)

    for idx in range(device_num):
        device = device_list[idx]

        # The wave vector w.r.t which, the device should be aligned.
        kout = kout_list[-1]

        # Align the device
        if device.type == "Crystal: Bragg Reflection":

            # Align the reciprocal lattice with the incident wave vector
            util.align_crystal_reciprocal_lattice(crystal=device,
                                                  axis=kout)

            # Estimate the Bragg angle
            bragg_estimation = util.get_bragg_angle(wave_length=two_pi / util.l2_norm(kin),
                                                    plane_distance=two_pi / device.h)

            # Align the crystal to the estimated Bragg angle
            rot_mat = util.rot_mat_in_yz_plane(theta=bragg_estimation * config_list[idx])
            device.rotate_wrt_point(rot_mat=rot_mat,
                                    ref_point=device.surface_point)

            # Second: calculate the rocking curve around the estimated angle
            (angles,
             reflect_s,
             reflect_p,
             b_array,
             kout_grid) = XRaySimulation.Crystal.get_bragg_rocking_curve(kin=kout,
                                                                         scan_range=searching_range,
                                                                         scan_number=searching_number,
                                                                         h_initial=device.h,
                                                                         normal_initial=device.normal,
                                                                         thickness=device.d,
                                                                         chi0=device.chi0,
                                                                         chih_sigma=device.chih_sigma,
                                                                         chihbar_sigma=device.chihbar_sigma,
                                                                         chih_pi=device.chih_pi,
                                                                         chihbar_pi=device.chihbar_pi)

            rocking_curve = np.square(np.abs(reflect_s)) / np.abs(b_array)

            # Third: find bandwidth of the rocking curve and the center of the rocking curve
            fwhm, angle_adjust = misc.get_fwhm(coordinate=angles, curve_values=rocking_curve)

            # Fourth: Align the crystal along that direction.
            rot_mat = util.rot_mat_in_yz_plane(theta=angle_adjust)
            device.rotate_wrt_point(rot_mat=rot_mat,
                                    ref_point=device.surface_point)

        if device.type == "Transmissive Grating":
            # Get rotation angle:
            util.align_grating_normal_direction(grating=device,
                                                  axis=kout)

        if device.type == "Transmission Telescope for CPA":
            util.align_telescope_optical_axis(telescope=device,
                                              axis=kout)

        # Update the kout vector for the alignment of the next device
        kout_list.append(util.get_kout(device=device, kin=kout))





def get_bragg_rocking_curve_bk(kin, scan_range, scan_number, h_initial, normal_initial, thickness,
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
        rot_mat = util.rot_mat_in_yz_plane(theta=angles[idx])
        h_array[idx] = rot_mat.dot(h_initial)
        normal_array[idx] = rot_mat.dot(normal_initial)

    # Create holder to save the reflectivity and output momentum
    kout_grid = np.zeros_like(h_array, dtype=np.float64)

    # ------------------------------------------------------------
    #          Step 1: Get output momentum wave vector
    # ------------------------------------------------------------
    # Get some info to facilitate the calculation
    klen = util.l2_norm(kin)
    dot_hn = np.dot(h_initial, normal_initial)
    h_square = util.l2_square(h_initial)

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
