####################################################
#       Align all the devices
####################################################
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
             kout_grid) = util.get_bragg_rocking_curve(kin=kout,
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