import numpy as np

from XRaySimulation import util
from XRaySimulation.util import get_bragg_reflection_array


####################################################
#       Single incident wave vector
####################################################
def get_kout(device_list, kin):
    """
    Get the output momentum vectors from each device.

    :param device_list:
    :param kin:
    :return:
    """

    # Create a variable for the kout list.
    # The reason to use is numpy array is that it's easy to determine the
    # total number of kouts generates and with numpy array, it might be more
    # efficient.
    kout_list = np.zeros((len(device_list), 3), dtype=np.float64)
    kout_list[0] = kin[:]

    for idx in range(len(device_list)):

        # Get the device
        device = device_list[idx]

        # Get output wave vector
        if device.type == "Crystal: Bragg Reflection":
            kout_list[idx + 1] = util.get_bragg_kout(kin=kout_list[idx],
                                                     h=device.h,
                                                     normal=device.normal)
        if device.type == "Transmissive Grating":
            kout_list[idx + 1] = kout_list[-1] + device.momentum_transfer

        if device.type == "Transmission Telescope for CPA":
            kout_list[idx + 1] = util.get_telescope_kout(optical_axis=device.lens_axis,
                                                         kin=kout_list[idx])


def get_lightpath(device_list, kin, initial_point, final_plane_point, final_plane_normal):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param final_plane_normal:
    :param final_plane_point:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Path length
    path_length = 0.

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device_list.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device_list.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

        if device.type == "Transmission Telescope for CPA":
            intersection_list.append(util.get_image_from_telescope_for_cpa(object_point=intersection_list[-1],
                                                                           lens_axis=device.lens_axis,
                                                                           lens_point=device.lens_point,
                                                                           focal_length=device.focal_length))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the output wave vector
            kout_list.append(util.get_telescope_kout(optical_axis=device.lens_axis,
                                                     kin=kout_list[-1]))

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                   k=kout_list[-1],
                                                   surface_point=final_plane_point,
                                                   normal=final_plane_normal))
    # Update the path length
    displacement = intersection_list[-1] - intersection_list[-2]
    path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

    return intersection_list, kout_list, path_length


def get_trajectory(device_list, kin, initial_point, final_plane_point, final_plane_normal):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param final_plane_normal:
    :param final_plane_point:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Path length
    path_length = 0.

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device_list.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device_list.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

        if device.type == "Transmission Telescope for CPA":
            # Find the intersection with the first lens
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.lens_axis,
                                                           surface_point=device_list.lens_point))

            # Find the image
            image = util.get_image_from_telescope_for_cpa(object_point=intersection_list[-1],
                                                          lens_axis=device.lens_axis,
                                                          lens_point=device.lens_point,
                                                          focal_length=device.focal_length)

            # Find the kout
            kout_list.append(util.get_telescope_kout(optical_axis=device.lens_axis,
                                                     kin=kout_list[-1]))

            # Find the intersection on the second lens
            point_on_seond_lens = device.lens_point + 2 * device.focal_length * device.lens_axis
            intersection_list.append(util.get_intersection(initial_position=image,
                                                           k=kout_list[-1],
                                                           normal=device.lens_axis,
                                                           surface_point=point_on_seond_lens))

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                   k=kout_list[-1],
                                                   surface_point=final_plane_point,
                                                   normal=final_plane_normal))
    # Update the path length
    displacement = intersection_list[-1] - intersection_list[-2]
    path_length += np.dot(displacement, kout_list[-1]) / util.l2_norm(kout_list[-1])

    return intersection_list, kout_list


def get_output_efficiency(device_list, kin):
    # Create a variable for the kout list.
    # The reason to use is numpy array is that it's easy to determine the
    # total number of kouts generates and with numpy array, it might be more
    # efficient.
    kout_list = np.zeros((len(device_list), 3), dtype=np.float64)
    kout_list[0] = kin[:]

    for idx in range(len(device_list)):

        # Get the device
        device = device_list[idx]

        # Get output wave vector
        if device.type == "Crystal: Bragg Reflection":
            kout_list[idx + 1] = util.get_bragg_kout(kin=kout_list[idx],
                                                     h=device.h,
                                                     normal=device.normal)
        if device.type == "Transmissive Grating":
            kout_list[idx + 1] = kout_list[-1] + device.momentum_transfer

        if device.type == "Transmission Telescope for CPA":
            kout_list[idx + 1] = util.get_telescope_kout(optical_axis=device.lens_axis,
                                                         kin=kout_list[idx])


####################################################
#       Multiple incident wave vector
####################################################
def get_output_efficiency_curve(device_list, kin_list):
    """
    Get the reflectivity for each kin.

    :param kin_list:
    :param device_list:
    :return:
    """
    k_num = kin_list.shape[0]
    x_num = len(device_list)

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
                                                d=device_list[x].d,
                                                h=device_list[x].h,
                                                n=device_list[x].normal,
                                                chi0=device_list[x].chi0,
                                                chih_sigma=device_list[x].chih_sigma,
                                                chihbar_sigma=device_list[x].chihbar_sigma,
                                                chih_pi=device_list[x].chih_pi,
                                                chihbar_pi=device_list[x].chihbar_pi)
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
