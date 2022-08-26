import time

import numpy as np
from numba import cuda

from XRaySimulation import util
from XRaySimulation.GPU import GPUSingleDevice


def get_diffraction_spectrum_sigma_phase_only(kin_grid,
                                              spectrum_in,
                                              device_list,
                                              total_path,
                                              initial_position,
                                              d_num=512):
    """
    This function assume that the incident beam is of sigma polarization
    and does not include polarization mixing effect.

    :param kin_grid:
    :param spectrum_in:
    :param device_list:
    :param total_path:
    :param initial_position: This is a (n,3) numpy array. It contains the trajectory of the center of the beam
                            through the setup.
    :param d_num:
    :return:
    """
    ############################################################################################################
    #                      Step 1: Create holders and get prepared for the simulation
    ############################################################################################################
    device_num = len(device_list)

    # Get meta data
    k_num = kin_grid.shape[0]

    tic = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(kin_grid)
    klen_grid = np.ascontiguousarray(util.l2_norm_batch(kin_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Holder for the phase term and the Jacob matrix term
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    phase_grid = (kin_grid[:, 0] * initial_position[0]
                  + kin_grid[:, 1] * initial_position[1]
                  + kin_grid[:, 2] * initial_position[2]
                  - total_path * klen_grid)

    jacobian_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacobian = cuda.to_device(jacobian_grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    spec_holder = np.ascontiguousarray(spectrum_in)
    cuda_spectrum = cuda.to_device(spec_holder)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))

    ############################################################################################################
    #                           Step 2: Calculate the field and save the data
    ############################################################################################################
    # d_num = 512
    b_num = (k_num + d_num - 1) // d_num

    # --------------------------------------------------------------------
    #  Step 7. Forward propagation
    # --------------------------------------------------------------------
    for device_idx in range(device_num):
        my_device = device_list[device_idx]

        if my_device.type == "Crystal: Bragg Reflection":
            # Get the reflectivity
            GPUSingleDevice.get_bragg_reflection_sigma_phase[b_num, d_num](cuda_reflect_sigma,
                                                                           cuda_phase,
                                                                           cuda_kin_grid,
                                                                           cuda_spectrum,
                                                                           cuda_jacobian,
                                                                           cuda_klen_grid,
                                                                           cuda_kin_grid,
                                                                           my_device.thickness,
                                                                           my_device.h,
                                                                           my_device.normal,
                                                                           np.dot(my_device.normal,
                                                                                  my_device.surface_point),
                                                                           my_device.dot_hn,
                                                                           my_device.h_square,
                                                                           my_device.chi0,
                                                                           my_device.chih_sigma,
                                                                           my_device.chihbar_sigma,
                                                                           k_num)

            GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         k_num)
        elif my_device.type == "Prism":
            # TODO: Finish the theory that include the prism effect
            # Diffracted by the prism
            GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                     cuda_kin_grid,
                                                     my_device.wavevec_delta,
                                                     k_num)

            # Update the wave number
            GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                            cuda_kin_grid,
                                                            3,
                                                            k_num)
        elif my_device.type == "Transmissive Grating":
            # TODO: Finish the theory that include the prism effect
            # Diffracted by the first grating
            GPUSingleDevice.get_square_grating_diffraction_scalar[b_num, d_num](cuda_kin_grid,
                                                                                cuda_spectrum,
                                                                                cuda_klen_grid,
                                                                                cuda_kin_grid,
                                                                                my_device.h,
                                                                                my_device.n,
                                                                                my_device.ab_ratio,
                                                                                my_device.thick_vec,
                                                                                my_device.order,
                                                                                my_device.base_wave_vector,
                                                                                k_num)
        else:
            pass

    # --------------------------------------------------------------------
    #  Step 8. Get the propagation phase
    # --------------------------------------------------------------------
    # Add the phase
    GPUSingleDevice.add_phase_to_scalar_spectrum[b_num, d_num](cuda_phase,
                                                               cuda_spectrum,
                                                               k_num)

    # Add the jacobian
    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_jacobian,
                                                                 cuda_spectrum,
                                                                 cuda_spectrum,
                                                                 k_num)

    ###################################################################################################
    #                                  Finish
    ###################################################################################################
    cuda_spectrum.to_host()
    cuda_reflect_sigma.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_phase.to_host()
    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()
    cuda_jacobian.to_host()

    # Create result dictionary
    sanity_check = {"phase_grid": phase_grid,
                    "jacob_grid": jacobian_grid,
                    "kout_grid": kin_grid
                    }

    reflectivity_holder = {"reflectivity_sigma_tot": reflect_total_sigma, }
    spectrum_holder = {"final_spectrum": spec_holder}

    return spectrum_holder, reflectivity_holder, sanity_check


def get_diffraction_spectrum_sigma_full(kin_grid,
                                        spectrum_in,
                                        device_list,
                                        total_path,
                                        initial_position,
                                        d_num=512):
    """
    This function assume that the incident beam is of sigma polarization
    and does not include polarization mixing effect.

    :param kin_grid:
    :param spectrum_in:
    :param device_list:
    :param total_path:
    :param initial_position:
    :param d_num:
    :return:
    """
    ############################################################################################################
    #                      Step 1: Create holders and get prepared for the simulation
    ############################################################################################################
    device_num = len(device_list)

    # Get meta data
    k_num = kin_grid.shape[0]

    tic = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(kin_grid)
    klen_grid = np.ascontiguousarray(util.l2_norm_batch(kin_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Holder for the phase term and the Jacob matrix term
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    phase_grid = (kin_grid[:, 0] * initial_position[0]
                  + kin_grid[:, 1] * initial_position[1]
                  + kin_grid[:, 2] * initial_position[2]
                  - total_path * klen_grid)

    jacobian_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacobian = cuda.to_device(jacobian_grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    spec_holder = np.ascontiguousarray(spectrum_in)
    cuda_spectrum = cuda.to_device(spec_holder)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))

    ############################################################################################################
    #                           Step 2: Calculate the field and save the data
    ############################################################################################################
    # d_num = 512
    b_num = (k_num + d_num - 1) // d_num

    # --------------------------------------------------------------------
    #  Step 7. Forward propagation
    # --------------------------------------------------------------------
    for device_idx in range(device_num):
        my_device = device_list[device_idx]

        if my_device.type == "Crystal: Bragg Reflection":
            # Get the reflectivity
            GPUSingleDevice.get_bragg_reflection_sigma_full[b_num, d_num](cuda_reflect_sigma,
                                                                          cuda_phase,
                                                                          cuda_kin_grid,
                                                                          cuda_spectrum,
                                                                          cuda_jacobian,
                                                                          cuda_klen_grid,
                                                                          cuda_kin_grid,
                                                                          my_device.thickness,
                                                                          my_device.h,
                                                                          my_device.normal,
                                                                          np.dot(my_device.normal,
                                                                                 my_device.surface_point),
                                                                          my_device.dot_hn,
                                                                          my_device.h_square,
                                                                          my_device.chi0,
                                                                          my_device.chih_sigma,
                                                                          my_device.chihbar_sigma,
                                                                          k_num)

            GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         k_num)
        elif my_device.type == "Prism":
            # TODO: Finish the theory that include the prism effect
            # Diffracted by the prism
            GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                     cuda_kin_grid,
                                                     my_device.wavevec_delta,
                                                     k_num)

            # Update the wave number
            GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                            cuda_kin_grid,
                                                            3,
                                                            k_num)
        elif my_device.type == "Transmissive Grating":
            # TODO: Finish the theory that include the prism effect
            # Diffracted by the first grating
            GPUSingleDevice.get_square_grating_diffraction_scalar[b_num, d_num](cuda_kin_grid,
                                                                                cuda_spectrum,
                                                                                cuda_klen_grid,
                                                                                cuda_kin_grid,
                                                                                my_device.h,
                                                                                my_device.n,
                                                                                my_device.ab_ratio,
                                                                                my_device.thick_vec,
                                                                                my_device.order,
                                                                                my_device.base_wave_vector,
                                                                                k_num)
        else:
            pass

    # --------------------------------------------------------------------
    #  Step 8. Get the propagation phase
    # --------------------------------------------------------------------
    # Add the phase
    GPUSingleDevice.add_phase_to_scalar_spectrum[b_num, d_num](cuda_phase,
                                                               cuda_spectrum,
                                                               k_num)

    # Add the jacobian
    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_jacobian,
                                                                 cuda_spectrum,
                                                                 cuda_spectrum,
                                                                 k_num)

    ###################################################################################################
    #                                  Finish
    ###################################################################################################
    spec_holder = cuda_spectrum.copy_to_host()
    reflect_sigma = cuda_reflect_sigma.copy_to_host()
    reflect_total_sigma = cuda_reflect_total_sigma.copy_to_host()

    phase_grid = cuda_phase.copy_to_host()
    kin_grid = cuda_kin_grid.copy_to_host()
    klen_grid = cuda_klen_grid.copy_to_host()
    jacobian_grid = cuda_jacobian.copy_to_host()

    # Create result dictionary
    sanity_check = {"phase_grid": phase_grid,
                    "jacob_grid": jacobian_grid,
                    "kout_grid": kin_grid
                    }

    reflectivity_holder = {"reflectivity_sigma_tot": reflect_total_sigma, }
    spectrum_holder = {"final_spectrum": spec_holder}

    return spectrum_holder, reflectivity_holder, sanity_check


########################################################################################################################
#        For arbitrary incident scalar spectrum, only consider the sigma polarization
########################################################################################################################
def get_diffracted_monochromatic_components_sigma_polarization(kin_grid,
                                                               spectrum_in,
                                                               device_list,
                                                               total_path,
                                                               ref_trajectory,
                                                               d_num=512):
    """
    This function assume that the incident beam is of sigma polarization
    and does not include polarization mixing effect.

    :param kin_grid:
    :param spectrum_in:
    :param device_list:
    :param total_path:
    :param ref_trajectory: This is a (n,3) numpy array. It contains the trajectory of the center of the beam
                            through the setup.
    :param d_num:
    :return:
    """
    ############################################################################################################
    #                      Step 1: Create holders and get prepared for the simulation
    ############################################################################################################
    device_num = len(device_list)

    # Get meta data
    k_num = kin_grid.shape[0]

    tic = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((k_num, 3), dtype=np.float64))
    intersect_points[:, 0] = ref_trajectory[0, 0]
    intersect_points[:, 1] = ref_trajectory[0, 1]
    intersect_points[:, 2] = ref_trajectory[0, 2]

    component_final_points = np.ascontiguousarray(np.zeros((k_num, 3), dtype=np.float64))
    remaining_length = np.ascontiguousarray(np.zeros(k_num, dtype=np.float64) + total_path)
    phase_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.float64))
    jacobian_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_intersect = cuda.to_device(intersect_points)
    cuda_final_points = cuda.to_device(component_final_points)
    cuda_remain_path = cuda.to_device(remaining_length)
    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacobian = cuda.to_device(jacobian_grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    spec_holder = np.ascontiguousarray(spectrum_in)
    cuda_spectrum = cuda.to_device(spec_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(kin_grid)
    klen_grid = np.ascontiguousarray(util.l2_norm_batch(kin_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))

    ############################################################################################################
    #                           Step 2: Calculate the field and save the data
    ############################################################################################################
    # d_num = 512
    b_num = (k_num + d_num - 1) // d_num

    # --------------------------------------------------------------------
    #  Step 7. Forward propagation
    # --------------------------------------------------------------------
    for device_idx in range(device_num):
        my_device = device_list[device_idx]

        # Get the intersection point from the previous intersection point to the next one
        # during this process, calculating the evolution phase
        GPUSingleDevice.get_intersection_point_with_evolution_phase[b_num, d_num](cuda_phase,
                                                                                  cuda_remain_path,
                                                                                  cuda_intersect,
                                                                                  cuda_kin_grid,
                                                                                  cuda_klen_grid,
                                                                                  cuda_intersect,
                                                                                  my_device.surface_point,
                                                                                  my_device.normal,
                                                                                  k_num)
        # Add the propagation phase
        GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase,
                                                        (ref_trajectory[device_idx + 1] -
                                                         ref_trajectory[device_idx]),
                                                        cuda_kin_grid,
                                                        k_num)

        if my_device.type == "Crystal: Bragg Reflection":

            # Get the reflectivity
            GPUSingleDevice.get_bragg_reflection_sigma_polarization[b_num, d_num](cuda_reflect_sigma,
                                                                                  cuda_kin_grid,
                                                                                  cuda_spectrum,
                                                                                  cuda_jacobian,
                                                                                  cuda_klen_grid,
                                                                                  cuda_kin_grid,
                                                                                  my_device.thickness,
                                                                                  my_device.h,
                                                                                  my_device.normal,
                                                                                  my_device.dot_hn,
                                                                                  my_device.h_square,
                                                                                  my_device.chi0,
                                                                                  my_device.chih_sigma,
                                                                                  my_device.chihbar_sigma,
                                                                                  k_num)

            GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         k_num)
        elif my_device.type == "Prism":

            # Diffracted by the prism
            GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                     cuda_kin_grid,
                                                     my_device.wavevec_delta,
                                                     k_num)

            # Update the wave number
            GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                            cuda_kin_grid,
                                                            3,
                                                            k_num)
        elif my_device.type == "Transmissive Grating":
            # Diffracted by the first grating
            GPUSingleDevice.get_square_grating_diffraction_scalar[b_num, d_num](cuda_kin_grid,
                                                                                cuda_spectrum,
                                                                                cuda_klen_grid,
                                                                                cuda_kin_grid,
                                                                                my_device.h,
                                                                                my_device.n,
                                                                                my_device.ab_ratio,
                                                                                my_device.thick_vec,
                                                                                my_device.order,
                                                                                my_device.base_wave_vector,
                                                                                k_num)
        else:
            pass

    # --------------------------------------------------------------------
    #  Step 8. Get the propagation phase
    # --------------------------------------------------------------------
    GPUSingleDevice.add_evolution_phase_elementwise[b_num, d_num](cuda_phase,
                                                                  cuda_remain_path,
                                                                  cuda_klen_grid,
                                                                  k_num)

    # Add the propagation phase
    GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase,
                                                    (ref_trajectory[-1] -
                                                     ref_trajectory[-2]),
                                                    cuda_kin_grid,
                                                    k_num)

    # Add the phase
    GPUSingleDevice.add_phase_to_scalar_spectrum[b_num, d_num](cuda_phase,
                                                               cuda_spectrum,
                                                               k_num)

    # Add the jacobian
    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_jacobian,
                                                                 cuda_spectrum,
                                                                 cuda_spectrum,
                                                                 k_num)

    ###################################################################################################
    #                                  Finish
    ###################################################################################################
    cuda_spectrum.to_host()

    # Move the arrays back to the device for debugging.
    cuda_final_points.to_host()
    cuda_remain_path.to_host()
    cuda_intersect.to_host()
    cuda_phase.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_sigma.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_jacobian.to_host()

    # Create result dictionary

    sanity_check = {"intersect_points": intersect_points,

                    "remaining_length": remaining_length,
                    "phase_grid": phase_grid,
                    "final_point": component_final_points,
                    "jacob_grid": jacobian_grid,
                    "kout_grid": kin_grid
                    }

    reflectivity_holder = {"reflectivity_sigma": reflect_sigma,
                           "reflectivity_sigma_tot": reflect_total_sigma,
                           }

    spectrum_holder = {"final_spectrum": spec_holder}

    return spectrum_holder, reflectivity_holder, sanity_check


def get_diffracted_monochromatic_components_sigma_polarization_bk(kin_grid,
                                                                  spectrum_in,
                                                                  device_list,
                                                                  total_path,
                                                                  ref_trajectory,
                                                                  d_num=512):
    """
    This function assume that the incident beam is of sigma polarization
    and does not include polarization mixing effect.

    :param kin_grid:
    :param spectrum_in:
    :param device_list:
    :param total_path:
    :param ref_trajectory: This is a (n,3) numpy array. It contains the trajectory of the center of the beam
                            through the setup.
    :param d_num:
    :return:
    """
    ############################################################################################################
    #                      Step 1: Create holders and get prepared for the simulation
    ############################################################################################################
    device_num = len(device_list)

    # Get meta data
    k_num = kin_grid.shape[0]

    tic = time.time()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((k_num, 3), dtype=np.float64))
    intersect_points[:, 0] = ref_trajectory[0, 0]
    intersect_points[:, 1] = ref_trajectory[0, 1]
    intersect_points[:, 2] = ref_trajectory[0, 2]

    component_final_points = np.ascontiguousarray(np.zeros((k_num, 3), dtype=np.float64))
    remaining_length = np.ascontiguousarray(np.zeros(k_num, dtype=np.float64) + total_path)
    phase_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.float64))
    jacobian_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_intersect = cuda.to_device(intersect_points)
    cuda_final_points = cuda.to_device(component_final_points)
    cuda_remain_path = cuda.to_device(remaining_length)
    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacobian = cuda.to_device(jacobian_grid)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    spec_holder = np.ascontiguousarray(spectrum_in)
    cuda_spectrum = cuda.to_device(spec_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(kin_grid)
    klen_grid = np.ascontiguousarray(util.l2_norm_batch(kin_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))

    ############################################################################################################
    #                           Step 2: Calculate the field and save the data
    ############################################################################################################
    # d_num = 512
    b_num = (k_num + d_num - 1) // d_num

    # --------------------------------------------------------------------
    #  Step 7. Forward propagation
    # --------------------------------------------------------------------
    for device_idx in range(device_num):
        my_device = device_list[device_idx]

        # Get the intersection point from the previous intersection point to the next one
        # during this process, calculating the evolution phase
        GPUSingleDevice.get_intersection_point_with_evolution_phase[b_num, d_num](cuda_phase,
                                                                                  cuda_remain_path,
                                                                                  cuda_intersect,
                                                                                  cuda_kin_grid,
                                                                                  cuda_klen_grid,
                                                                                  cuda_intersect,
                                                                                  my_device.surface_point,
                                                                                  my_device.normal,
                                                                                  k_num)
        # Add the propagation phase
        GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase,
                                                        (ref_trajectory[device_idx + 1] -
                                                         ref_trajectory[device_idx]),
                                                        cuda_kin_grid,
                                                        k_num)

        if my_device.type == "Crystal: Bragg Reflection":

            # Get the reflectivity
            GPUSingleDevice.get_bragg_reflection_sigma_polarization[b_num, d_num](cuda_reflect_sigma,
                                                                                  cuda_kin_grid,
                                                                                  cuda_spectrum,
                                                                                  cuda_jacobian,
                                                                                  cuda_klen_grid,
                                                                                  cuda_kin_grid,
                                                                                  my_device.thickness,
                                                                                  my_device.h,
                                                                                  my_device.normal,
                                                                                  my_device.dot_hn,
                                                                                  my_device.h_square,
                                                                                  my_device.chi0,
                                                                                  my_device.chih_sigma,
                                                                                  my_device.chihbar_sigma,
                                                                                  k_num)

            GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         cuda_reflect_total_sigma,
                                                                         k_num)
        elif my_device.type == "Prism":

            # Diffracted by the prism
            GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                     cuda_kin_grid,
                                                     my_device.wavevec_delta,
                                                     k_num)

            # Update the wave number
            GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                            cuda_kin_grid,
                                                            3,
                                                            k_num)
        elif my_device.type == "Transmissive Grating":
            # Diffracted by the first grating
            GPUSingleDevice.get_square_grating_diffraction_scalar[b_num, d_num](cuda_kin_grid,
                                                                                cuda_spectrum,
                                                                                cuda_klen_grid,
                                                                                cuda_kin_grid,
                                                                                my_device.h,
                                                                                my_device.n,
                                                                                my_device.ab_ratio,
                                                                                my_device.thick_vec,
                                                                                my_device.order,
                                                                                my_device.base_wave_vector,
                                                                                k_num)
        else:
            pass

    # --------------------------------------------------------------------
    #  Step 8. Get the propagation phase
    # --------------------------------------------------------------------
    GPUSingleDevice.add_evolution_phase_elementwise[b_num, d_num](cuda_phase,
                                                                  cuda_remain_path,
                                                                  cuda_klen_grid,
                                                                  k_num)

    # Add the propagation phase
    GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase,
                                                    (ref_trajectory[-1] -
                                                     ref_trajectory[-2]),
                                                    cuda_kin_grid,
                                                    k_num)

    # Add the phase
    GPUSingleDevice.add_phase_to_scalar_spectrum[b_num, d_num](cuda_phase,
                                                               cuda_spectrum,
                                                               k_num)

    # Add the jacobian
    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_jacobian,
                                                                 cuda_spectrum,
                                                                 cuda_spectrum,
                                                                 k_num)

    ###################################################################################################
    #                                  Finish
    ###################################################################################################
    cuda_spectrum.to_host()

    # Move the arrays back to the device for debugging.
    cuda_final_points.to_host()
    cuda_remain_path.to_host()
    cuda_intersect.to_host()
    cuda_phase.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_sigma.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_jacobian.to_host()

    # Create result dictionary

    sanity_check = {"intersect_points": intersect_points,

                    "remaining_length": remaining_length,
                    "phase_grid": phase_grid,
                    "final_point": component_final_points,
                    "jacob_grid": jacobian_grid,
                    "kout_grid": kin_grid
                    }

    reflectivity_holder = {"reflectivity_sigma": reflect_sigma,
                           "reflectivity_sigma_tot": reflect_total_sigma,
                           }

    spectrum_holder = {"final_spectrum": spec_holder}

    return spectrum_holder, reflectivity_holder, sanity_check


#########################################################################
#         Update 2021.8.29 new propagation phase calculation.
#########################################################################
def get_gaussian_source_diffraction(crystal_list,
                                    total_path,
                                    reference_trajectory,
                                    my_pulse,
                                    grating_orders,
                                    kx_grid,
                                    ky_grid,
                                    kz_grid,
                                    number_x,
                                    number_y,
                                    number_z,
                                    d_num=512):
    """

    :param crystal_list:
    :param total_path:
    :param my_pulse:
    :param reference_trajectory:
    :param grating_orders:
    :param kx_grid:
    :param ky_grid:
    :param kz_grid:
    :param number_x:
    :param number_y:
    :param number_z:
    :param d_num:
    :return:
    """
    crystal_num = len(crystal_list)

    tic = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [3D Blocks]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    efield_spec_3d = np.zeros((number_x, number_y, number_z, 3), dtype=np.complex128)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] Various intersection points, path length and phase
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    intersect_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    # component_final_points = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    # phase_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    phase_real_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.float64))
    jacob_grid = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    path_length_remain = np.ascontiguousarray(np.ones(number_z, dtype=np.float64))

    cuda_path_length = cuda.to_device(path_length_remain)
    cuda_intersect = cuda.to_device(intersect_points)
    # cuda_final_points = cuda.to_device(component_final_points)
    # cuda_phase = cuda.to_device(phase_grid)
    cuda_phase_real = cuda.to_device(phase_real_grid)
    cuda_jacob = cuda.to_device(jacob_grid)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] reflect and time response
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    reflect_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_pi = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))

    cuda_reflect_pi = cuda.to_device(reflect_pi)
    cuda_reflect_total_pi = cuda.to_device(reflect_total_pi)
    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # [1D slices] Vector field
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The reciprocal space
    coef_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.complex128))  # Input spectrum without Jacobian
    scalar_spec_holder = np.ascontiguousarray(np.ones(number_z, dtype=np.complex128))  # With Jacobian
    vector_spec_holder = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.complex128))

    cuda_coef = cuda.to_device(coef_grid)
    cuda_spec_scalar = cuda.to_device(scalar_spec_holder)
    cuda_spec_vec = cuda.to_device(vector_spec_holder)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #  [1D slices] k grid
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    kin_grid = np.ascontiguousarray(np.zeros((number_z, 3), dtype=np.float64))
    klen_grid = np.ascontiguousarray(np.zeros(number_z, dtype=np.float64))

    kz_grid = np.ascontiguousarray(kz_grid)
    kz_square = np.ascontiguousarray(np.square(kz_grid))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)
    cuda_kz_grid = cuda.to_device(kz_grid)
    cuda_kz_square = cuda.to_device(kz_square)

    toc = time.time()
    print("It takes {:.2f} seconds to prepare the variables.".format(toc - tic))

    ############################################################################################################
    # ----------------------------------------------------------------------------------------------------------
    #
    #                           Step 2: Calculate the field and save the data
    #
    # ----------------------------------------------------------------------------------------------------------
    ############################################################################################################
    # d_num = 512
    b_num = (number_z + d_num - 1) // d_num

    for x_idx in range(number_x):
        kx = kx_grid[x_idx]

        for y_idx in range(number_y):
            ky = ky_grid[y_idx]

            # --------------------------------------------------------------------
            #  Step 1. Get k_out mesh
            # --------------------------------------------------------------------
            GPUSingleDevice.init_kvec[b_num, d_num](cuda_kin_grid,
                                                    cuda_klen_grid,
                                                    cuda_kz_grid,
                                                    cuda_kz_square,
                                                    kx,
                                                    ky,
                                                    ky ** 2 + kx ** 2,
                                                    number_z)

            GPUSingleDevice.init_jacobian[b_num, d_num](cuda_jacob, number_z)
            GPUSingleDevice.init_scalar_grid[b_num, d_num](cuda_phase_real, 0., number_z)
            GPUSingleDevice.init_scalar_grid[b_num, d_num](cuda_path_length, total_path, number_z)
            GPUSingleDevice.init_vector_grid[b_num, d_num](cuda_intersect, my_pulse.x0, 3, number_z)

            # --------------------------------------------------------------------
            #  Step 2. Back propagate through all the crystals
            # --------------------------------------------------------------------
            # Define a variable to track the grating index for the grating order
            grating_idx = len(grating_orders) - 1

            for crystal_idx in range(crystal_num - 1, -1, -1):

                # Get the crystal
                my_crystal = crystal_list[crystal_idx]

                # Determine the type of the crystal
                if my_crystal.type == "Transmissive Grating":
                    GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                             cuda_kin_grid,
                                                             - grating_orders[
                                                                 grating_idx] * my_crystal.base_wave_vector,
                                                             number_z)

                    # Update the wave number
                    GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                                    cuda_kin_grid,
                                                                    3,
                                                                    number_z)

                    # Update the grating idx
                    grating_idx -= 1

                if my_crystal.type == "Prism":
                    GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                             cuda_kin_grid,
                                                             -my_crystal.wavevec_delta,
                                                             number_z)

                    # Update the wave number
                    GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                                    cuda_kin_grid,
                                                                    3,
                                                                    number_z)

                if my_crystal.type == "Crystal: Bragg Reflection":
                    # Calculate the incident wave vector
                    GPUSingleDevice.get_kin_and_jacobian[b_num, d_num](cuda_kin_grid,
                                                                       cuda_jacob,
                                                                       cuda_klen_grid,
                                                                       cuda_kin_grid,
                                                                       my_crystal.h,
                                                                       my_crystal.normal,
                                                                       my_crystal.dot_hn,
                                                                       my_crystal.h_square,
                                                                       number_z)
            # --------------------------------------------------------------------
            #  Step 5. Get the coefficient of each monochromatic component
            # --------------------------------------------------------------------
            # Calculate the corresponding coefficient in the incident pulse
            GPUSingleDevice.get_gaussian_pulse_spectrum[b_num, d_num](cuda_coef,
                                                                      cuda_kin_grid,
                                                                      0.,  # Initially time is 0.
                                                                      my_pulse.sigma_mat,
                                                                      my_pulse.scaling,
                                                                      np.zeros(3, dtype=np.float64),
                                                                      my_pulse.k0,
                                                                      my_pulse.omega0,
                                                                      my_pulse.n,
                                                                      number_z)

            # --------------------------------------------------------------------
            #  Step 6. Calculate the Jacobian weighted vector spectrum
            # --------------------------------------------------------------------
            # Add Jacobian
            GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_coef,
                                                                         cuda_jacob,
                                                                         cuda_spec_scalar,
                                                                         number_z
                                                                         )

            # Get the vector field
            GPUSingleDevice.scalar_vector_multiply_complex[b_num, d_num](cuda_spec_scalar,
                                                                         my_pulse.polar,
                                                                         cuda_spec_vec,
                                                                         number_z)

            # --------------------------------------------------------------------
            #  Step 7. Forward propagation
            # --------------------------------------------------------------------
            grating_idx = 0
            for crystal_idx in range(crystal_num):
                my_crystal = crystal_list[crystal_idx]
                if my_crystal.type == "Transmissive Grating":
                    # Get the intersection point on the first grating from the initial point
                    GPUSingleDevice.get_intersection_point_with_evolution_phase[b_num, d_num](cuda_phase_real,
                                                                                              cuda_path_length,
                                                                                              cuda_intersect,
                                                                                              cuda_kin_grid,
                                                                                              cuda_klen_grid,
                                                                                              cuda_intersect,
                                                                                              my_crystal.surface_point,
                                                                                              my_crystal.normal,
                                                                                              number_z)

                    # Add the propagation phase
                    GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase_real,
                                                                    (reference_trajectory[crystal_idx + 1] -
                                                                     reference_trajectory[crystal_idx]),
                                                                    cuda_kin_grid,
                                                                    number_z)

                    # Diffracted by the first grating
                    GPUSingleDevice.get_square_grating_effect_non_zero[b_num, d_num](cuda_kin_grid,
                                                                                     cuda_spec_vec,
                                                                                     cuda_klen_grid,
                                                                                     cuda_kin_grid,
                                                                                     my_crystal.h,
                                                                                     my_crystal.n,
                                                                                     my_crystal.ab_ratio,
                                                                                     my_crystal.thick_vec,
                                                                                     grating_orders[grating_idx],
                                                                                     my_crystal.base_wave_vector,
                                                                                     number_z)
                    # Update the grating_idx
                    grating_idx += 1

                if my_crystal.type == "Prism":
                    # Get the intersection point on the first grating from the initial point
                    GPUSingleDevice.get_intersection_point_with_evolution_phase[b_num, d_num](cuda_phase_real,
                                                                                              cuda_path_length,
                                                                                              cuda_intersect,
                                                                                              cuda_kin_grid,
                                                                                              cuda_klen_grid,
                                                                                              cuda_intersect,
                                                                                              my_crystal.surface_point,
                                                                                              my_crystal.normal,
                                                                                              number_z)

                    # Add the propagation phase
                    GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase_real,
                                                                    (reference_trajectory[crystal_idx + 1] -
                                                                     reference_trajectory[crystal_idx]),
                                                                    cuda_kin_grid,
                                                                    number_z)

                    # Diffracted by the prism
                    GPUSingleDevice.add_vector[b_num, d_num](cuda_kin_grid,
                                                             cuda_kin_grid,
                                                             my_crystal.wavevec_delta,
                                                             number_z)

                    # Update the wave number
                    GPUSingleDevice.get_vector_length[b_num, d_num](cuda_klen_grid,
                                                                    cuda_kin_grid,
                                                                    3,
                                                                    number_z)
                if my_crystal.type == "Crystal: Bragg Reflection":
                    # Get the intersection point on the first grating from the initial point
                    GPUSingleDevice.get_intersection_point_with_evolution_phase[b_num, d_num](cuda_phase_real,
                                                                                              cuda_path_length,
                                                                                              cuda_intersect,
                                                                                              cuda_kin_grid,
                                                                                              cuda_klen_grid,
                                                                                              cuda_intersect,
                                                                                              my_crystal.surface_point,
                                                                                              my_crystal.normal,
                                                                                              number_z)

                    # Add the propagation phase
                    GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase_real,
                                                                    (reference_trajectory[crystal_idx + 1] -
                                                                     reference_trajectory[crystal_idx]),
                                                                    cuda_kin_grid,
                                                                    number_z)

                    # Get the reflectivity
                    GPUSingleDevice.get_bragg_reflection[b_num, d_num](cuda_reflect_sigma,
                                                                       cuda_reflect_pi,
                                                                       cuda_kin_grid,
                                                                       cuda_spec_vec,
                                                                       cuda_klen_grid,
                                                                       cuda_kin_grid,
                                                                       my_crystal.thickness,
                                                                       my_crystal.h,
                                                                       my_crystal.normal,
                                                                       my_crystal.dot_hn,
                                                                       my_crystal.h_square,
                                                                       my_crystal.chi0,
                                                                       my_crystal.chih_sigma,
                                                                       my_crystal.chihbar_sigma,
                                                                       my_crystal.chih_pi,
                                                                       my_crystal.chihbar_pi,
                                                                       number_z)
                    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_sigma,
                                                                                 cuda_reflect_total_sigma,
                                                                                 cuda_reflect_total_sigma,
                                                                                 number_z)
                    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_reflect_pi,
                                                                                 cuda_reflect_total_pi,
                                                                                 cuda_reflect_total_pi,
                                                                                 number_z)
            # --------------------------------------------------------------------
            #  Step 8. Get the propagation phase
            # --------------------------------------------------------------------
            GPUSingleDevice.add_evolution_phase_elementwise[b_num, d_num](cuda_phase_real,
                                                                          cuda_path_length,
                                                                          cuda_klen_grid,
                                                                          number_z)

            # Add the propagation phase
            GPUSingleDevice.add_spatial_phase[b_num, d_num](cuda_phase_real,
                                                            (reference_trajectory[-1] -
                                                             reference_trajectory[-2]),
                                                            cuda_kin_grid,
                                                            number_z)

            # Add the phase
            GPUSingleDevice.add_phase_to_vector_spectrum[b_num, d_num](cuda_phase_real,
                                                                       cuda_spec_vec,
                                                                       cuda_spec_vec,
                                                                       number_z)

            # """
            ###################################################################################################
            #                          Clean up the calculation
            ###################################################################################################
            cuda_spec_vec.to_host()

            # Update the 3D variables.
            efield_spec_3d[x_idx, y_idx, :, 0] = vector_spec_holder[:, 0]
            efield_spec_3d[x_idx, y_idx, :, 1] = vector_spec_holder[:, 1]
            efield_spec_3d[x_idx, y_idx, :, 2] = vector_spec_holder[:, 2]

            cuda_spec_vec = cuda.to_device(vector_spec_holder)

    # Move the arrays back to the device for debugging.
    cuda_path_length.to_host()
    cuda_spec_scalar.to_host()
    cuda_intersect.to_host()
    cuda_phase_real.to_host()

    cuda_kin_grid.to_host()
    cuda_klen_grid.to_host()

    cuda_reflect_pi.to_host()
    cuda_reflect_sigma.to_host()
    cuda_reflect_total_pi.to_host()
    cuda_reflect_total_sigma.to_host()

    cuda_coef.to_host()
    cuda_spec_vec.to_host()

    # Create result dictionary
    sanity_check = {"intersect_points": intersect_points,
                    "remaining_length": path_length_remain,
                    "phase_grid": phase_real_grid,
                    "scalar_spec": scalar_spec_holder,
                    "coef_grid": coef_grid,
                    "jacob_grid": jacob_grid,
                    }

    result_3d = {"efield_spec_3d": efield_spec_3d, }

    reflectivty_holder = {"reflectivity_pi": reflect_pi,
                          "reflectivity_sigma": reflect_sigma,
                          "reflectivity_pi_tot": reflect_total_pi,
                          "reflectivity_sigma_tot": reflect_total_sigma, }

    return result_3d, reflectivty_holder, sanity_check
