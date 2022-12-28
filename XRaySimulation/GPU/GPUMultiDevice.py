import numpy as np
from numba import cuda

from XRaySimulation.GPU import GPUSingleDevice


def get_multicrystal_reflection(kin_grid,
                                spectrum_in,
                                device_list,
                                total_path,
                                initial_position,
                                d_num=512,
                                batch_num=int(1),
                                flag_reflectivity=False,
                                flag_jacobian=False,
                                flag_kout=True,
                                flag_kout_length=True,
                                flag_phase=False):
    """

    :param kin_grid:
    :param spectrum_in:
    :param device_list:
    :param total_path:
    :param initial_position:
    :param d_num:
    :param batch_num:
    :param flag_reflectivity:
    :param flag_jacobian:
    :param flag_kout:
    :param flag_kout_length:
    :param flag_phase:
    :return:
    """

    if batch_num == 1:
        output_dict = _get_multicrystal_reflection(kin_grid=kin_grid,
                                                   spectrum_in=spectrum_in,
                                                   device_list=device_list,
                                                   total_path=total_path,
                                                   initial_position=initial_position,
                                                   d_num=d_num,
                                                   flag_reflectivity=flag_reflectivity,
                                                   flag_jacobian=flag_jacobian,
                                                   flag_kout=flag_kout,
                                                   flag_kout_length=flag_kout_length,
                                                   flag_phase=flag_phase
                                                   )
        return output_dict

    elif batch_num > 1:

        output_dict = {}

        # Get a holder for the output from each batch
        output_list = []

        # Split the incident arrays
        kin_array_list = np.array_split(kin_grid, batch_num, axis=0)

        spectrum_array_list = np.array_split(spectrum_in, batch_num, axis=0)

        for batch_idx in range(batch_num):
            # Get the output from each batch
            output_list.append(
                _get_multicrystal_reflection(kin_grid=np.ascontiguousarray(kin_array_list[batch_idx]),
                                             spectrum_in=np.ascontiguousarray(spectrum_array_list[batch_idx]),
                                             device_list=device_list,
                                             total_path=total_path,
                                             initial_position=np.copy(initial_position),
                                             d_num=d_num,
                                             flag_reflectivity=flag_reflectivity,
                                             flag_jacobian=flag_jacobian,
                                             flag_kout=flag_kout,
                                             flag_kout_length=flag_kout_length,
                                             flag_phase=flag_phase
                                             ))

        # Assemble the output from each array
        output_dict.update({"spectrum_grid": np.concatenate([entry['spectrum_grid'] for entry in output_list],
                                                            axis=0)})

        if flag_reflectivity:
            output_dict.update({"reflectivity": np.concatenate([entry['reflectivity'] for entry in output_list],
                                                               axis=0)})
        if flag_jacobian:
            output_dict.update({"jacobian_grid": np.concatenate([entry['jacobian_grid'] for entry in output_list],
                                                                axis=0)})
        if flag_kout:
            output_dict.update({"kout_grid": np.concatenate([entry['kout_grid'] for entry in output_list],
                                                            axis=0)})
        if flag_kout_length:
            output_dict.update({"kout_len_grid": np.concatenate([entry['kout_len_grid'] for entry in output_list],
                                                                axis=0)})
        if flag_phase:
            output_dict.update({"phase_grid": np.concatenate([entry['phase_grid'] for entry in output_list],
                                                             axis=0)})
        return output_dict

    else:
        print("The batch_num has to be an integer larger than or equal to 1. ")


def _get_multicrystal_reflection(kin_grid,
                                 spectrum_in,
                                 device_list,
                                 total_path,
                                 initial_position,
                                 d_num=512,
                                 flag_reflectivity=False,
                                 flag_jacobian=False,
                                 flag_kout=True,
                                 flag_kout_length=True,
                                 flag_phase=False
                                 ):
    """
    This function assume that the incident beam is of sigma polarization
    and does not include polarization mixing effect.

    :param kin_grid: shape
    :param spectrum_in:
    :param device_list:
    :param total_path:
    :param initial_position:
    :param d_num: How many threads are calculating at the same time.
    :return:
    """

    ############################################################################################################
    #                      Step 1: Create holders and get prepared for the simulation
    ############################################################################################################
    device_num = len(device_list)

    # Amplitude for each wave-vector.
    cuda_spectrum = cuda.to_device(spectrum_in)

    # Get meta data
    k_num = kin_grid.shape[0]
    klen_grid = np.ascontiguousarray(np.linalg.norm(kin_grid, axis=-1))

    cuda_kin_grid = cuda.to_device(kin_grid)
    cuda_klen_grid = cuda.to_device(klen_grid)

    # Additional Bragg reflectivity phase
    phase_grid = (kin_grid[:, 0] * initial_position[0]
                  + kin_grid[:, 1] * initial_position[1]
                  + kin_grid[:, 2] * initial_position[2]
                  - total_path * klen_grid)

    # Jacobian matrix. This is equivalent to calculate the asymmetry factor for each wave-vector.
    jacobian_grid = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_phase = cuda.to_device(phase_grid)
    cuda_jacobian = cuda.to_device(jacobian_grid)

    # Holder for reflectivity from each crystal and the total reflectivity
    reflect_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))
    reflect_total_sigma = np.ascontiguousarray(np.ones(k_num, dtype=np.complex128))

    cuda_reflect_sigma = cuda.to_device(reflect_sigma)
    cuda_reflect_total_sigma = cuda.to_device(reflect_total_sigma)

    ############################################################################################################
    #                           Step 2: Calculate the field
    ############################################################################################################
    b_num = (k_num + d_num - 1) // d_num

    for device_idx in range(device_num):
        my_device = device_list[device_idx]

        if my_device.type == "Crystal: Bragg Reflection":
            # Get the reflectivity
            GPUSingleDevice.get_bragg_reflection_sigma[b_num, d_num](cuda_reflect_sigma,
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
            # TODO: Specifically, the prism should not change the wave-vector length
            # TODO: and, the prism should add a different phase to different wave-vector
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
    # Add the phase from bragg crystal
    GPUSingleDevice.add_phase_to_scalar_spectrum[b_num, d_num](cuda_phase,
                                                               cuda_spectrum,
                                                               k_num)

    # Add the Jacobian
    GPUSingleDevice.scalar_scalar_multiply_complex[b_num, d_num](cuda_jacobian,
                                                                 cuda_spectrum,
                                                                 cuda_spectrum,
                                                                 k_num)

    ############################################################################################################
    #                           Step 3: Save the data
    ############################################################################################################

    output_dict = {}
    output_dict.update({"spectrum_grid": cuda_spectrum.copy_to_host()})
    if flag_reflectivity:
        output_dict.update({"reflectivity": cuda_reflect_total_sigma.copy_to_host()})
    if flag_jacobian:
        output_dict.update({"jacobian_grid": cuda_jacobian.copy_to_host()})
    if flag_kout:
        output_dict.update({"kout_grid": cuda_kin_grid.copy_to_host()})
    if flag_kout_length:
        output_dict.update({"kout_len_grid": cuda_klen_grid.copy_to_host()})
    if flag_phase:
        output_dict.update({"phase_grid": cuda_phase.copy_to_host()})

    return output_dict
