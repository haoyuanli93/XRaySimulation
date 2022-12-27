import numpy as np
from numba import jit


@jit(nopython=True, parallel=True)
def get_1d_fresnel_diffraction(source, k_array, y_source_array, y_target_array, z_target_array):
    # Get the integration constant
    y_sampling = y_source_array[1] - y_source_array[0]
    k_sampling = k_array[1] - k_array[0]

    # Get the size of difference arrays
    y_source_num = y_source_array.shape[0]
    k_num = k_array.shape[0]

    y_target_num = y_target_array.shape[0]
    z_target_num = z_target_array.shape[0]

    # Create a phase holder
    phase_angle = np.zeros(y_source_num, dtype=np.float64)
    phase = np.ones(y_source_num, dtype=np.complex128)

    # Create a holder for the output field
    field_out = np.zeros((y_target_num, z_target_num), dtype=np.complex128)

    # Calculate the diffraction
    for z_idx in range(z_target_num):
        for k_idx in range(k_num):

            # Get the factor associated with k and z
            k = k_array[k_idx]
            z = z_target_array[z_idx]
            kz_dependent_factor = np.sqrt(k / z) * np.exp(1.j * k * z)

            # Get the fresnel integration
            for y_idx in range(y_target_num):
                integral = complex(0.)

                # Loop through the y source points
                for y_source_idx in range(y_source_num):
                    # Get the phase angle
                    phase_angle[y_source_idx] = (y_source_array[y_source_idx] - y_target_array[y_idx]) ** 2
                    phase_angle[y_source_idx] *= k_array[k_idx] / 2. / z_target_array[z_idx]

                    # Get the phse
                    phase[y_source_idx] = np.exp(1.j * phase_angle[y_source_idx])

                    # Add to the integration
                    integral += phase[y_source_idx] * source[y_source_idx, k_idx]

                # Get the field
                integral *= y_sampling * kz_dependent_factor

                # Add this to the total output field
                field_out[y_idx, z_idx] += integral * k_sampling

    # Add the overall constant to the field
    overall_factor = np.exp(-1.j * np.pi / 4.) / np.sqrt(2. * np.pi)
    field_out *= overall_factor

    return field_out
