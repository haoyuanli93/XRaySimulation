import numpy as np
from XRaySimulation import util


def get_propagation_phase(wavelength_loc, num_y, res_y, num_z, res_z, prop_distance):
    # Get k mesh
    ky_list = np.fft.fftshift(np.fft.fftfreq(num_y, d=res_y) * 2 * np.pi)
    kz_list = np.fft.fftshift(np.fft.fftfreq(num_z, d=res_z) * 2 * np.pi)
    kz_list += np.pi * 2. / wavelength_loc

    # Get frequency
    omega = np.zeros((num_y, num_z))
    omega += np.square(ky_list[:, np.newaxis])
    omega += np.square(kz_list[np.newaxis, :])
    omega = np.sqrt(omega) * util.c

    # Get time
    prop_time = prop_distance / util.c

    # Get the phase 
    propagation_phase = kz_list[np.newaxis, :] * prop_distance - omega * prop_time
    propagation_phase_complex = np.cos(propagation_phase) + 1.j * np.sin(propagation_phase)

    return propagation_phase_complex
