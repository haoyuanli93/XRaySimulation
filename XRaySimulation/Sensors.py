import numpy as np
import time


def get_intensity_with_spectrum_and_reflectivity(spectrum_array, dk_keV, reflectivity, relative_noise_level, ):
    """
    Get the readout value from the  diode.
    Include the transmission factor and diode noise.

    :param spectrum_array:
    :param dk_keV:
    :param reflectivity:
    :param relative_noise_level:
    :return:
    """
    randomSeed = int(time.time() * 1e6)
    np.random.seed(randomSeed)

    pulseEnergy = np.sum(np.square(np.abs(np.multiply(spectrum_array, reflectivity[np.newaxis, :]))), axis=-1)
    pulseEnergy *= dk_keV

    pulseEnergyReadOut = np.random.random(spectrum_array.shape[0]) + relative_noise_level
    pulseEnergyReadOut *= pulseEnergy

    return pulseEnergyReadOut, pulseEnergy
