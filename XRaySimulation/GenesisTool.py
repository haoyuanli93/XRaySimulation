"""
Code in this file are provided by James.

I have modified them to have the same style as my other code.
"""

import re
import time
import numpy as np


def get_simulation_info(out_file):
    # Get the spatial resolution along the propagation direction
    central_wavelength = float(get_input_param(out_file, 'xlamds').replace('D', 'E')) * 1e6

    tmp = float(get_input_param(out_file, 'zsep').replace('D', 'E'))
    dz = tmp * central_wavelength

    # Get the spatial resolution along the transverse direction
    gridpoints = "N/A"
    dx_and_dy = "N/A"

    # Loop through the file
    with open(out_file, 'r') as file_obj:

        for line in file_obj:

            stripped_line = line.strip()
            split_line = stripped_line.split()

            if len(split_line) >= 2:
                if split_line[-1] == "meshsize":
                    dx_and_dy = float(split_line[0]) * 1e6  # Convert to um

                if split_line[-1] == "gridpoints":
                    gridpoints = int(split_line[0])

    print("The central wavelength is {:.4f} nm".format(central_wavelength * 1000.))
    print("The spatial resolution along the propagation direction is {:.4f} nm".format(dz * 1000.))
    print("The gridpoints number is {:}".format(gridpoints))
    print("The size of the gridpoints mesh is {:.2f} um".format(dx_and_dy))

    return central_wavelength, dz, gridpoints, dx_and_dy


def get_input_param(base_in_file, param):
    """
    returns a text version of the desired parameter, or an empty string if the parameter is not found

    :param base_in_file:
    :param param:
    :return:
    """
    i_file = open(base_in_file, "r")
    out_value = False
    for line in i_file:
        if re.search(param, line):
            line = line.replace('\"', '')
            line = line.replace('\'', '')
            line = line.replace('\n', '').rstrip()
            out_value = re.split('[ =]', line)[-1].rstrip()
    i_file.close()
    return out_value


def read_field(field_file, out_file=None):
    """
    Read the output field file from genesis and convert that into a 3D complex numpy array

    :param field_file:
    :param out_file:
    :return:
    """
    # reads field output files from genesis, returns 3D array of complex field values
    raw_read = np.fromfile(field_file, dtype=float, count=-1)
    real = raw_read[0::2]
    imag = raw_read[1::2]

    if out_file is None:
        ncar = int(get_input_param(field_file[:-4], 'ncar'))
    else:
        ncar = int(get_input_param(out_file, 'ncar'))

    real = real.reshape((-1, ncar, ncar))
    imag = imag.reshape((-1, ncar, ncar))
    comp = real + np.multiply(1j, imag)
    return comp


def load_field(dfl_file, out_file, new_shape=None):
    """
    Load the field.
    select useful central region.
    Add some padding.

    :param dfl_file:
    :param out_file:
    :param new_shape:
    :return:
    """

    # Load the field
    tic = time.time()
    in_field = read_field(field_file=dfl_file, out_file=out_file)

    # Change the axes to make it compatible with my convention
    in_field = np.moveaxis(in_field, (0, 1, 2), (2, 0, 1))

    # Load simulation info
    (wavelength,
     dz,
     gradepoints,
     dy) = get_simulation_info(out_file=out_file)
    simu_info = {"wavelength": wavelength,
                 "gradepoints": gradepoints,
                 "dy": dy,
                 "dz": dz, }

    # Select the central region and add zero padding to the array if necessary
    if new_shape is not None:
        # Extract the shape of the incident field
        old_shape = np.array(in_field.shape, dtype=np.int64)
        tmp = old_shape // 2
        x_c = tmp[0]
        y_c = tmp[1]
        z_c = tmp[2]

        # Select the central region in the transverse direction
        in_field = in_field[
                   x_c - new_shape[0] // 2:x_c + new_shape[0] // 2,
                   y_c - new_shape[1] // 2:y_c + new_shape[1] // 2, :]

        # Pad zeros
        left_pad_num = (new_shape[2] - old_shape[2]) // 2
        right_pad_num = new_shape[2] - left_pad_num - old_shape[2]
        in_field = np.ascontiguousarray(np.pad(in_field,
                                               ((0, 0),
                                                (0, 0),
                                                (left_pad_num, right_pad_num)
                                                ),
                                               mode='constant',
                                               constant_values=0.))

        toc = time.time()

        print("It takes {:.2f} seconds to load the field".format(toc - tic))

    return simu_info, in_field
