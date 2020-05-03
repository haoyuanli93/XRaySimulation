import datetime
import time

import h5py as h5
import numpy as np


########################################################################################################################
#                    For I/O operation
########################################################################################################################
def save_branch_result_to_h5file(file_name, io_type, branch_name,
                                 result_3d_dict, result_2d_dict, check_dict):
    with h5.File(file_name, io_type) as h5file:
        group = h5file.create_group(branch_name)
        # Save the meta data
        group_check = group.create_group('check')
        for entry in list(check_dict.keys()):
            group_check.create_dataset(entry, data=check_dict[entry])

        group_2d = group.create_group('result_2d')
        for entry in list(result_2d_dict.keys()):
            group_2d.create_dataset(entry, data=result_2d_dict[entry])

        group_3d = group.create_group('result_3d')
        for entry in list(result_3d_dict.keys()):
            group_3d.create_dataset(entry, data=result_3d_dict[entry])


def time_stamp():
    """
    Get a time stamp
    :return: A time stamp of the form '%Y_%m_%d_%H_%M_%S'
    """
    stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    return stamp


########################################################################################################################
#                     Curve analysis
########################################################################################################################
def get_fwhm(coordinate, curve_values):
    """
    Get the FWHM in the straightforward way.
    However, notice that, when one calculate the FWHM in this way, the result
    is sensitive to small perturbations of the curve's shape.

    :param coordinate:
    :param curve_values:
    :return:
    """
    # Get the half max value
    half_max = np.max(curve_values) / 2.

    # Get the indexes for the range.
    indexes = np.arange(len(coordinate), dtype=np.int64)
    indexes_above = indexes[curve_values >= half_max]

    # Get the ends of the region
    left_idx = np.min(indexes_above)
    right_idx = np.max(indexes_above)

    return coordinate[right_idx] - coordinate[left_idx]
