import datetime
import time

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt


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
def get_fwhm(coordinate, curve_values, center=False):
    """
    Get the FWHM in the straightforward way.
    However, notice that, when one calculate the FWHM in this way, the result
    is sensitive to small perturbations of the curve's shape.

    :param coordinate:
    :param curve_values:
    :param center: Whether return the coordinate of the center of the region within FWHM
    :return:
    """
    # Get the half max value
    half_max = np.max(curve_values) / 2.

    # Get the indexes for the range.
    indexes = np.arange(len(coordinate), dtype=np.int64)
    mask = np.zeros_like(indexes, dtype=np.bool)
    mask[curve_values >= half_max] = True

    indexes_above = indexes[mask]

    # Get the ends of the region
    left_idx = np.min(indexes_above)
    right_idx = np.max(indexes_above)

    # Convert the indexes into coordinates
    fwhm = coordinate[right_idx] - coordinate[left_idx]

    if center:
        distribution = curve_values[mask]
        distribution /= np.sum(distribution)

        coordinate_roi = coordinate[mask]

        mean = np.sum(np.multiply(distribution, coordinate_roi))

        return fwhm, mean
    else:
        return fwhm


def get_statistics(distribution, coor=None):
    # Get a holder for the analysis result
    holder = {"2d slice": {},
              "2d projection": {},
              "1d slice": {},
              "1d projection": {},
              }

    # Get distribution shape
    dist_shape = np.array(distribution.shape, dtype=np.int64)
    center_position = dist_shape // 2
    x_c = center_position[0]
    y_c = center_position[1]
    z_c = center_position[2]

    # Get the 2d slice
    tmp_xy = distribution[:, :, z_c]
    tmp_xz = distribution[:, y_c, :]
    tmp_yz = distribution[x_c, :, :]
    holder['2d slice'].update({"xy": np.copy(tmp_xy),
                               "xz": np.copy(tmp_xz),
                               "yz": np.copy(tmp_yz),
                               })

    # Get 2d projection
    tmp_xy = np.sum(distribution, axis=2)
    tmp_xz = np.sum(distribution, axis=1)
    tmp_yz = np.sum(distribution, axis=0)
    holder['2d projection'].update({"xy": np.copy(tmp_xy),
                                    "xz": np.copy(tmp_xz),
                                    "yz": np.copy(tmp_yz),
                                    })

    # Get 1d slice
    holder['1d slice'].update({"x": np.copy(distribution[:, y_c, z_c]),
                               "y": np.copy(distribution[x_c, :, z_c]),
                               "z": np.copy(distribution[x_c, y_c, :]),
                               })

    # Get 1d projection
    holder['1d projection'].update({"x": np.copy(np.sum(tmp_xy, axis=1)),
                                    "y": np.copy(np.sum(tmp_xy, axis=0)),
                                    "z": np.copy(np.sum(tmp_xz, axis=0)),
                                    })

    if coor is not None:
        # Create an entry called sigma to get the sigma and FWHM
        holder.update({"sigma": {},
                       "fwhm": {}})

        for axis in ['x', 'y', 'z']:
            # Normalize to get the distribution
            tmp = np.copy(holder['1d projection'][axis])
            prob_dist = tmp / np.sum(tmp)

            # Get sigma
            mean = np.sum(np.multiply(prob_dist, coor[axis]))
            std = np.sum(np.multiply(np.square(prob_dist), coor[axis])) - np.square(mean)

            holder["sigma"].update({axis: np.copy(std)})

            # Get fwhm
            holder["fwhm"].update({axis: get_fwhm(coordinate=coor[axis], curve_values=prob_dist)})

    return holder


def get_gaussian_fit(curve, coordinate):
    """
    Fit the target curve with a Gaussian function
    :param curve:
    :param coordinate:
    :return:
    """
    total = np.sum(curve)
    distribution = curve / total

    mean = np.sum(np.multiply(distribution, coordinate))
    std = np.sum(np.multiply(distribution, np.square(coordinate))) - mean ** 2
    std = np.sqrt(std)

    gaussian_fit = np.exp(- np.square(coordinate - mean) / 2. / std ** 2)
    gaussian_fit /= np.sum(gaussian_fit)

    gaussian_fit *= total
    return gaussian_fit


############################################################
#     Show stats
############################################################
def show_stats_2d(stats_holder, fig_height, fig_width):
    fig, axes = plt.subplots(nrows=3, ncols=2)

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    #########################################
    #    xy slice and projection
    #########################################
    im00 = axes[0, 0].imshow(stats_holder['2d slice']['xy'], cmap='jet')
    axes[0, 0].set_title("XY Slice")
    axes[0, 0].set_axis_off()
    fig.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].imshow(stats_holder['2d projection']['xy'], cmap='jet')
    axes[0, 1].set_title("XY Projection")
    axes[0, 1].set_axis_off()
    fig.colorbar(im01, ax=axes[0, 1])

    #########################################
    #    xz slice and projection
    #########################################
    im10 = axes[1, 0].imshow(stats_holder['2d slice']['xz'], cmap='jet', aspect="auto")
    axes[1, 0].set_title("XZ Slice")
    axes[1, 0].set_axis_off()
    fig.colorbar(im10, ax=axes[1, 0])

    im11 = axes[1, 1].imshow(stats_holder['2d projection']['xz'], cmap='jet', aspect="auto")
    axes[1, 1].set_title("XZ Projection")
    axes[1, 1].set_axis_off()
    fig.colorbar(im11, ax=axes[1, 1])

    #########################################
    #    yz slice and projection
    #########################################
    im20 = axes[2, 0].imshow(stats_holder['2d slice']['yz'], cmap='jet', aspect="auto")
    axes[2, 0].set_title("YZ Slice")
    axes[2, 0].set_axis_off()
    fig.colorbar(im20, ax=axes[2, 0])

    im21 = axes[2, 1].imshow(stats_holder['2d projection']['yz'], cmap='jet', aspect="auto")
    axes[2, 1].set_title("YZ Projection")
    axes[2, 1].set_axis_off()
    fig.colorbar(im21, ax=axes[2, 1])

    plt.show()


def show_stats_1d(stats_holder, coor, fig_height, fig_width):
    fig, axes = plt.subplots(nrows=3, ncols=2)

    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    #########################################
    #    x slice and projection
    #########################################
    axes[0, 0].plot(coor['real axis']['x'], stats_holder['1d slice']['x'])
    axes[0, 0].set_title("X Slice")

    axes[0, 1].plot(coor['real axis']['x'], stats_holder['1d projection']['x'])
    axes[0, 1].set_title("X Projection")

    #########################################
    #    y slice and projection
    #########################################
    axes[1, 0].plot(coor['real axis']['y'], stats_holder['1d slice']['y'])
    axes[1, 0].set_title("Y Slice")

    axes[1, 1].plot(coor['real axis']['y'], stats_holder['1d projection']['y'])
    axes[1, 1].set_title("Y Projection")

    #########################################
    #    yz slice and projection
    #########################################
    axes[2, 0].plot(coor['real axis']['z'], stats_holder['1d slice']['z'])
    axes[2, 0].set_title("Z Slice")

    axes[2, 1].plot(coor['real axis']['z'], stats_holder['1d projection']['z'])
    axes[2, 1].set_title("Z Projection")

    plt.show()
