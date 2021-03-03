import numpy as np
import sys
import time
import matplotlib.pyplot as plt

sys.path.append("/cds/home/h/haoyuan/Documents/my_repos/XRaySimulation/")

from XRaySimulation import util, misc, GenesisTool, Crystal, MultiDevice
from XRaySimulation.GPU import GPUMultiDevice

# Load essential info for all simulation
seed_idx = 0
undulator_num = 11

# Get the example out file
parent_folder = "/reg/data/ana15/xpp/xppx42118/results/james/for_haoyuan/5_lens_10_seeds/"
out_file = parent_folder + "/2fs_15U_s1_{}_{:0>2d}.out".format(undulator_num, seed_idx)

# Get Resolution
(wavelength, dz, _, dy) = GenesisTool.get_simulation_info(out_file=out_file)

newshape = (32, 32, 2 ** 16)
#################################################################
#  Create crystals for the negative chirps
#################################################################
# Some meta data for the delay line.
h_length = 2. * np.pi / (3.1355 * 1e-4)

# Some crystal properties
chi_dict = {"chi0": complex(-0.10826E-04, 0.18209E-06),
            "chih_sigma": complex(0.57174E-05, -0.12694E-06),
            "chihbar_sigma": complex(0.57174E-05, -0.12694E-06),
            "chih_pi": complex(0.52222E-05, -0.11545E-06),
            "chihbar_pi": complex(0.52222E-05, -0.11545E-06)
            }

# Asymmetric angle
alphas = np.deg2rad(np.array([8.,
                              8.,
                              -8.,
                              -8., ]))

# Initialize the crystals
crystal_list_1 = [Crystal.CrystalBlock3D(h=np.array([0, h_length, 0]),
                                         normal=-np.array([0., np.cos(alphas[x]), np.sin(alphas[x])]),
                                         surface_point=np.zeros(3),
                                         thickness=1e6,
                                         chi_dict=chi_dict)
                  for x in range(4)]

#################################################################
#     Align the crystals
#################################################################
kin_c = np.array([0, 0, np.pi * 2. / wavelength])

# Set the initial position
initial_distance = 1e6
x_initial = np.array([0, 0, -initial_distance])

# Align the 1st crystal
util.align_crystal_dynamical_bragg_reflection(crystal=crystal_list_1[0],
                                              kin=kin_c,
                                              rot_direction=-1.,
                                              scan_range=0.0005,
                                              scan_number=10000)

# Align the 2nd crystal
util.align_crystal_reciprocal_lattice(crystal=crystal_list_1[1],
                                      axis=-crystal_list_1[0].h)

# Get the output from the first two crystals
kout_cc_1 = MultiDevice.get_kout(device_list=crystal_list_1[:2],
                                 kin=kin_c)

# Align the 3rd crystal
util.align_crystal_dynamical_bragg_reflection(crystal=crystal_list_1[2],
                                              kin=kout_cc_1[-1],
                                              rot_direction=1.,
                                              scan_range=0.0005,
                                              scan_number=10000)

# Align the 4th crystal
util.align_crystal_reciprocal_lattice(crystal=crystal_list_1[3],
                                      axis=-crystal_list_1[2].h)

# Get the output from the two channel-cuts
kout_cc_2 = MultiDevice.get_kout(device_list=crystal_list_1,
                                 kin=kin_c)

# Construct the device list
device_list_all = crystal_list_1

#################################################################
#     Change the separations between devices : Negative Chirp
#################################################################
# Get directions
directions = kout_cc_2 / util.l2_norm_batch(kout_cc_2)[:, np.newaxis]

surface_point = np.zeros(3)
crystal_list_1[0].set_surface_point(surface_point=np.copy(surface_point))

surface_point += directions[1] * 0.6e6
crystal_list_1[1].set_surface_point(surface_point=np.copy(surface_point))

surface_point += directions[2] * 0.1e6
crystal_list_1[2].set_surface_point(surface_point=np.copy(surface_point))

surface_point += directions[3] * 0.6e6
crystal_list_1[3].set_surface_point(surface_point=np.copy(surface_point))

observation = np.copy(surface_point) + directions[4] * 1e6

######################################################################
#            Calculate the trajectory
######################################################################
(_,
 kout_list) = MultiDevice.get_trajectory(device_list=device_list_all,
                                         kin=kin_c,
                                         initial_point=x_initial,
                                         final_plane_point=observation,
                                         final_plane_normal=directions[-1])

#  Get the total path legnth
(intersection_list,
 _,
 path_length) = MultiDevice.get_lightpath(device_list=device_list_all,
                                          kin=kin_c,
                                          initial_point=x_initial,
                                          final_plane_point=observation,
                                          final_plane_normal=directions[-1])

intersection_array = np.vstack(intersection_list)
######################################################################
#           Construct the k grid
######################################################################
# Build the k grid
kx_list = np.fft.fftshift(np.fft.fftfreq(newshape[0], d=dy) * 2 * np.pi)
ky_list = np.fft.fftshift(np.fft.fftfreq(newshape[1], d=dy) * 2 * np.pi)
kz_list = np.fft.fftshift(np.fft.fftfreq(newshape[2], d=dz) * 2 * np.pi)
kz_list += np.pi * 2. / wavelength

k_grid = np.zeros(newshape + (3,), dtype=np.float64)
k_grid[:, :, :, 0] = kx_list[:, np.newaxis, np.newaxis]
k_grid[:, :, :, 1] = ky_list[np.newaxis, :, np.newaxis]
k_grid[:, :, :, 2] = kz_list[np.newaxis, np.newaxis, :]

# Construct the k grid
k_grid_flat = np.ascontiguousarray(np.reshape(k_grid,
                                              (np.prod(k_grid.shape[:3]), 3)))
del k_grid

# Number of batches
batch_num = 5

# split the arrays
k_grid_list = np.array_split(k_grid_flat, batch_num)

######################################################################
#           Loop through all field
######################################################################
for undulator_num in [11, 13, 15, 17, 19]:
    for seed_idx in range(10):
        # Get file name
        parent_folder = "/reg/data/ana15/xpp/xppx42118/results/james/for_haoyuan/5_lens_10_seeds/"
        field_file = parent_folder + "/2fs_15U_s1_{}_{:0>2d}.out.dfl".format(undulator_num, seed_idx)
        out_file = parent_folder + "/2fs_15U_s1_{}_{:0>2d}.out".format(undulator_num, seed_idx)

        ###############################################################################
        #      Load the in file
        ###############################################################################
        # Load the field
        tic = time.time()
        in_field = np.moveaxis(GenesisTool.read_field(field_file=field_file, out_file=out_file),
                               (0, 1, 2), (2, 0, 1))

        z_len = in_field.shape[2]

        # Pad zeros
        in_field = in_field[75 - 16:75 + 16, 75 - 16:75 + 16, :]
        in_field = np.ascontiguousarray(np.pad(in_field,
                                               ((0, 0),
                                                (0, 0),
                                                ((2 ** 16 - z_len) // 2,
                                                 2 ** 16 - (2 ** 16 - z_len) // 2 - z_len)
                                                ),
                                               mode='constant',
                                               constant_values=0.))

        toc = time.time()
        print("It takes {:.2f} seconds to load {}".format(toc - tic, field_file))
        ###########################################################################

        # Get the spectrum
        initial_spectrum = np.fft.fftshift(np.fft.fftn(in_field))
        del in_field

        # reshape of the incident spectrum
        init_spectrum_flat = np.ascontiguousarray(np.reshape(initial_spectrum,
                                                             np.prod(initial_spectrum.shape)))
        del initial_spectrum

        # split the arrays
        field_list = np.array_split(init_spectrum_flat, batch_num, axis=0)

        field_holder = list(range(batch_num))
        reflectivity_holder = list(range(batch_num))
        sanity_check = list(range(batch_num))

        # Loop through all the batches
        # Get the time
        tic = time.time()
        for idx in range(batch_num):
            (field_holder[idx],
             reflectivity_holder[idx],
             sanity_check[idx]
             ) = GPUMultiDevice.get_diffracted_monochromatic_components_sigma_polarization(
                k_grid=np.ascontiguousarray(k_grid_list[idx]),
                spectrum_in=np.ascontiguousarray(field_list[idx]),
                device_list=device_list_all,
                total_path=path_length,
                observation=observation,
                initial_position=x_initial,
                pulse_k0_final=np.array([0., 0., np.pi * 2. / wavelength]),
                d_num=512)

        # Get the time
        toc = time.time()
        print("It takes {:.2f} seconds to get the field for field {}.".format(toc - tic, field_file))

        ###############################################################################
        #        Save the result
        ###############################################################################
        spectrum_assemble = np.concatenate([field_holder[x]["final_spectrum"] for x in range(batch_num)])
        np.save("../stretched_pulse_v1/" + "/2fs_15U_s1_{}_{:0>2d}.npy".format(undulator_num, seed_idx),
                np.reshape(spectrum_assemble, newshape))

        del spectrum_assemble
