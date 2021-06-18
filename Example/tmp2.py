import numpy as np
from XRaySimulation import util, Crystal, SpecialOptics, MultiDevice


def setup_devices(d1, d2, d3, d4, focal_length, wavelength):
    #####################################
    #   Load metadata
    #####################################
    kin = np.array([0, 0, np.pi * 2. / wavelength])

    # Set the initial position
    initial_distance = 1e6
    x0 = np.array([0, 0, -initial_distance])

    #####################################
    #   Define crystals
    #####################################
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
                                  -8,
                                  ]))

    # Initialize the crystals
    crystal_list = [Crystal.CrystalBlock3D(h=np.array([0, h_length, 0]),
                                           normal=-np.array([0., np.cos(alphas[x]), np.sin(alphas[x])]),
                                           surface_point=np.zeros(3),
                                           thickness=1e6,
                                           chi_dict=chi_dict)
                    for x in range(4)]

    # Telescope
    telescope = SpecialOptics.TelescopeForCPA()
    telescope.focal_length = focal_length
    telescope.efficiency = complex(1.)

    #####################################
    #   Align crystal and telescope
    #####################################
    # Align the 1st crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystal_list[0],
                                                  kin=kin,
                                                  rot_direction=-1.,
                                                  scan_range=0.0005,
                                                  scan_number=10000)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystal_list[1],
                                          axis=-crystal_list[0].h)

    # Get the kout from this crystal
    kout_3 = MultiDevice.get_kout(device_list=crystal_list[:2],
                                  kin=kin)
    kout_3 = kout_3[-1]

    # Align the telescope
    telescope.lens_axis = kout_3 / util.l2_norm(kout_3)

    # Get kout from the telescope
    kout_4 = util.get_kout(device=telescope,
                           kin=kout_3)

    # Align the 3rd crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystal_list[2],
                                                  kin=kout_4,
                                                  rot_direction=1.,
                                                  scan_range=0.0005,
                                                  scan_number=10000)

    # Adjust a bit
    angle1 = np.arccos(np.dot(crystal_list[1].normal, telescope.lens_axis))
    angle2 = np.arccos(-np.dot(crystal_list[2].normal, telescope.lens_axis))
    angles_adjust = angle2 - angle1
    rot_mat = util.rot_mat_in_yz_plane(theta=angles_adjust)
    crystal_list[2].rotate_wrt_point(rot_mat=rot_mat,
                                     ref_point=crystal_list[1].surface_point)

    # Align the 4th crystal
    util.align_crystal_reciprocal_lattice(crystal=crystal_list[3],
                                          axis=-crystal_list[2].h)

    device_list_all = [crystal_list[0],
                       crystal_list[1],
                       telescope,
                       crystal_list[2],
                       crystal_list[3]
                       ]

    # Get the output from the two channel-cuts
    kout_positive = MultiDevice.get_kout(device_list=device_list_all,
                                         kin=kin)

    #####################################
    #   Set relative positions
    #####################################
    # Get directions
    directions = kout_positive / util.l2_norm_batch(kout_positive)[:, np.newaxis]

    surface_point = np.zeros(3, dtype=np.float64)

    surface_point = np.copy(surface_point)
    crystal_list[0].set_surface_point(surface_point=np.copy(surface_point))

    surface_point += directions[1] * d1
    crystal_list[1].set_surface_point(surface_point=np.copy(surface_point))

    surface_point += directions[2] * d2
    telescope.lens_position = np.copy(surface_point)

    surface_point += directions[3] * (d3 + 2 * focal_length)
    crystal_list[2].set_surface_point(surface_point=np.copy(surface_point))

    surface_point += directions[4] * d4
    crystal_list[3].set_surface_point(surface_point=np.copy(surface_point))

    observation = np.copy(surface_point) + directions[5] * 1e6

    ###########################################
    #  Get trajectory
    ###########################################
    (_,
     kout_list) = MultiDevice.get_trajectory(device_list=device_list_all,
                                             kin=kin,
                                             initial_point=x0,
                                             final_plane_point=observation,
                                             final_plane_normal=directions[-1])

    #  Get the total path legnth
    (intersection_list,
     _,
     path_length) = MultiDevice.get_lightpath(device_list=device_list_all,
                                              kin=kin,
                                              initial_point=x0,
                                              final_plane_point=observation,
                                              final_plane_normal=directions[-1])

    intersection_array = np.vstack(intersection_list)

    return device_list_all, intersection_array, kin, x0
