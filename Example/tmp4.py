from XRaySimulation import Pulse, MultiDevice, util, Crystal
import numpy as np

grating_distance_1 = 3.01e6
grating_distance_2 = 3.24e6
grating_period = 1.


def get_grating_and_crystals():
    energy_center = 9.8
    pre_length = 1e6

    # Set up the pulse
    FWHM = 50.  # (um)

    my_pulse = Pulse.GaussianPulse3D()
    my_pulse.set_pulse_properties(central_energy=energy_center,
                                  polar=[1., 0., 0.],
                                  sigma_x=FWHM / 2. / np.sqrt(np.log(2)) / util.c,
                                  sigma_y=FWHM / 2. / np.sqrt(np.log(2)) / util.c,
                                  sigma_z=9.,
                                  x0=np.array([0., 0., -pre_length - grating_distance_1]))

    ############################################################################################
    # Define gratings
    ###########################################################################################
    grating_list = [Crystal.RectangleGrating(), Crystal.RectangleGrating()]
    grating_list[0].set_a(grating_period / 2.)
    grating_list[0].set_b(grating_period / 2.)
    grating_list[1].set_a(grating_period / 2.)
    grating_list[1].set_b(grating_period / 2.)

    grating_list[0].set_surface_point(np.array([0, 0, -grating_distance_1]))
    grating_list[1].set_surface_point(np.array([0, 0, grating_distance_2]))

    #################################################################################################
    #                         VCC branch
    #################################################################################################
    # Define physical parameters
    h_length = 2. * np.pi / (1.9201 * 1e-4)

    # Some crystal properties
    chi_dict = {"chi0": complex(-0.10169E-04, 0.16106E-06),
                "chih_sigma": complex(0.61786E-05, -0.15508E-06),
                "chihbar_sigma": complex(0.61786E-05, -0.15508E-06),
                "chih_pi": complex(0.48374E-05, -0.11996E-06),
                "chihbar_pi": complex(0.48374E-05, -0.11996E-06),
                }

    # Get crystal_list
    thetas = np.pi / 2. * np.array([1, -1, -1, 1, 1, -1, -1, 1], dtype=np.float64)

    rhos = thetas + np.pi
    rhos[[1, 6]] -= np.deg2rad(5)
    rhos[[2, 5]] += np.deg2rad(5)

    vcc_crystals = [Crystal.CrystalBlock3D(h=np.array([0, np.sin(thetas[x]), np.cos(thetas[x])]) * h_length,
                                           normal=np.array([0., np.sin(rhos[x]), np.cos(rhos[x])]),
                                           surface_point=np.zeros(3),
                                           thickness=1e6,
                                           chi_dict=chi_dict)
                    for x in range(8)]

    # ----------------------------------------------------------------------
    #          Crystal 1
    # ----------------------------------------------------------------------
    boundary = np.array([[0, 10, 0],
                         [50, 10, 0],
                         [50, 0, 0],
                         [0, 0, 0]], dtype=np.float64) * 1000

    vcc_crystals[0].boundary = np.copy(boundary[:, [2, 1, 0]])
    vcc_crystals[0].set_surface_point(np.copy(vcc_crystals[0].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 2
    # ----------------------------------------------------------------------
    boundary = np.array([[25.070, 26.40, 0],
                         [90.070, 20.753, 0],
                         [90.070, 36.40, 0],
                         [25.070, 36.40, 0]], dtype=np.float64) * 1000

    vcc_crystals[1].boundary = np.copy(boundary[:, [2, 1, 0]])
    vcc_crystals[1].set_surface_point(np.copy(vcc_crystals[1].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 3
    # ----------------------------------------------------------------------
    boundary = np.copy(vcc_crystals[1].boundary)
    boundary[:, 2] *= -1

    vcc_crystals[2].boundary = np.copy(boundary)
    vcc_crystals[2].set_surface_point(np.copy(vcc_crystals[2].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 4
    # ----------------------------------------------------------------------
    boundary = np.copy(vcc_crystals[0].boundary)
    boundary[:, 2] *= -1

    vcc_crystals[3].boundary = np.copy(boundary)
    vcc_crystals[3].set_surface_point(np.copy(vcc_crystals[3].boundary[0]))

    # ----------------------------------------------------------------------
    #          Crystal 5
    # ----------------------------------------------------------------------
    vcc_crystals[4].boundary = np.copy(np.copy(vcc_crystals[3].boundary))
    vcc_crystals[4].set_surface_point(np.copy(vcc_crystals[4].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 7
    # ----------------------------------------------------------------------
    boundary = np.array([[25.070, 26.40, 0],
                         [90.070, 20.753, 0],
                         [90.070, 30.753, 0],
                         [25.070, 36.40, 0]], dtype=np.float64) * 1000

    vcc_crystals[6].boundary = np.copy(boundary[:, [2, 1, 0]])
    vcc_crystals[6].set_surface_point(np.copy(vcc_crystals[6].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 6
    # ----------------------------------------------------------------------
    boundary = np.copy(vcc_crystals[6].boundary)
    boundary[:, 2] *= -1

    vcc_crystals[5].boundary = np.copy(boundary)
    vcc_crystals[5].set_surface_point(np.copy(vcc_crystals[5].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 8
    # ----------------------------------------------------------------------
    vcc_crystals[7].boundary = np.copy(vcc_crystals[0].boundary)
    vcc_crystals[7].set_surface_point(np.copy(vcc_crystals[7].boundary[0]))

    # ---------------------------------------------------------------------------

    #################################################################################################
    #                         CC branch
    thetas = np.array([-np.pi / 2,
                       np.pi / 2,
                       np.pi / 2,
                       -np.pi / 2])
    rhos = thetas + np.pi

    cc_crystals = [Crystal.CrystalBlock3D(h=np.array([0, np.sin(thetas[x]), np.cos(thetas[x])]) * h_length,
                                          normal=np.array([0., np.sin(rhos[x]), np.cos(rhos[x])]),
                                          surface_point=np.zeros(3),
                                          thickness=1e6,
                                          chi_dict=chi_dict)
                   for x in range(4)]
    # ----------------------------------------------------------------------
    #          Crystal 1
    # ----------------------------------------------------------------------
    boundary = np.array([[-60, 12.57, 0],
                         [-20, 12.57, 0],
                         [-20, 22.57, 0],
                         [-22.68, 22.57, 0]])
    boundary -= np.array([-60, -22.57, 0])[np.newaxis, :]
    boundary *= 1000

    cc_crystals[0].boundary = np.copy(boundary[:, [2, 1, 0]])
    cc_crystals[0].set_surface_point(np.copy(cc_crystals[0].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 2
    # ----------------------------------------------------------------------
    boundary = np.array([[-60, -12.58, 0],
                         [60, -12.58, 0],
                         [60, -22.57, 0],
                         [-60, -22.57, 0]])
    boundary -= np.array([-60, -22.57, 0])[np.newaxis, :]
    boundary *= 1000

    cc_crystals[1].boundary = np.copy(boundary[:, [2, 1, 0]])
    cc_crystals[1].set_surface_point(np.copy(cc_crystals[1].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 3
    # ----------------------------------------------------------------------
    boundary = np.array([[-65, -12.90, 0],
                         [65, -12.90, 0],
                         [65, -22.90, 0],
                         [-65, -22.90, 0]])
    boundary -= np.array([-65, -22.90, 0])[np.newaxis, :]
    boundary *= 1000

    cc_crystals[2].boundary = np.copy(boundary[:, [2, 1, 0]])
    cc_crystals[2].set_surface_point(np.copy(cc_crystals[2].boundary[0]))
    # ----------------------------------------------------------------------
    #          Crystal 4
    # ----------------------------------------------------------------------
    boundary = np.array([[45, 12.90, 0],
                         [65, 12.90, 0],
                         [65, 22.90, 0],
                         [45, 22.90, 0]])
    boundary -= np.array([-65, -22.90, 0])[np.newaxis, :]
    boundary *= 1000

    cc_crystals[3].boundary = np.copy(boundary[:, [2, 1, 0]])
    cc_crystals[3].set_surface_point(np.copy(cc_crystals[3].boundary[0]))

    return my_pulse, grating_list, vcc_crystals, cc_crystals


def get_analyzer():
    ####################################################################
    #     sampler
    ####################################################################
    # Define physical parameters
    h_length = 2. * np.pi / (1.9201 * 1e-4)

    # Some crystal properties
    chi_dict = {"chi0": complex(-0.10169E-04, 0.16106E-06),
                "chih_sigma": complex(0.61786E-05, -0.15508E-06),
                "chihbar_sigma": complex(0.61786E-05, -0.15508E-06),
                "chih_pi": complex(0.48374E-05, -0.11996E-06),
                "chihbar_pi": complex(0.48374E-05, -0.11996E-06),
                }

    # Get crystal_list
    thetas = np.pi / 2.

    rhos = thetas + np.pi

    sampler = Crystal.CrystalBlock3D(h=np.array([0, np.sin(thetas), np.cos(thetas)]) * h_length,
                                     normal=np.array([0., np.sin(rhos), np.cos(rhos)]),
                                     surface_point=np.zeros(3),
                                     thickness=1e6,
                                     chi_dict=chi_dict)

    ####################################################################
    #     Analyzer
    ####################################################################
    # Define physical parameters
    h_length = 2. * np.pi / (1.9201 * 1e-4)

    # Some crystal properties
    chi_dict = {"chi0": complex(-0.10169E-04, 0.16106E-06),
                "chih_sigma": complex(0.21718E-05, -0.12003E-06),
                "chihbar_sigma": complex(0.21718E-05, -0.12003E-06),
                "chih_pi": complex(0.20711E-05, -0.11420E-06),
                "chihbar_pi": complex(0.20711E-05, -0.11420E-06),
                }

    # Get crystal_list
    thetas = np.pi / 2.
    rhos = thetas + np.pi

    analyzer = Crystal.CrystalBlock3D(h=np.array([0, np.sin(thetas), np.cos(thetas)]) * h_length,
                                      normal=np.array([0., np.sin(rhos), np.cos(rhos)]),
                                      surface_point=np.zeros(3),
                                      thickness=1e6,
                                      chi_dict=chi_dict)

    return sampler, analyzer


def align_vcc_crystals(kin, crystals):
    kin_new = np.copy(kin)

    #####################################################
    #     Align CC1
    #####################################################
    rot_center = np.copy((crystals[0].boundary[0] + crystals[0].boundary[1]) / 2.)

    # Align the 1 crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystals[0],
                                                  kin=kin_new,
                                                  rot_direction=-1.,
                                                  rot_center=rot_center)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystals[1],
                                          axis=-crystals[0].h,
                                          rot_center=rot_center)

    # Get the kout from this crystal
    kin_new = MultiDevice.get_kout(device_list=crystals[:2],
                                   kin=kin)
    kin_new = kin_new[-1]

    crystals[0].shift(-rot_center)
    crystals[1].shift(-rot_center)

    #####################################################
    #     Align CC2
    #####################################################
    rot_center = np.copy((crystals[3].boundary[0] + crystals[3].boundary[1]) / 2.)

    # Align the 1 crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystals[2],
                                                  kin=kin_new,
                                                  rot_direction=1.,
                                                  rot_center=rot_center)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystals[3],
                                          axis=-crystals[2].h,
                                          rot_center=rot_center)

    # Get the kout from this crystal
    kin_new = MultiDevice.get_kout(device_list=crystals[:4],
                                   kin=kin)
    kin_new = kin_new[-1]

    crystals[2].shift(-rot_center)
    crystals[3].shift(-rot_center)
    #####################################################
    #     Align CC3
    #####################################################
    rot_center = (crystals[4].boundary[0] + crystals[4].boundary[1]) / 2.

    # Align the 1 crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystals[4],
                                                  kin=kin_new,
                                                  rot_direction=1.,

                                                  rot_center=rot_center)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystals[5],
                                          axis=-crystals[4].h,
                                          rot_center=rot_center)

    # Get the kout from this crystal
    kin_new = MultiDevice.get_kout(device_list=crystals[:6],
                                   kin=kin)
    kin_new = kin_new[-1]

    crystals[4].shift(-rot_center)
    crystals[5].shift(-rot_center)
    #####################################################
    #     Align CC4
    #####################################################
    rot_center = (crystals[7].boundary[0] + crystals[7].boundary[1]) / 2.

    # Align the 1 crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystals[6],
                                                  kin=kin_new,
                                                  rot_direction=-1.,
                                                  rot_center=rot_center)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystals[7],
                                          axis=-crystals[6].h,
                                          rot_center=rot_center)

    # Get the kout from this crystal
    kin_new = MultiDevice.get_kout(device_list=crystals,
                                   kin=kin)
    kin_new = kin_new[-1]
    crystals[6].shift(-rot_center)
    crystals[7].shift(-rot_center)

    return crystals, kin_new


def align_cc_crystals(kin, crystals):
    kin_new = np.copy(kin)

    #####################################################
    #     Align CC1
    #####################################################
    rot_center = np.copy((crystals[0].boundary[0]))

    # Align the 1 crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystals[0],
                                                  kin=kin_new,
                                                  rot_direction=1.,
                                                  rot_center=rot_center)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystals[1],
                                          axis=-crystals[0].h,
                                          rot_center=rot_center)

    # Get the kout from this crystal
    kin_new = MultiDevice.get_kout(device_list=crystals[:2],
                                   kin=kin)
    kin_new = kin_new[-1]

    crystals[0].shift(-rot_center)
    crystals[1].shift(-rot_center)
    #####################################################
    #     Align CC2
    #####################################################
    rot_center = np.copy((crystals[3].boundary[0]))

    # Align the 1 crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=crystals[2],
                                                  kin=kin_new,
                                                  rot_direction=-1.,
                                                  rot_center=rot_center)
    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=crystals[3],
                                          axis=-crystals[2].h,
                                          rot_center=rot_center)

    # Get the kout from this crystal
    kin_new = MultiDevice.get_kout(device_list=crystals[:4],
                                   kin=kin)
    kin_new = kin_new[-1]

    crystals[2].shift(-rot_center)
    crystals[3].shift(-rot_center)
    return crystals, kin_new


def set_crystal_positions(vcc, cc):
    # CC1 of the CC branch is at the assumed position

    # First shift VCC1
    vcc[0].shift(np.array([0, 0, 198.51]) * 1e3)
    vcc[1].shift(np.array([0, 0, 198.51]) * 1e3)

    # Second shift VCC2
    vcc[2].shift(np.array([0, 0, 198.51 + 178.]) * 1e3)
    vcc[3].shift(np.array([0, 0, 198.51 + 178.]) * 1e3)

    # Third shift VCC3
    vcc[4].shift(np.array([0, 0, 198.51 + 178. + 223.49]) * 1e3)
    vcc[5].shift(np.array([0, 0, 198.51 + 178. + 223.49]) * 1e3)

    # Forth shift VCC3
    vcc[6].shift(np.array([0, 0, 198.51 + 178. + 223.49 + 225.01]) * 1e3)
    vcc[7].shift(np.array([0, 0, 198.51 + 178. + 223.49 + 225.01]) * 1e3)

    # Shift the second CC in CC branch
    cc[2].shift(np.array([0, 0, 198.51 + 178. + 223.49 + 225.01 + 225.01]) * 1e3)
    cc[3].shift(np.array([0, 0, 198.51 + 178. + 223.49 + 225.01 + 225.01]) * 1e3)


def get_trajectory(vcc_motion=np.zeros(8), cc_motion=np.zeros(4)):
    # Create the setup
    (my_pulse,
     grating_list,
     vcc_crystals,
     cc_crystals) = get_grating_and_crystals()

    kin_new = grating_list[0].base_wave_vector + my_pulse.k0
    _, vcc_kout = align_vcc_crystals(kin=kin_new, crystals=vcc_crystals)

    kin_new = -grating_list[0].base_wave_vector + my_pulse.k0
    _, cc_kout = align_cc_crystals(kin=kin_new, crystals=cc_crystals)

    set_crystal_positions(vcc=vcc_crystals, cc=cc_crystals)

    # Move the crystals along the y axis
    for x in range(8):
        vcc_crystals[x].shift(displacement=np.array([0., vcc_motion[x], 0.], dtype=np.float64))
    for x in range(4):
        cc_crystals[x].shift(displacement=np.array([0., cc_motion[x], 0.], dtype=np.float64))

    # Get the trajectory of VCC and CC branch
    device_list = [grating_list[0], ] + vcc_crystals + [grating_list[1], ]
    device_list[0].momentum_transfer = device_list[0].base_wave_vector
    device_list[-1].momentum_transfer = -device_list[-1].base_wave_vector

    (vcc_trajectory,
     vcc_kout_list,
     vcc_path) = MultiDevice.get_lightpath(device_list=device_list,
                                           kin=my_pulse.k0,
                                           initial_point=my_pulse.x0,
                                           final_plane_point=np.array((0, 0, 10e6),
                                                                      dtype=np.float64),
                                           final_plane_normal=np.array((0, 0, -1),
                                                                       dtype=np.float64))
    vcc_trajectory = np.vstack(vcc_trajectory)
    vcc_kout_list = np.vstack(vcc_kout_list)

    device_list = [grating_list[0], ] + cc_crystals + [grating_list[1], ]
    device_list[0].momentum_transfer = -device_list[0].base_wave_vector
    device_list[-1].momentum_transfer = device_list[-1].base_wave_vector

    (cc_trajectory,
     cc_kout_list,
     cc_path) = MultiDevice.get_lightpath(device_list=device_list,
                                          kin=my_pulse.k0,
                                          initial_point=my_pulse.x0,
                                          final_plane_point=np.array((0, 0, 10e6),
                                                                     dtype=np.float64),
                                          final_plane_normal=np.array((0, 0, -1),
                                                                      dtype=np.float64))
    cc_trajectory = np.vstack(cc_trajectory)
    cc_kout_list = np.vstack(cc_kout_list)

    return (vcc_trajectory,
            vcc_kout_list,
            vcc_path,
            vcc_crystals,
            cc_trajectory,
            cc_kout_list,
            cc_path,
            cc_crystals,
            grating_list,
            my_pulse)


def tweak_horizontal_position_ratio(vcc_motion=np.zeros(8, dtype=np.float64),
                                    cc_motion=np.zeros(4, dtype=np.float64)):
    # Position1
    config1 = get_trajectory(vcc_motion=vcc_motion, cc_motion=cc_motion)

    # Position2
    config2 = get_trajectory(vcc_motion=vcc_motion + np.array([0, 0, 0, 0,
                                                               1, 1, -1, -1],
                                                              dtype=np.float64),
                             cc_motion=cc_motion)

    # Get the horizontal shift
    horizontal_shift = config2[0][-1][1] - config1[0][-1][1]

    print("t4x + 1um and t5x - 1um results in horizontal shift of {:.2f}um".format(horizontal_shift))
    return horizontal_shift, config1


def tweak_horizontal_overlap(vcc_motion=np.zeros(8, dtype=np.float64),
                             cc_motion=np.zeros(4, dtype=np.float64)):
    # Get ratio
    horizontal_shift_per_um, current_config = tweak_horizontal_position_ratio(vcc_motion=vcc_motion,
                                                                              cc_motion=cc_motion)

    # Get the distance
    tweak_size = (current_config[4][-1][1] - current_config[0][-1][1]) / horizontal_shift_per_um
    print("To get two pulse overlap, t4x should move {:.2f} um, t5x should move {:.2f} um".format(tweak_size,
                                                                                                  - tweak_size))
    return tweak_size


def tweak_temporal(vcc_motion=np.zeros(8, dtype=np.float64),
                   cc_motion=np.zeros(4, dtype=np.float64)):
    # Position1
    config1 = get_trajectory(vcc_motion=vcc_motion, cc_motion=cc_motion)

    # Position2
    config2 = get_trajectory(vcc_motion=vcc_motion + np.array([1, 1, 1, 1,
                                                               0, 0, 0, 0],
                                                              dtype=np.float64),
                             cc_motion=cc_motion)

    # Get the horizontal shift
    temporal_shift = config2[2] - config1[2]

    print("t23 +1 um results in temporal delay of {:.2f}fs".format(temporal_shift / util.c))
    return temporal_shift, config1


def tweak_temporal_overlap(vcc_motion=np.zeros(8, dtype=np.float64),
                           cc_motion=np.zeros(4, dtype=np.float64)):
    # Get ratio
    temporal_shift_per_um, current_config = tweak_temporal(vcc_motion=vcc_motion,
                                                           cc_motion=cc_motion)

    # Get the distance
    tweak_size = (current_config[6] - current_config[2]) / temporal_shift_per_um
    print("To get two pulse overlap, t23 should move {:.2f} um".format(tweak_size))
    return tweak_size
