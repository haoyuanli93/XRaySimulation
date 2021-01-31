import numpy as np
from scipy.spatial.transform import Rotation 
import sys

sys.path.append("C:/Users/haoyuan/Documents/GitHub/XRaySimulation")

from XRaySimulation import util, Crystal, MultiDevice

# Some meta data for the delay line.
h_length = 2. * np.pi / (1.9201 * 1e-4)

# Some crystal properties
chi_dict = {"chi0": complex(-0.10106E-04, 0.15913E-06),
            "chih_sigma": complex(0.61405E-05, -0.15323E-06),
            "chihbar_sigma": complex(0.61405E-05, -0.15323E-06),
            "chih_pi": complex(0.48157E-05, -0.11872E-06),
            "chihbar_pi": complex(0.48157E-05, -0.11872E-06)
            }

def get_crystals_in_delay_fixed_branch(pulse_in, grating, alignment_tweak):

    ########################################################################
    #       Create crystals for delay-fixed branch
    ########################################################################
    alphas = np.zeros(4)

    # Create 4 crystal list
    delay_fixed_crystals = [Crystal.CrystalBlock3D(h=np.array([0, h_length, 0]),
                                                   normal=-np.array([0, np.cos(alphas[x]), np.sin(alphas[x])]),
                                                   surface_point=np.zeros(3),
                                                   thickness=1e6,
                                                   chi_dict=chi_dict) for x in range(4)]

    # The following boundary information is taken from the solid edge model

    # ----------------------------------------------------------------------
    #          Crystal 1
    # ----------------------------------------------------------------------
    boundary = np.array([[-60, 12.57, 40],
                         [-20, 12.57, 40],
                         [-20, 22.57, 40],
                         [-22.68, 22.57, 40]])
    boundary -= np.array([-60, -22.57, 0])[np.newaxis, :]
    boundary *= 1000

    delay_fixed_crystals[0].boundary = boundary[:, [2, 1, 0]]
    delay_fixed_crystals[0].set_surface_point(delay_fixed_crystals[0].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 2
    # ----------------------------------------------------------------------
    boundary = np.array([[-60, -12.58, 40],
                         [60, -12.58, 40],
                         [60, -22.57, 40],
                         [-60, -22.57, 40]])
    boundary -= np.array([-60, -22.57, 0])[np.newaxis, :]
    boundary *= 1000

    delay_fixed_crystals[1].boundary = boundary[:, [2, 1, 0]]
    delay_fixed_crystals[1].set_surface_point(delay_fixed_crystals[1].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 3
    # ----------------------------------------------------------------------
    boundary = np.array([[-65, -12.90, 40],
                         [65, -12.90, 40],
                         [65, -22.90, 40],
                         [-65, -22.90, 40]])
    boundary -= np.array([-65, -22.90, 0])[np.newaxis, :]
    boundary *= 1000

    delay_fixed_crystals[2].boundary = boundary[:, [2, 1, 0]]
    delay_fixed_crystals[2].set_surface_point(delay_fixed_crystals[2].boundary[0])
    # ----------------------------------------------------------------------
    #          Crystal 4
    # ----------------------------------------------------------------------
    boundary = np.array([[45, 12.90, 40],
                         [65, 12.90, 40],
                         [65, 22.90, 40],
                         [45, 22.90, 40]])
    boundary -= np.array([-65, -22.90, 0])[np.newaxis, :]
    boundary *= 1000

    delay_fixed_crystals[3].boundary = boundary[:, [2, 1, 0]]
    delay_fixed_crystals[3].set_surface_point(delay_fixed_crystals[3].boundary[0])

    ########################################################################
    #       Move the crystal to the positions in the CAD model
    #       And align around the bragg angle
    ########################################################################
    # Get the wave vector from the first grating
    kout_g_1 = pulse_in.k0 + grating.base_wave_vector

    # Move the first crystal
    displacement = np.array([128.10, 627.61, 252.70]) * 1000
    delay_fixed_crystals[0].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    # Align the 1st crystal with the specified position
    util.align_crystal_dynamical_bragg_reflection(crystal=delay_fixed_crystals[0],
                                                  kin=kout_g_1,
                                                  rot_direction=1.,
                                                  scan_range=0.0005,
                                                  scan_number=10000)

    delay_fixed_crystals[0].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]], include_boundary=True)

    delay_fixed_crystals[1].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_crystals[1].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]], include_boundary=True)




    # Align the 2nd crystal
    util.align_crystal_reciprocal_lattice(crystal=delay_fixed_crystals[1],
                                          axis=-delay_fixed_crystals[0].h)

    # Get the output from the first two crystals
    kout_cc_1 = MultiDevice.get_kout(device_list=delay_fixed_crystals[:2],
                                     kin=kout_g_1)

    # Align the 3rd crystal
    util.align_crystal_dynamical_bragg_reflection(crystal=delay_fixed_crystals[2],
                                                  kin=kout_cc_1[-1],
                                                  rot_direction=-1.,
                                                  scan_range=0.0005,
                                                  scan_number=10000)

    # Align the 4th crystal
    util.align_crystal_reciprocal_lattice(crystal=delay_fixed_crystals[3],
                                          axis=-delay_fixed_crystals[2].h)

    # Get the output from the two channel-cuts
    kout_cc_2 = MultiDevice.get_kout(device_list=delay_fixed_crystals,
                                     kin=kout_g_1)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # Set up the angles
    angle_offset = -dtheta

    # Bragg angle
    bragg = np.radians(19.868) + 7e-6

    # ----------------------------------------------------------------------
    #          CC 1
    # ----------------------------------------------------------------------
    displacement = np.array([128.10, 627.61, 252.70])* 1000
    rot_mat = Rotation.from_euler('x', bragg + angle_offset)

    delay_fixed_crystals[0].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_crystals[0].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]], include_boundary=True)

    delay_fixed_crystals[1].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_crystals[1].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]], include_boundary=True)

    # ----------------------------------------------------------------------
    #          CC 2
    # ----------------------------------------------------------------------
    displacement = np.array([973.84, 588.51, 252.64])* 1000
    rot_mat = Rotation.from_euler('x', -bragg + angle_offset)

    delay_fixed_crystals[2].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_crystals[2].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]], include_boundary=True)

    delay_fixed_crystals[3].shift(displacement=displacement[[2, 1, 0]], include_boundary=True)
    delay_fixed_crystals[3].rotate_wrt_point(rot_mat.as_dcm(), ref_point=displacement[[2, 1, 0]], include_boundary=True)