import numpy as np
from numba import jit

from XRaySimulation import util

two_pi = 2. * np.pi


def get_bragg_reflectivity_fix_crystal(kin, thickness, crystal_h, normal, chi_dict):
    """
    Calculate the reflectivity with a fixed crystal.
    
    :param kin: wave vector array.  Numpy array of shape (n, 3)
    :param thickness: float
    :param crystal_h: Numpy array of shape (3,)
    :param normal: numpy array of shape (3,)
    :param chi_dict: The dictionary for parameters of electric susceptability.
    :return:
    """

    # Extract the parameter
    chi0 = chi_dict["chi0"]
    chih_sigma = chi_dict["chih_sigma"]
    chihbar_sigma = chi_dict["chih_sigma"]
    chih_pi = chi_dict["chih_pi"]
    chihbar_pi = chi_dict["chih_pi"]

    # ----------------------------------------------
    #    Get reflected wave-vectors
    # ----------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = np.linalg.norm(kin, axis=-1)

    # Get gamma and alpha and b
    dot_kn = np.dot(kin, normal)
    dot_kh = np.dot(kin, crystal_h)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + np.dot(crystal_h, normal), klen_grid)

    b_factor = np.divide(gamma_0, gamma_h).astype(np.complex128)
    alpha = np.divide(2 * dot_kh + np.sum(np.square(crystal_h)), np.square(klen_grid))

    # Get momentum tranfer
    delta = np.multiply(klen_grid, -gamma_h - np.sqrt(gamma_h ** 2 - alpha))

    # Get the output wave-vector
    kout = np.copy(kin) + crystal_h[np.newaxis, :] + delta[:, np.newaxis] * normal[np.newaxis, :]

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    alpha = alpha.astype(np.complex128)
    # Get alpha tidle
    alpha_tidle = (alpha * b_factor + chi0 * (complex(1., 0) - b_factor)) / complex(2., 0)

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** complex(2.0, 0) + b_factor * (chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = complex(1., 0) - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2., 0) - numerator)

    reflect_sigma = chih_sigma * b_factor * numerator / denominator

    # ------------------------------------------------------------
    # Step 3: Get the reflectivity for pi polarization
    # ------------------------------------------------------------
    # Notice that the polarization factor has been incorporated into the chi_pi

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + b_factor * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag
    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)
    reflect_pi = b_factor * chih_pi * numerator / denominator

    return reflect_sigma, reflect_pi, b_factor, kout


def get_bragg_reflectivity_per_entry(kin, thickness, crystal_h, normal, chi_dict):
    """
    Calculate the reflectivity for each element. 
    
    :param kin: wave vector array.  Numpy array of shape (n, 3)
    :param thickness: float
    :param crystal_h: Numpy array of shape (n, 3)
    :param normal: numpy array of shape (n, 3)
    :param chi_dict: The dictionary for parameters of electric susceptability.
    :return:
    """

    # Extract the parameter
    chi0 = chi_dict["chi0"]
    chih_sigma = chi_dict["chih_sigma"]
    chihbar_sigma = chi_dict["chih_sigma"]
    chih_pi = chi_dict["chih_pi"]
    chihbar_pi = chi_dict["chih_pi"]

    # ----------------------------------------------
    #    Get reflected wave-vectors
    # ----------------------------------------------
    # Get some info to facilitate the calculation
    klen_grid = np.linalg.norm(kin, axis=-1)

    # Get gamma and alpha and b
    dot_kn = np.sum(np.multiply(kin, normal), axis=-1)
    dot_kh = np.sum(np.multiply(kin, crystal_h), axis=-1)

    gamma_0 = np.divide(dot_kn, klen_grid)
    gamma_h = np.divide(dot_kn + np.sum(np.multiply(crystal_h, normal), axis=-1), klen_grid)

    b_factor = np.divide(gamma_0, gamma_h).astype(np.complex128)
    alpha = np.divide(2 * dot_kh + np.sum(np.square(crystal_h), axis=-1), np.square(klen_grid))

    # Get momentum tranfer
    delta = np.multiply(klen_grid, -gamma_h - np.sqrt(gamma_h ** 2 - alpha))

    # Get the output wave-vector
    kout = np.copy(kin) + crystal_h + delta[:, np.newaxis] * normal

    # ------------------------------------------------------------
    # Step 2: Get the reflectivity for input sigma polarization
    # ------------------------------------------------------------
    alpha = alpha.astype(np.complex128)
    # Get alpha tidle
    alpha_tidle = (alpha * b_factor + chi0 * (complex(1., 0) - b_factor)) / complex(2., 0)

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** complex(2.0, 0) + b_factor * (chih_sigma * chihbar_sigma))

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag

    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = complex(1., 0) - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2., 0) - numerator)

    reflect_sigma = chih_sigma * b_factor * numerator / denominator

    # ------------------------------------------------------------
    # Step 3: Get the reflectivity for pi polarization
    # ------------------------------------------------------------
    # Notice that the polarization factor has been incorporated into the chi_pi

    # Get sqrt(alpha**2 + beta**2) value
    sqrt_a2_b2 = np.sqrt(alpha_tidle ** 2 + b_factor * chih_pi * chihbar_pi)

    # Change the imaginary part sign
    mask = np.zeros_like(sqrt_a2_b2, dtype=bool)
    mask[sqrt_a2_b2.imag < 0] = True
    sqrt_a2_b2[mask] = - sqrt_a2_b2[mask]

    # Calculate the phase term
    re = klen_grid * thickness / gamma_0 * sqrt_a2_b2.real
    im = klen_grid * thickness / gamma_0 * sqrt_a2_b2.imag
    magnitude = np.exp(-im).astype(np.complex128)
    phase = np.cos(re) + np.sin(re) * 1.j

    # Calculate some intermediate part
    numerator = 1. - magnitude * phase
    denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)
    reflect_pi = b_factor * chih_pi * numerator / denominator

    return reflect_sigma, reflect_pi, b_factor, kout


def get_reflectivity_channel_cut(kin_array,
                                 channelCut):
    """

    :param kin_array:
    :param channelCut:
    :return:
    """
    (reflect_sigma1,
     reflect_pi1,
     b_factor1,
     kout1) = get_bragg_reflectivity_fix_crystal(kin=kin_array,
                                                 thickness=channelCut.crystal_list[0].thickness,
                                                 crystal_h=channelCut.crystal_list[0].h,
                                                 normal=channelCut.crystal_list[0].normal,
                                                 chi_dict=channelCut.crystal_list[0].chi_dict, )
    (reflect_sigma2,
     reflect_pi2,
     b_factor2,
     kout2) = get_bragg_reflectivity_fix_crystal(kin=kin_array,
                                                 thickness=channelCut.crystal_list[0].thickness,
                                                 crystal_h=channelCut.crystal_list[0].h,
                                                 normal=channelCut.crystal_list[0].normal,
                                                 chi_dict=channelCut.crystal_list[0].chi_dict, )

    return (reflect_sigma1 * reflect_sigma2,
            reflect_pi1 * reflect_pi2,
            b_factor1 * b_factor2,
            kout2)


def get_bragg_rocking_curve(kin,
                            scan_range,
                            scan_number,
                            h_initial,
                            normal_initial,
                            thickness,
                            chi_dict):
    """

    :param kin:
    :param scan_range:
    :param scan_number:
    :param h_initial:
    :param normal_initial:
    :param thickness:
    :param chi_dict:
    :return:
    """

    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning angle
    angles = np.linspace(start=-scan_range / 2, stop=scan_range / 2, num=scan_number)

    for idx in range(scan_number):
        rot_mat = util.rot_mat_in_yz_plane(theta=angles[idx])
        h_array[idx] = rot_mat.dot(h_initial)
        normal_array[idx] = rot_mat.dot(normal_initial)

    # Create holder to save the reflectivity and output momentum
    kin_grid = np.zeros_like(h_array, dtype=np.float64)
    kin_grid[:, 0] = kin[0]
    kin_grid[:, 1] = kin[1]
    kin_grid[:, 2] = kin[2]

    (reflect_sigma,
     reflect_pi,
     b_factor,
     kout) = get_bragg_reflectivity_per_entry(kin=kin_grid,
                                              thickness=thickness,
                                              crystal_h=h_array,
                                              normal=normal_array,
                                              chi_dict=chi_dict)

    return angles, reflect_sigma, reflect_pi, b_factor, kout


def get_bragg_rocking_curve_channelcut(kin,
                                       channelcut,
                                       scan_range,
                                       scan_number,
                                       ):
    """

    :param kin:
    :param channelcut
    :param scan_range:
    :param scan_number:
    :return:
    """

    # ------------------------------------------------------------
    #          Step 0: Generate h_array and normal_array for the scanning
    # ------------------------------------------------------------
    h_array_1 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_1 = np.zeros((scan_number, 3), dtype=np.float64)

    h_array_2 = np.zeros((scan_number, 3), dtype=np.float64)
    normal_array_2 = np.zeros((scan_number, 3), dtype=np.float64)

    # Get the scanning angle
    angles = np.linspace(start=-scan_range / 2, stop=scan_range / 2, num=scan_number)

    for idx in range(scan_number):
        rot_mat = util.rot_mat_in_yz_plane(theta=angles[idx])

        h_array_1[idx] = rot_mat.dot(channelcut.crystal_list[0].h)
        normal_array_1[idx] = rot_mat.dot(channelcut.crystal_list[0].normal)

        h_array_2[idx] = rot_mat.dot(channelcut.crystal_list[1].h)
        normal_array_2[idx] = rot_mat.dot(channelcut.crystal_list[1].normal)

    # Create holder to save the reflectivity and output momentum
    kin_grid = np.zeros_like(h_array_1, dtype=np.float64)
    kin_grid[:, 0] = kin[0]
    kin_grid[:, 1] = kin[1]
    kin_grid[:, 2] = kin[2]

    (reflect_sigma_1,
     reflect_pi_1,
     b_factor_1,
     kout_1) = get_bragg_reflectivity_per_entry(kin=kin_grid,
                                                thickness=channelcut.crystal_list[0].thickness,
                                                crystal_h=h_array_1,
                                                normal=normal_array_1,
                                                chi_dict=channelcut.crystal_list[0].chi_dict)

    (reflect_sigma_2,
     reflect_pi_2,
     b_factor_2,
     kout_2) = get_bragg_reflectivity_per_entry(kin=kout_1,
                                                thickness=channelcut.crystal_list[1].thickness,
                                                crystal_h=h_array_2,
                                                normal=normal_array_2,
                                                chi_dict=channelcut.crystal_list[1].chi_dict)

    return (angles,
            reflect_sigma_1 * reflect_sigma_2,
            reflect_pi_1 * reflect_pi_2,
            b_factor_1 * b_factor_2,
            kout_2)


# -------------------------------------------------------------
#               Alignment
# -------------------------------------------------------------
def align_crystal_reciprocal_lattice(crystal, axis, rot_center=None):
    """

    :param crystal: The crystal to align
    :param axis: The direction along which the reciprocal lattice will be aligned.
    :param rot_center:
    :return:
    """
    if rot_center is None:
        rot_center = crystal.surface_point

    # 1 Get the angle
    cos_val = np.dot(axis, crystal.h) / np.linalg.norm(axis) / np.linalg.norm(crystal.h)
    rot_angle = np.arccos(np.clip(cos_val, -1, 1))

    # print("rot_angle:{:.2e}".format(np.rad2deg(rot_angle)))

    # 2 Try the rotation
    rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, crystal.h)
    # print(new_h)

    if np.dot(new_h, axis) / np.linalg.norm(new_h) / np.linalg.norm(axis) < 0.999:
        # print("aaa")
        rot_mat = util.rot_mat_in_yz_plane(theta=-rot_angle)

    crystal.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=rot_center)


def align_crystal_geometric_bragg_reflection(crystal, kin, rot_direction=1, rot_center=None):
    if rot_center is None:
        rot_center = crystal.surface_point

    ###########################
    #   Align the recirpocal lattice with kin
    ###########################
    align_crystal_reciprocal_lattice(crystal=crystal, axis=kin, rot_center=rot_center)
    # print(crystal.h)

    ###########################
    #   Alignment based on geometric theory of bragg diffraction
    ###########################
    # Estimate the Bragg angle
    bragg_estimation = util.get_bragg_angle(wave_length=two_pi / np.linalg.norm(kin),
                                            plane_distance=two_pi / np.linalg.norm(crystal.h))

    # print("Bragg angle:{:.2e}".format(np.rad2deg(bragg_estimation)))

    # Align the crystal to the estimated Bragg angle
    rot_mat = util.rot_mat_in_yz_plane(theta=(bragg_estimation + np.pi / 2) * rot_direction)

    crystal.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=rot_center)


def align_crystal_dynamical_bragg_reflection(crystal, kin, rot_direction=1,
                                             scan_range=0.0005, scan_number=10000,
                                             rot_center=None,
                                             get_curve=False):
    """
    Align the crystal such that the incident wave vector is at the center of the
    reflectivity curve

    :param crystal:
    :param kin:
    :param rot_direction:
    :param scan_range:
    :param scan_number:
    :param rot_center:
    :param get_curve:
    :return:
    """
    if rot_center is None:
        rot_center = crystal.surface_point

    # Align the crystal with geometric bragg reflection theory
    align_crystal_geometric_bragg_reflection(crystal=crystal,
                                             kin=kin,
                                             rot_direction=rot_direction,
                                             rot_center=rot_center)

    # Align the crystal with dynamical diffraction theory
    (angles,
     reflect_s,
     reflect_p,
     b_array,
     kout_grid) = get_bragg_rocking_curve(kin=kin,
                                          scan_range=scan_range,
                                          scan_number=scan_number,
                                          h_initial=crystal.h,
                                          normal_initial=crystal.normal,
                                          thickness=crystal.thickness,
                                          chi_dict=crystal.chi_dict, )

    # rocking_curve = np.square(np.abs(reflect_s)) / np.abs(b_array)

    # Third: find bandwidth of the rocking curve and the center of the rocking curve
    fwhm, angle_adjust = util.get_fwhm(coordinate=angles,
                                       curve_values=np.square(np.abs(reflect_s)),
                                       center=True)

    # Fourth: Align the crystal along that direction.
    rot_mat = util.rot_mat_in_yz_plane(theta=angle_adjust)
    crystal.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=rot_center)
    if get_curve:
        return angles, np.square(np.abs(reflect_s))


def align_channel_cut_dynamical_bragg_reflection(channelcut,
                                                 kin,
                                                 scan_range=0.0005,
                                                 scan_number=10000,
                                                 rot_center=None,
                                                 get_curve=False):
    """
    Align the crystal such that the incident wave vector is at the center of the
    reflectivity curve

    :param channelcut:
    :param kin:
    :param scan_range:
    :param scan_number:
    :param rot_center:
    :param get_curve:
    :return:
    """
    if rot_center is None:
        rot_center = np.copy(channelcut.crystal_list[0].surface_point)

    # ------------------------------------------------------
    # Align the channel-cut such that the reciprocal lattice is anti-parallel to the kin
    # ------------------------------------------------------
    # 1 Get the angle
    cos_val = (np.dot(kin, channelcut.crystal_list[0].h)
               / np.linalg.norm(kin) / np.linalg.norm(channelcut.crystal_list[0].h))
    rot_angle = np.arccos(np.clip(cos_val, -1.0, 1.0))

    # 2 Try the rotation
    rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, channelcut.crystal_list[0].h)

    if np.dot(new_h, kin) / np.linalg.norm(new_h) / np.linalg.norm(kin) > - 0.999:
        # print("aaa")
        rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    channelcut.rotate_wrt_point(rot_mat=rot_mat,
                                ref_point=rot_center)

    # ------------------------------------------------------
    # Align the channel-cut according to the geometric Bragg angle.
    # The rotation direction is determined by the crystal geometry.
    # ------------------------------------------------------
    # Rotate according to the geometric Bragg angle
    geo_Bragg_angle = util.get_bragg_angle(wave_length=two_pi / np.linalg.norm(kin),
                                           plane_distance=two_pi / np.linalg.norm(channelcut.crystal_list[0].h))
    print("The geometric Bragg angle is {:.2f} deg".format(np.rad2deg(geo_Bragg_angle)))

    # Rotate the channel-cut according to the geometry of the channel-cut crystal
    if channelcut.first_crystal_loc == "lower left":
        rot_mat = util.rot_mat_in_yz_plane(theta=np.pi / 2. - geo_Bragg_angle)
    elif channelcut.first_crystal_loc == "upper left":
        rot_mat = util.rot_mat_in_yz_plane(theta=- np.pi / 2. + geo_Bragg_angle)
    else:
        print("The value of first_crystal_loc of the channel-cut can only be either lower left or uppper left."
              "Please check the value."
              "No rotation is implemented.")
        return 0

    channelcut.rotate_wrt_point(rot_mat=rot_mat,
                                ref_point=np.copy(rot_center),
                                include_boundary=True)

    # ------------------------------------------------------
    # Refine the alignment with dynamical diffraction theory.
    # ------------------------------------------------------
    # Align the crystal with dynamical diffraction theory
    (angles,
     reflect_s,
     reflect_p,
     b_array,
     kout_grid) = get_bragg_rocking_curve_channelcut(kin=kin,
                                                     channelcut=channelcut,
                                                     scan_range=scan_range,
                                                     scan_number=scan_number,
                                                     )

    rocking_curve = np.square(np.abs(reflect_s)) / np.abs(b_array)

    # Third: find bandwidth of the rocking curve and the center of the rocking curve
    fwhm, angle_adjust = util.get_fwhm(coordinate=angles,
                                       curve_values=rocking_curve,
                                       center=True)

    # Fourth: Align the crystal along that direction.
    rot_mat = util.rot_mat_in_yz_plane(theta=angle_adjust)
    channelcut.rotate_wrt_point(rot_mat=rot_mat,
                                ref_point=np.copy(rot_center))
    if get_curve:
        return angles, rocking_curve, b_array


def get_channel_cut_auto_align_rotMat(channelcut,
                                      kin,
                                      rotationAxis,
                                      scan_range=0.0005,
                                      scan_number=10000,
                                      rot_center=None,
                                      get_curve=False):
    """
    Align the crystal such that the incident wave vector is at the center of the
    reflectivity curve

    Assumption: The rotation axis is normal to the diffraction plane
    Assumption: Bragg reflection

    :param channelcut:
    :param kin:
    :param scan_range:
    :param scan_number:
    :param rot_center:
    :param get_curve:
    :return:
    """
    if not (rot_center):
        rot_center = channelcut.crystal_list[0].surface_point

    # Get the rotated kin
    geo_Bragg_angle = util.get_bragg_angle(wave_length=two_pi / np.linalg.norm(kin),
                                           plane_distance=two_pi / np.linalg.norm(channelcut.crystal_list[0].h))

    # Depending on the geometry of the channel-cut crystal, the rotation angle of the kin is different
    if channelcut.first_crystal_loc == "upper left":
        rotMat1 = util.get_rotmat_around_axis(angleRadian=geo_Bragg_angle + np.pi / 2,
                                              axis=rotationAxis)
    elif channelcut.first_crystal_loc == "lower left":
        rotMat1 = util.get_rotmat_around_axis(angleRadian=-geo_Bragg_angle - np.pi / 2,
                                              axis=rotationAxis)
    else:
        print("The value of first_crystal_loc of the channel-cut can only be either lower left or uppper left."
              "Please check the value."
              "No rotation is implemented.")
        return 0

    kin_rot = np.dot(rotMat1, kin)
    kin_rot /= np.linalg.norm(kin_rot)

    # Get the angle between current h and rotated kin
    h_dir = channelcut.crystal_list[0].h / np.linalg.norm(channelcut.crystal_list[0].h)
    rot_dir = np.cross(h_dir, kin_rot)
    # sin_ang = np.linalg.norm(rot_dir)
    cos_ang = np.dot(h_dir, kin_rot)
    if np.abs(np.abs(cos_ang) - 1) < 1e-6:
        rot_angle = np.pi
    else:  # These two are not parallel to each other
        # We can determine the angle from the arccos
        rot_angle = np.arccos(cos_ang)
        # We can determine whether it is clockwise or counter-clockwise
        rot_dir /= np.linalg.norm(rot_dir)

        # Add the rotation direction
        if np.dot(rot_dir, rotationAxis) < 0:
            rot_angle *= -1

    # Check if it satisfies the Bragg condition
    rotMat2 = util.get_rotmat_around_axis(angleRadian=rot_angle,
                                          axis=rotationAxis)
    h1 = np.dot(rotMat2, channelcut.crystal_list[0].h)
    n1 = np.dot(rotMat2, channelcut.crystal_list[0].normal)

    kout_test = kin + h1
    if (np.abs(np.linalg.norm(kout_test) - np.linalg.norm(kin)) / np.linalg.norm(kin) > 1e-6) or (
            np.dot(kout_test, n1) > 0):
        print("Error! The aligned result either does not meet the Bragg condition or is a Laue diffraction.")
        print(h1, n1)
        return 0

    # Rotate the channel-cut in this condition
    # TODO: I have tried not to do this, however, I failed.
    # TODO: In the future, I would like to have a function that do not rotate the crystal.

    channelcut.rotate_wrt_point(rot_mat=rotMat2,
                                ref_point=np.copy(rot_center),
                                include_boundary=True)

    # ------------------------------------------------------
    # Refine the alignment with dynamical diffraction theory.
    # ------------------------------------------------------
    # Align the crystal with dynamical diffraction theory
    (angles,
     reflect_s,
     reflect_p,
     b_array,
     kout_grid) = get_bragg_rocking_curve_channelcut(kin=kin,
                                                     channelcut=channelcut,
                                                     scan_range=scan_range,
                                                     scan_number=scan_number,
                                                     )

    rocking_curve = np.square(np.abs(reflect_s)) / np.abs(b_array)

    # Third: find bandwidth of the rocking curve and the center of the rocking curve
    fwhm, angle_adjust = util.get_fwhm(coordinate=angles,
                                       curve_values=rocking_curve,
                                       center=True)

    # Fourth: Align the crystal along that direction.
    rotMat3 = util.get_rotmat_around_axis(angleRadian=angle_adjust, axis=rotationAxis)

    # TODO: In the future, I would like to have a function that do not rotate the crystal.
    # TODO: In that case, I would not need to rotate the crystal back.
    channelcut.rotate_wrt_point(rot_mat=np.transpose(rotMat2),
                                ref_point=np.copy(rot_center))

    if get_curve:
        return np.matmul(rotMat3, rotMat2), angle_adjust + rot_angle, angles, rocking_curve, b_array
    else:
        return np.matmul(rotMat3, rotMat2)


def align_grating_normal_direction(grating, axis):
    # 1 Get the angle
    cos_val = np.dot(axis, grating.normal) / np.linalg.norm(axis) / np.linalg.norm(grating.normal)
    rot_angle = np.arccos(cos_val)

    # 2 Try the rotation
    rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, grating.normal)

    if np.dot(new_h, axis) < 0:
        rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    grating.rotate_wrt_point(rot_mat=rot_mat,
                             ref_point=grating.surface_point)


def align_telescope_optical_axis(telescope, axis):
    # 1 Get the angle
    cos_val = np.dot(axis, telescope.lens_axis) / np.linalg.norm(axis) / np.linalg.norm(telescope.lens_axis)
    rot_angle = np.arccos(cos_val)

    # 2 Try the rotation
    rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle)
    new_h = np.dot(rot_mat, telescope.lens_axis)

    if np.dot(new_h, axis) < 0:
        rot_mat = util.rot_mat_in_yz_plane(theta=rot_angle + np.pi)

    telescope.rotate_wrt_point(rot_mat=rot_mat,
                               ref_point=telescope.lens_point)


# --------------------------------------------------------------------------------------------------------------
#       Wrapper functions for different devices
# --------------------------------------------------------------------------------------------------------------
def get_kout_single_device(device, kin):
    """
    Get the output wave vector given the incident wave vector

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        kout = util.get_bragg_kout(kin=kin,
                                   h=device.h,
                                   normal=device.normal)
        return kout

    if device.type == "Transmissive Grating":
        kout = kin + device.momentum_transfer
        return kout

    if device.type == "Transmission Telescope for CPA":
        kout = util.get_telescope_kout(optical_axis=device.lens_axis,
                                       kin=kin)
        return kout


####################################################
#       Single incident wave vector
####################################################
def get_kout_multi_device(device_list, kin):
    """
    Get the output momentum vectors from each device.

    :param device_list:
    :param kin:
    :return:
    """

    # Create a variable for the kout list.
    # The reason to use is numpy array is that it's easy to determine the
    # total number of kouts generates and with numpy array, it might be more
    # efficient.
    kout_list = np.zeros((len(device_list) + 1, 3), dtype=np.float64)
    kout_list[0] = kin[:]

    for idx in range(len(device_list)):
        # Get the device
        device = device_list[idx]

        # Get the output wave vector
        kout_list[idx + 1] = get_kout_single_device(device=device,
                                                    kin=kout_list[idx])

    return kout_list


def get_lightpath(device_list, kin, initial_point, final_plane_point, final_plane_normal):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param final_plane_normal:
    :param final_plane_point:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Path length
    path_length = 0.

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

        if device.type == "Transmission Telescope for CPA":
            intersection_list.append(util.get_image_from_telescope_for_cpa(object_point=intersection_list[-1],
                                                                           lens_axis=device.lens_axis,
                                                                           lens_position=device.lens_position,
                                                                           focal_length=device.focal_length))
            # Find the path length
            # displacement = intersection_list[-1] - intersection_list[-2]
            # path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the output wave vector
            kout_list.append(util.get_telescope_kout(optical_axis=device.lens_axis,
                                                     kin=kout_list[-1]))

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                   k=kout_list[-1],
                                                   surface_point=final_plane_point,
                                                   normal=final_plane_normal))
    # Update the path length
    displacement = intersection_list[-1] - intersection_list[-2]
    path_length += np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

    return intersection_list, kout_list, path_length


def get_trajectory(device_list, kin, initial_point, path_length):
    """
    This function is used to generate the light path of the incident wave vector in the series of
    devices.

    This function correctly handles the light path through the telescopes.

    :param device_list:
    :param kin:
    :param initial_point:
    :param path_length:
    :return:
    """

    # Create a holder for kout vectors
    kout_list = [np.copy(kin), ]

    # Create a list for the intersection points
    intersection_list = [np.copy(initial_point)]

    # Loop through all the devices.
    for idx in range(len(device_list)):

        ###############################################################
        # Step 1: Get the device
        device = device_list[idx]

        ###############################################################
        # Step 2: Find the intersection and kout
        if device.type == "Crystal: Bragg Reflection":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length -= np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the output k vector
            kout_list.append(util.get_bragg_kout(kin=kout_list[-1],
                                                 h=device.h,
                                                 normal=device.normal))

        if device.type == "Transmissive Grating":
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.normal,
                                                           surface_point=device.surface_point))
            # Find the path length
            displacement = intersection_list[-1] - intersection_list[-2]
            path_length -= np.dot(displacement, kout_list[-1]) / util.np.linalg.norm(kout_list[-1])

            # Find the wave vecotr
            kout_list.append(kout_list[-1] + device.momentum_transfer)

        if device.type == "Transmission Telescope for CPA":
            # Find the intersection with the first lens
            intersection_list.append(util.get_intersection(initial_position=intersection_list[-1],
                                                           k=kout_list[-1],
                                                           normal=device.lens_axis,
                                                           surface_point=device.lens_position))

            # Find the image
            image = util.get_image_from_telescope_for_cpa(object_point=intersection_list[-1],
                                                          lens_axis=device.lens_axis,
                                                          lens_position=device.lens_position,
                                                          focal_length=device.focal_length)

            # Find the kout
            kout_list.append(util.get_telescope_kout(optical_axis=device.lens_axis,
                                                     kin=kout_list[-1]))

            # Find the intersection on the second lens
            point_on_seond_lens = device.lens_position + 2 * device.focal_length * device.lens_axis
            intersection_list.append(util.get_intersection(initial_position=image,
                                                           k=kout_list[-1],
                                                           normal=device.lens_axis,
                                                           surface_point=point_on_seond_lens))

    ################################################################
    # Step 3: Find the output position on the observation plane
    intersection_list.append(intersection_list[-1] + kout_list[-1] / util.np.linalg.norm(kout_list[-1]) * path_length)

    return intersection_list, kout_list


####################################################################################
#    Get intensity efficiency
####################################################################################

def get_intensity_efficiency_sigma_polarization_single_device(device, kin):
    """
    Get the output intensity efficiency for the given wave vector
    assuming a monochromatic plane incident wave.

    :param device:
    :param kin:
    :return:
    """
    # Get output wave vector
    if device.type == "Crystal: Bragg Reflection":
        tmp = np.zeros((1, 3))
        tmp[0, :] = kin

        (reflect_s,
         reflect_p,
         b,
         kout_grid) = util.get_bragg_reflection_array(kin_grid=tmp,
                                                      d=device.thickness,
                                                      h=device.h,
                                                      n=device.normal,
                                                      chi0=device.chi0,
                                                      chih_sigma=device.chih_sigma,
                                                      chihbar_sigma=device.chihbar_sigma,
                                                      chih_pi=device.chih_pi,
                                                      chihbar_pi=device.chihbar_pi)

        efficiency = np.square(np.abs(reflect_s)) / np.abs(b)
        return efficiency

    if device.type == "Transmissive Grating":

        # Determine the grating order
        if device.order == 0:
            efficiency = util.get_square_grating_0th_transmission(kin=kin,
                                                                  height_vec=device.h,
                                                                  refractive_index=device.n,
                                                                  ab_ratio=device.ab_ratio,
                                                                  base=device.thick_vec)
        else:
            efficiency, _, _ = util.get_square_grating_transmission(kin=kin,
                                                                    height_vec=device.h,
                                                                    ab_ratio=device.ab_ratio,
                                                                    base=device.thick_vec,
                                                                    refractive_index=device.n,
                                                                    order=device.order,
                                                                    grating_k=device.momentum_transfer)
        # Translate to the intensity efficiency
        return np.square(np.abs(efficiency))

    if device.type == "Transmission Telescope for CPA":
        return np.square(np.abs(device.efficiency))


def get_intensity_efficiency_sigma_polarization(device_list, kin):
    """
    Get the reflectivity of this kin.
    Notice that this function is not particularly useful.
    It just aims to make the function lists complete.

    :param device_list:
    :param kin:
    :return:
    """
    efficiency_list = np.zeros(len(device_list), dtype=np.float64)

    # Variable for the kout
    kout_list = np.zeros((len(device_list) + 1, 3), dtype=np.float64)
    kout_list[0] = kin[:]

    # Loop through all the devices
    for idx in range(len(device_list)):
        # Get the device
        device = device_list[idx]

        # Get the efficiency
        efficiency_list[idx] = get_intensity_efficiency_sigma_polarization_single_device(device=device,
                                                                                         kin=kout_list[idx])
        # Get the output wave vector
        kout_list[idx + 1] = get_kout_single_device(device=device, kin=kout_list[idx])

    # Get the overall efficiency
    total_efficiency = np.prod(efficiency_list)

    return total_efficiency, efficiency_list, kout_list


def get_output_efficiency_curve(device_list, kin_list):
    """
    Get the reflectivity for each kin.

    :param kin_list:
    :param device_list:
    :return:
    """
    d_num = len(device_list)  # number of devices
    k_num = kin_list.shape[0]  # number of kin vectors

    efficiency_holder = np.zeros((k_num, d_num))
    kout_holder = np.zeros((k_num, d_num + 1, 3))
    total_efficiency_holder = np.zeros(k_num)

    # Loop through all the kin
    for idx in range(k_num):
        (total_efficiency,
         efficiency_tmp,
         kout_tmp) = get_intensity_efficiency_sigma_polarization(device_list=device_list,
                                                                 kin=kin_list[idx])

        efficiency_holder[idx, :] = efficiency_tmp[:]
        kout_holder[idx, :, :] = kout_tmp[:, :]
        total_efficiency_holder[idx] = total_efficiency

    return total_efficiency_holder, efficiency_holder, kout_holder


######################################################
#     Get device list suitable for simulation
######################################################
def get_device_list_for_simulation():
    """
    Because I have several different crystal classes, not all of them can be understood by
    the simulation function, I add a conversion function here so that users can
    convert the crystals into devices that the program can simulate.
    :return:
    """
    # TODO: May ask Selene to complete this.
    pass


######################################################
#    Get propagation in free space
######################################################
def add_propagate_phase(kx, ky, kz, distance, spectrum):
    """

    :param kx:
    :param ky:
    :param kz:
    :param distance:
    :param spectrum:
    :return:
    """
    # get the wave-vector mesh
    # kx = two_pi / dx * np.linspace(start=-0.5, stop=0.5, num=nx)
    # ky = two_pi / dy * np.linspace(start=-0.5, stop=0.5, num=ny)
    # kz = two_pi / dz * np.linspace(start=-0.5, stop=0.5, num=nz)

    nx = kx.shape[0]
    ny = ky.shape[0]
    nz = kz.shape[0]

    # Get time
    t = distance / util.c

    # Get frequency
    omega = np.zeros((nx, ny, nz))
    omega += np.square(kx[:, np.newaxis, np.newaxis])
    omega += np.square(ky[np.newaxis, :, np.newaxis])
    omega += np.square(kz[np.newaxis, np.newaxis, :])
    omega = np.sqrt(omega) * util.c

    # get phase, to save memory, I'll just use omega
    omega *= t
    omega -= kz[np.newaxis, np.newaxis, :] * distance

    # Get the phase
    np.multiply(spectrum,
                np.exp(1.j * omega),
                out=spectrum,
                dtype=np.complex128)


def add_lens_transmission_function(x, y, kz, fx, fy, xy_kz_field, n=complex(1.0, 0)):
    """

    :param x:
    :param y:
    :param kz:
    :param fx:
    :param fy:
    :param xy_kz_field:
    :param n:
    :return:
    """
    phaseX = np.exp(complex(- n.imag / (1. - n.real), -1) * np.outer(np.square(x), kz / 2. / fx))
    phaseY = np.exp(complex(- n.imag / (1. - n.real), -1) * np.outer(np.square(y), kz / 2. / fy))

    # Add the transmission function to the electric field along each direction
    np.multiply(xy_kz_field, phaseX[:, np.newaxis, :], out=xy_kz_field)
    np.multiply(xy_kz_field, phaseY[np.newaxis, :, :], out=xy_kz_field)


def get_flat_wavevector_array(kx, ky, kz):
    nx = kx.shape[0]
    ny = ky.shape[0]
    nz = kz.shape[0]

    kVecArray = np.zeros((nx, ny, nz, 3))
    kVecArray[:, :, :, 0] = kx[:, np.newaxis, np.newaxis]
    kVecArray[:, :, :, 1] = ky[np.newaxis, :, np.newaxis]
    kVecArray[:, :, :, 2] = kz[np.newaxis, np.newaxis, :]

    return np.reshape(kVecArray, (nx * ny * nz, 3))
