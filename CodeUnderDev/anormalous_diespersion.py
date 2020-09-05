
def get_universal_anomalous_dispersion_curve_j(omega, omega_q, p):
    """
    This function get the Jq function value defined in paper
    "Anomalous Dispersion and Scattering of X-Rays" PhysRev.94.1593

    This function only find the real part of the Jq functions and have ignored
    all damping effects. It is unclear whether the damping effects will
    be included in future versions or not.

    :param p: The exponential value for the absorption. For the defition, see the paper.
    :param omega: Angular frequency of the x-ray
    :param omega_q: Angular frequency of the absorption edge.
    :return:
    """
    z = (omega_q / omega) ** 2
    xi = 0

    if np.abs(p - 7. / 3.) < 1e-6:
        return _get_re_jqm1_7_3(xi=xi, z=z)

    if np.abs(p - 2.5) < 1e-6:
        return _get_re_jqm1_5_2(xi=xi, z=z)

    if np.abs(p - 2.75) < 1e-6:
        return _get_re_jqm1_11_4(xi=xi, z=z)

    raise Exception("Values for p other than 7/3, 5/2, 11/4 are not implemented yet.")


def _get_re_jqm1_7_3(xi, z):
    """
    Get the value of the function (22) and (23) defined in paper
    "Anomalous Dispersion and Scattering of X-Rays" PhysRev.94.1593

    Notice that in this function, factors in xi larger than the second order were ignored.

    :param xi:
    :param z:
    :return:
    """
    theta = np.arctan(xi)
    a = np.power(z, 1. / 3.)

    if np.abs(z) < 1.:
        factor_1 = - 2. * (a ** 2) / 3.

        factor_2_1 = 0.5 * np.log((1. + a + a ** 2) /
                                  (1. - 2. * a * np.cos(theta / 3.) + a ** 2))
        factor_2_2 = np.sqrt(3.) * np.arctan(np.sqrt(3.) * a / (2. + a))
        factor_2_3 = - np.pi / np.sqrt(3) * (1. - 5. * xi / np.sqrt(3))
        factor_2_4 = - 5. * xi / 3. * np.arctan(a * np.sin(theta / 3.) / (1. - a * np.cos(theta / 3.)))

        return factor_1 * (factor_2_1 + factor_2_2 + factor_2_3 + factor_2_4)

    else:
        b = 1. / a

        factor_1 = -2. / (3. * (b ** 2))

        factor_2_1 = 0.5 * np.log((1. + b + b ** 2) / (1. - 2. * b * np.cos(theta / 3.) + b ** 2))
        factor_2_2 = -np.sqrt(3.) * np.arctan(np.sqrt(3.) * b / (2. + b))
        factor_2_3 = 5. * xi / 3. * np.arctan(b * np.sin(theta / 3.) / (1 - b * np.cos(theta / 3.)))

        return factor_1 * (factor_2_1 + factor_2_2 + factor_2_3)


def _get_re_jqm1_5_2(xi, z):
    """
    Get the value of the function (24) and (25) defined in paper
    "Anomalous Dispersion and Scattering of X-Rays" PhysRev.94.1593

    Notice that in this function, factors in xi larger than the second order were ignored.

    :param xi:
    :param z:
    :return:
    """
    theta = np.arctan(xi)
    a = np.power(z, 0.25)

    if np.abs(z) < 1.:
        factor_1 = - 0.75 * (a ** 3)

        factor_2_1 = 0.5 * np.log(((1. + a) ** 2) / (1. - 2. * a * np.cos(theta / 4.) + a ** 2))
        factor_2_2 = 2. * np.arctan(a)
        factor_2_3 = - np.pi * (1 - 7 * xi / 4.)
        factor_2_4 = -7. * xi / 4. * np.arctan(a * np.sin(theta / 4.) / (1 - a * np.cos(theta / 4.)))

        return factor_1 * (factor_2_1 + factor_2_2 + factor_2_3 + factor_2_4)
    else:
        b = 1 / a

        factor_1 = -0.75 / (b ** 3)

        factor_2_1 = 0.5 * np.log(((1 + b) ** 2) / (1 - 2 * b * np.cos(theta / 4.) + b ** 2))
        factor_2_2 = -2. * np.arctan(b)
        factor_2_3 = 7. * xi / 4. * np.arctan(b * np.sin(theta / 4.) / (1 - b * np.cos(theta / 4.)))

        return factor_1 * (factor_2_1 + factor_2_2 + factor_2_3)


def _get_re_jqm1_11_4(xi, z):
    """
    Get the value of the function (26) and (27) defined in paper
    "Anomalous Dispersion and Scattering of X-Rays" PhysRev.94.1593

    Notice that in this function, factors in xi larger than the second order were ignored.

    :param xi:
    :param z:
    :return:
    """
    theta = np.arctan(xi)
    a = np.power(z, 0.125)

    if np.abs(z) < 1.:
        factor_1 = -0.875 * (a ** 7)

        factor_2_1 = 0, 5 * np.log(((1 + a) ** 2) / (1 - 2 * a * np.cos(theta / 8.) + a ** 2))
        factor_2_2 = np.log((1 + np.sqrt(2) * a + a ** 2) / (1 - np.sqrt(2) * a + a ** 2)) / np.sqrt(2)
        factor_2_3 = 2 * np.arctan(a)
        factor_2_4 = np.sqrt(2) * np.arctan(np.sqrt(2) * a / (1 - a ** 2))
        factor_2_5 = -np.pi * cot_pi_8 * (1 - 1.875 * xi / cot_pi_8)
        factor_2_6 = -1.875 * xi * np.arctan(a * np.sin(theta / 8.) / (1 - a * np.cos(theta / 8.)))

        return factor_1 * (factor_2_1 + factor_2_2 + factor_2_3 + factor_2_4 + factor_2_5 + factor_2_6)

    else:
        b = 1. / a

        factor_1 = -0.875 / (b ** 7)

        factor_2_1 = 0.5 * np.log(((1 + b) ** 2) / (1 - 2 * b * np.cos(theta / 8.) + b ** 2))
        factor_2_2 = np.log((1 + np.sqrt(2) * b + b ** 2) / (1 - np.sqrt(2) * b + b ** 2)) / np.sqrt(2)
        factor_2_3 = - 2 * np.arctan(b)
        factor_2_4 = - np.sqrt(2) * np.arctan(np.sqrt(2) * b / (1 - b ** 2))
        factor_2_5 = 1.875 * xi * np.arctan(b * np.sin(theta / 8.) / (1 - b * np.cos(theta / 8.)))

        return factor_1 * (factor_2_1 + factor_2_2 + factor_2_3 + factor_2_4 + factor_2_5)
