import cmath
import math

import numpy as np
from numba import cuda

"""
                Format requirement
    For all the cuda functions, here is the format of the argument
    1. All the functions have void output
    2. The first arguments are the variables holding the output
    3. Following the output variables are the input variables. For the input variables, 
        1. First specify the input k vector or k length
        2. Then specify the k components 
        3. Then specify the crystal related properties
    4. The last argument is the number of scalars or vectors to calculate 
    
"""

c = 299792458. * 1e-9  # The speed of light in um / fs
two_pi = 2 * math.pi
eps = np.finfo(np.float64).eps


###################################################################################################
#         Variable Initialization 1 Dimension
###################################################################################################
@cuda.jit("void(float64[:,:], float64[:], float64[:], float64[:], float64, float64, float64, int64)")
def init_kvec(kvec_grid, klen_grid, kz_grid, kz_square_grid, kx, ky, square_kxy, num):
    """

    :param kvec_grid: The grid of wave vectors
    :param klen_grid: The grid of the length of the wave vectors
    :param kz_grid:
    :param kz_square_grid:
    :param kx:
    :param ky:
    :param square_kxy:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        kvec_grid[idx, 0] = kx
        kvec_grid[idx, 1] = ky
        kvec_grid[idx, 2] = kz_grid[idx]

        klen_grid[idx] = math.sqrt(kz_square_grid[idx] + square_kxy)


@cuda.jit("void(float64[:,:], float64[:], float64, float64, float64, int64)")
def init_kvec_dumond(kvec_grid, klen_grid, x_coef, y_coef, z_coef, num):
    """

    :param kvec_grid:
    :param klen_grid:
    :param x_coef:
    :param y_coef:
    :param z_coef:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        kvec_grid[idx, 0] = klen_grid[idx] * x_coef
        kvec_grid[idx, 1] = klen_grid[idx] * y_coef
        kvec_grid[idx, 2] = klen_grid[idx] * z_coef


@cuda.jit("void(complex128[:], int64)")
def init_jacobian(jacobian, num):
    """

    :param jacobian:
    :param num:
    :return:
    """

    idx = cuda.grid(1)
    if idx < num:
        jacobian[idx] = complex(1.)


@cuda.jit("void(complex128[:], int64)")
def init_phase(phase, num):
    idx = cuda.grid(1)
    if idx < num:
        phase[idx] = complex(1.)


@cuda.jit("void(float64[:], float64, int64)")
def init_scalar_grid(scalar_grid, scalar, num):
    idx = cuda.grid(1)
    if idx < num:
        scalar_grid[idx] = scalar


@cuda.jit("void(float64[:,:], float64[:], int64, int64)")
def init_vector_grid(vector_grid, vector, vec_size, num):
    idx = cuda.grid(1)
    if idx < num:
        for vec_idx in range(vec_size):
            vector_grid[idx, vec_idx] = vector[vec_idx]


###################################################################################################
#          Elementwise operation
###################################################################################################
@cuda.jit("void(complex128[:], complex128[:], complex128[:], int64)")
def scalar_scalar_multiply_complex(a, b, out, num):
    idx = cuda.grid(1)
    if idx < num:
        out[idx] = a[idx] * b[idx]


@cuda.jit("void(complex128[:], complex128[:], complex128[:,:], int64)")
def scalar_vector_multiply_complex(scalar_grid, vec, vec_grid, num):
    """

    :param scalar_grid:
    :param vec:
    :param vec_grid:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        vec_grid[idx, 0] = scalar_grid[idx] * vec[0]
        vec_grid[idx, 1] = scalar_grid[idx] * vec[1]
        vec_grid[idx, 2] = scalar_grid[idx] * vec[2]


@cuda.jit("void(complex128[:], complex128[:,:], complex128[:,:], int64)")
def scalar_vector_elementwise_multiply_complex(scalar_grid, vec, vec_grid, num):
    """

    :param scalar_grid:
    :param vec:
    :param vec_grid:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        vec_grid[idx, 0] = scalar_grid[idx] * vec[idx, 0]
        vec_grid[idx, 1] = scalar_grid[idx] * vec[idx, 1]
        vec_grid[idx, 2] = scalar_grid[idx] * vec[idx, 2]


@cuda.jit("void(float64[:], complex128[:,:], complex128[:,:], int64)")
def add_phase_to_vector_spectrum(phase_real, vec, vec_grid, num):
    """

    :param phase_real:
    :param vec:
    :param vec_grid:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        phase = complex(math.cos(phase_real[idx]), math.sin(phase_real[idx]))
        vec_grid[idx, 0] = phase * vec[idx, 0]
        vec_grid[idx, 1] = phase * vec[idx, 1]
        vec_grid[idx, 2] = phase * vec[idx, 2]


@cuda.jit("void(float64[:], complex128[:], int64)")
def add_phase_to_scalar_spectrum(phase_real, scalar_grid, num):
    """

    :param phase_real:
    :param scalar_grid:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        phase = complex(math.cos(phase_real[idx]), math.sin(phase_real[idx]))
        scalar_grid[idx] = phase * scalar_grid[idx]


@cuda.jit("void(complex128[:,:], complex128[:], complex128[:], complex128[:], int64)")
def vector_expansion(vector, x, y, z, num):
    """

    :param vector:
    :param x:
    :param y:
    :param z:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        x[idx] = vector[idx, 0]
        y[idx] = vector[idx, 1]
        z[idx] = vector[idx, 2]


@cuda.jit('void'
          '(float64[:,:], float64[:,:], float64[:], int64)')
def add_vector(vec_out_grid, vec_in_grid, delta_vec, num):
    """

    :param vec_out_grid:
    :param delta_vec:
    :param vec_in_grid:
    :param num:
    :return:
    """
    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:
        vec_out_grid[idx, 0] = vec_in_grid[idx, 0] + delta_vec[0]
        vec_out_grid[idx, 1] = vec_in_grid[idx, 1] + delta_vec[1]
        vec_out_grid[idx, 2] = vec_in_grid[idx, 2] + delta_vec[2]


@cuda.jit('void'
          '(float64[:], float64[:,:], int64, int64)')
def get_vector_length(vec_len_grid, vec_grid, vec_dim, num):
    """

    :param vec_len_grid:
    :param vec_grid:
    :param vec_dim:
    :param num:
    :return:
    """
    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:
        vec_len_grid[idx] = 0.
        for x in range(vec_dim):
            vec_len_grid[idx] += vec_grid[idx, x] ** 2
        vec_len_grid[idx] = math.sqrt(vec_len_grid[idx])


###################################################################################################
#          Data transfer
###################################################################################################
@cuda.jit("void(complex128[:,:], complex128[:], int64, int64, int64, int64, int64)")
def fill_column_complex_fftshift(holder, source, row_idx, idx_start1, num1, idx_start2, num2):
    """

    :param holder:
    :param source:
    :param row_idx:
    :param idx_start1:
    :param num1:
    :param idx_start2:
    :param num2:
    :return:
    """
    col_idx = cuda.grid(1)
    if col_idx < num1:
        holder[row_idx, col_idx] = source[idx_start1 + col_idx]
    if col_idx < num2:
        holder[row_idx, num1 + col_idx] = source[idx_start2 + col_idx]


@cuda.jit("void(complex128[:,:], complex128[:], int64, int64, int64)")
def fill_column_complex(holder, source, row_idx, idx_start, num):
    """
    This is a auxiliary function.

    Holder is the final function I would like to use to save the
    spatial-temporal function. I would like to fill in this variable
    with values from the source which would be a 1d response function.

    :param source: The source response function from which the data will be copied
    :param holder: The holder array to store the copied data
    :param idx_start: The first idx from which the data will be copied
    :param row_idx: The row idx of the holder variable to store the information
    :param num: The total number of data to store for each line.
    :return:
    """
    column = cuda.grid(1)
    if column < num:
        holder[row_idx, column] = source[column + idx_start]


@cuda.jit("void(float64[:,:], float64[:], int64, int64, int64)")
def fill_column_float(holder, source, row_idx, idx_start, num):
    """
    This is a auxiliary function.

    Holder is the final function I would like to use to save the
    spatial-temporal function. I would like to fill in this variable
    with values from the source which would be a 1d response function.

    :param source: The source response function from which the data will be copied
    :param holder: The holder array to store the copied data
    :param idx_start: The first idx from which the data will be copied
    :param row_idx: The row idx of the holder variable to store the information
    :param num: The total number of data to store for each line.
    :return:
    """
    column = cuda.grid(1)
    if column < num:
        holder[row_idx, column] = source[column + idx_start]


###################################################################################################
#         Forward propagation
###################################################################################################
@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128, complex128, complex128,'
          'int64)')
def get_bragg_reflection(reflectivity_sigma, reflectivity_pi, kout_grid, efield_grid,
                         klen_grid, kin_grid,
                         d, h, n,
                         dot_hn, h_square,
                         chi0, chih_sigma, chihbar_sigma, chih_pi, chihbar_pi,
                         num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # tmp_pos = abs(-gamma_h + sqrt_gamma_alpha)
        # tmp_neg = abs(-gamma_h - sqrt_gamma_alpha)
        # if tmp_pos > tmp_neg:
        #    m_trans = klen * (-gamma_h - sqrt_gamma_alpha)
        # else:
        #    m_trans = klen * (-gamma_h + sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # ----------------------------------------
        #     Get polarization component
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x) +
                        efield_grid[idx, 1] * complex(sigma_in_y) +
                        efield_grid[idx, 2] * complex(sigma_in_z))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x) +
                     efield_grid[idx, 1] * complex(pi_in_y) +
                     efield_grid[idx, 2] * complex(pi_in_z))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        # if im <= 200.:
        magnitude = complex(math.exp(-im))

        phase = complex(math.cos(re), math.sin(re))
        # Calculate some intermediate part
        numerator = 1. - magnitude * phase
        denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

        # Assemble everything
        reflectivity_sigma[idx] = b_complex * chih_sigma * numerator / denominator

        # else:
        #    # When the crystal is super thick, the numerator becomes 1 The exponential term becomes 0.
        #    # Calculate some intermediate part
        #    denominator = alpha_tidle + sqrt_a2_b2
        #
        #    # Assemble everything
        #    reflectivity_sigma[idx] = b_complex * chih_sigma / denominator

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex(1.)  # complex((kout_x * kin_x +
        # kout_y * kin_y +
        # kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 300.:
            magnitude = complex(math.exp(-im))
            phase = complex(math.cos(re), math.sin(re))

            # Calculate some intermediate part
            numerator = complex(1.) - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2.) - numerator)
            # Assemble everything
            reflectivity_pi[idx] = bp * chih_pi * numerator / denominator

        else:
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2
            # Assemble everything
            reflectivity_pi[idx] = bp * chih_pi / denominator

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z


@cuda.jit('void(complex128[:], complex128[:], float64[:,:], complex128[:,:], complex128[:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128, complex128, complex128,'
          'int64)')
def get_bragg_reflection_with_jacobian(reflectivity_sigma, reflectivity_pi, kout_grid, efield_grid, jacobian,
                                       klen_grid, kin_grid,
                                       d, h, n,
                                       dot_hn, h_square,
                                       chi0, chih_sigma, chihbar_sigma, chih_pi, chihbar_pi,
                                       num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param jacobian:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # tmp_pos = abs(-gamma_h + sqrt_gamma_alpha)
        # tmp_neg = abs(-gamma_h - sqrt_gamma_alpha)
        # if tmp_pos > tmp_neg:
        #    m_trans = klen * (-gamma_h - sqrt_gamma_alpha)
        # else:
        #    m_trans = klen * (-gamma_h + sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # Get the jacobian :   dot(kout, n) / dot(kin, n)
        jacobian[idx] *= complex(math.fabs((dot_kn + dot_hn + m_trans) / dot_kn))

        # ----------------------------------------
        #     Get polarization component
        # ----------------------------------------
        """
        sigma_in = kH cross k0     sigma_out = sigma_in  
        pi_in = k0 cross sigma0    pi_out = k_out cross sigma_out        
        """
        sigma_in_x = kout_y * kin_z - kout_z * kin_y
        sigma_in_y = kout_z * kin_x - kout_x * kin_z
        sigma_in_z = kout_x * kin_y - kout_y * kin_x
        tmp_len = math.sqrt(sigma_in_x ** 2 + sigma_in_y ** 2 + sigma_in_z ** 2)
        sigma_in_x /= tmp_len
        sigma_in_y /= tmp_len
        sigma_in_z /= tmp_len

        pi_in_x = kin_y * sigma_in_z - kin_z * sigma_in_y
        pi_in_y = kin_z * sigma_in_x - kin_x * sigma_in_z
        pi_in_z = kin_x * sigma_in_y - kin_y * sigma_in_x
        tmp_len = math.sqrt(pi_in_x ** 2 + pi_in_y ** 2 + pi_in_z ** 2)
        pi_in_x /= tmp_len
        pi_in_y /= tmp_len
        pi_in_z /= tmp_len

        pi_out_x = kout_y * sigma_in_z - kout_z * sigma_in_y
        pi_out_y = kout_z * sigma_in_x - kout_x * sigma_in_z
        pi_out_z = kout_x * sigma_in_y - kout_y * sigma_in_x
        tmp_len = math.sqrt(pi_out_x ** 2 + pi_out_y ** 2 + pi_out_z ** 2)
        pi_out_x /= tmp_len
        pi_out_y /= tmp_len
        pi_out_z /= tmp_len

        # Decompose the input electric field
        efield_sigma = (efield_grid[idx, 0] * complex(sigma_in_x) +
                        efield_grid[idx, 1] * complex(sigma_in_y) +
                        efield_grid[idx, 2] * complex(sigma_in_z))
        efield_pi = (efield_grid[idx, 0] * complex(pi_in_x) +
                     efield_grid[idx, 1] * complex(pi_in_y) +
                     efield_grid[idx, 2] * complex(pi_in_z))

        #####################################################################################################
        # Step 2: Get the reflectivity for input sigma polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        # if im <= 200.:
        magnitude = complex(math.exp(-im))

        phase = complex(math.cos(re), math.sin(re))
        # Calculate some intermediate part
        numerator = 1. - magnitude * phase
        denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

        # Assemble everything
        reflectivity_sigma[idx] = b_complex * chih_sigma * numerator / denominator

        # else:
        #    # When the crystal is super thick, the numerator becomes 1 The exponential term becomes 0.
        #    # Calculate some intermediate part
        #    denominator = alpha_tidle + sqrt_a2_b2
        #
        #    # Assemble everything
        #    reflectivity_sigma[idx] = b_complex * chih_sigma / denominator

        # Get the output electric field due to this component
        efield_out_sigma_x = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_x)
        efield_out_sigma_y = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_y)
        efield_out_sigma_z = reflectivity_sigma[idx] * efield_sigma * complex(sigma_in_z)

        #####################################################################################################
        # Step 3: Get the reflectivity for input pi polarization
        #####################################################################################################
        # ----------------------------------------
        #    Get reflectivity
        # ----------------------------------------

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex((kout_x * kin_x +
                           kout_y * kin_y +
                           kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 300.:
            magnitude = complex(math.exp(-im))
            phase = complex(math.cos(re), math.sin(re))

            # Calculate some intermediate part
            numerator = complex(1.) - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2.) - numerator)
            # Assemble everything
            reflectivity_pi[idx] = bp * chih_pi * numerator / denominator

        else:
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2
            # Assemble everything
            reflectivity_pi[idx] = bp * chih_pi / denominator

        # Get the output electric field due to this component
        efield_out_pi_x = reflectivity_pi[idx] * efield_pi * complex(pi_out_x)
        efield_out_pi_y = reflectivity_pi[idx] * efield_pi * complex(pi_out_y)
        efield_out_pi_z = reflectivity_pi[idx] * efield_pi * complex(pi_out_z)

        #####################################################################################################
        # Step 4: Assemble to get the output electric field
        #####################################################################################################
        efield_grid[idx, 0] = efield_out_sigma_x + efield_out_pi_x
        efield_grid[idx, 1] = efield_out_sigma_y + efield_out_pi_y
        efield_grid[idx, 2] = efield_out_sigma_z + efield_out_pi_z


@cuda.jit('void(complex128[:], float64[:,:], complex128[:], complex128[:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128, complex128,'
          'int64)')
def get_bragg_reflection_sigma_polarization(reflectivity_sigma,
                                            kout_grid,
                                            efield_grid,
                                            jacobian,
                                            klen_grid,
                                            kin_grid,
                                            d,
                                            h,
                                            n,
                                            dot_hn,
                                            h_square,
                                            chi0, chih_sigma,
                                            chihbar_sigma, num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param kout_grid:
    :param efield_grid:
    :param jacobian:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # Get the jacobian :   dot(kout, n) / dot(kin, n)
        jacobian[idx] *= complex(math.fabs((dot_kn + dot_hn + m_trans) / dot_kn))

        #####################################################################################################
        # Step 2: Get the reflectivity and field
        #####################################################################################################
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 400.:
            magnitude = complex(math.exp(-im))

            phase = complex(math.cos(re), math.sin(re))
            # Calculate some intermediate part
            numerator = 1. - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

            # Assemble everything
            reflectivity_sigma[idx] = b_complex * chih_sigma * numerator / denominator
        else:
            # When the crystal is super thick, the numerator becomes 1 The exponential term becomes 0.
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2

            # Assemble everything
            reflectivity_sigma[idx] = b_complex * chih_sigma / denominator

        # Get the field
        efield_grid[idx] *= reflectivity_sigma[idx]


@cuda.jit('void(complex128[:], float64[:], float64[:,:], complex128[:], complex128[:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:], float64,'
          'float64, float64,'
          'complex128, complex128, complex128,'
          'int64)')
def get_bragg_reflection_sigma_full(reflectivity_sigma,
                                    phase_grid,
                                    kout_grid,
                                    efield_grid,
                                    jacobian,
                                    klen_grid,
                                    kin_grid,
                                    d,
                                    h,
                                    n,
                                    dot_sn,
                                    dot_hn,
                                    h_square,
                                    chi0, chih_sigma,
                                    chihbar_sigma, num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param kout_grid:
    :param efield_grid:
    :param jacobian:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_sn:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # Get the jacobian :   dot(kout, n) / dot(kin, n)
        jacobian[idx] *= complex(math.fabs((dot_kn + dot_hn + m_trans) / dot_kn))

        #####################################################################################################
        # Step 2: Get the reflectivity and field
        #####################################################################################################
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + b_complex * chih_sigma * chihbar_sigma)

        if sqrt_a2_b2.imag < 0:
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 400.:
            magnitude = complex(math.exp(-im))

            phase = complex(math.cos(re), math.sin(re))
            # Calculate some intermediate part
            numerator = 1. - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (2. - numerator)

            # Assemble everything
            reflectivity_sigma[idx] = b_complex * chih_sigma * numerator / denominator
        else:
            # When the crystal is super thick, the numerator becomes 1 The exponential term becomes 0.
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2

            # Assemble everything
            reflectivity_sigma[idx] = b_complex * chih_sigma / denominator

        # Get the phase term:
        phase_grid[idx] -= m_trans * dot_sn

        # Get the field
        efield_grid[idx] *= reflectivity_sigma[idx]


# TODO: Add this function to only account for the phase
@cuda.jit('void(complex128[:], float64[:], float64[:,:], complex128[:], complex128[:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64, float64,'
          'complex128, complex128, complex128,'
          'int64)')
def get_bragg_reflection_sigma_phase(reflectivity_sigma,
                                     phase_grid,
                                     kout_grid,
                                     efield_grid,
                                     jacobian,
                                     klen_grid,
                                     kin_grid,
                                     d,
                                     h,
                                     n,
                                     dot_sn,
                                     dot_hn,
                                     h_square,
                                     chi0,
                                     chih_sigma,
                                     chihbar_sigma,
                                     num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_sigma:
    :param phase_grid:
    :param kout_grid:
    :param efield_grid:
    :param jacobian:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_sn:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_sigma:
    :param chihbar_sigma:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        #######################################################################
        #     In this very simple case, I only consider the phase change
        #######################################################################
        # Get the phase term:
        tmp = m_trans * dot_sn
        phase_grid[idx] -= tmp

        # Get the reflectivity
        reflectivity_sigma[idx] = complex(math.cos(tmp), -math.sin(tmp))

        # Get the jacobian :   dot(kout, n) / dot(kin, n)
        jacobian[idx] *= complex(math.fabs((dot_kn + dot_hn + m_trans) / dot_kn))


@cuda.jit('void(complex128[:], float64[:,:], complex128[:], complex128[:],'
          'float64[:], float64[:,:],'
          'float64, float64[:], float64[:],'
          'float64, float64,'
          'complex128, complex128,  complex128,'
          'int64)')
def get_bragg_reflection_pi_polarization(reflectivity_pi,
                                         kout_grid,
                                         efield_grid,
                                         jacobian,
                                         klen_grid,
                                         kin_grid,
                                         d,
                                         h,
                                         n,
                                         dot_hn,
                                         h_square,
                                         chi0, chih_pi,
                                         chihbar_pi, num):
    """
    Given the crystal info, the input electric field, this function returns the
    reflectivity for the sigma polarization and pi polarization and the
    diffracted electric field.

    :param reflectivity_pi:
    :param kout_grid:
    :param efield_grid:
    :param jacobian:
    :param klen_grid:
    :param kin_grid:
    :param d:
    :param h:
    :param n:
    :param dot_hn:
    :param h_square:
    :param chi0:
    :param chih_pi:
    :param chihbar_pi:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:

        #####################################################################################################
        # Step 1: Get parameters for reflectivity and decompose input field
        #####################################################################################################
        # ------------------------------------
        #     Get the diffracted wave number
        # ------------------------------------
        # Get k components
        kin_x = kin_grid[idx, 0]
        kin_y = kin_grid[idx, 1]
        kin_z = kin_grid[idx, 2]
        klen = klen_grid[idx]

        # Get gamma and alpha and b
        dot_kn = kin_x * n[0] + kin_y * n[1] + kin_z * n[2]
        dot_kh = kin_x * h[0] + kin_y * h[1] + kin_z * h[2]

        gamma_0 = dot_kn / klen
        gamma_h = (dot_kn + dot_hn) / klen
        b = gamma_0 / gamma_h
        b_complex = complex(b)
        alpha = (2 * dot_kh + h_square) / (klen ** 2)

        # Get momentum tranfer
        sqrt_gamma_alpha = math.sqrt(gamma_h ** 2 - alpha)
        m_trans = klen * (-gamma_h - sqrt_gamma_alpha)

        # Get output wave vector
        kout_x = kin_x + h[0] + m_trans * n[0]
        kout_y = kin_y + h[1] + m_trans * n[1]
        kout_z = kin_z + h[2] + m_trans * n[2]

        # Update the kout_grid
        kout_grid[idx, 0] = kout_x
        kout_grid[idx, 1] = kout_y
        kout_grid[idx, 2] = kout_z

        # Get the jacobian :   dot(kout, n) / dot(kin, n)
        jacobian[idx] *= complex(math.fabs((dot_kn + dot_hn + m_trans) / dot_kn))
        # jacobian[idx] *= complex(math.fabs(kout_x * n[0] + kout_y * n[1] + kout_z * n[2]) / dot_kn, 0)

        #####################################################################################################
        # Step 2: Get the reflectivity
        #####################################################################################################
        # Get alpha tidle
        alpha_tidle = complex((alpha * b + chi0.real * (1. - b)) / 2., chi0.imag * (1. - b) / 2.)

        # Get the polarization factor with the asymmetric factor b.
        p_value = complex((kout_x * kin_x +
                           kout_y * kin_y +
                           kout_z * kin_z) / (klen ** 2))
        bp = b_complex * p_value

        # Get sqrt(alpha**2 + beta**2) value
        sqrt_a2_b2 = cmath.sqrt(alpha_tidle ** 2 + bp * p_value * chih_pi * chihbar_pi)

        # Because this is a thick crystal, only one mode will be activated.
        if sqrt_a2_b2.imag < 0.:
            # Because only one mode is activated,
            sqrt_a2_b2 = - sqrt_a2_b2

        # Calculate the phase term
        re = klen * d / gamma_0 * sqrt_a2_b2.real
        im = klen * d / gamma_0 * sqrt_a2_b2.imag

        # Take care of the exponential
        if im <= 400.:
            magnitude = complex(math.exp(-im))
            phase = complex(math.cos(re), math.sin(re))

            # Calculate some intermediate part
            numerator = complex(1.) - magnitude * phase
            denominator = alpha_tidle * numerator + sqrt_a2_b2 * (complex(2.) - numerator)
            # Assemble everything
            reflectivity_pi[idx] = bp * chih_pi * numerator / denominator

        else:
            # Calculate some intermediate part
            denominator = alpha_tidle + sqrt_a2_b2
            # Assemble everything
            reflectivity_pi[idx] = bp * chih_pi / denominator

        # Get the field
        efield_grid[idx] *= reflectivity_pi[idx]


###################################################################################################
#          Backward propagation
###################################################################################################
@cuda.jit('void'
          '(float64[:,:], complex128[:],'
          'float64[:], float64[:,:], '
          'float64[:], float64[:], float64, float64, '
          'int64)')
def get_kin_and_jacobian(kin_grid, jacobian_grid,
                         klen_grid, kout_grid,
                         h, n, dot_hn, h_square,
                         num):
    """
    Given kout info, this function derives the corresponding kin info.

    :param kin_grid: This function derive the input wave vectors
    :param jacobian_grid:
    :param klen_grid: The wave vector length
    :param kout_grid:
    :param h: The crystal h vector
    :param n: The crystal normal direction
    :param dot_hn: The inner product between h and n
    :param h_square: The length of the h vector
    :param num:
    :return:
    """
    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:

        ##################################################################
        # Step 1: Get the corresponding parameters to get the reflectivity
        ##################################################################
        # Get k
        kout_x = kout_grid[idx, 0]
        kout_y = kout_grid[idx, 1]
        kout_z = kout_grid[idx, 2]

        k = klen_grid[idx]
        k_square = k ** 2

        # Get rho and epsilon
        dot_kn = kout_z * n[2] + kout_y * n[1] + kout_x * n[0]
        dot_kh = kout_z * h[2] + kout_y * h[1] + kout_x * h[0]
        rho = (dot_hn - dot_kn) / k
        epsilon = (h_square - 2 * dot_kh) / k_square

        # Decide the sign
        sqrt_rho_epsilon = math.sqrt(rho ** 2 - epsilon)
        tmp_pos = abs(-rho + sqrt_rho_epsilon)
        tmp_neg = abs(-rho - sqrt_rho_epsilon)
        if tmp_pos > tmp_neg:
            m_trans = k * (-rho - sqrt_rho_epsilon)
        else:
            m_trans = k * (-rho + sqrt_rho_epsilon)

        # Get the incident wave vector
        kin_grid[idx, 0] = kout_x - h[0] - m_trans * n[0]
        kin_grid[idx, 1] = kout_y - h[1] - m_trans * n[1]
        kin_grid[idx, 2] = kout_z - h[2] - m_trans * n[2]

        # Get the jacobian grid
        jacobian_grid[idx] *= complex(math.fabs(dot_kn / (dot_kn - dot_hn - m_trans)))


@cuda.jit("void(float64[:], float64[:,:],"
          "float64[:,:], float64[:], float64[:],"
          "float64[:,:], float64[:], float64[:], int64)")
def get_intersection_point(path_length_remain,
                           intersect_point,
                           kvec_grid,
                           klen_gird,
                           path_length,
                           source_point,
                           surface_position,
                           surface_normal,
                           num):
    """
    This function trace down the intersection point of the previous reflection plane.
    Then calculate the distance and then calculate the remaining distance to go
    to get to the initial point of this k component.

    Notice that, if the intersection point is along the positive direction of the
    propagation direction, then the path length is calculated to be positive.
    If the intersection point is along the negative direction of the propagation
    direction, then the path length is calculated to be negative.

    :param path_length_remain:
    :param intersect_point:
    :param kvec_grid: The incident k vector. Notice that I need this for all the
                    reflections. Therefore, I can not pre define kv ** 2 + ku ** 2
                    to reduce calculation
    :param klen_gird: Notice that all the reflections does not change the length
                        of the wave vectors. Therefore, I do not need to calculate
                        this value again and again.
    :param path_length:
    :param source_point:
    :param surface_position:
    :param surface_normal:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Get the coefficient before K
        coef_k = (surface_normal[0] * (surface_position[0] - source_point[idx, 0]) +
                  surface_normal[1] * (surface_position[1] - source_point[idx, 1]) +
                  surface_normal[2] * (surface_position[2] - source_point[idx, 2]))
        coef_k /= (surface_normal[0] * kvec_grid[idx, 0] +
                   surface_normal[1] * kvec_grid[idx, 1] +
                   surface_normal[2] * kvec_grid[idx, 2])

        # Assign the value
        intersect_point[idx, 0] = source_point[idx, 0] + coef_k * kvec_grid[idx, 0]
        intersect_point[idx, 1] = source_point[idx, 1] + coef_k * kvec_grid[idx, 1]
        intersect_point[idx, 2] = source_point[idx, 2] + coef_k * kvec_grid[idx, 2]

        # Get the distance change
        distance = coef_k * klen_gird[idx]
        path_length_remain[idx] = path_length[idx] - distance


@cuda.jit("void(float64[:], float64[:,:],"
          "float64[:,:], float64[:], float64[:],"
          "float64[:,:], float64[:], float64[:], int64)")
def get_intersection_point(path_length_remain,
                           intersect_point,
                           kvec_grid,
                           klen_gird,
                           path_length,
                           source_point,
                           surface_position,
                           surface_normal,
                           num):
    """
    This function trace down the intersection point of the previous reflection plane.
    Then calculate the distance and then calculate the remaining distance to go
    to get to the initial point of this k component.

    Notice that, if the intersection point is along the positive direction of the
    propagation direction, then the path length is calculated to be positive.
    If the intersection point is along the negative direction of the propagation
    direction, then the path length is calculated to be negative.

    :param path_length_remain:
    :param intersect_point:
    :param kvec_grid: The incident k vector. Notice that I need this for all the
                    reflections. Therefore, I can not pre define kv ** 2 + ku ** 2
                    to reduce calculation
    :param klen_gird: Notice that all the reflections does not change the length
                        of the wave vectors. Therefore, I do not need to calculate
                        this value again and again.
    :param path_length:
    :param source_point:
    :param surface_position:
    :param surface_normal:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Get the coefficient before K
        coef_k = (surface_normal[0] * (surface_position[0] - source_point[idx, 0]) +
                  surface_normal[1] * (surface_position[1] - source_point[idx, 1]) +
                  surface_normal[2] * (surface_position[2] - source_point[idx, 2]))
        coef_k /= (surface_normal[0] * kvec_grid[idx, 0] +
                   surface_normal[1] * kvec_grid[idx, 1] +
                   surface_normal[2] * kvec_grid[idx, 2])

        # Assign the value
        intersect_point[idx, 0] = source_point[idx, 0] + coef_k * kvec_grid[idx, 0]
        intersect_point[idx, 1] = source_point[idx, 1] + coef_k * kvec_grid[idx, 1]
        intersect_point[idx, 2] = source_point[idx, 2] + coef_k * kvec_grid[idx, 2]

        # Get the distance change
        distance = coef_k * klen_gird[idx]
        path_length_remain[idx] = path_length[idx] - distance


@cuda.jit("void(float64[:], float64[:], float64[:,:],"
          "float64[:,:], float64[:], "
          "float64[:,:], float64[:], float64[:], int64)")
def get_intersection_point_with_evolution_phase(phase_real_grid,
                                                path_length_remain,
                                                intersect_point,
                                                kvec_grid,
                                                klen_gird,
                                                source_point,
                                                surface_position,
                                                surface_normal,
                                                num):
    """
    This function trace down the intersection point of the previous reflection plane.
    Then calculate the distance and then calculate the remaining distance to go
    to get to the initial point of this k component.

    Notice that, if the intersection point is along the positive direction of the
    propagation direction, then the path length is calculated to be positive.
    If the intersection point is along the negative direction of the propagation
    direction, then the path length is calculated to be negative.

    :param phase_real_grid: This quantity contains the propagation phase of this specific components
                            The reason that I need to keep track of this is that
                            grating and prism can change the frequency.
    :param intersect_point:
    :param kvec_grid: The incident k vector. Notice that I need this for all the
                    reflections. Therefore, I can not pre define kv ** 2 + ku ** 2
                    to reduce calculation
    :param klen_gird: Notice that all the reflections does not change the length
                        of the wave vectors. Therefore, I do not need to calculate
                        this value again and again.
    :param source_point:
    :param path_length_remain
    :param surface_position:
    :param surface_normal:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Get the coefficient before K
        coef_k = (surface_normal[0] * (surface_position[0] - source_point[idx, 0]) +
                  surface_normal[1] * (surface_position[1] - source_point[idx, 1]) +
                  surface_normal[2] * (surface_position[2] - source_point[idx, 2]))
        coef_k /= (surface_normal[0] * kvec_grid[idx, 0] +
                   surface_normal[1] * kvec_grid[idx, 1] +
                   surface_normal[2] * kvec_grid[idx, 2])

        # Assign the value
        intersect_point[idx, 0] = source_point[idx, 0] + coef_k * kvec_grid[idx, 0]
        intersect_point[idx, 1] = source_point[idx, 1] + coef_k * kvec_grid[idx, 1]
        intersect_point[idx, 2] = source_point[idx, 2] + coef_k * kvec_grid[idx, 2]

        # Get the distance change
        distance = coef_k * klen_gird[idx]
        path_length_remain[idx] -= distance
        phase_real_grid[idx] -= klen_gird[idx] * distance


@cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64)")
def get_source_point(source_point, end_point, kvec_grid, klen_grid, path_length, num):
    """
    Find the source point of this wave vector component at time 0.

    :param source_point:
    :param end_point:
    :param kvec_grid:
    :param klen_grid
    :param path_length:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Normalize with the length of the wave number
        coef = path_length[idx] / klen_grid[idx]

        source_point[idx, 0] = end_point[idx, 0] - coef * kvec_grid[idx, 0]
        source_point[idx, 1] = end_point[idx, 1] - coef * kvec_grid[idx, 1]
        source_point[idx, 2] = end_point[idx, 2] - coef * kvec_grid[idx, 2]


@cuda.jit("void(float64[:,:], float64[:,:], float64[:,:], float64[:], float64[:], int64)")
def get_final_point(final_point, end_point, kvec_grid, klen_grid, path_length, num):
    """
    Find the source point of this wave vector component at time 0.

    :param final_point:
    :param end_point:
    :param kvec_grid:
    :param klen_grid
    :param path_length:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Normalize with the length of the wave number
        coef = path_length[idx] / klen_grid[idx]

        final_point[idx, 0] = end_point[idx, 0] + coef * kvec_grid[idx, 0]
        final_point[idx, 1] = end_point[idx, 1] + coef * kvec_grid[idx, 1]
        final_point[idx, 2] = end_point[idx, 2] + coef * kvec_grid[idx, 2]


@cuda.jit("void(float64[:], float64[:], float64[:,:], int64)")
def add_spatial_phase(phase, displacement, k_vec, num):
    """
    At present, I assume that the reference point is the same for all the components
    Then this function, calculate the relative phase of this wave component at the
    source point with respect to the reference point.

    The phase at the reference point is the phase that we have obtained when we
    take the fourier transformation of the gaussian field at the x0 position at time t=0

    :param phase:
    :param displacement:
    :param k_vec:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        phase[idx] += (k_vec[idx, 0] * displacement[0] +
                       k_vec[idx, 1] * displacement[1] +
                       k_vec[idx, 2] * displacement[2])


@cuda.jit("void(float64[:], float64, float64[:], int64)")
def add_evolution_phase(phase, path_length, k_len, num):
    """
    At present, I assume that the reference point is the same for all the components
    Then this function, calculate the relative phase of this wave component at the
    source point with respect to the reference point.

    The phase at the reference point is the phase that we have obtained when we
    take the fourier transformation of the gaussian field at the x0 position at time t=0

    :param phase:
    :param path_length:
    :param k_len:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        phase[idx] -= path_length * k_len[idx]  # * c


@cuda.jit("void(float64[:], float64[:], float64[:], int64)")
def add_evolution_phase_elementwise(phase, path_length, k_len, num):
    """
    At present, I assume that the reference point is the same for all the components
    Then this function, calculate the relative phase of this wave component at the
    source point with respect to the reference point.

    The phase at the reference point is the phase that we have obtained when we
    take the fourier transformation of the gaussian field at the x0 position at time t=0

    :param phase:
    :param path_length:
    :param k_len:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        phase[idx] -= path_length[idx] * k_len[idx]  # * c


@cuda.jit("void(complex128[:], float64[:,:], float64[:], float64[:,:], float64[:], int64)")
def get_relative_spatial_phase(phase, source_point, reference_point, k_vec, k_ref, num):
    """
    I keep it here only for the completeness of this package.
    This is not recommended.

    :param phase:
    :param source_point:
    :param reference_point:
    :param k_vec:
    :param k_ref:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        tmp = ((k_vec[idx, 0] - k_ref[0]) * (source_point[idx, 0] - reference_point[0]) +
               (k_vec[idx, 1] - k_ref[1]) * (source_point[idx, 1] - reference_point[1]) +
               (k_vec[idx, 2] - k_ref[2]) * (source_point[idx, 2] - reference_point[2]))
        tmp /= -2.  # This function is obselete. However,  If anyone would want to use this.
        # This factor is important.
        phase[idx] = complex(math.cos(tmp), math.sin(tmp))


@cuda.jit("void(complex128[:], float64[:,:], float64[:], float64[:,:], float64[:], int64)")
def get_relative_spatial_phase_backup(phase, source_point, reference_point, k_vec, k_ref, num):
    """
    This is an ancient function. It is obselete now. Also, it is wrong.
    I keep it here only for record.

    This is an old function which aims to calculate the propagation phase.
    However, I realized that this function might be wrong. Therefore
    I put it here.

    :param phase:
    :param source_point:
    :param reference_point:
    :param k_vec:
    :param k_ref:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        tmp = ((k_vec[idx, 0] - k_ref[0]) * (source_point[idx, 0] - reference_point[0]) +
               (k_vec[idx, 1] - k_ref[1]) * (source_point[idx, 1] - reference_point[1]) +
               (k_vec[idx, 2] - k_ref[2]) * (source_point[idx, 2] - reference_point[2]))

        phase[idx] = complex(math.cos(tmp), math.sin(tmp))


################################################################################
#  Get pulse spectrum
################################################################################
@cuda.jit('void'
          '(complex128[:], '
          'float64[:,:], float64, '
          'float64[:,:], complex128,'
          'float64[:], float64[:], float64, float64[:], int64)')
def get_gaussian_pulse_spectrum(coef,
                                k_vec, t,
                                sigma_mat, scaling,
                                x0, k0, omega0, n,
                                num):
    """
    Calculate the corresponding coefficient in the incident gaussian pulse
    for each wave vectors to investigate.

    :param coef: The coefficent to be calculated
    :param k_vec: The wave vectors to be calculated
    :param t: The time for the snapshot. This value will usually be 0.
    :param num: The number of wave vectors to calculate
    :param sigma_mat: The sigma matrix of the Gaussian pulse. Notice that here,
                        the elements in this matrix should have unit of um.
    :param scaling: A linear scaling coefficient to take the intensity
                    and some other factors in to consideration.
    :param x0: This is the displacement vector with respect to the origin of the incident pulse frame.
    :param k0:
    :param omega0:
    :param n: The direction of k0
    :return:
    """
    row = cuda.grid(1)
    if row < num:
        # Get the momentum difference
        dk_x = k0[0] - k_vec[row, 0]
        dk_y = k0[1] - k_vec[row, 1]
        dk_z = k0[2] - k_vec[row, 2]

        phase = -t * omega0

        # Notice that here, we are using the fourier component rather than the characteristic function
        # If it's characteristic function, then it should be +=
        phase -= ((x0[0] + c * t * n[0]) * dk_x +
                  (x0[1] + c * t * n[1]) * dk_y +
                  (x0[2] + c * t * n[2]) * dk_z)

        phase_term = complex(math.cos(phase), math.sin(phase))

        # Get the quadratic term
        quad_term = - (dk_x * sigma_mat[0, 0] * dk_x + dk_x * sigma_mat[0, 1] * dk_y +
                       dk_x * sigma_mat[0, 2] * dk_z +
                       dk_y * sigma_mat[1, 0] * dk_x + dk_y * sigma_mat[1, 1] * dk_y +
                       dk_y * sigma_mat[1, 2] * dk_z +
                       dk_z * sigma_mat[2, 0] * dk_x + dk_z * sigma_mat[2, 1] * dk_y +
                       dk_z * sigma_mat[2, 2] * dk_z
                       ) / 2.

        # if quad_term >= -200:
        magnitude = scaling * complex(math.exp(quad_term), 0)
        coef[row] = magnitude * phase_term


@cuda.jit('void'
          '(complex128[:], float64[:,:],'
          'float64[:], float64, float64, float64, complex128,'
          'int64)')
def get_square_pulse_spectrum(coef,
                              k_vec,
                              k0, a_val, b_val, c_val, scaling,
                              num):
    """
    Calculate the spectrum of a square pulse. 
    
    :param coef: 
    :param k_vec: 
    :param k0: 
    :param a_val: 
    :param b_val: 
    :param c_val: 
    :param scaling: 
    :param num: 
    :return: 
    """
    row = cuda.grid(1)
    if row < num:
        # Get the momentum difference
        dk_x = a_val * (k_vec[row, 0] - k0[0]) / 2.
        dk_y = b_val * (k_vec[row, 1] - k0[1]) / 2.
        dk_z = c_val * (k_vec[row, 2] - k0[2]) / 2.

        holder = 1.

        # Get the contribution from the x component
        if math.fabs(dk_x) <= eps:
            holder *= (0.05 * dk_x ** 2 - 1.) * (dk_x ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_x) / dk_x

        # Get the contribution from the y component
        if math.fabs(dk_y) <= eps:
            holder *= (0.05 * dk_y ** 2 - 1.) * (dk_y ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_y) / dk_y

        # Get the contribution from the z component
        if math.fabs(dk_z) <= eps:
            holder *= (0.05 * dk_z ** 2 - 1.) * (dk_z ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_z) / dk_z

        coef[row] = scaling * complex(holder)


@cuda.jit('void'
          '(complex128[:], float64[:,:],'
          'float64[:], float64, float64, float64, complex128, float64, '
          'int64)')
def get_square_pulse_spectrum_smooth(coef,
                                     k_vec,
                                     k0, a_val, b_val, c_val, scaling, sigma,
                                     num):
    """
    Calculate the spectrum of a square pulse.

    :param coef:
    :param k_vec:
    :param k0:
    :param a_val:
    :param b_val:
    :param c_val:
    :param scaling:
    :param sigma: The sigma value of the gaussian filter.
    :param num:
    :return:
    """
    row = cuda.grid(1)
    if row < num:
        # Get the momentum difference
        dk_x = a_val * (k_vec[row, 0] - k0[0]) / 2.
        dk_y = b_val * (k_vec[row, 1] - k0[1]) / 2.
        dk_z = c_val * (k_vec[row, 2] - k0[2]) / 2.

        holder = 1.

        # Get the contribution from the x component
        if math.fabs(dk_x) <= eps:
            holder *= (0.05 * dk_x ** 2 - 1.) * (dk_x ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_x) / dk_x

        # Get the contribution from the y component
        if math.fabs(dk_y) <= eps:
            holder *= (0.05 * dk_y ** 2 - 1.) * (dk_y ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_y) / dk_y

        # Get the contribution from the z component
        if math.fabs(dk_z) <= eps:
            holder *= (0.05 * dk_z ** 2 - 1.) * (dk_z ** 2) / 6. + 1.
        else:
            holder *= math.sin(dk_z) / dk_z

        gaussian = math.exp(-(dk_x ** 2 + dk_y ** 2 + dk_z ** 2) * (sigma ** 2) / 2.)
        coef[row] = scaling * complex(holder * gaussian)


################################################################################
#  Get grating effects
################################################################################
@cuda.jit('void'
          '(float64[:,:], complex128[:,:],'
          'float64[:],'
          'float64[:,:],'
          'float64[:], complex128, float64, float64[:], float64, float64[:],'
          'int64)')
def get_square_grating_effect_non_zero(kout_grid, efield_grid,
                                       klen_grid,
                                       kin_grid,
                                       grating_h,
                                       grating_n,
                                       grating_ab_ratio,
                                       grating_base,
                                       order,
                                       grating_k,
                                       num):
    """
    This function add the grating effect to the pulse. Including the phase change and the momentum change.

    Notice that this function can not handle the zeroth order

    :param kout_grid: The output momentum grid
    :param efield_grid: The output coefficient for each monochromatic component
    :param klen_grid: The length of each incident wave vector. Notice that this value will update for the grating.
    :param kin_grid: The incident wave vector grid.
    :param grating_h: The height vector of the grating
    :param grating_n: The refraction index of the grating.
    :param grating_ab_ratio: The b / (a + b) where a is the width of the groove while b the width of the tooth
    :param grating_base: The thickness of the base of the grating
    :param order: The order of diffraction to investigate.
                    Notice that this variable has to be an integer mathematically.
                    However, in numerical calculation, this is used as a float.
    :param grating_k: The base wave vector of the grating
    :param num: The number of momenta to calculate.
    :return: None
    """

    row = cuda.grid(1)
    if row < num:
        # Step 1: Calculate the effect of the grating on magnitude and phase for each component

        # The argument for exp(ik(n-1)h)
        nhk = complex(grating_h[0] * kin_grid[row, 0] +
                      grating_h[1] * kin_grid[row, 1] +
                      grating_h[2] * kin_grid[row, 2]) * (grating_n - complex(1.))

        # The argument for exp(ik(n-1)t) for the phase different and absorption from
        # the base of the grating
        thick_k_n = complex(grating_base[0] * kin_grid[row, 0] +
                            grating_base[1] * kin_grid[row, 1] +
                            grating_base[2] * kin_grid[row, 2]) * (grating_n - complex(1.))

        first_factor = complex(1.
                               - math.cos(two_pi * order * grating_ab_ratio),
                               - math.sin(two_pi * order * grating_ab_ratio))
        second_factor = complex(1.) - complex(math.exp(-nhk.imag) * math.cos(nhk.real),
                                              math.exp(-nhk.imag) * math.sin(nhk.real))

        # Factor from the base
        factor_base = complex(math.cos(thick_k_n.real) * math.exp(-thick_k_n.imag),
                              math.sin(thick_k_n.real) * math.exp(-thick_k_n.imag))

        factor = 1.j / complex(2. * math.pi * order) * first_factor * second_factor * factor_base

        # Step 2: Update the coefficient
        efield_grid[row, 0] = factor * efield_grid[row, 0]
        efield_grid[row, 1] = factor * efield_grid[row, 1]
        efield_grid[row, 2] = factor * efield_grid[row, 2]

        # Step 3: Update the momentum and the length of the momentum
        kout_grid[row, 0] = kin_grid[row, 0] + order * grating_k[0]
        kout_grid[row, 1] = kin_grid[row, 1] + order * grating_k[1]
        kout_grid[row, 2] = kin_grid[row, 2] + order * grating_k[2]

        klen_grid[row] = math.sqrt(kout_grid[row, 0] * kout_grid[row, 0] +
                                   kout_grid[row, 1] * kout_grid[row, 1] +
                                   kout_grid[row, 2] * kout_grid[row, 2])


@cuda.jit('void'
          '(float64[:,:], complex128[:],'
          'float64[:],'
          'float64[:,:],'
          'float64[:], complex128, float64, float64[:], float64, float64[:],'
          'int64)')
def get_square_grating_diffraction_scalar(kout_grid,
                                          efield_grid,
                                          klen_grid,
                                          kin_grid,
                                          grating_h,
                                          grating_n,
                                          grating_ab_ratio,
                                          grating_base,
                                          order,
                                          grating_k,
                                          num):
    """
    This function add the grating effect to the pulse. Including the phase change and the momentum change.

    Notice that this function can not handle the zeroth order

    :param kout_grid: The output momentum grid
    :param efield_grid: The output coefficient for each monochromatic component
    :param klen_grid: The length of each incident wave vector. Notice that this value will update for the grating.
    :param kin_grid: The incident wave vector grid.
    :param grating_h: The height vector of the grating
    :param grating_n: The refraction index of the grating.
    :param grating_ab_ratio: The b / (a + b) where a is the width of the groove while b the width of the tooth
    :param grating_base: The thickness of the base of the grating
    :param order: The order of diffraction to investigate.
                    Notice that this variable has to be an integer mathematically.
                    However, in numerical calculation, this is used as a float.
    :param grating_k: The base wave vector of the grating
    :param num: The number of momenta to calculate.
    :return: None
    """

    row = cuda.grid(1)
    if row < num:
        # Step 1: Calculate the effect of the grating on magnitude and phase for each component

        # The argument for exp(ik(n-1)h)
        nhk = complex(grating_h[0] * kin_grid[row, 0] +
                      grating_h[1] * kin_grid[row, 1] +
                      grating_h[2] * kin_grid[row, 2]) * (grating_n - complex(1.))

        # The argument for exp(ik(n-1)t) for the phase different and absorption from
        # the base of the grating
        thick_k_n = complex(grating_base[0] * kin_grid[row, 0] +
                            grating_base[1] * kin_grid[row, 1] +
                            grating_base[2] * kin_grid[row, 2]) * (grating_n - complex(1.))

        first_factor = complex(1.
                               - math.cos(two_pi * order * grating_ab_ratio),
                               - math.sin(two_pi * order * grating_ab_ratio))
        second_factor = complex(1.) - complex(math.exp(-nhk.imag) * math.cos(nhk.real),
                                              math.exp(-nhk.imag) * math.sin(nhk.real))

        # Factor from the base
        factor_base = complex(math.cos(thick_k_n.real) * math.exp(-thick_k_n.imag),
                              math.sin(thick_k_n.real) * math.exp(-thick_k_n.imag))

        factor = 1.j / complex(2. * math.pi * order) * first_factor * second_factor * factor_base

        # Step 2: Update the coefficient
        efield_grid[row] = factor * efield_grid[row]

        # Step 3: Update the momentum and the length of the momentum
        kout_grid[row, 0] = kin_grid[row, 0] + order * grating_k[0]
        kout_grid[row, 1] = kin_grid[row, 1] + order * grating_k[1]
        kout_grid[row, 2] = kin_grid[row, 2] + order * grating_k[2]

        klen_grid[row] = math.sqrt(kout_grid[row, 0] * kout_grid[row, 0] +
                                   kout_grid[row, 1] * kout_grid[row, 1] +
                                   kout_grid[row, 2] * kout_grid[row, 2])


@cuda.jit('void'
          '(complex128[:,:],'
          'float64[:,:],'
          'float64[:], complex128, float64,'
          'int64)')
def get_square_grating_effect_zero(efield_grid,
                                   kin_grid,
                                   grating_h, grating_n, grating_ab_ratio,
                                   num):
    """
    This function add the grating effect to the pulse. Including the phase change and the momentum change.

    Notice that this function only handle the zeroth order

    :param efield_grid: The output coefficient for each monochromatic component
    :param kin_grid: The incident wave vector grid.
    :param grating_h: The height vector of the grating
    :param grating_n: The refraction index of the grating.
    :param grating_ab_ratio: The b / (a + b) where a is the width of the groove while b the width of the tooth
    :param num: The number of momenta to calculate.
    :return: None
    """

    row = cuda.grid(1)
    if row < num:
        # Step 1: Calculate the effect of the grating on magnitude and phase for each component

        # The argument for exp(ik(n-1)h)
        nhk = complex(grating_h[0] * kin_grid[row, 0] +
                      grating_h[1] * kin_grid[row, 1] +
                      grating_h[2] * kin_grid[row, 2]) * (grating_n - complex(1.))

        pre_factor = complex(1.) - complex(math.exp(-nhk.imag) * math.cos(nhk.real),
                                           math.exp(-nhk.imag) * math.sin(nhk.real))

        factor = complex(1.) - complex(grating_ab_ratio) * pre_factor

        # Step 2: Update the coefficient
        efield_grid[row, 0] *= factor
        efield_grid[row, 1] *= factor
        efield_grid[row, 2] *= factor


###################################################################################################
#          For telescope
###################################################################################################
@cuda.jit('void'
          '(float64[:,:],'
          'float64[:,:], '
          'float64[:], int64)')
def get_kin_telescope(kin_grid,
                      kout_grid,
                      optical_axis,
                      num):
    """
    Given the kout grid, this function return the corresponding kin grid for
    the telescope of the chirped pulse amplification device.

    :param kin_grid: This function derive the input wave vectors
    :param kout_grid:
    :param optical_axis: The optical axis of the telescope
    :param num:
    :return:
    """
    # Step 0: Get the cuda grid idx
    idx = cuda.grid(1)
    if idx < num:
        k_parallel = (kout_grid[idx, 0] * optical_axis[0] +
                      kout_grid[idx, 1] * optical_axis[1] +
                      kout_grid[idx, 2] * optical_axis[2])

        kin_grid[idx, 0] = 2 * k_parallel * optical_axis[0] - kout_grid[idx, 0]
        kin_grid[idx, 1] = 2 * k_parallel * optical_axis[1] - kout_grid[idx, 1]
        kin_grid[idx, 2] = 2 * k_parallel * optical_axis[2] - kout_grid[idx, 2]


@cuda.jit('void('
          'float64[:,:],'
          'complex128[:,:],'
          'float64[:,:],'
          'float64[:,:],'
          'float64[:,:],'
          'float64[:],'
          'float64,'
          'float64[:],'
          'complex128,'
          'int64)')
def get_telescope_diffraction(kout_grid,
                              efield_grid,
                              output_position,
                              kin_grid,
                              input_position,
                              optical_axis,
                              focal_length,
                              telescope_position,
                              efficiency,
                              num):
    """
    This function update the information for the diffraction from the telescope

    :param kout_grid:
    :param efield_grid:
    :param output_position:
    :param kin_grid:
    :param input_position:
    :param optical_axis:
    :param focal_length:
    :param telescope_position:
    :param efficiency:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Update the wave vector
        k_parallel = (kin_grid[idx, 0] * optical_axis[0] +
                      kin_grid[idx, 1] * optical_axis[1] +
                      kin_grid[idx, 2] * optical_axis[2])

        kout_grid[idx, 0] = 2 * k_parallel * optical_axis[0] - kin_grid[idx, 0]
        kout_grid[idx, 1] = 2 * k_parallel * optical_axis[1] - kin_grid[idx, 1]
        kout_grid[idx, 2] = 2 * k_parallel * optical_axis[2] - kin_grid[idx, 2]

        # Transmission of the electric field
        efield_grid[idx, 0] *= efficiency
        efield_grid[idx, 1] *= efficiency
        efield_grid[idx, 2] *= efficiency

        #######################################################################
        # Change of the position for the phase calculation
        #######################################################################
        # Object position
        tmp_x = telescope_position[0] - input_position[idx, 0]
        tmp_y = telescope_position[1] - input_position[idx, 1]
        tmp_z = telescope_position[2] - input_position[idx, 2]

        object_distance_1 = (tmp_x * optical_axis[0] +
                             tmp_y * optical_axis[1] +
                             tmp_z * optical_axis[2])

        image_vector_x = tmp_x - object_distance_1 * optical_axis[0]
        image_vector_y = tmp_y - object_distance_1 * optical_axis[1]
        image_vector_z = tmp_z - object_distance_1 * optical_axis[2]

        # Image position
        tmp_length = 4 * focal_length - object_distance_1
        output_position[idx, 0] = telescope_position[0] + tmp_length * optical_axis[0]
        output_position[idx, 1] = telescope_position[1] + tmp_length * optical_axis[1]
        output_position[idx, 2] = telescope_position[2] + tmp_length * optical_axis[2]

        # Change the image position due to the deviation from the optical axis
        output_position[idx, 0] -= image_vector_x
        output_position[idx, 1] -= image_vector_y
        output_position[idx, 2] -= image_vector_z


@cuda.jit('void('
          'float64[:,:],'
          'complex128[:],'
          'float64[:,:],'
          'float64[:,:],'
          'float64[:,:],'
          'float64[:],'
          'float64,'
          'float64[:],'
          'complex128,'
          'int64)')
def get_telescope_scalar_diffraction(kout_grid,
                                     efield_grid,
                                     output_position,
                                     kin_grid,
                                     input_position,
                                     optical_axis,
                                     focal_length,
                                     telescope_position,
                                     efficiency,
                                     num):
    """
    This function update the information for the diffraction from the telescope

    :param kout_grid:
    :param efield_grid:
    :param output_position:
    :param kin_grid:
    :param input_position:
    :param optical_axis:
    :param focal_length:
    :param telescope_position:
    :param efficiency:
    :param num:
    :return:
    """
    idx = cuda.grid(1)
    if idx < num:
        # Update the wave vector
        k_parallel = (kin_grid[idx, 0] * optical_axis[0] +
                      kin_grid[idx, 1] * optical_axis[1] +
                      kin_grid[idx, 2] * optical_axis[2])

        kout_grid[idx, 0] = 2 * k_parallel * optical_axis[0] - kin_grid[idx, 0]
        kout_grid[idx, 1] = 2 * k_parallel * optical_axis[1] - kin_grid[idx, 1]
        kout_grid[idx, 2] = 2 * k_parallel * optical_axis[2] - kin_grid[idx, 2]

        # Transmission of the electric field
        efield_grid[idx] *= efficiency

        #######################################################################
        # Change of the position for the phase calculation
        #######################################################################
        # Object position
        tmp_x = telescope_position[0] - input_position[idx, 0]
        tmp_y = telescope_position[1] - input_position[idx, 1]
        tmp_z = telescope_position[2] - input_position[idx, 2]

        object_distance_1 = (tmp_x * optical_axis[0] +
                             tmp_y * optical_axis[1] +
                             tmp_z * optical_axis[2])

        image_vector_x = tmp_x - object_distance_1 * optical_axis[0]
        image_vector_y = tmp_y - object_distance_1 * optical_axis[1]
        image_vector_z = tmp_z - object_distance_1 * optical_axis[2]

        # Image position
        tmp_length = 4 * focal_length - object_distance_1
        output_position[idx, 0] = telescope_position[0] + tmp_length * optical_axis[0]
        output_position[idx, 1] = telescope_position[1] + tmp_length * optical_axis[1]
        output_position[idx, 2] = telescope_position[2] + tmp_length * optical_axis[2]

        # Change the image position due to the deviation from the optical axis
        output_position[idx, 0] -= image_vector_x
        output_position[idx, 1] -= image_vector_y
        output_position[idx, 2] -= image_vector_z


@cuda.jit('void('
          'complex128[:,:],'
          'complex128[:,:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'complex128,'
          'complex128,'
          'int64,'
          'int64,'
          'int64,'
          'int64)')
def get_1d_fresnel_diffraction(field_out,
                               source,
                               k_array,
                               y_source_array,
                               y_target_array,
                               z_target_array,
                               y_sampling,
                               k_sampling,
                               y_source_num,
                               k_num,
                               y_target_num,
                               z_target_num):
    # Calculate diffraction field at different distances in parallel
    y_idx, z_idx = cuda.grid(2)

    if z_idx < z_target_num and y_idx < y_target_num:
        # Get an overall constant factor
        overall_factor = complex(0.5 / math.sqrt(math.pi), -0.5 / math.sqrt(math.pi))

        z = z_target_array[z_idx]
        y = y_target_array[y_idx]

        # Calculate the effect from different wave length
        for k_idx in range(k_num):

            # Get the factor associated with k and z
            k = k_array[k_idx]

            kz = k * z
            kz_sqrt = math.sqrt(k / z)

            kz_dependent_factor = complex(kz_sqrt * math.cos(kz),
                                          kz_sqrt * math.sin(kz))

            # Loop through the y source points
            integral = complex(0.)
            for y_source_idx in range(y_source_num):
                # Get the phase angle
                phase_angle = (y_source_array[y_source_idx] - y) ** 2
                phase_angle *= k / 2. / z

                # Get the phse
                phase = complex(math.cos(phase_angle), math.sin(phase_angle))

                # Add to the integration
                integral += phase * source[y_source_idx, k_idx]

            # Get the field
            integral *= y_sampling * kz_dependent_factor

            # Add this to the total output field
            field_out[y_idx, z_idx] += integral * k_sampling

        # Add the overall constant to the field
        field_out[y_idx, z_idx] *= overall_factor


@cuda.jit('void('
          'complex128[:,:,:],'
          'complex128[:,:,:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64,'
          'complex128,'
          'complex128,'
          'complex128,'
          'int64,'
          'int64,'
          'int64,'
          'int64,'
          'int64,'
          'int64)')
def get_2d_fresnel_diffraction(field_out,
                               source,
                               k_array,
                               x_source_array,
                               y_source_array,
                               x_target_array,
                               y_target_array,
                               z_target_array,
                               z_ref,
                               x_sampling,
                               y_sampling,
                               k_sampling,
                               x_source_num,
                               y_source_num,
                               k_num,
                               x_target_num,
                               y_target_num,
                               z_target_num):
    # Calculate diffraction field at different distances in parallel
    x_idx, y_idx, z_idx = cuda.grid(3)

    if x_idx < x_target_num and z_idx < z_target_num and y_idx < y_target_num:

        z = z_target_array[z_idx]
        y = y_target_array[y_idx]
        x = x_target_array[x_idx]

        # Calculate the effect from different wave length
        for k_idx in range(k_num):

            # Get the factor associated with k and z
            k = k_array[k_idx]

            kz = k * (z - z_ref)
            kz_p = complex(k / z_ref)
            kz_p2 = k / z_ref / 2.

            # Loop through the x and y source points
            integral = complex(0.)
            for x_source_idx in range(x_source_num):
                # Get the phase from x direction
                phase_angle_1 = (x_source_array[x_source_idx] - x) ** 2

                for y_source_idx in range(y_source_num):
                    # Get the phase angle from y direction
                    phase_angle_2 = phase_angle_1 + (y_source_array[y_source_idx] - y) ** 2
                    phase_angle_2 *= kz_p2

                    # Get the phase
                    phase = complex(math.cos(phase_angle_2), math.sin(phase_angle_2))

                    # Add to the integration
                    integral += phase * source[x_source_idx, y_source_idx, k_idx]

            # Get the field
            integral *= kz_p * complex(math.cos(kz), math.sin(kz))

            # Add this to the total output field
            field_out[x_idx, y_idx, z_idx] += integral

        # Add the overall constant to the field
        overall_constant = - 1.j / math.pow((2. * math.pi), 1.5)
        overall_constant *= k_sampling * x_sampling * y_sampling
        field_out[x_idx, y_idx, z_idx] *= overall_constant


@cuda.jit('void('
          'complex128[:,:,:],'
          'complex128[:,:,:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64[:],'
          'float64,'
          'complex128,'
          'complex128,'
          'complex128,'
          'int64,'
          'int64,'
          'int64,'
          'int64,'
          'int64,'
          'int64)')
def get_2d_fresnel_diffraction_v2(field_out,
                                  source,
                                  k_array,
                                  x_source_array,
                                  y_source_array,
                                  x_target_array,
                                  y_target_array,
                                  z_target_array,
                                  z_ref,
                                  x_sampling,
                                  y_sampling,
                                  k_sampling,
                                  x_source_num,
                                  y_source_num,
                                  k_num,
                                  x_target_num,
                                  y_target_num,
                                  z_target_num):
    # Calculate diffraction field at different distances in parallel
    x_idx, y_idx, z_idx = cuda.grid(3)

    if x_idx < x_target_num and z_idx < z_target_num and y_idx < y_target_num:

        z = z_target_array[z_idx]
        y = y_target_array[y_idx]
        x = x_target_array[x_idx]

        # Calculate the effect from different wave length
        for k_idx in range(k_num):

            # Get the factor associated with k and z
            k = k_array[k_idx]

            kz = k * (z - z_ref)
            kz_p = complex(k / z_ref)
            kz_p2 = k / z_ref / 2.

            # Loop through the x and y source points
            integral = complex(0.)
            for x_source_idx in range(x_source_num):
                # Get the phase from x direction
                phase_angle_1 = (x_source_array[x_source_idx] - x) ** 2

                for y_source_idx in range(y_source_num):
                    # Get the phase angle from y direction
                    phase_angle_2 = phase_angle_1 + (y_source_array[y_source_idx] - y) ** 2
                    phase_angle_2 *= kz_p2

                    # Get the phase
                    phase = complex(math.cos(phase_angle_2), math.sin(phase_angle_2))

                    # Add to the integration
                    integral += phase * source[x_source_idx, y_source_idx, k_idx]

            # Get the field
            integral *= kz_p * complex(math.cos(kz), math.sin(kz))

            # Add this to the total output field
            field_out[x_idx, y_idx, z_idx] += integral

        # Add the overall constant to the field
        overall_constant = - 1.j / math.pow((2. * math.pi), 1.5)
        overall_constant *= k_sampling * x_sampling * y_sampling
        field_out[x_idx, y_idx, z_idx] *= overall_constant
