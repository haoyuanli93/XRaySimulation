import cmath
import math

import numpy as np
from numba import cuda


#########################################################################################
#                         Method 1
#
#     Get the time-averaged mutual coherence function first and then get the contrast
#########################################################################################

def getContrastMethod1(eFieldComplexFiles, qVec, k0, nx, ny, nz, dx, dy, dz, nSampleZ, dSampleZ, out="contrast"):
    """
    This is much harder than I have previously expected.
    The array is so large that I have to divide them into several batches.

    The calculation strategy of this method is the following:

    1. Create a cuda array to store the following information
                    1/N_p \sum_(n, tIdx)  U'(x1,y1, z3 + Q*r1/ck) U(x2,y2,z3 + Q*r2/ck)
    2. Move the array back to memory (because the I/O of the x-ray pulse is more expensive)
    3. Loop through all the pulses
    4. Calculate the integral of the cude array over each sub-array and assemble them in the end

    Notice that, the cuda array is a 3D array rather than a 4D array because cude only accept 3D grid.
    i.e., I have flatten the x,y spatial dimensions.

    :return:
    """
    # Step1, prepare the
    pass


@cuda.jit('()')
def getCoherenceFunctionXY_GPU_Method1(nSpatial,
                                       nTemporal,
                                       nSample,
                                       deltaZx,
                                       deltaZy,
                                       deltaZz,
                                       weight,
                                       eFieldReal,
                                       eFieldImag,
                                       holderReal,
                                       holderImag):
    """

    :param nSpatial:  The length of the spatial index, which = nx * ny
    :param nTemporal:   The ends of the summation of the mutual coherence function
    :param nSample:
    :param deltaZx:
    :param deltaZy:
    :param deltaZz:
    :param weight:
    :param eFieldReal:
    :param eFieldImag:
    :param holderReal:
    :param holderImag:
    :return:
    """
    idx1, idx2 = cuda.grid(2)
    if (idx1 < nSpatial) & (idx2 < nSpatial):
        deltaZxy = deltaZx[idx2] + deltaZy[idx2] - (deltaZx[idx1] + deltaZy[idx1])  # get the contribution from the xy direction in Q*(r2-r1)/k0

        for sIdx in range(nSample):
            deltaZ = int(deltaZxy + deltaZz(sIdx))  # Get the contribution from the z dimension

            # Add the temporary holder
            holderRealTmp = 0
            holderImagTmp = 0

            # Do the Time averaging
            for tIdx in range(nTemporal[deltaZ, 0], nTemporal[deltaZ, 1]):
                holderRealTmp += eFieldReal[idx1, tIdx] * eFieldReal[idx2, tIdx + deltaZ]
                holderRealTmp += eFieldImag[idx1, tIdx] * eFieldImag[idx2, tIdx + deltaZ]

                holderImagTmp -= eFieldReal[idx1, tIdx] * eFieldImag[idx2, tIdx + deltaZ]
                holderImagTmp += eFieldImag[idx1, tIdx] * eFieldReal[idx2, tIdx + deltaZ]

            # Because for the same z2 - z1, the summation is the same. One only needs to add a weight
            holderReal[idx1, idx2] += holderRealTmp * weight[sIdx]
            holderImag[idx1, idx2] += holderImagTmp * weight[sIdx]

#########################################################################################
#                         Method 2
#
#            Get the contrast of each pulse individually.
#########################################################################################
@numba.njit
def calc_gamma_t(nx, ny, nz, nsz, dx, dy, dz, Q_vec, eFieldReal, eFieldImag):
    gamma_t = 0
    for i1 in range(nx):
        for j1 in range(ny):
            for m1 in range(nsz):
                for i2 in range(nx):
                    for j2 in range(ny):
                        for m2 in range(nsz):
                            delta_x = samx[i1, j1, m1] - samx[i2, j2, m2]
                            delta_y = samy[i1, j1, m1] - samy[i2, j2, m2]
                            delta_z = samz[i1, j1, m1] - samz[i2, j2, m2]
                            delta_r = np.array([delta_x, delta_y, delta_z])
                            tau = np.dot(Q_vec, delta_r) / w0
                            delta_t = int(np.round(tau / dt))
                            if (delta_t >= 0) & ((delta_t < nz)):
                                gamma_t += np.square(np.abs(np.mean(np.conjugate(eField[i1, j1, 0:nz - delta_t]) * eField[i2, j2, delta_t:nz])))
                            elif (delta_t < 0) & ((delta_t > -nz)):
                                gamma_t += np.square(np.abs(np.mean(np.conjugate(eField[i1, j1, -delta_t:]) * eField[i2, j2, :nz + delta_t])))
    return gamma_t
