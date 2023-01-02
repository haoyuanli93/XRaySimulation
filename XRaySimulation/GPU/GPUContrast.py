import math

import numpy as np
from numba import cuda


#########################################################################################
#                         Method 2
#
#            Get the contrast of each pulse individually.
#########################################################################################
def getContrastMethod2(eFieldComplexFiles, qVec, k0, nx, ny, nz, dx, dy, dz, nSampleZ, ):
    """
    Very challenging calculation.
    Need to check with Yanwen about the definition of the calculation.

    :param eFieldComplexFiles:
    :param qVec:
    :param k0:
    :param nx:
    :param ny:
    :param nz:
    :param dx:
    :param dy:
    :param dz:
    :param nSampleZ:
    :return:
    """
    # Step1, prepare the variables
    numXY = nx * ny

    # The phase change according to Q/k0 * r
    deltaZx = np.zeros((nx, ny))
    deltaZx[:, :] = (np.arange(nx) * dx * qVec[0] / k0 / dz)[:, np.newaxis]
    deltaZx = np.ascontiguousarray(np.reshape(deltaZx, numXY))

    deltaZy = np.zeros((nx, ny))
    deltaZy[:, :] = (np.arange(ny) * dy * qVec[1] / k0 / dz)[np.newaxis, :]
    deltaZy = np.ascontiguousarray(np.reshape(deltaZy, numXY))

    deltaZz = np.ascontiguousarray(np.arange(-(nSampleZ - 1), nSampleZ, 1) * dz * qVec[2] / k0 / dz)

    # Get the weight of the summation over z2-z1
    weight = np.ascontiguousarray((nSampleZ - np.abs(np.arange(-(nSampleZ - 1), nSampleZ, 1))).astype(np.float64))

    # Move the gpu to reduce traffic
    cuDeltaZx = cuda.to_device(deltaZx)
    cuDeltaZy = cuda.to_device(deltaZy)
    cuDeltaZz = cuda.to_device(deltaZz)
    cuWeight = cuda.to_device(weight)

    # Step2, Loop through the electric field
    contrastArray = np.zeros(len(eFieldComplexFiles), dtype=np.float64)

    for eFieldIdx in range(len(eFieldComplexFiles)):
        # Load the electric field
        fileName = eFieldComplexFiles[eFieldIdx]
        eFieldComplex = np.load(fileName)

        eFieldRealFlat = np.ascontiguousarray(np.reshape(eFieldComplex.real, (numXY, nz)))
        eFieldImagFlat = np.ascontiguousarray(np.reshape(eFieldComplex.imag, (numXY, nz)))
        del eFieldComplex

        # Define gpu calculation batch
        threadsperblock = (int(16), int(16))
        blockspergrid_x = int(math.ceil(numXY / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(numXY / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        contrastLocal = np.zeros(1, dtype=np.float64)
        cuContrast = cuda.to_device(contrastLocal)
        # Update the coherence function
        getCoherenceFunctionXY_GPU_Method2[[blockspergrid, threadsperblock]](
            numXY,
            nz,
            nSampleZ,
            cuDeltaZx,
            cuDeltaZy,
            cuDeltaZz,
            cuWeight,
            eFieldRealFlat,
            eFieldImagFlat,
            cuContrast
        )

        contrastArray[eFieldIdx] = cuContrast.copy_to_host()[0]

    return contrastArray


@cuda.jit(
    'void(int64, int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:,:], float64[:,:], float64[:])')
def getCoherenceFunctionXY_GPU_Method2(nSpatial,
                                       nz,
                                       nSample,
                                       deltaZx,
                                       deltaZy,
                                       deltaZz,
                                       weight,
                                       eFieldReal,
                                       eFieldImag,
                                       contrastHolder):
    """
    We divide the reshaped time-averaged coherence function along the first dimension.

    :param nSpatial:  The length of the spatial index, which = nx * ny
    :param nz:   The ends of the summation of the mutual coherence function
    :param nSample:
    :param deltaZx:
    :param deltaZy:
    :param deltaZz:
    :param weight:
    :param eFieldReal:
    :param eFieldImag:
    :param contrastHolder:
    :return:
    """
    idx1, idx2 = cuda.grid(2)

    if (idx1 < nSpatial) & (idx2 < nSpatial):
        oldDeltaZ = 0
        oldValue = 0.0

        deltaZxy = deltaZx[idx2] + deltaZy[idx2] - (
                deltaZx[idx1] + deltaZy[idx1])  # get the contribution from the xy direction in Q*(r2-r1)/k0

        for sIdx in range(2 * nSample - 1):
            deltaZ = int(deltaZxy + deltaZz[sIdx])  # Get the contribution from the z dimension

            if abs(deltaZ) >= nz:
                continue

            # When deltaZ does not change, one does not need to calculate the time average again.
            if deltaZ == oldDeltaZ:
                # Because for the same z2 - z1, the summation is the same. One only needs to add a weight
                # Add to the contrast
                tmp = oldValue * weight[sIdx]
                cuda.atomic.add(contrastHolder, 0, tmp)
                continue

            # The delta Z determines the range over which time we calculate the average
            zStart = max(0, -deltaZ)
            zStop = min(nz, nz - deltaZ)

            # Add the temporary holder
            holderRealTmp = 0.0
            holderImagTmp = 0.0

            # Do the Time averaging
            for tIdx in range(zStart, zStop):
                holderRealTmp += eFieldReal[idx1, tIdx] * eFieldReal[idx2, tIdx + deltaZ]
                holderRealTmp += eFieldImag[idx1, tIdx] * eFieldImag[idx2, tIdx + deltaZ]

                holderImagTmp -= eFieldReal[idx1, tIdx] * eFieldImag[idx2, tIdx + deltaZ]
                holderImagTmp += eFieldImag[idx1, tIdx] * eFieldReal[idx2, tIdx + deltaZ]

            newValue = holderRealTmp ** 2 + holderImagTmp ** 2
            cuda.atomic.add(contrastHolder, 0, newValue * weight[sIdx])

            # Update the infor for the same deltaZ
            oldDeltaZ = int(deltaZ)
            oldValue = float(newValue)


#########################################################################################
#                         Method 3
#
#            Get the contrast of each pulse individually.
#########################################################################################
def getContrastMethod3(eFieldPairFiles, qVec, k0, nx, ny, nz, dx, dy, dz, nSampleZ, dSampleZ, ):
    """
    Very challenging calculation.
    Need to check with Yanwen about the definition of the calculation.

    :param eFieldPairFiles:
    :param qVec:
    :param k0:
    :param nx:
    :param ny:
    :param nz:
    :param dx:
    :param dy:
    :param dz:
    :param nSampleZ:
    :param dSampleZ:
    :return:
    """
    # Step1, prepare the variables
    numXY = nx * ny

    # The phase change according to Q/k0 * r
    deltaZx = np.zeros((nx, ny))
    deltaZx[:, :] = (np.arange(nx) * dx * qVec[0] / k0 / dz)[:, np.newaxis]
    deltaZx = np.ascontiguousarray(np.reshape(deltaZx, nx * ny))

    deltaZy = np.zeros((nx, ny))
    deltaZy[:, :] = (np.arange(ny) * dy * qVec[1] / k0 / dz)[np.newaxis, :]
    deltaZy = np.ascontiguousarray(np.reshape(deltaZy, nx * ny))

    deltaZz = np.ascontiguousarray(np.arange(nz) * dz * qVec[2] / k0 / dz)

    # Get the weight of the summation over z2-z1
    weight = np.ascontiguousarray(dSampleZ - np.abs(np.arange(-(dSampleZ - 1), dSampleZ, 1)))

    # Move the gpu to reduce traffic
    cuDeltaZx = cuda.to_device(deltaZx)
    cuDeltaZy = cuda.to_device(deltaZy)
    cuDeltaZz = cuda.to_device(deltaZz)
    cuWeight = cuda.to_device(weight)

    # Step2, Loop through the electric field
    contrastArray = np.zeros((len(eFieldPairFiles), 2), dtype=np.float64)

    for eFieldIdx in range(len(eFieldPairFiles)):
        # Load the electric field
        fileName1 = eFieldPairFiles[eFieldIdx][0]
        eFieldComplex1 = np.load(fileName1)

        eFieldRealFlat1 = np.ascontiguousarray(np.reshape(eFieldComplex1.real, (nx * ny, nz)))
        eFieldImagFlat1 = np.ascontiguousarray(np.reshape(eFieldComplex1.imag, (nx * ny, nz)))
        del eFieldComplex1

        # Load the electric field
        fileName2 = eFieldPairFiles[eFieldIdx][1]
        eFieldComplex2 = np.load(fileName2)

        eFieldRealFlat2 = np.ascontiguousarray(np.reshape(eFieldComplex2.real, (nx * ny, nz)))
        eFieldImagFlat2 = np.ascontiguousarray(np.reshape(eFieldComplex2.imag, (nx * ny, nz)))
        del eFieldComplex2

        # Define gpu calculation batch
        threadsperblock = (int(32), int(32))
        blockspergrid_x = int(math.ceil(numXY / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(numXY / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        contrastLocal = np.zeros(1, dtype=np.float64)
        cuContrast = cuda.to_device(contrastLocal)

        # Update the coherence function
        getCoherenceFunctionXY_GPU_Method2[[blockspergrid, threadsperblock]](
            numXY,
            nz,
            nSampleZ,
            cuDeltaZx,
            cuDeltaZy,
            cuDeltaZz,
            cuWeight,
            eFieldRealFlat1,
            eFieldImagFlat1,
            eFieldRealFlat2,
            eFieldImagFlat2,
            cuContrast
        )

        contrastLocal = cuContrast.copy_to_host()
        contrastArray[eFieldIdx, :] = contrastLocal[:]

    return contrastArray


@cuda.jit('void(int64, int64, int64,' +
          ' float64[:], float64[:], float64[:], float64[:], ' +
          'float64[:,:], float64[:,:], float64[:,:], float64[:,:], float64[:])')
def getCoherenceFunctionXY_GPU_Method3(nSpatial,
                                       nz,
                                       nSample,
                                       deltaZx,
                                       deltaZy,
                                       deltaZz,
                                       weight,
                                       eFieldReal1,
                                       eFieldImag1,
                                       eFieldReal2,
                                       eFieldImag2,
                                       contrast
                                       ):
    """
    We divide the reshaped time-averaged coherence function along the first dimension.

    :param nSpatial:  The length of the spatial index, which = nx * ny
    :param nz:   The ends of the summation of the mutual coherence function
    :param nSample:
    :param deltaZx:
    :param deltaZy:
    :param deltaZz:
    :param weight:
    :param eFieldReal1:
    :param eFieldImag1:
    :param eFieldReal2:
    :param eFieldImag2:
    :param contrast:
    :return:
    """
    idx1, idx2 = cuda.grid(2)

    if (idx1 < nSpatial) & (idx2 < nSpatial):
        oldDeltaZ = 0
        oldValueReal = 0.0
        oldValueImag = 0.0

        deltaZxy = deltaZx[idx2] + deltaZy[idx2] - (
                deltaZx[idx1] + deltaZy[idx1])  # get the contribution from the xy direction in Q*(r2-r1)/k0

        for sIdx in range(nSample):
            deltaZ = int(deltaZxy + deltaZz[sIdx])  # Get the contribution from the z dimension

            if abs(deltaZ) >= nz:
                continue

            # When deltaZ does not change, one does not need to calculate the time average again.
            if deltaZ == oldDeltaZ:
                # Because for the same z2 - z1, the summation is the same. One only needs to add a weight
                # Add to the contrast
                tmpReal = oldValueReal * weight[sIdx]
                tmpImag = oldValueImag * weight[sIdx]

                cuda.atomic.add(contrast, 0, tmpReal)
                cuda.atomic.add(contrast, 1, tmpImag)
                continue

            # The delta Z determines the range over which time we calculate the average
            zStart = max(0, -deltaZ)
            zStop = min(nz, nz - deltaZ)

            # Add the temporary holder
            holderRealTmp1 = 0.0
            holderImagTmp1 = 0.0

            holderRealTmp2 = 0.0
            holderImagTmp2 = 0.0

            # Do the Time averaging
            for tIdx in range(zStart, zStop):
                holderRealTmp1 += eFieldReal1[idx1, tIdx] * eFieldReal1[idx2, tIdx + deltaZ]
                holderRealTmp1 += eFieldImag1[idx1, tIdx] * eFieldImag1[idx2, tIdx + deltaZ]

                holderImagTmp1 -= eFieldReal1[idx1, tIdx] * eFieldImag1[idx2, tIdx + deltaZ]
                holderImagTmp1 += eFieldImag1[idx1, tIdx] * eFieldReal1[idx2, tIdx + deltaZ]

                holderRealTmp2 += eFieldReal2[idx1, tIdx] * eFieldReal2[idx2, tIdx + deltaZ]
                holderRealTmp2 += eFieldImag2[idx1, tIdx] * eFieldImag2[idx2, tIdx + deltaZ]

                holderImagTmp2 -= eFieldReal2[idx1, tIdx] * eFieldImag2[idx2, tIdx + deltaZ]
                holderImagTmp2 += eFieldImag2[idx1, tIdx] * eFieldReal2[idx2, tIdx + deltaZ]

            newValueReal = holderRealTmp1 * holderRealTmp2 + holderImagTmp1 * holderImagTmp2
            newValueImag = holderRealTmp1 * holderRealTmp2 - holderImagTmp1 * holderImagTmp2
            cuda.atomic.add(contrast, 0, newValueReal * weight[sIdx])
            cuda.atomic.add(contrast, 1, newValueImag * weight[sIdx])

            # Update the infor for the same deltaZ
            oldDeltaZ = int(deltaZ)
            oldValueReal = float(newValueReal)
            oldValueImag = float(newValueImag)


@cuda.jit('void(int64, int64, int64,'
          ' float64, float64, float64,'
          ' float64, float64, float64,'
          ' float64, float64, float64,' 
          'float64[:,:,:], float64[:,:,:], float64[:])')
def getCoherence_kSpace(numX, numY, numZ,
                        qx, qy, qz,
                        delta_kx, delta_ky, delta_kz,
                        d, a, aSec,
                        spectrumReal, spectrumImag, contrast):
    zIdx1, zIdx2 = cuda.grid(2)

    if (zIdx1 < numZ) & (zIdx2 < numZ):
        # get eta
        eta = float(zIdx2 - zIdx1) * delta_kz * (1 - qz)

        # Get the pre factor
        factor1 = (math.exp(- 2. * d * a) + math.exp(- 2. * d * aSec)
                   - 2 * math.exp(- d * (a + aSec)) * math.cos(d * eta))
        factor1 /= eta ** 2 + (aSec - a) ** 2

        # Get the displacement of the wave-vector
        deltaX = int(float(zIdx2 - zIdx1) * delta_kz * qx / delta_kx)
        deltaY = int(float(zIdx2 - zIdx1) * delta_kz * qy / delta_ky)

        if abs(deltaX) < numX:
            if abs(deltaY) < numY:

                # This displacement change the summation region of the spectrum
                xStart = max(0, -deltaX)
                xStop = min(numX, numX - deltaX)

                yStart = max(0, -deltaY)
                yStop = min(numX, numY - deltaY)

                # holder spectrum
                real_holder = 0
                imag_holder = 0

                # Loop through the kx, ky component of the electric field
                for yIdx in range(yStart, yStop):
                    for xIdx in range(xStart, xStop):
                        real_holder += (spectrumReal[xIdx, yIdx, zIdx1]
                                        * spectrumReal[xIdx + deltaX, yIdx + deltaY, zIdx2])
                        real_holder += (spectrumImag[xIdx, yIdx, zIdx1]
                                        * spectrumImag[xIdx + deltaX, yIdx + deltaY, zIdx2])

                        imag_holder -= (spectrumImag[xIdx, yIdx, zIdx1]
                                        * spectrumReal[xIdx + deltaX, yIdx + deltaY, zIdx2])
                        imag_holder += (spectrumReal[xIdx, yIdx, zIdx1]
                                        * spectrumImag[xIdx + deltaX, yIdx + deltaY, zIdx2])

                cuda.atomic.add(contrast, 0, (real_holder ** 2 + imag_holder ** 2) * factor1)
            else:
                pass
        else:
            pass


#########################################################################################
#            Yanwen's example code
#########################################################################################
def calc_gamma_t(nx, ny, nz, nsz, samx, samy, samz, Q_vec, eField, dt, w0):
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
                            if (delta_t >= 0) & (delta_t < nz):
                                gamma_t += np.square(np.abs(
                                    np.mean(np.conjugate(eField[i1, j1, 0:nz - delta_t]) * eField[i2, j2, delta_t:nz])))
                            elif (delta_t < 0) & (delta_t > -nz):
                                gamma_t += np.square(np.abs(
                                    np.mean(np.conjugate(eField[i1, j1, -delta_t:]) * eField[i2, j2, :nz + delta_t])))
    return gamma_t
