import math

import numpy as np
from numba import cuda


#########################################################################################
#                         Method 1
#
#     Get the time-averaged mutual coherence function first and then get the contrast
#########################################################################################

def getContrastMethod1(eFieldComplexFiles, qVec, k0, nx, ny, nz, dx, dy, dz, nSampleZ, dSampleZ,
                       out="contrast",
                       spatialBatchSize=1024,
                       workDir="./"):
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
    # Step1, prepare the variables

    # Get many batches of np.array to store the mutual coherence function
    numXY = nx * ny
    batchNum = (numXY - 1) // int(spatialBatchSize) + 1
    batchSizeS = [int(spatialBatchSize), ] * (batchNum - 1) + [numXY - (batchNum - 1) * int(spatialBatchSize), ]
    batchEnds = np.sum([0, ] + batchSizeS)

    # Create the arrays and save to the disk
    for batchIdx in range(batchNum):
        cohFunR = np.zeros((batchSizeS[batchIdx], numXY), dtype=np.float64)
        cohFunI = np.zeros((batchSizeS[batchIdx], numXY), dtype=np.float64)

        np.save(workDir + "cohFunR_{}.npy".format(batchIdx), cohFunR)
        np.save(workDir + "cohFunI_{}.npy".format(batchIdx), cohFunI)

    # The phase change according to Q/k0 * r
    deltaZx = np.zeros((nx, ny))
    deltaZx[:, :] = (np.arange(nx) * dx * qVec[0] / k0 / dz)[:, np.newaxis]
    deltaZx = np.reshape(deltaZx, nx * ny)

    deltaZy = np.zeros((nx, ny))
    deltaZy[:, :] = (np.arange(ny) * dy * qVec[1] / k0 / dz)[np.newaxis, :]
    deltaZy = np.reshape(deltaZy, nx * ny)

    deltaZz = np.arange(nz) * dz * qVec[2] / k0 / dz

    # Get the weight of the summation over z2-z1
    weight = dSampleZ - np.abs(np.arange(-(dSampleZ - 1), dSampleZ, 1))

    # Step2, Loop through the electric field
    for eFieldIdx in range(len(eFieldComplexFiles)):
        # Load the electric field
        fileName = eFieldComplexFiles[eFieldIdx]
        eFieldComplex = np.load(fileName)

        eFieldRealFlat = np.reshape(eFieldComplex.real, (nx * ny, nz))
        eFieldImagFlat = np.reshape(eFieldComplex.imag, (nx * ny, nz))
        del eFieldComplex

        # Loop through the calcluation of the mutual coherence function
        for batchIdx in range(batchNum):
            # Load the coherence function
            cohFunR = np.load(workDir + "cohFunR_{}.npy".format(batchIdx))
            cohFunI = np.load(workDir + "cohFunI_{}.npy".format(batchIdx))

            cudaCohFunR = cuda.to_device(cohFunR)
            cudaCohFunI = cuda.to_device(cohFunI)

            # Define gpu calculation batch
            threadsperblock = (16, 32)
            blockspergrid_x = math.ceil(batchSizeS[batchIdx] / threadsperblock[0])
            blockspergrid_y = math.ceil(numXY / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            # Update the coherence function
            getCoherenceFunctionXY_GPU_Method1[[blockspergrid, threadsperblock]](
                batchSizeS[batchIdx],
                numXY,
                nz,
                nSampleZ,
                np.ascontiguousarray(deltaZx[batchEnds[batchIdx]:batchEnds[batchIdx + 1]]),
                np.ascontiguousarray(deltaZy[batchEnds[batchIdx]:batchEnds[batchIdx + 1]]),
                np.ascontiguousarray(deltaZz[batchEnds[batchIdx]:batchEnds[batchIdx + 1]]),
                weight,
                np.ascontiguousarray(eFieldRealFlat[batchEnds[batchIdx]:batchEnds[batchIdx + 1], :]),
                np.ascontiguousarray(eFieldImagFlat[batchEnds[batchIdx]:batchEnds[batchIdx + 1], :]),
                cudaCohFunR,
                cudaCohFunI
            )

            # Move the array to the host
            cohFunR = cudaCohFunR.copy_to_host()
            cohFunI = cudaCohFunI.copy_to_host()

            # Save the coherence function
            np.save(workDir + "cohFunR_{}.npy".format(batchIdx), cohFunR)
            np.save(workDir + "cohFunI_{}.npy".format(batchIdx), cohFunI)

    if out == "contrast":
        contrast = 0.0
        for batchIdx in range(batchNum):
            cohFunR = np.load(workDir + "cohFunR_{}.npy".format(batchIdx))
            cohFunI = np.load(workDir + "cohFunI_{}.npy".format(batchIdx))

            contrast += np.square(np.abs(cohFunR)) + np.square(np.abs(cohFunI))
        return contrast
    else:
        print("The coherence function has been save to {}".format(workDir))


@cuda.jit('void(int64, int64, int64, int64,' +
          ' float64[:], float64[:], float64[:], float64[:], ' +
          'float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def getCoherenceFunctionXY_GPU_Method1(nSpatial1,
                                       nSpatial2,
                                       nz,
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
    We divide the reshaped time-averaged coherence function along the first dimension.

    :param nSpatial1:  The batch size
    :param nSpatial2:  The length of the spatial index, which = nx * ny
    :param nz:   The ends of the summation of the mutual coherence function
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
    if (idx1 < nSpatial1) & (idx2 < nSpatial2):
        oldDeltaZ = 0
        oldHolderRealTmp = 0.0
        oldHolderImagTmp = 0.0

        deltaZxy = deltaZx[idx2] + deltaZy[idx2] - (
                deltaZx[idx1] + deltaZy[idx1])  # get the contribution from the xy direction in Q*(r2-r1)/k0

        for sIdx in range(nSample):
            deltaZ = int(deltaZxy + deltaZz(sIdx))  # Get the contribution from the z dimension

            if abs(deltaZ) >= nz:
                continue

            # When deltaZ does not change, one does not need to calculate the time average again.
            if deltaZ == oldDeltaZ:
                # Because for the same z2 - z1, the summation is the same. One only needs to add a weight
                holderReal[idx1, idx2] += oldHolderRealTmp * weight[sIdx]
                holderImag[idx1, idx2] += oldHolderImagTmp * weight[sIdx]
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

            # Because for the same z2 - z1, the summation is the same. One only needs to add a weight
            holderReal[idx1, idx2] += holderRealTmp * weight[sIdx]
            holderImag[idx1, idx2] += holderImagTmp * weight[sIdx]

            # Update the infor for the same deltaZ
            oldDeltaZ = int(deltaZ)
            oldHolderRealTmp = float(holderRealTmp)
            oldHolderImagTmp = float(holderImagTmp)


#########################################################################################
#                         Method 2
#
#            Get the contrast of each pulse individually.
#########################################################################################
def getContrastMethod2(eFieldComplexFiles, qVec, k0, nx, ny, nz, dx, dy, dz, nSampleZ, dSampleZ, ):
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
    contrastArray = np.zeros(len(eFieldComplexFiles), dtype=np.float64)

    for eFieldIdx in range(len(eFieldComplexFiles)):
        # Load the electric field
        fileName = eFieldComplexFiles[eFieldIdx]
        eFieldComplex = np.load(fileName)

        eFieldRealFlat = np.ascontiguousarray(np.reshape(eFieldComplex.real, (nx * ny, nz)))
        eFieldImagFlat = np.ascontiguousarray(np.reshape(eFieldComplex.imag, (nx * ny, nz)))
        del eFieldComplex

        # Define gpu calculation batch
        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(numXY / threadsperblock[0])
        blockspergrid_y = math.ceil(numXY / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        # Update the coherence function
        contrastArray[eFieldIdx] = getCoherenceFunctionXY_GPU_Method2[[blockspergrid, threadsperblock]](
            numXY,
            nz,
            nSampleZ,
            cuDeltaZx,
            cuDeltaZy,
            cuDeltaZz,
            cuWeight,
            eFieldRealFlat,
            eFieldImagFlat,
        ).copy_to_host()[0]

    return contrastArray


@cuda.jit('void(int64, int64, int64,' +
          ' float64[:], float64[:], float64[:], float64[:], ' +
          'float64[:,:], float64[:,:], float64[:,:], float64[:,:])')
def getCoherenceFunctionXY_GPU_Method2(nSpatial,
                                       nz,
                                       nSample,
                                       deltaZx,
                                       deltaZy,
                                       deltaZz,
                                       weight,
                                       eFieldReal,
                                       eFieldImag, ):
    """
    We divide the reshaped time-averaged coherence function along the first dimension.

    :param nSpatial1:  The length of the spatial index, which = nx * ny
    :param nz:   The ends of the summation of the mutual coherence function
    :param nSample:
    :param deltaZx:
    :param deltaZy:
    :param deltaZz:
    :param weight:
    :param eFieldReal:
    :param eFieldImag:
    :return:
    """
    idx1, idx2 = cuda.grid(2)
    contrast = cuda.device_array(1, np.float64)

    if (idx1 < nSpatial) & (idx2 < nSpatial):
        oldDeltaZ = 0
        oldValue = 0.0

        deltaZxy = deltaZx[idx2] + deltaZy[idx2] - (
                deltaZx[idx1] + deltaZy[idx1])  # get the contribution from the xy direction in Q*(r2-r1)/k0

        for sIdx in range(nSample):
            deltaZ = int(deltaZxy + deltaZz(sIdx))  # Get the contribution from the z dimension

            if abs(deltaZ) >= nz:
                continue

            # When deltaZ does not change, one does not need to calculate the time average again.
            if deltaZ == oldDeltaZ:
                # Because for the same z2 - z1, the summation is the same. One only needs to add a weight
                # Add to the contrast
                tmp = oldValue * weight[sIdx]
                cuda.atomic.add(contrast, 0, tmp)
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
            contrast += newValue * weight[sIdx]

            # Update the infor for the same deltaZ
            oldDeltaZ = int(deltaZ)
            oldValue = float(newValue)

    return contrast


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
