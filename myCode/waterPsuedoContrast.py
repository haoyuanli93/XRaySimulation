import time

import numpy as np


def assemble_image(imgs):
    """
    return the assembled image of size [2080,2080]
    with the 4 epix
    input: list of 4 epix map
    output: map of size (2080,2080) assembled
    """
    shape = [704, 768]
    edge = [170, 140]
    frame = np.zeros([2080, 2080])

    # epix2
    frame[edge[0]:shape[0] + edge[0], edge[1]:shape[1] + edge[1]] = np.rot90(imgs[1], 2)
    # epix1
    frame[edge[0]:shape[0] + edge[0], -edge[1] - shape[1]:-edge[1]] = np.rot90(imgs[0], 2)
    # epix4
    frame[-edge[0] - shape[0]:-edge[0], edge[1]:shape[1] + edge[1]] = imgs[3]
    # epix3
    frame[-edge[0] - shape[0]:-edge[0], -edge[1] - shape[1]:-edge[1]] = imgs[2]

    return frame


def reconstructImage2D(photons_x, photons_y, shape):
    nx, ny = shape
    phot_img1, _, _ = np.histogram2d(photons_y[0] + 0.5, photons_x[0] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])
    phot_img2, _, _ = np.histogram2d(photons_y[1] + 0.5, photons_x[1] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])
    phot_img3, _, _ = np.histogram2d(photons_y[2] + 0.5, photons_x[2] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])
    phot_img4, _, _ = np.histogram2d(photons_y[3] + 0.5, photons_x[3] + 0.5,
                                     bins=[np.arange(nx + 1), np.arange(ny + 1)])

    return assemble_image([phot_img1, phot_img2, phot_img3, phot_img4])


def getPhotonCount(photonX, photonY, catMap2DStack, catNum):
    # Get the pattern
    pattern_num = photonX.shape[0]
    print("There are totally {:.2e} patterns to analyze".format(pattern_num))

    # Get the photon count holder
    photonCount = np.zeros((pattern_num, catNum, 10), dtype=np.float64)
    photonSum = np.zeros((pattern_num, catNum, 10), dtype=np.float64)

    # Loop through each pattern and get the corresponding 2d image
    tic = time.time()
    for patternIdx in range(pattern_num):
        pattern_holder = reconstructImage2D(photons_x=photonX[patternIdx],
                                            photons_y=photonY[patternIdx],
                                            shape=(704, 768))

        # Loop though category num
        for catIdx in range(catNum):
            hist, bins = np.histogram(pattern_holder[catMap2DStack[catIdx]],
                                      bins=5,
                                      range=(0.5, 5.5))
            photonCount[patternIdx, catIdx, :] = hist[:]

            photonSum[patternIdx, catIdx] = np.sum(pattern_holder[catMap2DStack[catIdx]])

        # If processed 1000 patterns: print time
        if ((patternIdx + 1) % 1000) == 0:
            toc = time.time()
            print("It takes {:.2f} seconds to process 1000 patterns".format(toc - tic))
            tic = time.time()

    return photonCount, photonSum


def getPhotonCountNew(photonX, photonY, roiMask, shape):
    # parse shape
    nx, ny = shape

    # Get the pattern
    pattern_num = photonX.shape[0]
    print("There are totally {:.2e} patterns to analyze".format(pattern_num))

    photonCountLimit = 10

    # Get the photon count holder
    photonCount = np.zeros((pattern_num, photonCountLimit), dtype=np.int64)
    photonSum = np.zeros(pattern_num, dtype=np.int64)

    # Loop through each pattern and get the corresponding 2d image
    tic = time.time()
    for patternIdx in range(pattern_num):

        pattern, _, _ = np.histogram2d(photonY[patternIdx] + 0.5, photonX[patternIdx] + 0.5,
                                       bins=[np.arange(nx + 1), np.arange(ny + 1)])

        roiData = pattern[roiMask]
        hist, bins = np.histogram(roiData, bins=photonCountLimit, range=(0.5, photonCountLimit + 0.5))
        photonCount[patternIdx, :] = hist[:]

        photonSum[patternIdx] = np.sum(roiData)

        # If processed 1000 patterns: print time
        if ((patternIdx + 1) % 1000) == 0:
            toc = time.time()
            print("It takes {:.2f} seconds to process 1000 patterns".format(toc - tic))
            tic = time.time()

    return photonCount, photonSum
