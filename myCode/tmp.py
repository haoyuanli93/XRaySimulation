import time

import h5py as h5
import numpy as np

## For different ratio category generate random number until we have enough patterns
batchNum = 64
batchSize = 2 ** 17  # 1e6 pattern per category
ratioEnds = np.array([[0.35, 0.36],
                      [0.40, 0.41],
                      [0.45, 0.46],
                      [0.49, 0.495],
                      [0.505, 0.51],
                      [0.54, 0.55],
                      [0.59, 0.60],
                      [0.64, 0.65], ])

ratioCatNum = ratioEnds.shape[0]
np.random.seed(19)

with h5.File("./output/randomSumIndex.h5", 'w') as h5File:
    # Loop through each batch
    for batchIdx in range(batchNum):
        ratioIdx = batchIdx // ratioCatNum

        tic = time.time()

        # Create holders to save the index and ratio information
        indexPairHolder = []
        ratioHolder = []
        kbarSumHolder = []

        # Try until we find enough patterns
        for attempt in range(1000):
            # Get a new batch
            randomPair = np.random.randint(0, ipm2AllRuns.shape[0], size=(batchSize, 2))

            # Get the ratio
            kbarsum = kbarAllRuns[randomPair[:, 0]] + kbarAllRuns[randomPair[:, 1]]
            ratioTmp = kbarAllRuns[randomPair[:, 0]] / kbarsum

            # Get the mask for pair with in the
            ratioMask = np.ones_like(ratioTmp, dtype=bool)
            ratioMask[ratioTmp > ratioEnds[ratioIdx, 1]] = False
            ratioMask[ratioTmp < ratioEnds[ratioIdx, 0]] = False

            # Get those with in the category
            indexPairHolder.append(randomPair[ratioMask])
            ratioHolder.append(ratioTmp[ratioMask])
            kbarSumHolder.append(kbarsum[ratioMask])

            # Check the number of pattern
            if np.concatenate(ratioHolder, axis=0).shape[0] > batchSize:
                indexPairHolder = np.concatenate(indexPairHolder, axis=0)[:batchSize]
                ratioHolder = np.concatenate(ratioHolder, axis=0)[:batchSize]
                kbarSumHolder = np.concatenate(kbarSumHolder, axis=0)[:batchSize]

                break
        toc = time.time()
        print("{:.2f}".format(toc - tic))

        # Convert the index into the run number and local index information
        outputInfo = np.zeros((batchSize, 4), dtype=np.int64)  # run1, idx1, run2, idx2
        outputInfo[:, 0] = runAndIdxList[indexPairHolder[:, 0]][:, 0]
        outputInfo[:, 1] = runAndIdxList[indexPairHolder[:, 0]][:, 1]
        outputInfo[:, 2] = runAndIdxList[indexPairHolder[:, 1]][:, 0]
        outputInfo[:, 3] = runAndIdxList[indexPairHolder[:, 1]][:, 1]

        # Sort the output along the O axis
        outputInfo = np.sort(outputInfo, axis=0)

        # Save the result to the output
        h5File.create_dataset("batch_{}/p1/run".format(batchIdx), data=outputInfo[:, 0])
        h5File.create_dataset("batch_{}/p1/localIdx".format(batchIdx), data=outputInfo[:, 1])
        h5File.create_dataset("batch_{}/p2/run".format(batchIdx), data=outputInfo[:, 2])
        h5File.create_dataset("batch_{}/p2/localIdx".format(batchIdx), data=outputInfo[:, 3])
        h5File.create_dataset("batch_{}/ratio".format(batchIdx), data=ratioHolder)
        h5File.create_dataset("batch_{}/kbarSum".format(batchIdx), data=kbarSumHolder)