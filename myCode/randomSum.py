import socket
import time

import h5py as h5
import numpy as np
import psana
from dropletCode.convert_img import *
from dropletCode.getProb import *
from dropletCode.loopdrops import *
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

hostname = socket.gethostname()
expname = 'xppc00120'

tic = time.time()

###################################################################
#     Load run, mask on each process
###################################################################
ePixNum = 1

# Define the runList
runList = [174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187]
runNum = len(runList)

# Load the time and fiducial from each run to construct the data



# general mask for droplet
mask = np.load('/reg/d/psdm/xpp/xppc00120/results/haoyuan/masks/epix_{}/droplet_roi.npy'.format(ePixNum))

# this should be the one Tomoki identified, the grand mask
roi = np.load('./output/roiMask.npy')

# values for droplets, 9.5 keV
thres = 15
aduspphot = 162
offset = 81
# This is photon separation which I do not understand what it is.
# Yanwen: this is an overkill, but to make it more general for all count rates
photpts = np.arange(
    1000000) * aduspphot - aduspphot + offset

# For all the runs initialize the data access
dsList = [psana.DataSource('exp={}:run={}:idx'.format(expname, run)) for run in runList]
detList = [psana.Detector('epix_alc{}'.format(ePixNum)) for runIdx in range(runNum)]  # need to do it for 1,2,3,4
# events = [evt for evt in ds.events()]
runList = [next(dsList[runIdx].runs()) for runIdx in range(runNum)]
timesList = [runList[runIdx].times() for runIdx in range(runNum)]

# photons numbers range to consider, from 0 to Np-1 is the probability of i photons/pixel
photonNumRange = 12

print('rank{}, initialization done'.format(rank))

###################################################################
#     Load indexes and assign jobs to each node
###################################################################

# Each node load its own index to process
with h5.File("./output/randomSumIndex.h5", 'r') as h5File:
    # Get infor for the first pattern in this batch
    p1_runList = np.array(h5File['batch_{}/p1/run'.format(rank)])
    p1_idxList = np.array(h5File['batch_{}/p1/localIdx'.format(rank)])

    # Get info for the second pattern in this batch
    p2_runList = np.array(h5File['batch_{}/p2/run'.format(rank)])
    p2_idxList = np.array(h5File['batch_{}/p2/localIdx'.format(rank)])

# Get the random pair number
pairNum = p1_idxList.shape[0]
print("There are {} patterns in batch {}".format(pairNum, rank))

# Get the holder for the photon statistics in this pattern
photonCountList = np.zeros((pairNum, photonNumRange))

###################################################################
#     Load patterns to process
###################################################################
# Loop through patterns
for pairIdx in range(pairNum):
    #print(pairIdx)
    # Get the run number and index of this pair
    run1 = p1_runList[pairIdx]
    idx1 = p1_idxList[pairIdx]

    run2 = p2_runList[pairIdx]
    idx2 = p2_idxList[pairIdx]

    if pairIdx == 0:
        # Load the first pattern
        evt1 = runList[run1].event(timesList[run1][idx1])
        img1 = detList[run1].calib(evt1)
        if img1 is None:
            print("img1 is None. Run {}, Event number: {}".format(run1, idx1))
            continue
        elif np.isnan(img1).any():
            img1 = np.nan_to_num(img1, copy=False, nan=0.0)
        else:
            pass

        img1[img1 < thres] = 0.

    # If the pattern is the same as the previous one, skip
    elif (p1_runList[pairIdx] == p1_runList[pairIdx - 1]) and (p1_idxList[pairIdx] == p1_idxList[pairIdx - 1]):
        pass
    else:  # if this pattern is not the same as the previous one
        evt1 = runList[run1].event(timesList[run1][idx1])
        img1 = detList[run1].calib(evt1)
        if img1 is None:
            print("img1 is None. Run {}, Event number: {}".format(run1, idx1))
            continue
        elif np.isnan(img1).any():
            img1 = np.nan_to_num(img1, copy=False, nan=0.0)
        else:
            pass
        img1[img1 < thres] = 0.

    # Always load the second pattern, since the second pattern is rarely the same
    evt2 = runList[run2].event(timesList[run2][idx2])
    img2 = detList[run2].calib(evt2)
    if img2 is None:
        print("img2 is None. Run {}, Event number: {}".format(run2, idx2))
        continue
    elif np.isnan(img2).any():
        img2 = np.nan_to_num(img2, copy=False, nan=0.0)
    else:
        pass

    img2[img2 < thres] = 0.

    # Get the sum pattern
    imgSum = img1 + img2

    #############################################
    # Apply the droplet algorithm   (I do not understand the meaning of each variables.)
    #############################################
    # make droplets
    ones, ts, pts, h, b = convert_img(imgSum, thres, photpts, mask=mask)
    # find photons, greedy guess
    photonlist = loopdrops(ones, ts, pts, aduspphot, photpts)
    photonlist = np.append(ones[:, [0, 2, 1]], photonlist, axis=0)
    # get probability, p has Np elements, the last one is the count rate
    p = getProb_img(photonlist, roi, Np=photonNumRange)

    #############################################
    # Save the probability infomation
    #############################################
    photonCountList[pairIdx, :] = p[:]

toc = time.time()
print("It takes {:.2f} seconds to process batch {}".format(toc - tic, rank))

# Save the photon count information
np.save("./output/photonProb_batch_{}.npy".format(rank), photonCountList)
