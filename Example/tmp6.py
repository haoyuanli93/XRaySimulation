##########################################################
#
# I create this file to help Arijit to read the ADU data
#             2022-1-11
##########################################################

# Load psana. This is the most crucial module for operation
import psana
import tqdm

# Load other modules for common operations.
# Technically, these modules are not absolutely necessary.
# you can replace them with others as long as they can plot and manipulate numpy arrays
import numpy as np
import matplotlib.pyplot as plt

############################################
#    Define objects to access the XTC file
############################################
exp_name = 'xpplw3319'  # The experiment name
run_num = 325  # The run number you would like to analysis
det_name = 'zyla_0'  # The detector name you would like to access

pattern_num = 10  # Number of patterns to read


"""
If instead of the zyla or epix detector,
you would like to know the incident pulse energy,
then set
det_name = XPP-SB2-BMMON
"""

# Create the DataSource object. This object is the gate to the XTC file
ds = psana.DataSource("exp={}:run={}:smd".format(exp_name, run_num))

# Below is some variables needed for the data extraction.
# I do not understand why it is designed in this way.
runid = ds.runs().next()
det = psana.Detector(det_name)

# Get the time when each pattern is obtained
seconds = []
nanoseconds = []
fiducials = []
for nevt, evt in tqdm.tqdm(enumerate(ds.events())):
    if nevt == pattern_num:
        break
    evtId = evt.get(psana.EventId)
    seconds.append(evtId.time()[0])
    nanoseconds.append(evtId.time()[1])
    fiducials.append(evtId.fiducials())

# Read the first image and get the detector image size
et = psana.EventTime(int((seconds[0] << 32) | nanoseconds[0]), fiducials[0])
evt = runid.event(et)
img = det.calib(evt)  # This is the ADU value

det_shape = img.shape

# Create a holder to save images
image_holder = np.zeros((pattern_num,) + det_shape)

# Create the index for each pattern
idx = 0
for sec, nsec, fid in tqdm.tqdm(zip(seconds, nanoseconds, fiducials)):
    et = psana.EventTime(int((sec << 32) | nsec), fid)
    evt = runid.event(et)
    image_holder[idx] = det.calib(evt)
    idx += 1
