import numpy as np
import json
import matplotlib.pyplot as plt


# Function to load a dictionary that has all the run numbers documented
def load_runs(filename):
    with open(filename) as f:
        runs = json.loads(f.read())
    return runs


# Filter the shots by the intensity threshold
def filterIntensity(inten, low_thres=0.05, high_thres=0.25):
    # Get the values that lie within the percentile specified
    # Ilow = np.nanpercentile(inten, 100-percentile)
    # Ihigh = np.nanpercentile(inten, percentile)
    indices = np.where(np.logical_and(inten <= high_thres, inten >= low_thres))[0]

    # Return the indices that satisfy the constraint
    return indices


# Get the mask according to the run number
def getMask(run_num):
    if run_num < 1283787:
        mask = np.load('mask/det_general_mask.npy')
    else:
        mask = np.load('mask/det_general_mask_May_31_afternoon.npy')

    return mask


# Get the center based on the run number
def getCenter(run_num):
    if run_num < 1283787:
        center = [714 - 630, 1245]
    else:
        center = [425, 1245]

    return center


# Get the dark run based on time of day
def getDark(dark_run=1283392):
    return np.load('dark/run_{}.npz'.format(dark_run))['dark'][:]


# Get the ROI for the mask
def getROI(shape, center, rmin=0, rmax=2000):
    x, y = np.indices((shape[0], shape[1]))
    r = np.hypot(x - center[1], y - center[0])
    mask = x * 0
    mask[(r > rmin) & (r < rmax)] = 1
    return mask


# Bin the intensity values based on the range specified and the number of bins
# Return the indices of the bins for each value in the array
def binIntensity(inten, range_min=0, range_max=0.3, nb_bins=4):
    h, b = np.histogram(inten, range=(range_min, range_max), bins=nb_bins)
    plt.hist(inten, range=(range_min, range_max), bins=nb_bins)
    pos = np.digitize(inten, b)

    # This only returns values from 1 to nb_bins
    return pos


def getPumpProbeBins(Ipump, Iprobe, Itot, bin_pump, bin_probe, nb_bins):
    i0s = np.zeros((nb_bins, nb_bins, 3))

    for i in range(1, nb_bins + 1):
        for j in range(1, nb_bins + 1):
            w = np.where((bin_pump == i) & (bin_probe == j))[0]
            if w.size > 0:
                i0s[i - 1, j - 1, 0] = np.nanmean(Itot[w])
                i0s[i - 1, j - 1, 1] = np.nanmean(Ipump[w])
                i0s[i - 1, j - 1, 2] = np.nanmean(Iprobe[w])

    return i0s


def normalizeImagesBins(raw_imgs, count_imgs, Inorm, bin_pump, bin_probe, nb_bins):
    for i in range(1, nb_bins + 1):
        for j in range(1, nb_bins + 1):
            w = np.where((bin_pump == i) & (bin_probe == j))[0]
            # Normalize by the total, averaged intensity in each bin
            if w.size > 0:
                raw_imgs[i - 1, j - 1] /= (Inorm[i - 1, j - 1, 0] * count_imgs[i - 1, j - 1])

    return raw_imgs


def get_radial_average_short_distance(img, shape, center, mask, nrs=120, E=10, L=0.13, pixelsize=50e-6):
    lam = 12.398 / E
    k = 2 * np.pi / lam
    det_angle = np.deg2rad(7)
    y, x = np.indices(shape)
    x = x - center[0]
    y = y - center[1]
    ll = np.sqrt(
        (pixelsize * x * np.cos(det_angle)) ** 2 + (pixelsize * y) ** 2 + (L - x * pixelsize * np.sin(det_angle)) ** 2)
    tth = np.arccos((L - x * pixelsize * np.sin(det_angle)) / ll)
    qmap = 2 * k * np.sin(tth / 2.)
    qmin = 0
    qmax = qmap[mask > 0].max()
    qs = np.linspace(qmin, qmax, nrs + 1)
    q_values = (qs[1:] + qs[:-1]) / 2.
    ra = np.zeros((nrs))
    for i in range(nrs):
        ra[i] = np.nanmean(img[(mask > 0) & (qmap >= qs[i]) & (qmap < qs[i + 1])])
    return q_values, ra
