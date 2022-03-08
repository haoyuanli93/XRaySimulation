#!/usr/bin/env python

'''
purpose:
- read genesis 1.3 v. 2 output files into python
'''

# Some constants used later
c = 299792458
h = 6.62607004e-34
echarge = 1.60217662e-19
# pgf_with_rc_fonts = {
#    "font.family": "serif",
#    "font.serif": [],                   # use latex default serif font
#    "font.sans-serif": ["DejaVu Sans"], # use a specific sans-serif font
# }
import matplotlib
# matplotlib.rcParams.update(pgf_with_rc_fonts)
from matplotlib.backends.backend_pgf import FigureCanvasPgf

matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import matplotlib.pyplot as plt
import numpy as np
# import sys
import pandas as pd
import glob
import re
import os
from subprocess import call
from subprocess import check_output
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import griddata
from scipy import optimize
import scipy.stats
import time
import io
import random
import pickle

try:
    import larch
    from larch_plugins.xray import *
    from larch_plugins.xray.cromer_liberman import f1f2
    from larch_plugins.xray.xraydb_plugin import xray_delta_beta

    session = larch.Interpreter()
except ImportError:
    print("larch import failed: this is ok, just means functions using larch wont work")


# import matplotlib
# matplotlib.rcParams.update({'font.family': 'serif'})

def read_out(out_file):
    # out_file = sys.argv[1]
    zsteps = nslices = wavelength = separation = None
    searchfile = open(out_file, 'r')
    pos = 0
    while zsteps is None:
        pos += 1
        line = searchfile.readline()
        if "entries per record" in line:
            zsteps = np.int(line.rsplit()[0])
            nslices = np.int(searchfile.readline().rsplit()[0])
            wavelength = np.float(searchfile.readline().rsplit()[0])
            separation = np.float(searchfile.readline().rsplit()[0])
    searchfile.close()
    print("zsteps: " + str(zsteps))
    print("nslices: " + str(nslices))
    print("wavelength (m): " + str(wavelength))
    print("slice sep (m): " + str(separation))

    headline = pos + zsteps + 17
    rows_to_skip = []
    rows_to_skip_curr = list(range(0, headline - 3))
    for slice in range(1, nslices):
        rows_to_skip += list(range(headline + (zsteps + 7) * slice - 6, headline + (zsteps + 7) * slice + 1))
    for slice in range(1, nslices + 1):
        rows_to_skip_curr += list(
            range(headline - 2 + (zsteps + 7) * (slice - 1), headline + (zsteps + 7) * (slice) - 3))
    pdout = pd.read_csv(out_file, header=headline, delim_whitespace=1, skip_blank_lines=0, skiprows=rows_to_skip,
                        dtype=np.float32, engine='c')
    zinfo = pd.read_csv(out_file, header=pos + 10, delim_whitespace=1, skip_blank_lines=0, nrows=zsteps,
                        dtype=np.float32, engine='c')
    curr = pd.read_csv(out_file, names=['current'], delim_whitespace=1, skip_blank_lines=0, skiprows=rows_to_skip_curr,
                       usecols=[0], engine='c', dtype=np.float32)
    pdout['zstep'] = nslices * list(range(1, zsteps + 1))
    pdout['z'] = nslices * list(zinfo.iloc[::, 0])
    pdout['aw'] = nslices * list(zinfo.aw)
    pdout['qfld'] = nslices * list(zinfo.qfld)

    pdout['slice'] = np.repeat(range(1, nslices + 1), zsteps)
    pdout['s'] = np.repeat(np.arange(0, nslices) * separation, zsteps)
    pdout['t'] = np.repeat(np.arange(0, nslices) * separation / c, zsteps)
    pdout['current'] = np.repeat(np.array(curr.current), zsteps)
    pdout.rename(index=str, columns={"<x>": "x", "<y>": "y"}, inplace=True)
    pdout = pdout.set_index(['slice', 'zstep'])

    return pdout
    # d.mean(level='zstep')


class GenOut(object):
    'genesis output object'

    def __init__(self, out_file):
        self.out_file = out_file
        self.pd_out = read_out(self.out_file)

    def plotz(self, col_str, pl=False):
        z = self.pd_out.z.xs(1, level='slice')
        y = getattr(self.pd_out, col_str).groupby(level='zstep').mean()
        if pl:
            plt.figure()
            plt.plot(z, y)
            plt.xlabel('z (m)')
            plt.ylabel(col_str)
            # plt.tight_layout()
            # plt.autoscale(enable=True, axis='x', tight=True)
            plt.show(block=False)
        return z, y

    def plott(self, col_str, zstep=-1, pl=False):
        if zstep < 0:
            zstep = self.pd_out.index[-1][-1] + zstep + 1
        t = self.pd_out.t.xs(zstep, level='zstep')
        y = getattr(self.pd_out, col_str).xs(zstep, level='zstep')
        if pl:
            plt.figure()
            plt.plot(1e15 * t, y)
            plt.xlabel('t (fs)')
            plt.ylabel(col_str)
            plt.title('at z = ' + str(self.pd_out.z[1, zstep]) + ' m')
            plt.tight_layout()
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.show(block=False)
        return t, y

    def plot_multt(self, col_str, zsteps=[1, -1]):
        zvals = []
        plt.figure()
        for step in zsteps:
            if step < 0:
                step = self.pd_out.index[-1][-1] + step + 1
            t, p = self.plott(col_str, step, False)
            zval = self.pd_out.z[1, step]
            zvals.append(zval)
            plt.plot(1e15 * t, p, label=str(zval))
        plt.xlabel('t (fs)')
        plt.ylabel(col_str)
        plt.legend(loc='upper left', shadow=False)
        plt.tight_layout()
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.show(block=False)

    def compress(self, zsteps=[-1], end_of_uds=False):
        # returns compressed version of output
        if end_of_uds:
            aw = self.pd_out.aw.xs(1, level='slice')
            zsteps = [1]
            for n in range(2, aw.size + 1):
                if (aw[n] - aw[n - 1] < -1 or n == aw.size):  # and z[n]>16:
                    zsteps.append(n)
            zsteps.append(-1)
        print('compressing at z=' + str(zsteps))
        zsteps_max = self.pd_out.count(level=0).power[1]
        zsteps = [zsteps_max if x == -1 else x for x in zsteps]
        comp = []
        comp.append(self.pd_out.mean(level='zstep'))
        for zstep in zsteps:
            comp.append(self.pd_out.xs(zstep, level='zstep'))
        pickle.dump(comp, open(self.out_file[0:-3] + 'pkl', "wb"))

    def delete_output(self):
        print('removing output file: ', self.out_file)
        os.remove(self.out_file)


# class GenCompressedOut(object):
#  def __init__(self, genout_object):


def compare_slice_power(first_file, second_file):
    a = read_out(first_file)
    b = read_out(second_file)
    plt.style.use('fivethirtyeight')
    fig = plt.figure()
    a2 = a.power.reshape((len(a.t), round(len(a.power) / len(a.t))))
    b2 = b.power.reshape((len(b.t), round(len(b.power) / len(b.t))))
    a3 = a2[:, -1]
    b3 = b2[:, -1]
    plt.plot(1e15 * (a.t), a3, label='first segment')
    plt.plot(1e15 * b.t, b3, label='afterburner')
    plt.xlabel('t (fs)')
    plt.ylabel('power (W)')
    plt.title('4+1, 60 uRad kick (W)')
    plt.legend(loc='upper left', shadow=False)
    # plt.tight_layout()
    plt.show(block=False)


def r56_part_file(part_file, r56, input_file, new_part_file=False):
    # in mm
    if not new_part_file:
        new_part_file = part_file + '.r'
    raw_read = np.fromfile(part_file, dtype=float, count=-1)
    gamma0 = float(get_input_param(input_file, 'gamma0'))
    nslice = int(get_input_param(input_file, 'nslice'))
    npart = int(get_input_param(input_file, 'npart'))
    lam = float(get_input_param(input_file, 'xlamds').replace('D', 'E'))
    d3 = raw_read.reshape(-1, 6, npart)
    d3[:, 1] = d3[:, 1] + r56 / 1000 * 2 * np.pi / lam * (d3[:, 0] - gamma0) / gamma0
    d3.tofile(new_part_file)
    return


def read_part_file(part_file, input_file):
    # depreciated, use read_dpa_file instead
    raw_read = np.fromfile(part_file, dtype=float, count=-1)
    print(np.shape(raw_read))
    gamma0 = float(get_input_param(input_file, 'gamma0').replace('D', 'e'))
    npart = int(get_input_param(input_file, 'npart'))
    data = raw_read.reshape(-1, 6, npart)
    return data


def read_dpa_file(out_file, dpa_file=None):
    if not dpa_file:
        dpa_file = out_file + '.dpa'
    raw_read = np.fromfile(dpa_file, dtype=float, count=-1)
    gamma0 = float(get_input_param(out_file, 'gamma0').replace('D', 'e'))
    npart = int(get_input_param(out_file, 'npart'))
    data = raw_read.reshape(-1, 6, npart)
    return data


def read_par_file(out_file):
    # assumes an out.par file exists
    # the z-dependent verison of read_part_file
    # output: data(slices, zsteps, {gamma,theta,x,y,px,py}, npart)
    # NOTE: the phase of a .par file is different from that of a dpa file, in that the phase in a .par is relative to the radiation phase, whereas .dpa is absolute. WHY DID HE DO THIS I DONT KNOW
    par_file = out_file + '.par'
    raw_read = np.fromfile(par_file, dtype=float, count=-1)
    print(np.shape(raw_read))
    gamma0 = float(get_input_param(out_file, 'gamma0').replace('D', 'e'))
    npart = int(get_input_param(out_file, 'npart'))
    nslice = int(get_input_param(out_file, 'nslice'))
    data = raw_read.reshape(nslice, -1, 6, npart)
    return data


def twoD_Gaussian(xy, amplitude, x0, y0, sigma_x, sigma_y):
    amplitude = float(amplitude)
    x0 = float(x0)
    y0 = float(y0)
    sigma_x = float(sigma_x)
    sigma_y = float(sigma_y)
    x, y = xy
    return amplitude * np.exp(- (x - x0) ** 2 / (2 * sigma_x ** 2) - (y - y0) ** 2 / (2 * sigma_y ** 2))


def fit_2d_to_gaussian(z2d, window_size=10):
    # guesses are [amp, x,y,sx,sy]
    xs = z2d.shape[0]
    x1d = np.arange(0, z2d.shape[0])
    y1d = np.arange(0, z2d.shape[1])
    x, y = np.meshgrid(x1d, y1d)
    gy, gx = np.unravel_index(np.argmax(z2d), z2d.shape)
    z2d = z2d * (np.abs(x - gx) < window_size) * (np.abs(y - gy) < window_size)
    maxz = np.max(z2d)
    # gy2 = gy
    # gy = gx
    # gx = gy2
    guesses = maxz, gx, gy, window_size, window_size
    print('guesses: ')
    print(guesses)
    print(np.max(z2d))
    popt, pcov = optimize.curve_fit(twoD_Gaussian, (x.flatten(), y.flatten()), z2d.flatten(), p0=guesses, bounds=(
    [.1 * maxz, gx - window_size, gy - window_size, 1, 1],
    [1.5 * maxz, gx + window_size, gy + window_size, 4 * window_size, 4 * window_size]),
                                    sigma=1 / np.float64(z2d.flatten()), max_nfev=1e3)
    return popt


def plot_angular_distribution(part_file, out_file, pl=False, po=1):
    # calculates the far field angular bunching spectrum from a given particle file
    # similar to far_field function
    # uses b(x,y) = exp(i*theta(x,y)) for each slice
    # assumes increasing theta towards tail of beam, I believe. See note in in call to histogram2d
    # currently doesn't handle variable current, but this could be added
    # returns 2d array and gaussian fit parameters. amplitude of the fit is the peak bunching! (if po=1)
    parts = read_part_file(part_file, out_file)
    lam = np.float(get_input_param(out_file, 'xlamds').replace('D', 'e'))
    nbins = 201
    meanx = np.mean(np.ndarray.flatten(parts[:, 2, :]))
    meany = np.mean(np.ndarray.flatten(parts[:, 3, :]))

    parts[:, 2, :] = parts[:, 2, :] - meanx
    parts[:, 3, :] = parts[:, 3, :] - meany
    mi_x = np.min(np.ndarray.flatten(parts[:, 2, :]))
    ma_x = np.max(np.ndarray.flatten(parts[:, 2, :]))
    mi_y = np.min(np.ndarray.flatten(parts[:, 3, :]))
    ma_y = np.max(np.ndarray.flatten(parts[:, 3, :]))
    freq_mult = 4
    rangexy = [[mi_x - freq_mult * (ma_x - mi_x), ma_x + freq_mult * (ma_x - mi_x)],
               [mi_y - freq_mult * (ma_y - mi_y), ma_y + freq_mult * (ma_y - mi_y)]]
    bfft = np.zeros((parts.shape[0], nbins, nbins), dtype='complex64')
    for sli in range(parts.shape[0]):
        hr, xedge, yedge = np.histogram2d(np.ndarray.flatten(parts[sli, 2, :]), np.ndarray.flatten(parts[sli, 3, :]),
                                          bins=nbins, weights=np.pi * np.pi / 2 * np.real(
                np.exp(-1j * np.ndarray.flatten(parts[sli, 1, :]))),
                                          range=rangexy)  # note: negative in exponent was required
        hi, xedge, yedge = np.histogram2d(np.ndarray.flatten(parts[sli, 2, :]), np.ndarray.flatten(parts[sli, 3, :]),
                                          bins=nbins, weights=np.pi * np.pi / 2 * np.imag(
                np.exp(-1j * np.ndarray.flatten(parts[sli, 1, :]))), range=rangexy)
        h = hr + 1j * hi
        h = h.T
        bfft[sli, :, :] = np.fft.fftshift(np.fft.fft2(h)) / nbins ** 2
    xmiddle = (xedge[1:] + xedge[:-1]) / 2
    ymiddle = (yedge[1:] + yedge[:-1]) / 2
    freqx = np.fft.fftfreq(xmiddle.shape[0], xmiddle[1] - xmiddle[0]) * lam * 1e6
    freqy = np.fft.fftfreq(ymiddle.shape[0], ymiddle[1] - ymiddle[0]) * lam * 1e6
    toplot = np.mean(np.abs(bfft) ** po, 0)
    x, y = np.unravel_index(np.argmax(toplot), toplot.shape)
    params = fit_2d_to_gaussian(toplot, 20)
    print('Fir parameters before scaling:')
    print(params)
    dfx = freqx[1] - freqx[0]
    dfy = freqy[1] - freqy[0]
    params[1] = (params[1] - (nbins - 1) / 2) * dfx
    params[2] = (params[2] - (nbins - 1) / 2) * dfy
    params[3] = params[3] * dfx
    params[4] = params[4] * dfy
    print('Fit parameters:')
    print(params)
    if pl:
        plt.figure()
        ex = (np.min(freqx), np.max(freqx), np.min(freqy), np.max(freqy))
        plt.imshow(toplot, extent=ex, origin='lower', cmap='jet', aspect=1)
        locs = np.unravel_index(np.argmax(toplot), np.shape(toplot))
        freqs = np.linspace(ex[0], ex[1], np.shape(toplot)[0])
        plt.xlabel('$\\theta_x$ ($\\mu$Rad)')
        plt.ylabel('$\\theta_y$ ($\\mu$Rad)')
        plt.title('$|b(\\theta_x,\\theta_y)|^' + str(po) + '$')
        plt.show(block=False)
    return toplot, params


def plot_dpa_density(out_file, x_axis, y_axis, t_slice, dpa_file=None, bins=60, ylim=None, xlim=None, returnarrs=False):
    # uses dpa file to plot 2d histogram of particle distribution
    if not dpa_file:
        dpa_file = out_file + '.dpa'
    dpa = read_dpa_file(out_file, dpa_file=dpa_file)
    dpa[:, 1, :] = dpa[:, 1, :] % (2 * np.pi)
    if x_axis == 'gamma':
        x_idx = 0
    elif x_axis == 'theta':
        x_idx = 1
    elif x_axis == 'x':
        x_idx = 2
    elif x_axis == 'y':
        x_idx = 3
    elif x_axis == 'px':
        x_idx = 4
    elif x_axis == 'py':
        x_idx = 5
    if y_axis == 'gamma':
        y_idx = 0
    elif y_axis == 'theta':
        y_idx = 1
    elif y_axis == 'x':
        y_idx = 2
    elif y_axis == 'y':
        y_idx = 3
    elif y_axis == 'px':
        y_idx = 4
    elif y_axis == 'py':
        y_idx = 5
    # sli = 780
    if not ylim:
        ylim = [np.min(np.min(dpa[:, y_idx, :])), np.max(np.max(dpa[:, y_idx, :]))]
    if not xlim:
        xlim = [np.min(np.min(dpa[:, x_idx, :])), np.max(np.max(dpa[:, x_idx, :]))]

    edgesy = np.linspace(ylim[0], ylim[1], bins)
    edgesx = np.linspace(xlim[0], xlim[1], bins)
    h, xedge, yedge = np.histogram2d(np.ndarray.flatten(dpa[t_slice, x_idx, :]),
                                     np.ndarray.flatten(dpa[t_slice, y_idx, :]), bins=[edgesx, edgesy])

    plt.figure()
    plt.imshow(h.T, extent=(xedge[0], xedge[-1], yedge[0], yedge[-1]), aspect='auto', origin='low')
    plt.title(dpa_file)
    plt.show(block=False)
    if returnarrs:
        return dpa[t_slice, x_idx, :], dpa[t_slice, y_idx, :]
    else:
        return


def plot_part_evolution(out_file, x_axis, y_axis, t_slice, ylim=None, xlim=None, base_name=None, remove_pngs=True):
    # makes a movie of how the y vs x evolves in time
    # assumes a .par file exists for this out_file
    # mostly useless because .par files have phase relative to the radiation phase.
    if not base_name:
        base_name = y_axis + '_vs_' + x_axis + '_s_' + str(t_slice)
    par = read_par_file(out_file)
    out = GenOut(out_file)
    if x_axis == 'gamma':
        x_idx = 0
    elif x_axis == 'theta':
        x_idx = 1
    elif x_axis == 'x':
        x_idx = 2
    elif x_axis == 'y':
        x_idx = 3
    elif x_axis == 'px':
        x_idx = 4
    elif x_axis == 'py':
        x_idx = 5
    if y_axis == 'gamma':
        y_idx = 0
    elif y_axis == 'theta':
        y_idx = 1
    elif y_axis == 'x':
        y_idx = 2
    elif y_axis == 'y':
        y_idx = 3
    elif y_axis == 'px':
        y_idx = 4
    elif y_axis == 'py':
        y_idx = 5
    zs = np.array(out.pd_out.z[1])
    par[:, :, 1, :] = par[:, :, 1, :] % (2 * np.pi)
    if not ylim:
        ylim = [np.min(np.min(par[t_slice, :, y_idx, :])), np.max(np.max(par[t_slice, :, y_idx, :]))]
    if not xlim:
        xlim = [np.min(np.min(par[t_slice, :, x_idx, :])), np.max(np.max(par[t_slice, :, x_idx, :]))]
    for zstep in range(np.size(zs)):
        plt.figure(212)
        x = par[t_slice, zstep, x_idx, :]
        y = par[t_slice, zstep, y_idx, :]
        plt.scatter(x, y, s=1)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title('z = ' + str(zs[zstep]) + ' m')
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        plt.savefig(base_name + '_{:04d}'.format(zstep) + '.png')
        plt.clf()
        print(str(zstep + 1) + ' step(s) done')
    call('ffmpeg -r 1 -i ' + base_name + '_\%04d.png ' + base_name + '.mp4', shell=True)
    if remove_pngs:
        call('rm ' + base_name + '*.png', shell=True)
    call('mplayer ' + base_name + '.mp4', shell=True)
    # plt.show(block=False)


def kick_all_particles(part_file, input_file, kick_urad_x, kick_urad_y, new_part_file):
    particles = read_part_file(part_file, input_file)
    gamma0 = float(get_input_param(input_file, 'gamma0').replace('D', 'E'))
    particles[:, 4] = particles[:, 4] + gamma0 * np.sin(kick_urad_x / 1e6)
    particles[:, 5] = particles[:, 5] + gamma0 * np.sin(kick_urad_y / 1e6)
    particles.tofile(new_part_file)
    return


def read_field(field_file, out_file=None, extent=None):
    # reads field output files from genesis, returns 3D array of complex field values
    # extent is transverse extent about the middle pixel
    # returns array with 2*extent width in transverse dims
    raw_read = np.fromfile(field_file, dtype=float, count=-1)
    real = raw_read[0::2]
    imag = raw_read[1::2]
    if not out_file:
        ncar = int(get_input_param(field_file[:-4], 'ncar'))
    else:
        ncar = int(get_input_param(out_file, 'ncar'))
    real = real.reshape((-1, ncar, ncar))
    imag = imag.reshape((-1, ncar, ncar))
    comp = real + np.multiply(1j, imag)
    if not extent:
        return comp
    else:
        return comp[:, ncar // 2 - extent:ncar // 2 + extent, ncar // 2 - extent:ncar // 2 + extent]


def read_fld(out_file):
    # reads z-dependent field output files from genesis, returns 4D array of complex field values
    raw_read = np.fromfile(out_file + '.fld', dtype=float, count=-1)
    ncar = int(get_input_param(out_file, 'ncar'))
    nslice = int(get_input_param(out_file, 'nslice'))
    # real = raw_read[0::2]
    # imag = raw_read[1::2]
    # real = real.reshape((nslice,-1,ncar,ncar))
    # imag = imag.reshape((nslice,-1,ncar,ncar))
    # comp = real + np.multiply(1j, imag)
    # comp = raw_read[0::2].reshape((nslice,-1,ncar,ncar)) + np.multiply(1j, raw_read[1::2].reshape((nslice,-1,ncar,ncar)))
    # print(np.shape(comp))
    # print(np.shape(real))
    print(np.shape(raw_read))
    print(raw_read[-1])
    print(raw_read[1])
    # return real + np.multiply(1j, imag)
    comp = raw_read.reshape((nslice, -1, ncar, ncar))
    return comp


def write_field(field, field_file):
    # takes a (a,b,c) dimension array of complex numbers, and writes to a field file. a=#slices, b,c = x,y = transverse grid
    # currently writes only dfl, not fld
    field.tofile(field_file)
    return


def sqrt_field(field_file, new_file):
    f = read_field(field_file)
    write_field(f / np.sqrt(2), new_file)
    return


def gaussian_field(input_file, fwhm_t=600e-18, sigma_x=60e-6, peak_power=1e10):
    # gaussian field with a peak power of peak_power
    # sigma t is power (because this is realistic)
    # sigma x is field (to match genesis r_size, which is std of field dist)
    sigma_t = fwhm_t / (2 * np.sqrt(2 * np.log(2)))
    ncar = float(get_input_param(input_file, 'ncar').replace('D', 'E'))
    dgrid = float(get_input_param(input_file, 'dgrid').replace('D', 'E'))
    nslice = float(get_input_param(input_file, 'nslice').replace('D', 'E'))
    zsep = float(get_input_param(input_file, 'zsep').replace('D', 'E'))
    xlamds = float(get_input_param(input_file, 'xlamds').replace('D', 'E'))
    txy = np.stack(np.mgrid[0:nslice, 0:ncar, 0:ncar], axis=3)
    sigma_t = (sigma_t / (xlamds * zsep / c)) * np.sqrt(2)
    sigma_x = sigma_x / (2 * dgrid / ncar)
    cm = [[sigma_t ** 2, 0, 0], [0, sigma_x ** 2, 0], [0, 0, sigma_x ** 2]]
    mn = [(nslice - 1) / 2, (ncar - 1) / 2, (ncar - 1) / 2]
    rv = scipy.stats.multivariate_normal(mean=mn, cov=cm)
    field = rv.pdf(txy)
    low_vals = field < np.max(field) / 1e10
    field[low_vals] = np.max(field) / 1e10
    fs = np.abs(field) ** 2
    norm = np.sum(np.abs(field[round((nslice - 1) / 2), :, :]) ** 2)
    field = np.sqrt(peak_power / norm) * field * np.exp(1j * 0)
    return field


def pad_field(field):
    # takes field from read_field and pads zero field to the end, doubling length
    return np.concatenate((field, 0 * field))


def dechirp(field_file, out_file, omegadots):
    with open(out_file) as search:
        for line in search:
            if 'wavelength' in line:
                lam = float(line.rsplit()[0])
            if 'seperation of output slices' in line:
                dt = float(line.rsplit()[0]) / c
                break
    e = read_field(field_file)
    ef = np.fft.fft(e, axis=0)
    efnew = ef * 0
    f = (np.fft.fftfreq(n=e.shape[0], d=dt)) * 2 * np.pi
    print(dt)

    overall_fwhms = []
    on_axis_fwhms = []

    for omega_dot in omegadots:
        for w in range(ef.shape[0]):
            efnew[w, :, :] = ef[w, :, :] * np.exp(-1j * omega_dot * f[w] ** 2 / 2)
        enew = np.fft.ifft(efnew, axis=0)
        pw2 = np.sum(np.sum(np.abs(enew) ** 2, 1), 1)
        overall = fwhm(np.arange(pw2.shape[0]), pw2)
        onaxis = fwhm(np.arange(pw2.shape[0]), np.abs(enew[:, 75, 75]) ** 2)
        overall_fwhms.append(overall)
        on_axis_fwhms.append(onaxis)

    plt.figure()
    a = np.angle(enew[:, 75, 75])
    b = np.angle(e[:, 75, 75])
    plt.plot(a)
    plt.plot(b)
    plt.show(block=False)

    plt.figure()
    a = np.angle(enew[:, 75, 75])
    b = np.angle(e[:, 75, 75])
    plt.plot(np.diff(unwrap_phase(a)))
    plt.plot(np.diff(unwrap_phase(b)))
    plt.show(block=False)

    plt.figure()
    a = enew[:, 75, 75]
    b = e[:, 75, 75]
    plt.plot(np.abs(a) ** 2)
    plt.plot(np.abs(b) ** 2)
    plt.show(block=False)

    return overall_fwhms, on_axis_fwhms


def get_spectrum(field_file, out_file, pl=False, start_slice=None, end_slice=None, extent=None):
    # field file is a dfl from genesis
    # returns: ft of field, frequencies in ev, field, time separation of slices
    ncar = int(get_input_param(out_file, 'ncar'))
    lam = float(get_input_param(out_file, 'xlamds').replace('D', 'E'))
    zsep = float(get_input_param(out_file, 'zsep').replace('D', 'E'))
    dt = zsep * lam / c
    e = read_field(field_file, extent=None)
    e = e[start_slice:end_slice, :, :]

    ef = np.fft.fft(e, axis=0)
    f = (np.fft.fftfreq(n=e.shape[0], d=dt) + c / lam) * h / echarge
    lams = h * c / echarge / f
    if pl:
        plt.figure()
        plt.plot(f, np.sum(np.sum(np.abs(ef) ** 2, 1), 1))
        plt.show(block=False)
    return ef, f, e, dt, lams


def compress_with_foil(field_file, out_file, start_slice=None, end_slice=None, element='Cr', thickness_um=1.0,
                       add_ev=73):
    spec, freq, efield, dt, lams = get_spectrum(field_file, out_file, False, start_slice, end_slice)
    # plt.figure()
    # plt.plot(freq, np.sum(np.sum(np.abs(spec)**2,1),1))
    # plt.show(block=False)
    freq = freq + add_ev
    specn, ekz, ref_idx = propagate_pulse_through_foil(spec, freq, element, thickness_um, 0)

    spec_sum = np.sum(np.sum(np.abs(spec) ** 2, 1), 1)
    spec_sumn = np.sum(np.sum(np.abs(specn) ** 2, 1), 1)
    efield_sum = np.sum(np.sum(np.abs(efield) ** 2, 1), 1)
    efield_new = np.fft.ifft(specn, axis=0)
    efield_new_sum = np.sum(np.sum(np.abs(efield_new) ** 2, 1), 1)

    f, axarr = plt.subplots(4, sharex=True)
    se = np.argsort(freq)
    phbefore = np.angle(spec[:, int(251 / 2), int(251 / 2)][se])
    phafter = np.angle(specn[:, int(251 / 2), int(251 / 2)][se])
    axarr[0].plot(freq[se], spec_sum[se])
    axarr[0].plot(freq[se], spec_sumn[se])
    axarr[1].plot(freq[se], np.abs(ekz ** 2)[se])
    axarr[2].plot(freq[se], 1 - np.real(ref_idx)[se])
    axarr[3].plot(freq[se], phbefore)
    axarr[3].plot(freq[se], phafter)
    axarr[0].set_ylabel('spectrum, before, after')
    axarr[1].set_ylabel('transmission')
    axarr[2].set_ylabel('delta')
    axarr[3].set_ylabel('spectral phase')
    axarr[1].set_xlabel('eV')
    plt.show(block=False)

    f, axarr = plt.subplots(3, sharex=True)
    xax = np.arange(efield_sum.size) * dt
    axarr[0].plot(xax, efield_sum)
    axarr[0].plot(xax, efield_new_sum)
    axarr[1].plot(xax, efield_sum / np.max(efield_sum))
    axarr[1].plot(xax, efield_new_sum / np.max(efield_new_sum))
    phbefore = unwrap_phase(np.angle(efield[:, int(251 / 2), int(251 / 2)]))
    phafter = unwrap_phase(np.angle(efield_new[:, int(251 / 2), int(251 / 2)]))
    axarr[2].plot(xax[1:], np.diff(phbefore) / dt)
    axarr[2].plot(xax[1:], np.diff(phafter) / dt)
    axarr[2].set_ylim((-3e16, 3e16))
    axarr[0].set_ylabel('power')
    axarr[1].set_ylabel('rel power')
    axarr[2].set_xlabel('phase der')
    plt.show(block=False)

    print('fwhm before: ' + str(fwhm(xax, efield_sum)))
    print('fwhm after : ' + str(fwhm(xax, efield_new_sum)))


def propagate_pulse_through_foil(ef, ev_ef, element='Cr', thickness_um=1.0, add_ev=0, pl=False):
    # takes input spectrum, gives output spectrum
    # get input spectrum from get_spectrum
    # uses larch for propagation

    formula = element
    thickness = thickness_um / 1e6
    try:
        den = atomic_density(formula, _larch=session)
    except:
        den = material_get(formula, _larch=session)[1]
    print('density: ' + str(den))
    electron_charge = 1.60217662e-19
    hbar = 1.0545718e-34
    c = 299792458

    energies = ev_ef + add_ev
    omegas = energies * electron_charge / hbar
    # delta = np.array([])
    # beta = np.array([])

    delta, beta, tatten = xray_delta_beta(formula, density=den, energy=energies, _larch=session)
    # for en in energies:
    #  print(en)
    #  de, be, ta = xray_delta_beta(formula,density=den,energy=en, _larch=session)
    #  delta = np.append(delta, de)
    #  beta = np.append(beta, de)
    ref_idx = 1 + delta - 1j * beta
    ekz = np.exp(-1j * ref_idx * omegas / c * thickness)
    efnew = ef * 0;
    for n in range(ef.shape[1]):
        for m in range(ef.shape[2]):
            efnew[:, n, m] = ef[:, n, m] * ekz
    if pl:
        plt.figure()
        se = np.argsort(energies)
        plt.plot(energies[se], np.abs(ekz ** 2)[se])
        plt.xlabel('eV')
        plt.ylabel('transmission (of |E|^2)')
        plt.show(block=False)

        # plt.figure()
        # plt.plot(energies[se], delta[se])
        # plt.plot(energies[se], beta[se])
        # plt.xlabel('eV')
        # plt.ylabel('delta, beta')
        # plt.show(block=False)

        plt.figure()
        plt.plot(energies[se], np.mean(np.mean(np.abs(efnew), 1), 1)[se] ** 2)
        plt.plot(energies[se], np.mean(np.mean(np.abs(ef), 1), 1)[se] ** 2)
        plt.xlabel('eV')
        plt.ylabel('new spectrum')
        plt.show(block=False)

    return efnew, ekz, ref_idx


def monochromator(field_file, out_file, mono_center_ev, mono_width_ev):
    # takes field from read_field (nominally E(x,y,t)) and transforms it into E(x,y,w)
    # the field file must be a dfl from genesis
    # applies filter, saves
    ef, f, e, dt, lams = get_spectrum(field_file, out_file, pl=False)
    plt.figure()
    plt.plot(f, np.sum(np.sum(np.abs(ef) ** 2, 1), 1))
    for i in range(len(f)):
        if abs(f[i] - mono_center_ev) > mono_width_ev / 2:
            ef[i, :, :] = 0 * ef[i, :, :]
    plt.plot(f, np.sum(np.sum(np.abs(ef) ** 2, 1), 1))
    plt.show(block=False)

    enew = np.fft.ifft(ef, axis=0)

    plt.figure()
    plt.plot(np.arange(len(f)) * dt, np.sum(np.sum(np.abs(e) ** 2, 1), 1))
    plt.plot(np.arange(len(f)) * dt, np.sum(np.sum(np.abs(enew) ** 2, 1), 1))
    x = np.arange(len(f)) * dt
    y = np.sum(np.sum(np.abs(e) ** 2, 1), 1)
    yn = np.sum(np.sum(np.abs(enew) ** 2, 1), 1)
    print('old pulse energy: ' + str(np.trapz(y, x=x)))
    print('new pulse energy: ' + str(np.trapz(yn, x=x)))
    plt.yscale('log')
    plt.gca().set_ylim(bottom=100)
    plt.show(block=False)

    write_field(enew, field_file + '.mono')

    return


def unwrap_phase(phase):
    cum_phase = 0
    new_phase = np.zeros(len(phase))
    for index in range(1, len(phase)):
        if phase[index] - phase[index - 1] > 5:
            cum_phase += -2 * np.pi
        if phase[index] - phase[index - 1] < -5:
            cum_phase += +2 * np.pi
        new_phase[index] = phase[index] + cum_phase

    return new_phase


def plot_near_field(fieldfile, out_file, pl=False, powerorefield=2):
    searchfile = open(out_file, 'r')
    pos = 0
    sep = None
    while sep is None:
        pos += 1
        line = searchfile.readline()
        if "history records" in line:
            wavelength = np.float(searchfile.readline().rsplit()[0])
            sep = np.float(searchfile.readline().rsplit()[0])
            gridpoints = np.int(searchfile.readline().rsplit()[0])
            meshsize = np.float(searchfile.readline().rsplit()[0])
    searchfile.close()
    field = read_field(fieldfile)
    power = np.power(np.abs(field), powerorefield)
    if pl:
        plt.figure()
        plt.style.use('fivethirtyeight')
        plt.imshow(power.sum(0), extent=(
        -gridpoints * meshsize / 2, gridpoints * meshsize / 2, -gridpoints * meshsize / 2, gridpoints * meshsize / 2),
                   origin='lower')
        plt.show(block=False)
    return field


def far_field(field_file, out_file, pl=False, powerorefield=2):
    # plots the far field radiation distribution
    searchfile = open(out_file, 'r')
    pos = 0
    sep = None
    while sep is None:
        pos += 1
        line = searchfile.readline()
        if "history records" in line:
            wavelength = np.float(searchfile.readline().rsplit()[0])
            sep = np.float(searchfile.readline().rsplit()[0])
            gridpoints = np.int(searchfile.readline().rsplit()[0])
            meshsize = np.float(searchfile.readline().rsplit()[0])
    searchfile.close()
    freq = np.fft.fftfreq(gridpoints, meshsize) / (1 / (wavelength)) * 1e6
    field = read_field(field_file=field_file, out_file=out_file)
    sh = np.shape(field)
    # fft_field = np.fft.fftshift(np.fft.fftn(field))
    field = np.fft.fftshift(np.fft.fftn(field))
    tp = np.power(np.abs(field), powerorefield).sum(0)
    # power = np.power(np.abs(fft_field),powerorefield)
    # tp = power.sum(0)
    max_idx = np.unravel_index(tp.argmax(), tp.shape)
    mx = (np.max(freq) - np.min(freq)) / np.shape(tp)[0] * (0.5 + max_idx[0]) + np.min(freq)
    my = (np.max(freq) - np.min(freq)) / np.shape(tp)[1] * (0.5 + max_idx[1]) + np.min(freq)
    if pl:
        plt.figure()
        plt.imshow(tp, extent=(np.min(freq), np.max(freq), np.min(freq), np.max(freq)), origin='lower', cmap='jet')
        plt.xlabel('$\\theta_x$ (uRad)')
        plt.ylabel('$\\theta_y$ (uRad)')
        plt.title('$|E(\\theta_x,\\theta_y)|^2$')
        plt.show(block=False)
        print('max x: ' + str(mx))
        print('max y: ' + str(my))
    return field, np.min(freq), np.max(freq), mx, my
    # return fft_field, np.min(freq), np.max(freq), mx, my


def propagate_field(field_file, out_file, delz_meter=87, pl=False):
    # plots the field intensity at a given point in the distance
    # differs from far_field in that it calculates the exact field at a distance, whereas far_field just performs an angular decomposition at the source and plots the angular field. These two functions should be quite similar, unless you're in the near filed. Also, propagate_field plots three different fields for comparison. See plots for details
    searchfile = open(out_file, 'r')
    pos = 0
    sep = None
    print('yes')
    while sep is None:
        pos += 1
        line = searchfile.readline()
        if "history records" in line:
            wavelength = np.float(searchfile.readline().rsplit()[0])
            dt = np.float(searchfile.readline().rsplit()[0]) / c
            gridpoints = np.int(searchfile.readline().rsplit()[0])
            meshsize = np.float(searchfile.readline().rsplit()[0])
    searchfile.close()
    theta_x = np.fft.fftfreq(gridpoints, meshsize) / (1 / (wavelength))
    nu = (np.fft.fftfreq(n=field.shape[0], d=dt) + c / lam)
    field = read_field(field_file)
    sh = np.shape(field)
    fft_field = np.fft.fftshift(np.fft.fftn(field))
    k = 2 * pi * nu / c
    kz = np.zeros(sh)
    print('yes')
    for j in range(sh[0]):
        for x in range(sh[1]):
            for y in range(sh[2]):
                kz[j, :, :] = k[j] * np.sqrt(1 - theta_x[x] ** 2 - theta_x[y] ** 2)
    field_new = np.multiply(field, np.exp(1j * kz * delz_meter))  # the field in kx ky w at the detector
    field_real = np.fft.fftshift(np.fft.fftn(field_new))  # the field in xyz at the detector
    power = np.square(np.abs(fft_field))
    power_kkw = np.square(np.abs(field_new))
    power_xyz = np.square(np.abs(field_real))
    if pl:
        plt.figure()
        plt.imshow(power.sum(0), extent=(np.min(freq), np.max(freq), np.min(freq), np.max(freq)), origin='lower')
        plt.show(block=False)
        plt.figure()
        plt.imshow(power_kkw.sum(0), extent=(np.min(freq), np.max(freq), np.min(freq), np.max(freq)), origin='lower')
        plt.show(block=False)
        plt.figure()
        plt.imshow(power_xyz.sum(0), extent=(np.min(freq), np.max(freq), np.min(freq), np.max(freq)), origin='lower')
        plt.show(block=False)
    return field_real


def plot_field_slice(field_file, out_file=None):
    comp_field = read_field(field_file, out_file)
    fig = plt.figure()
    plt.style.use('fivethirtyeight')
    power = np.square(np.abs(comp_field))
    plt.plot(power.sum(1).sum(1))
    plt.show(block=False)


def plot_power_image(field_file):
    comp_field = read_field(field_file)
    fig = plt.figure()
    plt.style.use('fivethirtyeight')
    power = np.square(np.abs(comp_field))
    plt.imshow(power.sum(0), origin='lower')
    plt.show(block=False)


def plot_field_image(field_file):
    comp_field = read_field(field_file)
    fig = plt.figure()
    plt.style.use('fivethirtyeight')
    power = np.abs(comp_field)
    plt.imshow(power.sum(0))
    plt.show(block=False)


def submit_job(file_base, ncores, queue='beamphysics'):
    # returns job number
    bjobs_out = check_output(['bjobs']).decode()
    print('submitting', flush=True)
    while 'LSF is down' in bjobs_out or 'LSF is processing' in bjobs_out:
        print('LSF is down or processing, waiting to submit job', flush=True)
        time.wait(10)
        bjobs_out = check_output(['bjobs']).decode()
    if 'bullet' in check_output('hostname').decode():
        ex = 'genesis2_mpi_1p8'
    else:
        ex = 'genesis2_mpi_1p8'
    if queue == 'beamphysics-mpi':
        bsub_out = check_output(
            ['bsub', '-x', '-a', 'mympi', '-sla', 'bpmpi', '-q', queue, '-R', 'select[hname!=bullet0075]', '-R',
             'select[hname!=bullet0062]', '-n', str(ncores), '-o', file_base + '.log', '-J', file_base, ex,
             file_base + '.in']).decode()
    else:
        bsub_out = check_output(
            ['bsub', '-x', '-a', 'mympi', '-q', queue, '-n', str(ncores), '-o', file_base + '.log', '-J', file_base, ex,
             file_base + '.in']).decode()
    job = bsub_out.rsplit()[1][1:-1]
    print('submitting job # ' + job + ' for ' + file_base + '.out', flush=True)
    return job


def kill_job(job):
    print('Attempted to kill job ' + str(job), flush=True)


def submit_and_complete_job(file_base, ncores, moveall=False, queue=False, complete=True):
    # returns the time taken to submit and finish job
    # also moves output files to beamphsyics-sims, returning them to beamphsyics after completion
    if queue:
        print('Requested queue: ' + queue)
    else:
        queue = get_better_queue(ncores)
    ts = time.time()
    tmp_base = '/nfs/slac/g/beamphysics-simu/jmacart/' + file_base + '_' + str(int(time.time()))
    # tmp_base = '/lustre/ki/pfs/beamphysics/jmacart/'+file_base+'_'+str(int(time.time()))
    check_output(['mkdir', tmp_base])
    cwd = os.getcwd()
    # check_output('ln -s '+cwd+'/'+file_base+'* '+tmp_base+'/', shell=True)

    # call('ln -s '+cwd+'/'+file_base+'* '+tmp_base+'/', shell=True)
    ff = get_input_param(file_base + '.in', 'fieldfile')
    if ff:
        call('cp -r ' + cwd + '/' + ff + ' ' + tmp_base + '/' + ff, shell=True)
    pf = get_input_param(file_base + '.in', 'partfile')
    if pf:
        call('cp -r ' + cwd + '/' + pf + ' ' + tmp_base + '/' + pf, shell=True)
    mf = get_input_param(file_base + '.in', 'maginfile')
    df = get_input_param(file_base + '.in', 'distfile')
    bf = get_input_param(file_base + '.in', 'beamfile')
    call('cp -r ' + cwd + '/' + file_base + '.in' + ' ' + tmp_base + '/' + file_base + '.in', shell=True)
    call('cp -r ' + cwd + '/' + mf + ' ' + tmp_base + '/' + mf, shell=True)
    if df:
        call('cp -r ' + cwd + '/' + df + ' ' + tmp_base + '/' + df, shell=True)
    call('cp -r ' + cwd + '/' + bf + ' ' + tmp_base + '/' + bf, shell=True)
    os.chdir(tmp_base)
    job = submit_job(file_base, ncores, queue=queue)
    bjobs_out = check_output(['bjobs', job]).decode()
    time_reset = 0

    if complete:
        while 'DONE' not in bjobs_out or 'not found' in bjobs_out:
            bjobs_out = check_output(['bjobs', job]).decode()
            time.sleep(2)
            time_reset += 2
            if time_reset > 240 and bjobs_out == '':
                kill_job(job)  # jut in case it isn't dead
                time.sleep(30)
                job = submit_job(file_base, ncores, queue=queue)
                time_reset = 0
            try:
                out_size = os.path.getsize(file_base + '.out') / 2 ** 20
            except:
                out_size = 0
            if 'Done' in bjobs_out and out_size < .5:
                # output file is too small or doesn't exist
                kill_job(job)  # jut in case it isn't dead
                time.sleep(30)
                job = submit_job(file_base, ncores, queue=queue)
                time_reset = 0
                bjobs_out = 'reset'
        os.chdir(cwd)
        if moveall:
            move_all_files(tmp_base, cwd)
        else:
            print('copying out and log files', flush=True)
            check_output(['cp', '-f', tmp_base + '/' + file_base + '.out', '.'])
            check_output(['cp', '-f', tmp_base + '/' + file_base + '.log', '.'])
            print('removing files from: ' + tmp_base, flush=True)
            check_output(['rm', '-rf', tmp_base])
        st = time.time()
        return st - ts
    else:
        while 'not found' in bjobs_out:
            bjobs_out = check_output(['bjobs', job]).decode()
            time.sleep(1)
            time_reset += 1
            if time_reset > 240 and bjobs_out == '':
                kill_job(job)  # jut in case it isn't dead
                time.sleep(30)
                job = submit_job(file_base, ncores, queue=queue)
                time_reset = 0
        os.chdir(cwd)
        return [job, tmp_base]


def move_all_files(source_dir, target_dir):
    check_output('cp -f ' + source_dir + '/* ' + target_dir, shell=True)


def submit_and_wait_for_pend(file_base, ncores, queue='beamphysics'):
    if queue == 'beamphysics':
        tmp_base = '/nfs/slac/g/beamphysics-simu/jmacart/' + file_base + '_' + str(int(time.time()))
    else:
        tmp_base = '/lustre/ki/pfs/beamphysics/jmacart/' + file_base + '_' + str(int(time.time()))
    check_output(['mkdir', tmp_base])
    cwd = os.getcwd()
    bf = get_input_param(file_base + '.in', 'fieldfile')
    if bf:
        call('cp -r ' + cwd + '/' + bf + ' ' + tmp_base + '/' + bf, shell=True)
    pf = get_input_param(file_base + '.in', 'partfile')
    if pf:
        call('cp -r ' + cwd + '/' + pf + ' ' + tmp_base + '/' + pf, shell=True)
    mf = get_input_param(file_base + '.in', 'maginfile')
    df = get_input_param(file_base + '.in', 'distfile')
    if df:
        call('cp -r ' + cwd + '/' + df + ' ' + tmp_base + '/' + df, shell=True)
    bf = get_input_param(file_base + '.in', 'beamfile')
    call('cp -r ' + cwd + '/' + file_base + '.in' + ' ' + tmp_base + '/' + file_base + '.in', shell=True)
    call('cp -r ' + cwd + '/' + mf + ' ' + tmp_base + '/' + mf, shell=True)
    call('cp -r ' + cwd + '/' + bf + ' ' + tmp_base + '/' + bf, shell=True)
    os.chdir(tmp_base)
    job = submit_job(file_base, ncores, queue=queue)
    os.chdir(cwd)
    bjobs_out = check_output(['bjobs', job]).decode()
    time_reset = 0
    while 'not found' in bjobs_out:
        bjobs_out = check_output(['bjobs', job]).decode()
        time.sleep(1)
        time_reset += 1
        if time_reset > 240 and bjobs_out == '':
            kill_job(job)  # jut in case it isn't dead
            time.sleep(30)
            job = submit_job(file_base, ncores, queue=queue)
            time_reset = 0
    return [job, tmp_base]


def check_if_job_finished(job):
    bjobs_out = check_output(['bjobs', job]).decode()
    if 'DONE' in bjobs_out:
        return 1
    elif 'RUN' in bjobs_out:
        return 0
    elif 'PEND' in bjobs_out:
        return 0
    else:
        print(bjobs_out)
        print('job appears to have died, returning -1 to calling function')
        print('job in question: ' + str(job), flush=True)
        return -1
    # bjobs_out = check_output(['bjobs', job]).decode()
    # t0 = time.time()
    # while 'DONE' not in bjobs_out:
    #  time.sleep(2)
    #  bjobs_out = check_output(['bjobs', job]).decode()
    #  t = time.time()
    #  if t-t0 > 0.5*60*60 and 'is not found' in bjobs_out:
    #    print('Jobs numbering may have reset, returning',flush=True)
    #    return


def get_empty_cores(host):
    while True:
        try:
            # parse = check_output('bhosts | grep '+host+' | grep ok',shell=True).decode()
            bhosts = check_output('bhosts', shell=True).decode().splitlines()
            break
        except:
            time.sleep(10)
            print('LSF may be down, host info not available: ' + host, flush=True)
    empty = 0
    for bhost in bhosts:
        if host in bhost.rsplit()[0]:
            empty += int(bhost.rsplit()[3]) - int(bhost.rsplit()[4])
    # d = np.loadtxt(io.StringIO(parse), dtype=int, usecols=(3,4))
    return empty  # np.sum(d[:,0]-d[:,1])


def get_better_queue(ncores=16):
    # return best queue
    # or, put a file in directory called queue.txt with the line:
    # use this queue: beamphysics-mpi
    try:
        queue = np.loadtxt('queue.txt', dtype=str)[-1][2:-1]
        print('using queue.txt to submit job to queue: ' + queue, flush=True)
        if not call('bqueues ' + queue, shell=True):
            return queue
        else:
            print('queue requeted in queue.txt not available, proceeding', flush=True)
    except:
        print('getting queue info, will use best queue', flush=True)
    oak = get_empty_cores('oak')

    bul = get_empty_cores('bullet')
    while True:
        try:
            oakpend = int(check_output('bqueues beamphysics', shell=True).decode().rsplit()[-3])
            break
        except:
            oakpend = 0
            break
    while True:
        try:
            bminjobs = int(check_output('bqueues beamphysics-mpi', shell=True).decode().rsplit()[-4])
            break
        except:
            bminjobs = 0
            break
    while True:
        try:
            bulpend = int(check_output('bqueues bulletmpi', shell=True).decode().rsplit()[-3])
            break
        except:
            bulpend = 0
            break
    tot_oak = oak - oakpend
    tot_bpi = 257 - bminjobs
    tot_bul = bul - bulpend
    print("tot oak: " + str(tot_oak), flush=True)
    print("tot bpi: " + str(tot_bpi), flush=True)
    print("tot bul: " + str(tot_bul), flush=True)
    if tot_oak >= ncores and tot_bpi >= ncores:
        print('submitting random', flush=True)
        # rc = random.choice(['beamphysics','beamphysics-mpi'])
        rc = 'beamphysics-mpi'
        print(rc)
        return rc
    elif tot_oak >= ncores:
        print('submitting oak', flush=True)
        return 'beamphysics'
    elif tot_bpi >= ncores:
        print('submitting beamphysics-mpi', flush=True)
        return 'beamphysics-mpi'
    elif tot_bul >= ncores:
        print('submitting beamphysics-mpi', flush=True)
        return 'beamphysics-mpi'
    elif tot_oak > tot_bpi:
        print('submitting oak', flush=True)
        return 'beamphysics'
    else:
        print('submitting beamphysics-mpi', flush=True)
        return 'beamphysics-mpi'
        # if oakpend < bulpend:
        #  return 'beamphysics'
        # else:
        #  return 'beamphysics-mpi'


def remove_previous(base_file):
    # removes old in, out, dfl, dpa, dfl, log, lat, etc files
    files = glob.glob(base_file + "*")
    for file in files:
        call(['rm', file])


def remove_big_files(base_file):
    # removes old out, dfl, dpa, par files
    files = glob.glob(base_file + "*par")
    for file in files:
        call(['rm', file])


def set_input_param(base_in_file, param, value):
    # sets param to value in base_in_file
    if not value:
        call('sed -i /' + param + '/d ' + base_in_file, shell=True)
    if isinstance(value, str):
        value = '\"' + value + '\"'
    if get_input_param(base_in_file, param) == False:
        call('sed -i \'$i\ ' + param + ' = ' + str(value) + '\' ' + base_in_file, shell=True)
    else:
        call(['sed', '-i', '/' + param + '/c\ ' + param + ' = ' + str(value), base_in_file])
    # call(['sed', '-i', '104s/.*/ maginfile = "'+latf+'"/', new_in_file_f+'.in'])


def get_input_param(base_in_file, param):
    # returns a text version of the desired parameter, or an empty string if the parameter is not found
    i_file = open(base_in_file, "r")
    out_value = False
    for line in i_file:
        if re.search(param, line):
            line = line.replace('\"', '')
            line = line.replace('\'', '')
            line = line.replace('\n', '').rstrip()
            out_value = re.split(' |=', line)[-1].rstrip()
    i_file.close()
    return out_value


def split_lattice(base_lat_file, in_file, n1, n2, t):
    # makes new files, called base_lat_file_f.lat, _s, _t
    # or, if n2 = 0, copies the first section to in_file.lat
    d = 0
    cx = 0.1 * t / 2.2847767
    f_lat = in_file + "_f.lat"
    s_lat = in_file + "_s.lat"
    call(['cp', base_lat_file, f_lat])
    call(['cp', base_lat_file, s_lat])
    call(['sed', '-i', str(8 + n1 - d) + ',44s/2.4749/0.0000/', f_lat])
    call(['sed', '-i', str(8 + n1) + ',44d;' + str(n1 * 4 + 49 - 4) + ',191d;' + str(n1 * 2 + 196 - 2) + ',266d;' + str(
        n1 * 2 + 271 - 2) + ',341d', f_lat])
    if n2 == 0:
        call(['rm', s_lat])
        call(['mv', f_lat, in_file[:-3] + '.lat'])
    else:
        call(['sed', '-i', '8,' + str(8 + n1 - 1) + 'd;' + '48,' + str(n1 * 4 + 49 - 4 - 1) + 'd;' + '195,' + str(
            n1 * 2 + 196 - 2 - 1) + 'd;' + str(8 + n1 + n2) + ',44d;' + str((n1 + n2) * 4 + 49 - 4) + ',191d;' + str(
            (n1 + n2) * 2 + 196 - 2) + ',266d;' + '270,' + str(n1 * 2 + 271 - 2 - 1) + 'd;' + str(
            (n1 + n2) * 2 + 271 - 2) + ',341d', s_lat])
        call(['sed', '-i', '0,/CX  0.0       5.0/{s/CX  0.0       5.0/' + 'CX  ' + str(cx) + '       5.0/}', s_lat])

    return f_lat, s_lat


def lcls_lattice(new_lat_file='test.lat', nu=36, start_u=1, zero_drifts=True, awo=2.4749, daw=0.0, start_taper=1):
    # creates an LCLS-like lattice of nu undulators, starting at undulator
    # start_taper of 1 means und 1 is at the start value, usually 2.4749. subsequent undulators are higher in K
    # start taper is in absolute undulator number
    if start_taper == 0:
        start_taper = 1
    h = '# BasicLCLS Lattice  every third section, there is a long break\n' + \
        '? VERSION = 1.0\n' + \
        '? UNITLENGTH = 3.000000D-02\n' + \
        '# Total undulator length is    m with this lattice\n' + \
        '#\n' + \
        '#       Undulator Field\n' + \
        '#\n'
    h2 = '#\n' + \
         '#       Quadrupole Field\n' + \
         '#\n'
    h3 = '#\n' + \
         '#       Section Gap AD\n' + \
         '#\n'
    a = ''
    q = ''
    adr = ''
    for n in range(nu):
        und_num = n + start_u
        und_len = 110.0

        if n == 0:
            pre_drift = 0.0
        elif und_num % 3 == 1:
            pre_drift = 30.0
        else:
            pre_drift = 20.0
        # if und_num == 9 or und_num==16:
        if und_num > start_taper:
            aw = awo + (und_num - start_taper) * daw
        else:
            aw = awo

        a = a + 'AW' + fmt(aw) + fmt(110) + fmt(pre_drift) + '\n'
        if und_num % 2 == 0:
            quad = -9.64
        else:
            quad = 9.84

        if pre_drift > 0:
            q = q + 'QF' + fmt(0) + fmt(0.0) + fmt((pre_drift - 10) / 2) + '\n'
            q = q + 'QF' + fmt(quad) + fmt(10.0) + fmt(0.0) + '\n'
            q = q + 'QF' + fmt(0) + fmt(0.0) + fmt((pre_drift - 10) / 2) + '\n'
            adr = adr + 'AD' + fmt(0.65194) + fmt(pre_drift) + fmt(0.0) + '\n'
        adr = adr + 'AD' + fmt(0.65194) + fmt(0) + fmt(110) + '\n'
        q = q + 'QF' + fmt(0) + fmt(0.0) + fmt(110.0) + '\n'

    if quad == -9.64:
        quad = 9.84
    else:
        quad = -9.64
    if (und_num + 1) % 3 == 1:
        pre_drift = 30.0
    else:
        pre_drift = 20.0
    a = a + 'AW' + fmt(0.0) + fmt(0) + fmt(pre_drift) + '\n'
    q = q + 'QF' + fmt(0) + fmt(0) + fmt((pre_drift - 10) / 2) + '\n'
    q = q + 'QF' + fmt(quad) + fmt(10.0) + fmt(0) + '\n'
    q = q + 'QF' + fmt(0) + fmt(0) + fmt((pre_drift - 10) / 2) + '\n'
    adr = adr + 'AD' + fmt(0.65194) + fmt(pre_drift) + fmt(0.0) + '\n'
    with open(new_lat_file, "w") as text_file:
        text_file.write(h + a + h2 + q + h3 + adr)


def genesis_informed_taper(outfile, suppress_output=True, trap_phase_deg=20.0, lin_phase=False):
    # takes an output file, returns a recommended taper and z
    o = GenOut(outfile)
    z, pmid = o.plotz('p_mid', False)
    ncar = float(get_input_param(outfile, 'ncar'))
    dgri = float(get_input_param(outfile, 'dgrid').replace('D', 'E'))
    delz = float(get_input_param(outfile, 'delz').replace('D', 'E'))
    gamma0 = float(get_input_param(outfile, 'gamma0').replace('D', 'E'))
    mesh_size = 2 * dgri / (ncar - 1)  # meters
    z, aw = o.plotz('aw', False)
    z, power = o.plotz('power', False)
    z, bunching = o.plotz('bunching', False)
    if not suppress_output:
        print('ending power = {:.2f} GW'.format(1e-9 * power[len(power)]))
        print('ending bunching = {:.6f}'.format(bunching[len(bunching)]))
        print('the taper was')
        print(np.array(aw))
    K0 = np.max(aw) * np.sqrt(2)
    z0 = 376.73
    eomc2 = 1.95695118e-6
    jj = scipy.special.jv(0, K0 ** 2 / (4 + 2 * K0 ** 2)) - scipy.special.jv(1, K0 ** 2 / (4 + 2 * K0 ** 2))
    field = np.sqrt(2 * z0 * pmid / mesh_size ** 2.0)
    trap_phase = trap_phase_deg * np.pi / 180.
    if lin_phase:  # makes linear phase from zero to trap_phase
        trap_phase = trap_phase * np.linspace(0, 1, len(z))
    dKdz = -2 * eomc2 * jj * field * (1 + .5 * K0 ** 2) / (2 * gamma0 ** 2.0) * np.sin(trap_phase)
    Kofz = np.cumsum(np.array(dKdz)) * (z[2] - z[1]) + K0
    awofz = Kofz / np.sqrt(2)
    return z, awofz


def lclsII_HXR_lattice(new_lat_file='test.lat', nu=32, zero_drifts=True, aw_every_period=None, delzmax=1,
                       qsign_start=1.0, ad=None):
    # creates an LCLSIIHXR-like lattice of nu undulators, starting at undulator
    # aw_every_period is a list of aw for each period * delz
    # ad is calculated to optimize phase matching in drift with minimal r56
    # aw in the drift is ignored
    # must set delzmax to be equal to or larger than delz.
    # right now delzmax should be 1 or 5 since the quad is 5 periods long
    h = '# BasicLCLS Lattice  every third section, there is a long break\n' + \
        '? VERSION = 1.0\n' + \
        '? UNITLENGTH = 2.600000D-02\n'
    nper = (nu * 130 + 25 * nu) // delzmax
    if aw_every_period is None:
        aw_every_period = 2.44 / np.sqrt(2.0) * np.ones(nper)
    aw_current = aw_every_period[0]
    out_text = h
    one_per = (130 + 25) // delzmax
    ulen = 130 // delzmax

    for n in range(nper):
        if n % (one_per) < ulen:
            uline = 'AW   {:.6f}   {:.0f}   0\n'.format(aw_every_period[n], delzmax)
            adline = 'AD   0.0   0   {:.0f}\n'.format(delzmax)
            qfline = 'QF   0.0   0   {:.0f}\n'.format(delzmax)
            aw_current = aw_every_period[n]
        else:
            uline = 'AW   0.0   0   {:.0f}\n'.format(delzmax)
            if not ad:
                mult_fact = np.ceil(25. / (1 + aw_current ** 2.0))
                ad = np.sqrt(-1 + mult_fact * (1 + aw_current ** 2.0) / 25.)

            adline = 'AD   {:.6f}   {:.0f}   0\n'.format(ad, delzmax)
            if (n % one_per >= ulen + 10 // delzmax) and (n % one_per < ulen + 15 // delzmax):
                qsign = n // (one_per) % 2
                if qsign == 0:
                    qfline = 'QF   {:.6f}   {:.0f}   0\n'.format(-23.077 * qsign_start, delzmax)
                else:
                    qfline = 'QF   {:.6f}   {:.0f}   0\n'.format(23.077 * qsign_start, delzmax)
                qsign = -qsign
            else:
                qfline = 'QF   0.0   0   {:.0f}\n'.format(delzmax)
        out_text = out_text + uline + adline + qfline

    with open(new_lat_file, "w") as text_file:
        text_file.write(out_text)


def fmt(num):
    return '{:10.4f}'.format(num)


def change_aw_value(lat_file, und_number, new_aw_value):
    read = np.genfromtxt(lat_file, dtype=str, delimiter='\n', comments='&!2')
    ct = 0
    for n in range(read.size):
        if 'AW' in read[n]:
            ct += 1
            if ct == und_number:
                linesp = read[n].rsplit()
                linesp[1] = str(new_aw_value)
                read[n] = '    '.join(linesp)
    np.savetxt(lat_file, read, fmt='%s')


def make_taper_lattice(base_lat, new_lat_file_name, daw, start_taper=0, awo=2.4749):
    # makes a new Lattice file
    if base_lat == new_lat_file_name:
        print('overwriting lattice file with new taper')
    else:
        call(['cp', base_lat, new_lat_file_name])
    for k in range(8 + start_taper, 44):
        old = check_output(['sed', str(k) + 'q;d', base_lat]).decode()
        new = old.replace('2.4749', '{0:.4f}'.format(round(awo + daw * (k - 8 - start_taper), 4)))
        call(['sed', '-i', str(k) + 's/.*/' + new[0:-2] + '/', new_lat_file_name])


def fwhm(x, y):
    spline = UnivariateSpline(x, y - np.max(y) / 2.0, s=0)
    r = spline.roots()  # find the roots
    if r.size > 2:
        print('error: more than 2 roots in fwhm function')
        print('returning farthest roots')
        print(r)
    elif r.size == 1:
        print('error: only one root found')
        print('returning large value for fwhm')
        return x[-1] - x[0]
    return r[-1] - r[0]


def track_contour(fn='xleap_optimize.track', x='amp', y='r56', z='meanfw', bin1='daw', bin1_range=[0, .005],
                  bin2='curr', bin2_range=[1000, 5000], pp=False, topfew=8, title='', plotTrue=True):
    track = pd.read_csv(fn, delim_whitespace=1)
    track.wt50 = track[track.columns[
        track.columns.to_series().str.contains('^w')]]  # np.mean(np.sort(np.array(track.ix[:,8:8+16]))[:,0:0+topfew],1)
    track.wt50 = np.median(np.sort(np.array(track.wt50))[:, 0:topfew], 1)
    track.pt50 = track[track.columns[track.columns.to_series().str.contains(
        '^p')]]  # np.mean(np.sort(np.array(track.ix[:,8+16:8+16+16]))[:,0:0+topfew],1)
    track.pt50 = np.median(np.sort(np.array(track.pt50))[:, -topfew:], 1)
    # print(track)
    # return track
    dx = np.array(getattr(track, x))
    dy = np.array(getattr(track, y))
    dz = np.array(getattr(track, z))
    lab = ''
    if x == 'amp':
        dx = dx * .511
        x = 'modulation amplitude (MeV)'
    if y == 'amp':
        dy = dy * .511
        y = 'modulation amplitude (MeV)'
    if z == 'amp':
        dz = dz * .511
        z = 'modulation amplitude (MeV)'
    if x == 'r56':
        x = '$R_{56}$ (mm)'
    if y == 'r56':
        y = '$R_{56}$ (mm)'
    if z == 'r56':
        z = '$R_{56}$ (mm)'
    if z == 'meanfw':
        dz = dz * 1e15
        lab = 'median duration (fs)'
    if z == 'meanpow':
        dz = dz * 1e-9
        lab = 'median power (GW)'
    if y == 'daw':
        dy = dy * 6 / 2.4749
        y = '$\Delta a_u / a_u$'
    b1 = np.array(getattr(track, bin1))
    b2 = np.array(getattr(track, bin2))
    b1i = (b1 > bin1_range[0]) & (b1 < bin1_range[1]) & (b2 > bin2_range[0]) & (b2 < bin2_range[1])
    dx = dx[b1i]
    dy = dy[b1i]
    dz = dz[b1i]
    gridx, gridy = np.mgrid[np.min(dx):np.max(dx):(np.max(dx) - np.min(dx)) / 1000,
                   np.min(dy):np.max(dy):(np.max(dy) - np.min(dy)) / 1000]
    gd = griddata((dx, dy), dz, (gridx, gridy), method='nearest', rescale=True)
    if plotTrue:
        fig = plt.figure()
        if z == 'meanfw' or z == 'wt50':
            plt.imshow(gd.T, extent=(np.min(dx), np.max(dx), np.min(dy), np.max(dy)), origin='lower', aspect='auto')
        else:
            plt.imshow(gd.T, extent=(np.min(dx), np.max(dx), np.min(dy), np.max(dy)), origin='lower', aspect='auto')
        if pp:
            plt.plot(dx, dy, 'k.', markersize=2)
        if x == 'modulation amplitude (MeV)' and y == '$R_{56}$ (mm)':
            # plot expected location for good simulations to exist (i.e. produce peak current)
            x_arr = np.linspace(np.min(dx), np.max(dx), 200)
            dy_expected = float(get_input_param('base.in', 'gamma0')) / (x_arr / .511) * 2e-6 / (2 * np.pi) * 1e3
            plt.plot(x_arr, dy_expected, 'k-')
        plt.xlim(np.min(dx), np.max(dx))
        plt.ylim(np.min(dy), np.max(dy))
        plt.xlabel(x)
        plt.ylabel(y)
        # plt.title(z+' with '+bin1+' in range '+str(bin1_range)+', and\n '+bin2+' in range'+str(bin2_range))
        cbar = plt.colorbar(label=lab)
        plt.tight_layout()
        plt.set_cmap('jet')
        if y == '$\Delta a_u / a_u$':
            ax = plt.gca()
            labels = [item.get_text() for item in ax.get_yticklabels()]
            labels[1] = str(-float(labels[5]))
            labels[2] = str(-float(labels[4]))
            # print(labels[2])
            ax.set_yticklabels(labels)
            # plt.title(title)
        plt.show(block=False)
    return gd.T, dx, dy
