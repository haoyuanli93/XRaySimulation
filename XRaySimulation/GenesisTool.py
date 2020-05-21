"""
Code in this file are provided by James.

I have modified them to have the same style as my other code.
"""

import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Some constants used later
c = 299792458
h = 6.62607004e-34
echarge = 1.60217662e-19


def get_simulation_info(out_file):
    # Set variables for the result
    central_wavelength = "N/A"
    dz = "N/A"

    gridpoints = "N/A"
    dx_and_dy = "N/A"

    # Loop through the file
    with open(out_file, 'r') as file_obj:

        for line in file_obj:

            stripped_line = line.strip()
            split_line = stripped_line.split()

            if len(split_line) >= 2:
                if split_line[-1] == "meshsize":
                    dx_and_dy = float(split_line[0]) * 1e6  # Convert to um

                if split_line[-1] == "gridpoints":
                    gridpoints = int(split_line[0])

                if split_line[-1] == "wavelength":
                    central_wavelength = float(split_line[0]) * 1e6  # Convert to um

                if split_line[1] == "seperation":
                    dz = float(split_line[0]) * 1e6  # Convert to um

    print("The central wavelength is {:.2f} nm".format(central_wavelength * 1000.))
    print("The spatial resolution along the propagation direction is {:.2e} nm".format(dz * 1000.))
    print("The gridpoints number is {:}".format(gridpoints))
    print("The size of the gridpoints mesh is {:.2f} um".format(dx_and_dy))

    return central_wavelength, dz, gridpoints, dx_and_dy


def read_out(out_file):
    # Initialize variables
    zsteps = nslices = wavelength = separation = None

    # Search the file to extract information
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

    for tmp_slice in range(1, nslices):
        rows_to_skip += list(range(headline + (zsteps + 7) * tmp_slice - 6, headline + (zsteps + 7) * tmp_slice + 1))
    for tmp_slice in range(1, nslices + 1):
        rows_to_skip_curr += list(
            range(headline - 2 + (zsteps + 7) * (tmp_slice - 1), headline + (zsteps + 7) * tmp_slice - 3))

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


def get_input_param(base_in_file, param):
    """
    returns a text version of the desired parameter, or an empty string if the parameter is not found

    :param base_in_file:
    :param param:
    :return:
    """
    i_file = open(base_in_file, "r")
    out_value = False
    for line in i_file:
        if re.search(param, line):
            line = line.replace('\"', '')
            line = line.replace('\'', '')
            line = line.replace('\n', '').rstrip()
            out_value = re.split('[ =]', line)[-1].rstrip()
    i_file.close()
    return out_value


def read_field(field_file, out_file=None):
    """
    Read the output field file from genesis and convert that into a 3D complex numpy array

    :param field_file:
    :param out_file:
    :return:
    """
    # reads field output files from genesis, returns 3D array of complex field values
    raw_read = np.fromfile(field_file, dtype=float, count=-1)
    real = raw_read[0::2]
    imag = raw_read[1::2]

    if out_file is None:
        ncar = int(get_input_param(field_file[:-4], 'ncar'))
    else:
        ncar = int(get_input_param(out_file, 'ncar'))

    real = real.reshape((-1, ncar, ncar))
    imag = imag.reshape((-1, ncar, ncar))
    comp = real + np.multiply(1j, imag)
    return comp


class GenOut(object):

    def __init__(self, out_file):
        self.out_file = out_file
        self.pd_out = read_out(self.out_file)

    def plotz(self, col_str, pl=False):
        z = self.pd_out.z.xs(1, level='slice')
        y = getattr(self.pd_out, col_str).mean(level='zstep')
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

    def plot_multt(self, col_str, zsteps=None):

        if zsteps is None:
            zsteps = [1, -1]

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

    def compress(self, zsteps=None, end_of_uds=False):
        if zsteps is None:
            zsteps = [-1]

        # returns compressed version of output
        if end_of_uds:
            aw = self.pd_out.aw.xs(1, level='slice')
            zsteps = [1]
            for n in range(2, aw.size + 1):
                if aw[n] - aw[n - 1] < -1 or n == aw.size:  # and z[n]>16:
                    zsteps.append(n)
            zsteps.append(-1)
        print('compressing at z=' + str(zsteps))
        zsteps_max = self.pd_out.count(level=0).power[1]
        zsteps = [zsteps_max if x == -1 else x for x in zsteps]

        comp = [self.pd_out.mean(level='zstep'), ]
        for zstep in zsteps:
            comp.append(self.pd_out.xs(zstep, level='zstep'))
        pickle.dump(comp, open(self.out_file[0:-3] + 'pkl', "wb"))

    def delete_output(self):
        print('removing output file: ', self.out_file)
        os.remove(self.out_file)
