import sys
from psmon import publish
from psmon.plots import XYPlot,Image, MultiPlot
import numpy as np
import collections
import time
from mpidata import mpidata
from utilities import *


# user parameters
updaterate = 1 # plot-push frequency, measured in "client updates"
Npts = 1000
def runmaster(nClients):
    while (1):
        print('**** New Run ****')
        nClientsInRun = nClients
        myplotter = Plotter()
        while nClientsInRun > 0:
            md = mpidata()
            md.recv()
            if md.small.endrun:
                nClientsInRun -= 1
            else:
                myplotter.update(md)

class Plotter:
    def __init__(self):
        self.nupdate=0
        self.lasttime = None
    def update(self,md):
        self.nupdate+=1
        self.th1s = md.th1s
        self.i0s = md.i0s
        self.lom_EC = md.lom_EC
        self.ipm2s = md.ipm2s
        self.th2s = md.th2s
        #update
        if self.nupdate%updaterate==0:
            self.lasttime = time.time()
            #fitting horizontal
            p0 = [self.i0s.max(), self.th1s.mean(), 0.04, 0]
            p02 = [self.ipm2s.max(), self.th2s.mean(), 0.0004, 0]
            p0E = [self.i0s.max(), self.lom_EC.mean(), 0.05, 0]
            #print (self.th1s, self.i0s)
            try:
                th10,FWHM_th1, xplot,yplot = fit_gaussian_rocking_curve(self.th1s, self.i0s, p0)
                print ('first crystal angle {}, {}'.format(th10, FWHM_th1))
            #plotting
            except:
                print ("fitting failed")
                xplot = self.th1s
                yplot = self.i0s
                th0 = 0.
            try:
                E0,FWHM_E, xplot2,yplot2 = fit_gaussian_rocking_curve(self.lom_EC, self.i0s, p0E)
                print ('center photon energy: {} {} keV'.format(E0, FWHM_E))
            #plotting
            except:
                print ("fitting failed")
                xplot2 = self.lom_EC
                yplot2 = self.i0s

            try:
                th20,FWHM_th2, xplot3,yplot3 = fit_gaussian_rocking_curve(self.th2s, self.ipm2s, p02)
                print ('second crystal angle {}, {}'.format(th20, FWHM_th2))
            #plotting
            except:
                print ("fitting failed")
                xplot3 = self.th2s
                yplot3 = self.ipm2s
            plot1 = XYPlot(self.nupdate,"rocking curve", [self.th1s, xplot],[self.i0s, yplot],formats = ['bo', 'b-'],leg_label=['measurement','fit'],xlabel = 'lom th1 (degree)',ylabel = 'intensity')
            plot2 = XYPlot(self.nupdate,"lom EC", [self.lom_EC, xplot2],[self.i0s,yplot2],formats = ['bo','b-'],xlabel = 'lom EC (keV)',ylabel = 'intensity')
            plot3 = XYPlot(self.nupdate,"rocking curve", [self.th2s, xplot3],[self.ipm2s, yplot3],formats = ['ro', 'r-'],leg_label=['measurement','fit'],xlabel = 'lom th2 (degree)',ylabel = 'intensity')
            rockingcurve_plot = MultiPlot(self.nupdate,"rocking curve plot", ncols=2)
            rockingcurve_plot.add(plot1)
            rockingcurve_plot.add(plot2)
            rockingcurve_plot.add(plot3)
            publish.send('ROCKING_CURVE', rockingcurve_plot)
