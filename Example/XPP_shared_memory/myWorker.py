import numpy as np
from mpidata import mpidata
import psana

"""
This is the script used to collect the sensors of the miniSD 
to push to the master node for later on analysis.
"""

# TODO: Due to the instability of the data system of SLAC I think it is better to
#  have an option to specify which mode is used to


# user parameters
updaterate = 30  # how often we push to master, in events

i0thresh = 5
i0thresl = 0.1


def runclient(args):
    print("starting client...%s" % args.exprun)
    psana.setOption("psana.calib-dir", "/reg/d/psdm/xpp/xpplw3319/calib")

    # Define the data source and specify the data we would like to collect.
    ds = psana.DataSource(args.exprun)

    # XPP infrastructure
    det_xpp_i0 = psana.Detector("XppMon_Pim1")  # Intensity before the XPP monochrometer
    det_xpp_ipm2 = psana.Detector("XPP-SB2-BMMON")  # Intensity after the XPP mono
    det_xpp_th1 = psana.Detector("lom_th1")  # The angle of the first crystal of the XPP mono
    det_xpp_th2 = psana.Detector("lom_th2")  # The angle of the second crystal of the XPP mono
    det_xpp_pe = psana.Detector("FEEGasDetEnergy")  # The gas detector measurement of the x-ray pulse energy
    det_xpp_lom_ec = psana.Detector("lom_EC")  # The nominal photon energy defined by the LODCM

    # Define the holders for the parameters from the XPP infrastructure
    xpp_th1s = np.array([])
    xpp_i0s = np.array([])
    xpp_count = np.array([])
    xpp_lom_ec = np.array([])
    xpp_th2s = np.array([])
    xpp_ipm2s = np.array([])
    xpp_count2 = np.array([])

    # Define the user device in the psana data source for the miniSD
    # Photo diodes read out value
    det_user_i0 = psana.Detector("XppEnds_Ipm0:FEX:CH2")
    det_user_d1 = psana.Detector("XppSb3_Pim:FEX:CHO")
    det_user_d2 = psana.Detector("XppSb3_Pim:FEX:CH1")
    det_user_d3 = psana.Detector("XppSb3_Pim:FEX:CH2")
    det_user_d4 = psana.Detector("XppSb3_Pim:FEX:CH3")
    det_user_d5 = psana.Detector("XppEnds_Ipm0:FEX:CHO")
    det_user_d6 = psana.Detector("XppEnds_Ipm0:FEX:CH1")

    # Shutter Status
    det_user_shutter1 = psana.Detector("XppMon_Pim1")
    det_user_shutter2 = psana.Detector("XppMon_Pim1")

    for run in ds.runs():
        for nevent, evt in enumerate(run.events()):
            th1 = det_xpp_th1(evt)
            th2 = det_xpp_th2(evt)
            try:
                pe0 = det_xpp_pe.get(evt).f_11_ENRC()
            except:
                continue
            lom_EC0 = det_xpp_lom_ec(evt)
            if (pe0 <= i0thresl) or (pe0 >= i0thresh):
                print(pe0)
                continue
            try:
                i0 = det_xpp_i0.channel(evt)[0]
            except:
                continue
            try:
                ipm2 = det_xpp_ipm2.get(evt).TotalIntensity()
            except:
                continue

            if i0 is None or th1 is None or th2 is None:
                print("no i0 or angle readout")
                continue
            if (pe0 < i0thresl) or (pe0 > i0thresh):
                continue
            _delta = lom_EC0 - xpp_lom_ec
            w = np.where(np.isclose(_delta, 0, atol=2e-3))[0]
            if len(w) == 0:
                xpp_th1s = np.append(xpp_th1s, th1)
                xpp_i0s = np.append(xpp_i0s, 0)
                xpp_count = np.append(xpp_count, 0)
                xpp_lom_ec = np.append(xpp_lom_ec, lom_EC0)
                _delta = lom_EC0 - xpp_lom_ec
            w = np.where(np.isclose(_delta, 0, atol=2e-3))[0]
            xpp_i0s[w] += i0
            xpp_count[w] += 1
            # if th2 scan
            _delta2 = th2 - xpp_th2s
            w2 = np.where(np.isclose(_delta2, 0, atol=1e-4))[0]
            if len(w2) == 0:
                xpp_th2s = np.append(xpp_th2s, th2)
                xpp_ipm2s = np.append(xpp_ipm2s, 0)
                xpp_count2 = np.append(xpp_count2, 0)
                _delta2 = th2 - xpp_th2s
            w2 = np.where(np.isclose(_delta2, 0, atol=1e-4))[0]
            xpp_ipm2s[w2] += ipm2
            xpp_count2[w2] += 1
            if (
                    nevent
            ) % updaterate == 0:  # send mpi data object to master when desired
                print("send data to master")
                senddata = mpidata()
                senddata.addarray("th1s", xpp_th1s)
                senddata.addarray("i0s", xpp_i0s / np.float_(xpp_count))
                senddata.addarray("th2s", xpp_th2s)
                senddata.addarray("ipm2s", xpp_ipm2s / np.float_(xpp_count2))
                senddata.addarray("lom_EC", xpp_lom_ec)
                senddata.send()
        md = mpidata()
        md.endrun()
