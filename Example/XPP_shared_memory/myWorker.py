import numpy as np
from mpidata import mpidata
import psana

# user parameters
updaterate=30 # how often we push to master, in events

i0thresh = 5
i0thresl = 0.1

def runclient(args):
    print('starting client...%s'%args.exprun)
    psana.setOption('psana.calib-dir','/reg/d/psdm/xpp/xpplw3319/calib')
    ds = psana.DataSource(args.exprun)
    det_i0 = psana.Detector('XppMon_Pim1')
    det_ipm2 = psana.Detector('XPP-SB2-BMMON')
    det_th1 = psana.Detector('lom_th1')
    det_th2 = psana.Detector('lom_th2')
    det_pe=psana.Detector('FEEGasDetEnergy')
    det_lom_EC = psana.Detector('lom_EC')
    th1s = np.array([])
    i0s = np.array([])
    count = np.array([])
    lom_EC = np.array([])
    th2s = np.array([])
    ipm2s = np.array([])
    count2 = np.array([])
    for run in ds.runs():
        for nevent,evt in enumerate(run.events()):
            th1 = det_th1(evt)
            th2 = det_th2(evt)
            try:
                pe0 = det_pe.get(evt).f_11_ENRC()
            except:
                continue
            lom_EC0 = det_lom_EC(evt)
            if (pe0<=i0thresl) or (pe0>=i0thresh):
                print (pe0)
                continue
            try:
                i0 = det_i0.channel(evt)[0]
            except:
                continue
            try:
                ipm2 = det_ipm2.get(evt).TotalIntensity()
            except:
                continue

            if i0 is None or th1 is None or th2 is None:
                print('no i0 or angle readout')
                continue
            if (pe0<i0thresl) or (pe0>i0thresh):
                continue
            _delta = lom_EC0-lom_EC
            w = np.where(np.isclose(_delta, 0, atol = 2e-3))[0]
            if len(w)==0:
                th1s = np.append(th1s, th1)
                i0s = np.append(i0s, 0)
                count = np.append(count,0)
                lom_EC = np.append(lom_EC, lom_EC0)
                _delta = lom_EC0 - lom_EC
            w = np.where(np.isclose(_delta, 0, atol = 2e-3))[0]
            i0s[w] += i0
            count[w] +=1
            #if th2 scan
            _delta2 = th2- th2s
            w2 = np.where(np.isclose(_delta2, 0, atol = 1e-4))[0]
            if len(w2)==0:
                th2s = np.append(th2s, th2)
                ipm2s = np.append(ipm2s, 0)
                count2 = np.append(count2, 0)
                _delta2 = th2-th2s
            w2 = np.where(np.isclose(_delta2, 0, atol = 1e-4))[0]
            ipm2s[w2] += ipm2
            count2[w2] += 1
            if ((nevent)%updaterate == 0): # send mpi data object to master when desired
                print ('send data to master')
                senddata=mpidata()
                senddata.addarray('th1s',th1s)
                senddata.addarray('i0s',i0s/np.float_(count))
                senddata.addarray('th2s',th2s)
                senddata.addarray('ipm2s',ipm2s/np.float_(count2))
                senddata.addarray('lom_EC',lom_EC)
                senddata.send()
        md = mpidata()
        md.endrun()
