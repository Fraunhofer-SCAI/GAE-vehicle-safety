import os
import glob
import re
import time
import numpy as np
import pandas as pd
import networkx as nx
# from qd.cae.dyna import Binout, D3plot, KeyFile
from lasso.dyna import Binout, D3plot  # KeyFile
from ld_data.models import Sim, Part, Node, Model, Barr, Imp
# from model_func import MthdF
# import update_graph as UG
# from energy_model import grp_nrg_fts
import gae.oems

import matplotlib.pyplot as plt


class PopulateSim():

    def __init__(self, job, pids=[], coords=[]):

        self.job = job
        self.pids = pids
        self.coords = coords

    def populate_sim(self):

        """
        Generate node (Sim) and its relation for the Job class.

        It will delete the sim node if it exist and re-populate/pupulate it.

        > properties:
          sim_name, sim_description, sim_path_pre, sim_path_post

        > relationships:
            model, parameters, energy_pid
        """

        sim_old = Sim.nodes.get_or_none(sim_name=self.Job.name)
        self.sim_new = Sim(
            sim_oem=self.Job.oem,
            sim_name=self.Job.name,
            sim_abb=self.Job.abb,
            sim_lc=self.Job.loadcase,
            sim_rel=self.Job.release,
            sim_dspln=self.Job.dspln,
            sim_description=self.Job.description,
            sim_path_post=self.Job.pathPost,
            sim_path_pre=self.Job.pathPre,)

        if sim_old:
            delete_list_uid(sim_old.get_childNodes('uid', 'Node'), Node)
            delete_list_uid(sim_old.get_childNodes('uid', 'Part'), Part)
            sim_old.delete()
        self.sim_new.save()

        self.binout = get_binout(self.Job.pathPost)
        self._populate_sim_fts_parts(self.Job, )

    def _populate_sim_fts_parts(self):

        print(self.Job.cnfg)
        for func, vals in self.Job.cnfg.items():
            eval(
                'self._populate_sim_{}(vals, func)'.format(func))

    def _populate_sim_nrg_parts(self, vals, func):
        """
        read the model and its N part with max energy
        """
        start = time.time()

        nrgFtr, boxdf = self._energy_feature()
        if nrgFtr:
            for ip, pid in enumerate(nrgFtr.pid):
                part = Part(
                    part_id=pid,
                    part_sim_name=self.Job.name,
                    part_sim_abb=self.Job.abb,

                    nrg_max=nrgFtr.nrgMax[ip],
                    ti_ll=nrgFtr.tiLL[ip],
                    ti_grad=nrgFtr.tiGrad[ip],
                    tn_pct=nrgFtr.tnPct[ip],
                    tn_max=nrgFtr.tnMax[ip])
                
                part.save()
                self.sim.sim_nrg_part.connect(part)

        # connect the parts based on box merg
            # if not boxdf is None:
            #     grp_nrg_fts.grp_box(boxdf, sim)

        # to save the new property it is needed to regreb the node
        # sim = Sim.nodes.get_or_none(sim_name=Job.name)
            sim.sim_ie_tot_max = nrgFtr.IEtotMax
            sim.sim_ie_prt_max = nrgFtr.IEprtMax
            sim.sim_ke_tot_max = nrgFtr.KEtotMax
            sim.sim_ie_ti_ll = nrgFtr.IEtiLL
            sim.sim_ie_ti_grad = nrgFtr.IEtiGrad
            sim.sim_ie_tn_pct = nrgFtr.IEtnPct
            sim.sim_ie_tn_max = nrgFtr.IEtnMax
            sim.sim_tn_pct_max = nrgFtr.tnPctMax
            sim.sim_tn_max_max = nrgFtr.tnMaxMax
            sim.sim_ti_ll_pp = nrgFtr.tiLLpp
            sim.sim_ti_grad_pp = nrgFtr.tiGradpp
            sim.sim_t_max = nrgFtr.tMax
            sim.save()
        else:
            return()
            # sim.delete()
        end = time.time()
        _trace_time(end - start, 'populate_sim_nrg_parts')

    def _populate_sim_nrg_parts_grid(self, value, func):
        """
        if IE_grid split the t_i to t_n with the specified
        segments
        """
        start = time.time()

        for vi in value:
            IEs, ts, pids = self._energy_feature_grid(vi)
            mthd = {
                'mthd_name': func,
                'mthd_cnfg_keys': ['num_smpl'],
                'mthd_cnfg_vals': [vi],
                'mthd_keys': ['IE', 't']
            }

            # for ip, pid in enumerate(pids):

            #     fts = IEs[ip] + ts[ip]
            #     a = MthdF.cnct_part_fts(mthd, pid, self.sim.sim_name, fts)

        end = time.time()
        _trace_time(end - start, 'populate_sim_nrg_parts_grid')

    def _energy_feature(self, out=None):

        def curv_mrg(Job):
            curv_mrg_l = []
            box_mrg_df = pd.read_json(Job.boxGrp, dtype={'PID_merg': str})

            for pid_m in box_mrg_df.PID_merg:
                pid_m = [int(x) for x in pid_m.split(',')]
                indx_mrg = np.isin(ids, pid_m)
                ids_mrg = ids[indx_mrg]
                curv_mrg_i = curv.T[indx_mrg].T
                curv_mrg_sum = curv_mrg_i.sum(axis=1)
                curv_mrg_l.append(curv_mrg_sum)


            curv_mrg = np.array(curv_mrg_l).T
            curvAbs = np.absolute(curv_mrg.T)
            _, tiGrad, _, _ = ti(curv_mrg, t, cfg)
            _, tnPct = tn(curvAbs, t, cfg)

            box_mrg_df['nrg_max'] = np.amax(curv_mrg, axis=0)
            box_mrg_df['tn_pct'] = tnPct
            box_mrg_df['ti_grad'] = tiGrad

            return(box_mrg_df)

        def tn(curv, t, cfg):

            try:
                tnPct = np.asarray([t[
                    np.where(x > np.max(x) * cfg.pctMax)][0] for x in curv])
            except IndexError:
                tnPct = 0
                print('01-TIME ISSUE, NOT CLEAR ERROR, tnPct=0')
            try:
                tnMax = np.asarray([t[np.where(x == np.max(x))][0] for x in curv])
            except IndexError:
                tnMax = 0
                print('01-TIME ISSUE, NOT CLEAR ERROR, tnMax=0')

            return(tnMax, tnPct)

        def ti(curv, t, cfg):

            def _curv_filter(cg):
                from scipy import signal
                n = 75
                b = [1.0 / n] * n
                a = 1

                cg_f = signal.lfilter(b, a, cg, axis=0)
                normV = np.max(np.absolute(cg_f), axis=0)
                # remove zeros
                normV = np.where(normV == 0, 1, normV)
                cg_fn = cg_f / normV  # norm-filtered
                return(cg_fn)

            LL = np.sum(np.max(curv, axis=0)) * cfg.pctLL
            tiLL = t[np.argmax(curv > LL, axis=0)]
            tiLLpp = np.ptp(t[np.argmax(curv > LL, axis=0)])

            cGrad = np.gradient(curv, t, axis=0)
            cGradFltNorm = _curv_filter(cGrad)  # filtered and normalized
            tiGrad = t[np.argmax(cGradFltNorm > cfg.pctGrad, axis=0)]

            tiGradpp = np.ptp(t[np.argmax(cGradFltNorm > cfg.pctGrad, axis=0)])

            return(tiLL, tiGrad, tiGradpp, tiLLpp)

        class NrgPid:

            def __init__(self, curv, ids):
                # # curve processing
                curvAbs = np.absolute(curv.T)
                idsMax = np.argsort([max(x) for x in curvAbs])
                idsMax = idsMax[::-1]
                idsMax = rmBarrier(ids, idsMax, cfg.barrier)

                idsK = idsMax[cfg.pidMin:cfg.pidMax]
                print([cfg.pidMin, cfg.pidMax])
                curvK = curv.T[idsK].T
                curvAbsK = np.absolute(curvK)
                pidK = ids[idsK]

                # feature extraction
                tiLL, tiGrad, tiGradpp, tiLLpp = ti(curv, t, cfg)
                tnMax, tnPct = tn(curvAbs, t, cfg)
                nrgMax = [curvK[ci, i]
                          for i, ci in enumerate(np.argmax(curvAbsK, axis=0))]
                nrgMax = np.asarray(nrgMax)

                self.pid = pidK
                self.nrgMax = nrgMax
                self.tMax = t[-1]

                self.tiLL = tiLL[idsK]
                self.tiGrad = tiGrad[idsK]

                self.tnPct = tnPct[idsK]
                self.tnPctMax = np.max(tnPct)

                self.tnMax = tnMax[idsK]
                self.tnMaxMax = np.max(tnMax)

                self.tiLLpp = tiLLpp
                self.tiGradpp = tiGradpp

            def set_global(self):
                # total IE
                curvSum = np.sum(curv, axis=1)
                curvSum2 = np.sum(curv2, axis=1)

                IEtiLL, IEtiGrad, _, _ = ti(curvSum, t, cfg)

                try:
                    IEtnPct = t[np.where(
                        curvSum > np.max(curvSum) * cfg.pctMax)][0]
                    IEtnMax = t[np.where(
                        curvSum > np.max(curvSum) * cfg.pctMax)][0]

                    self.IEtotMax = np.max(curvSum)  # max of the total IE
                    self.IEprtMax = np.max(np.max(curv, axis=0))  # parts IE maxmax
                    self.KEtotMax = np.max(curvSum2)  # max of the total maxmax
                    self.IEtiLL = IEtiLL
                    self.IEtiGrad = IEtiGrad

                    self.IEtnPct = IEtnPct
                    self.IEtnMax = IEtnMax

                except IndexError:
                    print('02-TIME ISSUE, NOT CLEAR ERROR')
                    return()

            def box_fts(self):

                # mrg curves
                if os.path.isfile(self.Job.boxGrp):
                    self.box_mrg_df = curv_mrg(self.Job)
                else:
                    print('NO JSON FILE TO MERGE PARTS')
                    print('path:', self.Job.boxGrp)
                    self.box_mrg_df = None

        start = time.time()
        cfg = self.Job.energyCurve
        try:
            curv, t, ids, _, curv2 = binout_vals(
                self.binout, cfg.var1, cfg.var2, cfg.var3)
        except TypeError:
            return (None, None)

        fts = NrgPid(curv, ids)
        fts.set_global()
        fts.box_fts()

        end = time.time()
        _trace_time(end - start, 'nrgFtr')

        if out:
            return(t, ids, curv, fts)
        return(fts, fts.box_mrg_df)

    def _energy_feature_grid(self, n):
        start = time.time()
        t, ids, curv, fts = self._energy_feature(out=True)

        ti = fts.tiGrad
        tn = fts.tnPct
        IE = fts.nrgMax

        d_smpl_t = (tn-ti)/n

        ts, IEs = [], []
        # for pid, tii in enumerate(ti):
        for ip, pid in enumerate(fts.pid):
            ci = np.where(pid == ids)

            tsi, IEsi = [], []
            IE0 = float(curv[np.abs(t - ti[ip]).argmin(), ci])

            tsi.append(ti[ip])
            IEsi.append(IE0)
            d_smpl_nrg = IE[ip]/n

            for ie in np.arange(d_smpl_nrg, IE[ip]+1, d_smpl_nrg):

                id = np.abs(curv.T[ci] - ie).argmin()
                tsi.append(float(t.flat[id]))
                IEsi.append(float(curv[id, ci]))

            IEsi.append(IE[ip])
            tsi.append(tn[ip])

            IEs.append(IEsi)
            ts.append(tsi)

        end = time.time()
        _trace_time(end - start, 'IE grid')

        return(IEs, ts, fts.pid)


class File:

    def __init__(self, root, pattern, act=None):

        pathList = glob.glob(os.path.join(root, pattern))
        if len(pathList) > 1:
            for p in pathList:
                fName = p.split('/')[-1].split('.')[0]
                dirName = p.split('/')[-3]
                if fName == dirName:
                    pathList = [p]
                    # break
        if len(pathList) == 1 or act == 'append':
            # file = pathList[0]
            txt = []
            file = pathList
            for f in pathList:
                with open(f, 'rb') as f:
                    TXT = f.read()
                    TXT = TXT.decode('utf-8', 'ignore')
                    TXT = TXT.strip()
                    txt.append(TXT)

        else:
            txt = ''
            file = 'None' + root + '/' + pattern

        self.txt = txt
        self.path = file


class DataBinout:

    def energy(self):
        self.var1 = 'matsum'
        self.var2 = 'internal_energy'
        self.var3 = 'kinetic_energy'
        self.pctLL = 0.0007  # lower limit percentage of energy
        self.pctMax = 0.95
        self.pctGrad = 0.0001

    def nrgPAG(self):
        self.energy()
        self.pidExc = np.array([20005200, 20001800])
        self.barrier = [0, 10009999]
        self.pidMin, self.pidMax = 0, 20

    def nrgCEVT(self):
        self.energy()
        self.pidExc = np.array([])
        self.barrier = [98000000, 99999999]
        self.pidMin, self.pidMax = 0, 20

    def nrgYARIS(self):
        self.energy()
        self.pidExc = np.array([])
        self.barrier = [10000000, 10001062]
        self.pidMin, self.pidMax = 0, 50

    def nrgTL2PID(self):
        self.energy()
        self.pidExc = np.array([970101])
        self.barrier = [10001000, 10001001]
        self.pidMin, self.pidMax = 0, 2

        self.pctMax = 0.98

    def nrgENVS(self):
        self.energy()
        self.pidExc = np.array([])
        self.barrier = [1, 1]
        self.pidMin, self.pidMax = 0, 1000

        self.pctMax = 0.98

    def nrgPEDPRO(self):
        self.energy()
        self.pidExc = np.array([])
        self.barrier = [10000000, 10001062]
        self.pidMin, self.pidMax = 0, 50


class DataD3plot:
    def __init__(self, oemf):
        eval('self.{}()'.format(oemf))

    def displacement(self):
        self.var1 = "node_displacement"

    def dispENVS(self):
        self.displacement()

    def read_disp(self, pathd, states=None, part_ids=None):
        from lasso.dyna import D3plot, FilterType, ArrayType

        if states:
            d3plot = D3plot(pathd + '/d3plot', state_array_filter=[
                self.var1], state_filter=states)
        else:
            d3plot = D3plot(pathd + '/d3plot', state_array_filter=[
                self.var1])

        disp = d3plot.arrays[self.var1]

        if part_ids:
            mask = d3plot.get_part_filter(FilterType.NODE, part_ids)
            disp = disp[:, mask]
        return (disp)


class CaeSim:

    def __init__(self, oem):
        self.oem = oem

    def dataOEM(self, rPath):

        # loop = ''
        # design = ''
        runName = rPath.split('/')[-1]
        try:
            loadcase = runName.split('_')[1]
        except IndexError:
            loadcase = ''
        release = ''

        abb = runName.split('_00')[-1]
        abb = runName.split('_')[-1]

        self.name = runName
        self.model = runName
        self.imp = ''
        self.barr = 'full_front'
        self.dev = 'pre_study'
        self.abb = abb
        self.design = ''
        self.path = rPath
        self.description = ''
        self.reference = ''
        self.refPath = ''
        self.id = ''
        self.media = []
        self.pathPost = rPath + '/'
        self.pathPre = rPath + '/'

        self.release = release
        self.loadcase = loadcase
        self.dspln = 'front'

        self.energyCurve = DataBinout()
        self.energyCurve.nrgYARIS()

        self.nodeStatus = DataBinout()
        self.boxGrp = 'energy_model/src/bumper/box.json'  # YARIS_BUMPER
        # self.boxGrp = rPath + '/{}_box.json'.format(runName)

        self.cnfg = {
            'nrg_parts': [],
        }



def get_binout(src):

    try:
        binout = Binout(os.path.join(src, "binout*"))
        return(binout)
    except OSError:
        print(src)
        print('NO BINOUT')
        print(os.path.join(src, "binout*"))
        return(None)

def delete_list_uid(idList, NodeName):

    for id in idList:
        old_node = NodeName.nodes.get_or_none(uid=id)
        if old_node:
            old_node.delete()

def update_part(Job, pids=[], rng=[], cog=[]):
    if pids == []:
        pids, rng, cog = make_box(sim)

    pids = pids.tolist()
    # get the simulation node
    sim_curr = Sim.nodes.get_or_none(sim_name=Job.name)
    # get parts of the simulation
    pids_curr = sim_curr.get_parts(':NRG_PART')
    for p in pids_curr:
        print(p)
        try:
            i = pids.index(p)
        except ValueError:
            continue

        part_uid = sim_curr.get_childByID(p, 'Part')
        part_curr = Part.nodes.get_or_none(uid=part_uid)
        part_curr.part_cog = cog[i][0].tolist()
        part_curr.part_min = rng[i][0].tolist()
        part_curr.part_max = rng[i][1].tolist()
        part_curr.save()

def read_binout(src, var1, var2, var3=None):

    start = time.time()
    try:
        binout = Binout(os.path.join(src, "binout*"))
        try:
            t = binout.read(var1, 'time')
        except ValueError:
            print('BINOUT ISSUE')
            return(None)
    except OSError:
        print(src)
        print('NO BINOUT')
        print(os.path.join(src, "binout*"))
        return(None)
    if len(t) == 1:
        print('Error Term - one time step')
        return(None)

    ids = binout.read(var1, 'ids')
    curv = binout.read(var1, var2)

    # prinet(binout.read(var1))
    if var3:
        curv2 = binout.read(var1, var3)
    else:
        curv2 = []

    names = binout.read(var1, 'legend')
    names2 = []

    # get part names
    for x in re.findall('.{70}', names):
        try:
            names2.append(x.split()[0])
        except IndexError:
            names2.append('')
    # d3plot = D3plot(os.path.join(src, "*.fz"), use_femzip=True)
    # part = d3plot.get_partByID(ids[0])

    end = time.time()
    _trace_time(end - start, 'read_binout')
    return(curv, t, ids, names2, curv2)

def binout_vals(binout, var1, var2, var3=None):

    start = time.time()
    try:
        t = binout.read(var1, 'time')
    except:
        print('BINOUT ISSUE')
        return(None)
    if len(t) == 1:
        print('Error Term - one time step')
        return(None)

    ids = binout.read(var1, 'ids')
    curv = binout.read(var1, var2)

    # prinet(binout.read(var1))
    if var3:
        curv2 = binout.read(var1, var3)
    else:
        curv2 = []

    names = binout.read(var1, 'legend')
    names2 = []

    # get part names
    for x in re.findall('.{70}', names):
        try:
            names2.append(x.split()[0])
        except IndexError:
            names2.append('')

    # remove zero curves
    nonZeroIds = ~np.all(curv == 0, axis=0)
    curv = curv.T[nonZeroIds].T
    ids = ids[nonZeroIds]

    end = time.time()
    _trace_time(end - start, 'read_binout')
    return(curv, t, ids, names2, curv2)

def rmBarrier(ids, idsMax, barrier):
    idsMax1 = idsMax[np.where(ids[idsMax] < barrier[0])]
    idsMax2 = idsMax[np.where(ids[idsMax] > barrier[1])]

    if idsMax1.size == 0:
        return(idsMax2)
    elif idsMax2.size == 0:
        return(idsMax1)
    else:
        idsMax = np.concatenate(idsMax1, idsMax2)
        return(idsMax)

def nrg_nodes(method, lc='', rls0='', dspln=''):
    feat_list = [
        "nrg_max",  # 0

        "ti_grad",  # 1
        "ti_ll",  # 2

        "tn_pct",  # 3
        "tn_max"]  # 4
 # ------------------------------------------------------------
    fts_opt = [feat_list[i] for i in [0, 1, 3]]
    # fts_opt = [feat_list[i] for i in [0, 3]]
    # fts_id = ['IE', 'ti', 'tn']

    UG.neomodelUpdate.group_pid(lc, rls0, dspln)
    # for f in fts_opt:
    #     UG.neomodelUpdate.nrg_behavior(f, method, lc, rls0, dspln)
    # UG.neomodelUpdate.connect_sim_behav(lc, rls0, dspln)
    UG.neomodelUpdate.connect_sim_des(lc, rls0, dspln)
    # UG.neomodelUpdate.connect_behav_des(lc, rls0, dspln)




if __name__ == '__main__':
    # OEM = 'PAG'
    # OEM = 'CEVT'
    # OEM = 'YARIS_BUMPER'
    OEM = 'YARIS'
    # OEM = 'TL2PID'
    # OEM = 'envs'
    # OEM = 'PEDPRO'

    print('--------------------------------------')
    print('                                    ')
    print('LODING DATA FOR {}'.format(OEM))
    print('                                    ')
    input('PRESS ENTER TO CONTINUE')

# PAG
    if OEM == 'PAG':
        neo4j_bolt('3687', 'ivory')
        data_path = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_*'
        # data_path = '/home/anahita/Projects/PAG/ROB_VOWA_505*'
        # data_path = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_*'

        dir_list = glob.glob(data_path)

        with open('src/key_out.npy', 'rb') as f:
            pids_nodout = np.load(f)
            coords_nodout = np.load(f)

        with open('src/box.npy', 'rb') as f:
            pids_box = np.load(f)
            rng_box = np.load(f)
            cog_box = np.load(f)

        for dir in dir_list:
            print(dir)
            i = 0
            count = 10
            sims = glob.glob(dir + '/*')
            # sims = glob.glob(dir + '/Design0017')
            ''' load or update all CAE simulations'''
            for s in sims:
                print(s)
                sim = CaeSim(OEM)
                sim.dataPAG(s)
                # sim.read_energy_pid(energy_static)
                # sim.energy_pic(media_path)
                populate_sim(sim, pids_nodout, coords_nodout)
                # update_part(sim, pids_box, rng_box, cog_box)
                i += 1
                if i == count:
                    break

# CEVT
    if OEM == 'CEVT':
        neo4j_bolt('7687', 'localhost')
        runSet = [
            ['3_stv02', ['fp3', 'fo5', 'fod']],
            ['3_stv0', ['fp3', 'fo5', 'fod']],
            ['2_stcr', ['fp3', 'fo5', 'fod']],
            ['3_m1', ['fp3', 'fo5', 'fod']],
            ['3_stv02', ['fp3', 'fo5', 'fod']],
            ['3_stv03', ['fp3', 'fo5', 'fod']],
        ]
        runSet = [
            ['3_stv0', ['fp3']],
            # ['2_stcr', ['fp3']],
            # ['3_m1', ['fp3']],
            # ['3_stv03', ['fp3']],
        ]

        # runSet=[
        # ['3_stv0', ['si9a', 'sp2b', 'si10a']],
        # ['3_stv02', ['si9a', 'sp2b', 'si10a']],
        #     ['3_stv03', ['si9a', 'sp2b', 'si10a']],
        #     ['3_m1', ['si9a', 'sp2b', 'si10a']],
        # ]

        # runSet = [
        #     ['3_stv0', ['rd8l', 'rd8r', 'ri4']],
        #     ['3_stv02', ['rd8l', 'rd8r', 'ri4']],
        #     ['3_stv03', ['rd8l', 'rd8r', 'ri4']],
        #     ['3_m1', ['rd8l', 'rd8r', 'ri4']],
        # ]

        dspln = 'front'  # 'side' #'rear' # 'front'
        print(runSet)
        # nrg_nodes('IE_ti_tn')
        # UG.connect_sim_des_weighted('nrg_max')
        for s in runSet:
            # break
            rls = s[0]
            rls0 = rls.split('_')[1]
            for lc in s[1]:
                # break
                proj = 'mma'
                data = '{2}\\{0}\\{3}\\runs\\*{1}_*'.format(
                    rls, lc, proj, dspln)
                root_path = 'S:\\nobackup\\safety\\projects\\'
                # data = '{1}/{0}/{2}/runs/cm1*'.format(rls, proj, dsp)
                data = '{2}/{0}/{3}/runs/*{1}_*'.format(rls, lc, proj, dspln)
                root_path = '/cevt/cae/backup/safety/users/anahita.pakiman1/'

                # 3_stv0, fp3-fo5(084,145-matsum value error), fod
                # 2_stcr, fp3(019), fo5, fod
                # recomended fod
                data_path = os.path.join(root_path, data)
                # data_path = 'src/CEVT/*fp3*'

                dir_list = glob.glob(data_path)
                print(data_path)
                i = 0
                count = 1
                for dir in dir_list:
                    pid = os.getpid()
                    print(dir)
                    start = time.time()
                    s = dir
                    ''' load or update all CAE simulations'''
                    sim = CaeSim(OEM)
                    sim.dataCEVT(s, lc, rls0, dspln)
                    print('pop done')
                    # sim.read_energy_pid(energy_static)
                    # sim.energy_pic(media_path)
                    populate_sim(sim)
                    # update_part(sim, pids_box, rng_box, cog_box)

                    i += 1
                    if i == count:
                        break
                print('----------------------------')
                print(dspln, rls, lc)
                print(len(dir_list))
            # connect nodes
            # print('----------------------------')
            # print('conect enrgy nodes')
            # print('----------------------------')
            # nrg_nodes('IE_ti_tn', lc, rls0, dspln)
        print(time.time()-start)
        # Getting usage of virtual_memory in GB ( 4th field)
        import psutil
        print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)

        python_process = psutil.Process(pid)
        # memory use in GB...I think
        memoryUse = python_process.memory_info()[0]/2.**30
        print('memory use:', memoryUse)

# YARIS
    if OEM == 'YARIS':
        oem = oems.oems(OEM)
        oem.backend_server()
        lc = 'CCSA_submodel_0'
        sims = glob.glob(oem.data_path)

        ''' load or update all CAE simulations'''
        for s in sims:
            # break
            print(s)
            sim = CaeSim(OEM)
            sim.dataYARIS(s)
            # sim.read_energy_pid(energy_static)
            # sim.energy_pic(media_path)
            populate_sim(sim)
            # break
            # update_part(sim, pids_box, rng_box, cog_box)

        # connect nodes
        nrg_nodes('IE_ti_tn', lc=lc)

        # UG.add_sym_des([
        #     [2000001, 2000501],
        #     [2000002, 2000502],
        #     [2000011, 2000511],
        #     [2000012, 2000512],
        #     [3000001, 3000501],
        #     [3000002, 3000502],
        #     [3000003, 3000503],
        #     [3000004, 3000504],
        #     [3000005, 3000505],
        #     [3000006, 3000506],
        #     [3000007, 3000507],
        #     [3000008, 3000508],
        #     [3000009, 3000509],
        # ])
        # UG.neomodelUpdate.connect_sim_des_weighted('nrg_max', lc=lc)


# YARIS_BUMPER
    if OEM == 'YARIS_BUMPER':
        neo4j_bolt('7687', 'localhost')
        data_path = '/home/ndv/data/BUMPER/MAIN_SIM/*.draco'
        sims = glob.glob(data_path)

        ''' load or update all CAE simulations'''
        for s in sims:
            break
            print(s)
            sim = CaeSim(OEM)
            sim.dataYARIS(s)
            # sim.read_energy_pid(energy_static)
            # sim.energy_pic(media_path)
            populate_sim(sim)
            # update_part(sim, pids_box, rng_box, cog_box)

        # connect nodes
        # nrg_nodes('IE_ti_tn')
        UG.connect_sim_des_weighted('nrg_max')

# TL2PID
    if OEM == 'TL2PID':
        vehList = {
            1: {
                'name': 'TL2PID_12_01',
                'cnfg': {
                    'ubdy_name': '01',
                    'ubdy_spec': 0.08,
                    'pltf_name': '01',
                    'pltf_spec': ''}
            },
            2: {
                'name': 'TL2PID_12_02',
                'cnfg': {
                    'ubdy_name': '02',
                    'ubdy_spec': 0.112,
                    'pltf_name': '01',
                    'pltf_spec': ''}
            },
            3: {
                'name': 'TL2PID_12_03',
                'cnfg': {
                    'ubdy_name': '03',
                    'ubdy_spec': 0.25,
                    'pltf_name': '01',
                    'pltf_spec': ''}
            },
            4: {
                'name': 'TL2PID_12_04',
                'cnfg': {
                    'ubdy_name': '04',
                    'ubdy_spec': 0.35,
                    'pltf_name': '01',
                    'pltf_spec': ''}
            },
            5: {
                'name': 'TL2PID_12_05',
                'cnfg': {
                    'ubdy_name': '05',
                    'ubdy_spec': 0.576,
                    'pltf_name': '01',
                    'pltf_spec': ''}
            },
            6: {
                'name': 'TL2PID_12_06',
                'cnfg': {
                    'ubdy_name': '06',
                    'ubdy_spec': 0.96,
                    'pltf_name': '01',
                    'pltf_spec': ''}
            }
        }
        strc = {
            'edge': {
                'src': 2000501,
                'trgt': 2000500
            }
        }
        oem = oems.oems(OEM)
        oem.backend_server()

        for v in vehList:
            print(v)
            if v in []:  # [2, 3, 4, 6]:  # [1,2,3,4]:
                continue
            veh = vehList[v]['name']
            cnfg = vehList[v]['cnfg']
            print(veh)
            p1 = glob.glob(oem.data_path.format(veh))
            p2 = glob.glob(oem.data_path_val.format(veh))
            sims = p1 + p2

            for s in sims:
                # s = '/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/TL2PID_12_01/TL2PID_12_01_0043'  # 235
                print(s)
                break
                sim = CaeSim(OEM)
                sim.dataTL2PID(s)
                populate_sim(sim)

        #  load the development tree structure
            root = oem.root.format(veh)
            simsTree = glob.glob(os.path.join(root,  "sims.txt"))[0]
            G = nx.read_gpickle(simsTree)
            UG.cyUpdate.devTree(G, veh, cnfg, strc, model=False, mm=True)
            # UG.cyUpdate.TL2PID(G, veh, model=True)

        # add wieght
            # w = '[p.ti_grad, p.tn_max, p.nrg_max]'
            # w = 'p.nrg_max * (p.tn_max -p.ti_grad)/s.sim_ke_tot_max/2'
            # UG.cyUpdate.nrgWeight(w)

# envs
    if OEM == 'envs':
        oem = oems.oems(OEM)
        oem.backend_server()
        sims = glob.glob(oem.data_path)

        for s in sims:
            print(s)
            sim = CaeSim(OEM)
            sim.dataENVS(s)
            populate_sim(sim)
            break


# PEDPRO
    if OEM == 'PEDPRO':
        oem = oems.oems(OEM)
        oem.backend_server()
        sims = glob.glob(oem.data_path)

        for s in sims:
            print(s)
            sim = CaeSim(OEM)
            sim.dataPEDPRO(s)
            populate_sim(sim)

        # connect nodes
        lc = ''
        nrg_nodes('IE_ti_tn', lc=lc)
