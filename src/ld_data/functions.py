import os
import glob
import re
import time
import numpy as np
import pandas as pd
import networkx as nx
# from qd.cae.dyna import Binout, D3plot, KeyFile
from lasso.dyna import Binout, D3plot  # KeyFile
from ld_data.models import Sim, Part, Node, Model, Barr, Imp, Behav, Des
# from model_func import MthdF
# from energy_model import grp_nrg_fts
import nrg_fts.functions
from neomodel import config, Q

import matplotlib.pyplot as plt


class PopulateSim:

    def __init__(self, job, pids=[], coords=[]):

        self.Job = job
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
        self.sim = Sim(
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
        self.sim.save()

        self.binout = get_binout(self.Job.pathPost)
        self._populate_sim_fts_parts()
        return ('test')

    def cnct_nodes(self, method, lc='', rls0='', dspln=''):
        feat_list = [
            "nrg_max",  # 0

            "ti_grad",  # 1
            "ti_ll",  # 2

            "tn_pct",  # 3
            "tn_max"]  # 4
     # ------------------------------------------------------------
        fts_opt = [feat_list[i] for i in [0, 1, 3]]

        neomodelUpdate.group_pid(lc, rls0, dspln)
        for f in fts_opt:
            neomodelUpdate.nrg_behavior(f, method, lc, rls0, dspln)
        neomodelUpdate.connect_sim_behav(lc, rls0, dspln)
        neomodelUpdate.connect_sim_des(lc, rls0, dspln)
        neomodelUpdate.connect_behav_des(lc, rls0, dspln)

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
            self.sim.sim_ie_tot_max = nrgFtr.IEtotMax
            self.sim.sim_ie_prt_max = nrgFtr.IEprtMax
            self.sim.sim_ke_tot_max = nrgFtr.KEtotMax
            self.sim.sim_ie_ti_ll = nrgFtr.IEtiLL
            self.sim.sim_ie_ti_grad = nrgFtr.IEtiGrad
            self.sim.sim_ie_tn_pct = nrgFtr.IEtnPct
            self.sim.sim_ie_tn_max = nrgFtr.IEtnMax
            self.sim.sim_tn_pct_max = nrgFtr.tnPctMax
            self.sim.sim_tn_max_max = nrgFtr.tnMaxMax
            self.sim.sim_ti_ll_pp = nrgFtr.tiLLpp
            self.sim.sim_ti_grad_pp = nrgFtr.tiGradpp
            self.sim.sim_t_max = nrgFtr.tMax
            self.sim.save()
        else:
            return()
            # sim.delete()
        end = time.time()
        trace_time(end - start, 'populate_sim_nrg_parts')

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
        trace_time(end - start, 'populate_sim_nrg_parts_grid')

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
                if os.path.isfile(Job.boxGrp):
                    self.box_mrg_df = curv_mrg(self.Job)
                else:
                    print('NO JSON FILE TO MERGE PARTS')
                    print('path:', Job.boxGrp)
                    self.box_mrg_df = None

        start = time.time()
        Job = self.Job
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
        trace_time(end - start, 'nrgFtr')

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
        trace_time(end - start, 'IE grid')

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

    def nrgOem(self):
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
        self.energyCurve.nrgOem()

        self.nodeStatus = DataBinout()
        self.boxGrp = 'energy_model/src/bumper/box.json'  # YARIS_BUMPER
        # self.boxGrp = rPath + '/{}_box.json'.format(runName)

        self.cnfg = {
            'nrg_parts': [],
        }

class neomodelUpdate:

    def group_pid(rel='', lc='', dspln=''):
        print('Group PID')
        parts = Part.nodes.filter(
            Q(part_sim_name__contains=lc)
            & Q(part_sim_name__contains=rel)
            & Q(part_sim_name__contains=dspln)
        )
        print(parts)
        for p in parts:
            pid = p.part_id
            # print(p.part_sim_name)
            # print(pid)

            print(p)
            des_curr = Des.nodes.get_or_none(des_pid=pid)
            if not des_curr:
                des_curr = Des(
                    des_type='pid',
                    des_pid=pid)
                des_curr.save()
            p.part_des.connect(des_curr)

    def nrg_behavior(fts_opt, method, type='nrg', lc='', rel='', dspln=''):
        print('NRG_behavior: ' + fts_opt)
        parts = Part.nodes.filter(
            Q(part_sim_name__contains=lc)
            & Q(part_sim_name__contains=rel)
            & Q(part_sim_name__contains=dspln)
        )

        for part in parts:
            fts = part.part_nrg(fts_opt)
            behav_old = Behav.nodes.get_or_none(
                behav_embd=fts,
                behav_method=method,
                behav_id=fts_opt)
            if behav_old:
                # print('behavior exist', behav_old.uid)
                behav_uid = behav_old.uid
                con = part.part_behav_con(behav_uid)
                if not con:
                    part.part_behav.connect(behav_old)
            else:
                # print('add behav for part', part.uid)
                behav = Behav(
                    behav_type=type,
                    behav_embd=fts,
                    behav_id=fts_opt,
                    behav_method=method)
                behav.save()
                part.part_behav.connect(behav)

    def connect_behav_des(lc='', rel='', dspln='', mType='pid'):
        print('connect behavior to design')
        sims = Sim.nodes.filter(
            Q(sim_name__contains=lc)
            & Q(sim_name__contains=rel)
            & Q(sim_name__contains=dspln)
        )
        for s in sims:
            # print(s.sim_name)
            deses = s.get_deses()
            for m in deses:
                des = Des.nodes.get_or_none(uid=m)
                behavs = des.get_behav()

                for e in behavs:
                    behav = Behav.nodes.get_or_none(uid=e)
                    # print(behav.uid)
                    des.des_behav.connect(behav)

    def connect_sim_behav(lc='', rel='', dspln=''):
        print('connect sim to behavior')
        sims = Sim.nodes.filter(
            Q(sim_name__contains=lc)
            & Q(sim_name__contains=rel)
            & Q(sim_name__contains=dspln)
        )
        for s in sims:
            # print(s.sim_name)
            behavs = s.get_behavs()
            for e in behavs:
                behav = Behav.nodes.get_or_none(uid=e)
                s.sim_behav.connect(behav)

    def connect_sim_des(lc='', rel='', dspln=''):
        print('connect sim to design')
        sims = Sim.nodes.filter(
            Q(sim_name__contains=lc)
            & Q(sim_name__contains=rel)
            & Q(sim_name__contains=dspln)
        )
        for s in sims:
            # print(s.sim_name)
            deses = s.get_deses()
            for m in deses:
                des = Des.nodes.get_or_none(uid=m)
                s.sim_des.connect(des)

    def connect_sim_des_weighted(ft='nrg_max', lc='', rel='', dspln=''):
        '''
            match (d:Des)-[r]-(s:Sim)-[]-(p:Part)
        where d.des_pid = p.part_id
        set r.weight=p.nrg_max
        //set r.weight=p.nrg_max/(p.tn_pct - p.ti_grad)
        //return max(r.weight), max(p.nrg_max ),
        return r.weight, p.nrg_max ,d.des_pid, p.part_id
        //return p
        '''
        print('connect sim to design')
        parts = Part.nodes.filter(
            Q(part_sim_name__contains=lc)
            & Q(part_sim_name__contains=rel)
            & Q(part_sim_name__contains=dspln)
        )
        for p in parts:
            pid = p.part_id
            w = p.part_nrg(ft)
            w = p.part_power()
            print(w)

            sim = Sim.nodes.get_or_none(uid=p.get_sim())
            des = Des.nodes.get_or_none(uid=p.get_des('pid'))
            sm = sim.sim_des.connect(des)
            sm.weight = w
            # print(sm)
            sm.save()

    def add_sym_des(pair_list, lc='', rel='', dspln=''):

        print('generate sym design')
        for pair in pair_list:
            pair.sort()
            parts = Part.nodes.filter(
                Q(part_id__in=pair)
            )
            pid = '_'.join([str(p) for p in pair])
            des_curr = Des.nodes.get_or_none(des_pid=pid)
            if not des_curr:
                des_curr = Des(
                    des_type='sym',
                    des_pid=pid)
                des_curr.save()

            for p in parts:
                p.part_des.connect(des_curr)

        print(type(parts))
        # print()

    def part_add_sim_name(lc='', rel='', dspln=''):
        print('add sim name to part')
        parts = Part.nodes.filter(
            Q(part_sim_name__contains=lc)
            & Q(part_sim_name__contains=rel)
            & Q(part_sim_name__contains=dspln)
        )
        for p in parts:
            sim_name = p.get_sim_name()
            # p(part_sim_name=sim_name)

            p.part_sim_name = sim_name[0]
            p.part_sim_abb = sim_name[1]
            p.save()

    def sim_add_info(OEM):
        print('update sim info')
        sims = Sim.nodes.all()
        for s in sims:
            if OEM == 'CEVT':
                if s.sim_path_post.endswith('/'):
                    s.sim_path_post = s.sim_path_post[:-1]
                rPath_split = s.sim_path_post.split('/')
                if len(rPath_split) == 1:
                    rPath_split = s.sim_path_post.split('\\')
                    s.sim_path_post = s.sim_path_post.replace('\\', '/')

                runName = s.sim_name
                release = rPath_split[-4].split('_')[1]
                dspln = rPath_split[-3]

                abb = runName.split('_')
                if len(abb) > 3:
                    loadcase = runName.split('_')[3]
                    abb = '_'.join([abb[2], abb[-1]])
                else:
                    loadcase = runName.split('_')[1]
                    abb = '_'.join([abb[1], abb[-1]])

                s.sim_abb = abb
                s.sim_lc = loadcase
                s.sim_rel = release
                s.sim_dspln = dspln
                # print(s)
                s.save()


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
    trace_time(end - start, 'read_binout')
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
    trace_time(end - start, 'read_binout')
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

def trace_time(t, name, flag=False):
    if flag:
        print('--------------------------------------------------------------')
        print('{}:'.format(name), t)
        print('--------------------------------------------------------------')