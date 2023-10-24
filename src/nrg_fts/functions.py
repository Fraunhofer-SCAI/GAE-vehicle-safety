# PYTHON script
from gae.constants import OEM_NAME, IE_UNIT, TIME_UNIT, DATA_PATH, SIM_PATTERN, PIC_NAME

import numpy as np
import pandas as pd
import re
from ld_data.models import Sim
import oems



class QueyNrgFts:
    def __init__(self, oem):
        oem = oems.oems(OEM_NAME, IE_UNIT, TIME_UNIT, DATA_PATH, SIM_PATTERN)
        oem.backend_server()
        self.oem = oem

    def make_simList(self, simReg, limit=None):
        driver = self.oem.driver
        simList = []
        with driver.session() as session:
            for sr in simReg:
                txt = '''
                    MATCH (n:Sim) WHERE n.sim_name =~ $name
                    RETURN distinct n.sim_name
                    '''
                if limit:
                    txt = txt + 'limit {}'.format(limit)
                
                result = session.run(txt, name=sr)
                for record in result:
                    simList.append(record.values()[0])
        return(simList)
    
    def make_pidList(self, simReg, pidList):
        driver = self.oem.driver
        cypherTxt = '''
                    MATCH (n:Part)-[]-(s:Sim)
                    where s.sim_name =~ "{0}" and n.part_id in {1}
                    RETURN distinct n.part_id, count(n)'''.format(simReg, repr(pidList))
        results = driver.session().run(cypherTxt)
        pidList, count = [], []
        for r in results:
            pidList.append(r[0])
            count.append(r[1])
        return(pidList, count)
    
    def feed_normalization(self):

        def nrg_normalized(nameReg, fts, norm_opt):
            driver = self.oem.driver

            with driver.session() as session:
                func = ["max", "min"]
                result_norm = session.run(
                    nrg_cypher(norm_opt, func), name=nameReg)
                for r in result_norm:
                    ss = len(norm_opt)
                    MAX = np.asarray(r.values()[0:ss])
                    # MIN = np.asarray(r.values()[ss:])
                MAX[2] = MAX[2] - MAX[1]
                fts[:, 2] = (fts[:, 2] - fts[:, 1])  # dt
                fts_nrm = (fts) / (MAX)
                return(fts_nrm)
        
        sims_nrg, sims_pid, sims = [], [], []
        
        for s in self.simList:
            sim = Sim.nodes.get_or_none(sim_name=s)
            if self.pids_sel:
                fts = sim.embed_nrg_sel(self.ft_opt, self.pids_sel)
                if not len(fts) == 0:
                    pids = fts[:, -1]
                    fts = fts[:, :-1]
                    diff_len = len(self.pids_sel) - len(pids)
                else:
                    continue
                if diff_len > 0:
                    for di in range(0, diff_len):
                        pids = np.append(pids, 0)
                        fts = np.append(fts, [[0, 0, 0]], axis=0)
            else:
                fts = sim.embed_nrg(self.ft_opt)
                if not len(fts) == 0:
                    pids = fts[:, -1]
                    fts = fts[:, :-1]
                else:
                    continue
            if not self.norm_opt == []:
                fts_nrm = nrg_normalized(self.nrmList, fts, self.norm_opt)
            else:
                fts_nrm = fts
            sims_nrg.append(fts_nrm)
            sims_pid.append(np.array(pids))
            sims.append(s)
        if sims_pid == []:
            print('''
            --------------------------------------------------------
            edit filtering seting, no sim is selected
            --------------------------------------------------------
            ''')
            return 
        sims_nrg = np.array(sims_nrg)
        sims_pid = np.array(sims_pid)
        return(sims_nrg, sims_pid, sims)
    
    def rmv_err(self):
        errList = []
        if not self.oem.err == {}:
            for rls in self.oem.err['release']:
                for lc in self.oem.err['release'][rls]:
                    errList += self.oem.err['release'][rls][lc]['errList']
            if not self.oem.rls == '' or not self.oem.lc == '':
                errList = self.oem.err['release'][self.oem.rls][self.oem.lc]['errList']
            for s in errList:
                try:
                    self.simList.remove(s)  # early termination
                except ValueError:
                    continue
    
    def out_dataframe(self, ns, nPID, nOrd, regs='.*', regp='.*'):

        def get_color(n):
            from random import randint
            colors = []
            for i in range(n):
                colors.append('#%06X' % randint(0, 0xFFFFFF))
            return(colors)
        
        self.ft_opt = ["nrg_max", "ti_grad", "tn_pct"]
        self.norm_opt = []
        self.nrmList = '.*'
        if regp == '.*' or regp == '':
            self.pids_sel = None
        else:
            self.pids_sel = [int(x) for x in regp.split(',')]
        regs = regs.replace(' ', '')
        regs_list = regs.split(',')
        self.simList = self.make_simList(
            regs_list,
            limit=ns)
        
        self.rmv_err()
        out_normalized = self.feed_normalization()
        if out_normalized:
            sims_nrg, sims_pid, sims = out_normalized
        else:
            return 
        
        try:
            sims_pid = sims_pid[:, :nPID]
        except IndexError:
            pass
        simListAbb = [s.split('_')[2] for s in sims]
        c_sim = get_color(len(sims))
        rls_list = [s.split('_')[self.oem.rls_pos] for s in sims]
        lc_list = [s.split('_')[self.oem.lc_pos] for s in sims]
        lcU = np.unique(lc_list).tolist()
        rlsU = np.unique(rls_list).tolist()
        c_lc = lcU
        c_rls = rlsU
        
        try:
            pidU = np.sort(np.unique(sims_pid))
            ntPid = sims_pid.shape[1]
        except:
            sims_pid_flat = []
            for s in sims_pid:
                sims_pid_flat += list(s)
            pidU = np.sort(np.unique(np.array(sims_pid_flat)))
            ntPid = min([s.shape for s in sims_pid])[0]

        pidU, countPid = self.make_pidList('.*', pidU.tolist())
        pidU = np.array(pidU)
        pidU = np.delete(pidU, np.argwhere(pidU == 0))
        c_pid = get_color(len(pidU))

        def ordinal(n): return "%d%s" % (
            n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
        c_ord = np.repeat(
            np.array([[ordinal(x)
                       for x in range(1, ntPid + 1)]], dtype=object),
            len(sims), axis=0)
        c_ord[:, nOrd:] = '{0} to {1}'.format(
            ordinal(nOrd + 1), ordinal(ntPid))
        
        df = pd.DataFrame()
        for i, pi in enumerate(pidU):
            id = np.where(sims_pid == pi)
            dfi = pd.DataFrame(sims_nrg[id], columns=['IE', 'ti', 'tn'])


            dfi['dt'] = sims_nrg[id][:, 2] - sims_nrg[id][:, 1]
            dfi['PID'] = int(pi)
            sim = np.array(sims)[id[0]]
            dfi['sim'] = sim
            dfi['sim_abb'] = np.array(simListAbb)[id[0]]
            dfi['c_grPID'] = c_pid[i]
            dfi['c_grSim'] = np.array(c_sim)[id[0]]
            dfi['c_grOrd'] = c_ord[id]
            dfi['count'] = countPid[i]
            rls_list_pi = [s.split('_')[self.oem.rls_pos] for s in sim]
            lc_list_pi = [s.split('_')[self.oem.lc_pos] for s in sim]
            dfi['c_lc'] = [c_lc[lcU.index(lci)] for lci in lc_list_pi]
            dfi['c_rls'] = [c_rls[rlsU.index(ri)] for ri in rls_list_pi]
            # relPath = '/'.join(self.oem.save_path.split('/')[-2:])
            dfi['pic'] = ['YARIS/{0}_{1}_iso.png'.format(
                PIC_NAME, int(pi)) for s in sim]
            
            df = pd.concat([df, dfi])
            df = df.reset_index(drop=True)
        df = df.sort_values(by=['PID'])
        df['PID'] = df['PID'].astype(str)
        return(df)
    
    def cnv_usys(self, df):
        df['IE'] = df['IE'] * self.oem.unit['IE']
        df['ti'] = df['ti'] * self.oem.unit['t']
        df['tn'] = df['tn'] * self.oem.unit['t']
        return(df)
    
    def nrg_fts(self, sim, lmt):
        import pandas as pd
        cypherTxt = '''
         match (s:Sim) where s.sim_name={}
            match (s)-[:NRG_PART]-(p:Part)
            return
                s.sim_abb as id,
                p.part_id as PID,
                s.sim_name as s_name,
                p.ti_grad as ti,
                p.tn_pct as tn,
                p.nrg_max as IE
                order by IE DESC
                limit {}
            '''.format(sim, lmt)
        # input(cypherTxt)
        results = self.driver.session().run(cypherTxt)
        df = pd.DataFrame([dict(record) for record in results])
        return(df)
    
    def simrank_G(
        self, sims, simM, lmt,
        simM2='""', simList='"[]"',
        rel='SIM_DES', trgt='Des'
    ):
        cypherTxt = '''
        //embd_nrg_graph
        match (s:Sim)
        where s.sim_name=~ {0} or s.sim_name=~{1} or s.sim_name=~{2}
        or s.sim_name in {3}
        CALL{{
            with s
            match p=(s)-[rm:{5}]-(m:{6})
            return m,rm order by rm.w_e_value desc limit {4}
        }}
        return s,m, rm order by s.sim_name
            '''.format(sims, simM, simM2, simList, lmt, rel, trgt)
        return(cypherTxt)
