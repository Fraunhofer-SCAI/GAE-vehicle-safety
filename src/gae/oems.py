# PYTHON script
import os
import glob
import sys


class oems:

    def __init__(self, name, rls='', lc=''):
        self.name = name
        eval('self.{}()'.format(name))
        self.rls = rls
        self.lc = lc
        self.set_data_path()
        self.set_post()

    def backend_server(self):
        from neo4j import GraphDatabase

        def neo4j_bolt(svr_id, host):
            from neomodel import config
            config.DATABASE_URL = os.environ.get(
                'NEO4J_BOLT_URL', 'bolt://neo4j:ivory123@{0}:{1}'.format(host, svr_id))

        if self.name == 'CEVT':
            neo4j_bolt('7687', 'localhost')
            # KG.neo4j_bolt('7687', 'ivory')
            uri = "bolt://localhost:7687"
            self.driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))

        elif self.name == 'Porsche':
            uri = "neo4j://localhost:3687"
            self.driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
            neo4j_bolt('3687', 'ivory')

        elif self.name in ['envs', 'PEDPRO']:
            uri = "neo4j://localhost:3687"
            self.driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
            neo4j_bolt('3687', 'localhost')

        elif self.name in ['YARIS_BUMPER', 'TL2PID', 'YARIS']:
            uri = "neo4j://localhost:7687"
            self.driver = GraphDatabase.driver(uri, auth=("neo4j", "ivory123"))
            neo4j_bolt('7687', 'localhost')

    def set_data_path(self, inPath=False):

        if self.name == 'Porsche':
            self.data_path = '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/ROB_VOWA_*'

            # pids_nodout, coords_nodout
            self.model = 'src/key_out.npy'

            # pids_box, rng_box, cog_box
            self.box = 'src/box.npy', 'rb'

        if self.name == 'CEVT':

            proj = 'mma'
            self.data = '{2}\\*{0}\\front\\runs\\*{1}_*'.format(
                self.rls, self.lc, proj)
            root_path = 'S:\\nobackup\\safety\\projects\\'

            if os.path.isdir(root_path):
                self.root_path = root_path
            else:
                self.root_path0 = '/share/nobackup/safety/projects/'
                self.root_path = '/cevt/cae/nobackup/safety/projects/'
                self.data = self.data.replace('\\', '/')

            self.data_path = os.path.join(self.root_path, self.data)

        if self.name == 'YARIS':
            self.data_path = '/home/ndv/stud/data/YARIS/full_front/CCSA_submodel/crash_modes/'
            if not os.path.exists(self.data_path):
                self.data_path = 'C:/Users/apakiman/data/CCSA_submodel/'
            self.data_path += 'CCSA_submodel_60*'

            # '/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/CCSA_*/*/CCSA_*'

        if self.name == 'YARIS_BUMPER':
            self.data_path = '/home/ndv/data/BUMPER/MAIN_SIM/*'

        if self.name == 'TL2PID':
            self.root = '/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/{}/'
            sim_check = self.name + '_*'
            self.data_path = self.root + sim_check
            self.data_path_val = self.root + '/val/*' + sim_check

        if self.name == 'PEDPRO':
            self.data_path = '/home/ndv/stud/data/YARIS/pedestrian/head_EN/runs/yaris_ccsa_pp46*/yaris_ccsa_pp46*[0-9]'

        if inPath:
            self.data_path = inPath

        if self.name == 'envs':
            self.data_path = '/home/ndv/stud/data/LOAD_PATH_DTCT/env_0/*'

        if self.name == 'NISSAN':

            self.data_path = '/home/ndv/stud/data/NISSAN/full_front/NISSAN_ff_*'

    def set_post(self):
        if self.name == 'CEVT':
            self.lc_pose = 3
            self.views = ['top', 'front', 'right', 'btm']
            self.states = '5/10/15/20/25/30/33'
            self.states = '5/10/15/20/25/30/35/40/45/50/55/57/33'

            # import sys
            # sys.path.append(os.path.abspath('../src/cevt_vis'))
            # sys.path.append(os.path.abspath('src/cevt_vis'))
            # import pids
            # pid_txt = 'pids.{}_{}'.format(self.rls, self.lc)
            # try:
            #     self.pid_list = eval(pid_txt)
            # except AttributeError:
            #     # print(pid_txt, 'is pid_txt')
            #     pass

            self.save_path = '/home/apakiman/kg01/dash-nrg/assets/CEVT'
            if not os.path.isdir(self.save_path):
                self.save_path = '/cevt/cae/backup/safety/users/anahita.pakiman1/kg01/dash-nrg/assets/CEVT'
            sys.path.append(os.path.abspath('../src/cevt_vis'))

        if self.name == 'YARIS':
            self.lc_pose = 3
            self.views = ['top', 'front', 'right', 'btm']
            self.states = '5/10/15/20/25/30/33'
            self.states = '5/10/15/20/25/30/35/40/45/50/55/60'

            self.save_path = '/home/apakiman/Projects/kg01/dash-nrg/assets/YARIS'
            self.pid_list = '''
            2000511, 3000508, 2000011, 2000512, 3000001, 3000502,
            2000001, 3000501, 2000501, 3000002, 2000000, 4000486,
            3000504, 3000004, 3000003, 3000005, 3000503, 3000505,
            3000506, 3000006, 3000007, 3000507, 3000509, 2000002,
            2000502, 2000012, 3000008, 3000009'''

        if self.name == 'TL2PID':
            self.save_path = '/home/apakiman/Projects/kg01/transferDOE/apps/assets/TL2PID'
            self.lc_pose = 3
            self.views = ['top', 'front', 'right', 'btm']
            self.states = '5/10/15/20/25/30/35/40/45/50/55/60'

            self.pid_list = '''2000500, 2000501,10001001'''

        if self.name == 'ENVS':
            # self.lc_pose = 3
            self.views = ['top']
            self.states = '5/10/15/20/25/30/35/40/45/50/55/60'

            self.pid_list = 'all'

            self.save_path = '/home/apakiman/Projects/kg01/energy_model/apps/assets/ENVS'

        if self.name == 'NISSAN':
            self.save_path = '/home/apakiman/Projects/kg01/dash-nrg/assets/NISSAN'

    def metapost(self):

        def model_vis(self):
            from meta import utils
            dirs = (glob.glob(self.data_path))
            # lc_dic = {}
            for d in dirs:
                runName = d.split('/')[-1]
                # loadcase = runName.split('_')[self.lc_pose]

                modelPath = d + '/' + runName + '.fz'
                print(modelPath)
                if not os.path.isfile(modelPath):
                    modelPath = d + '/' + 'd3plot'

                print(modelPath)
                if os.path.isfile(modelPath):
                    utils.MetaCommand('window delete "MetaPost"')
                    utils.MetaCommand('window create "MetaPost"')
                    utils.MetaCommand('window active "MetaPost"')
                    utils.MetaCommand('model active all')

                    utils.MetaCommand('options var add runName ' + runName)
                    utils.MetaCommand(
                        'options var add savePath ' + self.save_path)
                    utils.MetaCommand(
                        'options var add pidList ' + self.pid_list)

                    utils.MetaCommand('read geom Dyna3d ' + modelPath)
                    utils.MetaCommand(
                        'read dis Dyna3d ' + modelPath + '  ' +
                        self.states + '  Displacements')

                    for v in self.views:
                        print('views:', v)
                        utils.MetaCommand('options var add pos ' + v)
                        utils.MetaCommand('options var add aniOn 1 ')
                        utils.MetaCommand('read session ses/pic_ani.ses')

                        nState = len(self.states.split('/'))
                        for s in range(0, nState + 1):
                            print('states: ' + str(s))
                            utils.MetaCommand('options var add pos ' + v)
                            utils.MetaCommand('options var add st ' + str(s))
                            utils.MetaCommand('options var add aniOn 0 ')
                            utils.MetaCommand('read session ses/pic_ani.ses')

        def part_vis(self):
            from meta import utils

            dirs = (glob.glob(self.data_path))
            print(dirs)
            for d in dirs:
                print(d)
                runName = d.split('/')[-1]

                modelPath = d + '/' + runName + '.fz'
                if not os.path.isfile(modelPath):
                    modelPath = d + '/' + 'd3plot'

                if os.path.isfile(modelPath):
                    print(runName)
                    utils.MetaCommand('window delete "MetaPost"')
                    utils.MetaCommand('window create "MetaPost"')
                    utils.MetaCommand('window active "MetaPost"')
                    utils.MetaCommand('model active all')

                    utils.MetaCommand('options var add runName ' + runName)
                    utils.MetaCommand(
                        'options var add savePath ' + self.save_path)
                    utils.MetaCommand(
                        'options var add pidList ' + self.pid_list)

                    utils.MetaCommand('read geom Dyna3d ' + modelPath)

                    pid_list = self.pid_list.split(',')
                    for pid in pid_list:
                        print(pid)
                        utils.MetaCommand('options var add pid ' + pid)
                        utils.MetaCommand('read session ses/vis_part.ses')
                else:

                    print('missing file')
                    print(modelPath)

        def status(self, lc):
            from meta import utils

            dirs = (glob.glob(self.data_path))
            print(dirs)
            for d in dirs:
                print(d)
                runName = d.split('/')[-1]

                modelPath = d + '/' + runName + '.fz'
                if not os.path.isfile(modelPath):
                    modelPath = d + '/' + 'd3plot'

                if os.path.isfile(modelPath):
                    print(runName)
                    utils.MetaCommand('window delete "MetaPost"')
                    utils.MetaCommand('window create "MetaPost"')
                    utils.MetaCommand('window active "MetaPost"')
                    utils.MetaCommand('model active all')

                    utils.MetaCommand('options var add runName ' + runName)
                    utils.MetaCommand('options var add my_path ' + d)
                    utils.MetaCommand(
                        'options var add savePath ' + self.save_path)

                    utils.MetaCommand(
                        'read session ses/' + lc + '_status_values.ses')
                else:

                    print('missing file')
                    print(modelPath)

        # model_vis(self)
        # part_vis(self)
        status(self, 'front')

    def keyInp(self):
        if self.name == 'Porsche':
            import codecs
            inKey = glob.glob(self.data_path + '/Design*i1')[0]
            with codecs.open(inKey, 'r', encoding='utf-8', errors='ignore') as file:
                keyTxt = file.read()
            inKey = inKey.split('002_Daten_Kaan/')[-1]
            txtIn = (
                '*KEYWORD\n'
                '$--------1---------2---------3---------4---------5---------6'
                + '---------7---------8\n'
                + '*INCLUDE_PATH\n'
                + '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan/INPUT_Keyfiles\n'
                + '/export/PAG_DATA/VORDERWAGEN/002_Daten_Kaan\n'
                + '{1}\n').format(inKey, src)
            keyTxt = keyTxt.replace('*KEYWORD', txtIn)
            includes_trans = re.findall(
                r'(\*INCLUDE_TRANSFORM[\S\s]*?)[\*\$]', keyTxt)
            for i in includes_trans:
                null_trans = re.findall(
                    '         0         0         0         0'
                    + '         0         0         0\n         0', i)
                if len(null_trans) == 1:
                    new_txt = re.findall(r'\*INCLUDE_TRANSFORM\n(.*)', i)[0]
                    keyTxt = keyTxt.replace(
                        i, '*INCLUDE\n{}\n'.format(new_txt))
            with open('src/key_assem.key', 'w')as file:
                file.write(keyTxt)
            print(keyTxt)
            return('src/key_assem.key')

        if self.name == 'YARIS':
            keys = glob.glob(self.data_path + '/*.key')
            return(keys)
        if self.name == 'YARIS_BUMPER':
            keys = glob.glob(self.data_path + '/main*.key')
            return(keys)
        if self.name == 'envs':
            keys = glob.glob(self.data_path + '/main*.k')
            return(keys)

    def CEVT(self):
        self.unit = {"IE": 1e-3, "t": 1}  # from Nmm, ms to kNmm,ms
        self.err = {"release": {
            'stcr': {
                    'fp3': {
                        "errList": [
                            'cm1e_stcr_173_fp3__001',  # HH-LL
                            'cm1e_stcr_276_fp3__001',  # early termination
                            # early termination, not bipartite graph,  all releases 960810
                            'cm1e_stcr_248_fp3__903',
                            'cm1e_stcr_248_fp3__001',  # early termination, HH-LL,  all releases 960810
                        ]
                    },
                    'fod': {
                        "errList": [
                            'cm1e_stcr_075_fod__001',  # HH-LL
                            'cm1e_stcr_230_fod__001',  # DOE fingerprint, negative IE
                            'cm1e_stcr_332_fod__001',  # DOE fingerprint, negative IE
                        ]
                    },
                    'fo5': {
                        "errList": [
                            'cm1e_stcr_386_fo5__001',
                            'cm1e_stcr_237_fo5__001',  # OLD and HH-LL
                            'cm1e_stcr_003_fo5__001',  # graph vis outlier
                            'cm1e_stcr_035_fo5__001',  # graph vis outlier
                            'cm1e_stcr_090_fo5__001',  # graph vis outlier
                            'cm1e_stcr_094_fo5__001',  # graph vis outlier
                            'cm1e_stcr_206_fo5__001',  # graph vis outlier
                            'cm1e_stcr_221_fo5__001',  # graph vis outlier
                            'cm1e_stcr_239_fo5__001',  # graph vis outlier
                            'cm1e_stcr_245_fo5__001',  # graph vis outlier
                            'cm1e_stcr_320_fo5__001',  # graph vis outlier
                            'cm1e_stcr_018_fo5__001',  # graph vis outlier
                        ]
                    },
                    'fodo': {
                        "errList": [
                            'cm1e_stcr_068_fodo__001',
                            'cm1e_stcr_075_fodo__001',
                            'cm1e_stcr_204_fodo__001',
                        ]
                    }

                    },
            'stv0': {
                'fp3': {
                    "errList": [
                        'cm1e_stv0_104_fp3__001',   # manual, LL-HH
                        'cm1e_stv0_055_fp3__001',  # manual,  all releases 960810
                        'cm1e_stv0_105_fp3__001',  # manual
                        'cm1e_stv0_100_fp3__001',  # manual
                        'cm1e_stv0_095_fp3__001',  # manual
                        'cm1e_stv0_068_fp3__001',  # manual
                        'cm1e_stv0_065_fp3__001',  # manual
                        'cm1e_stv0_060_fp3__001',  # manual
                        'cm1e_stv0_059_fp3__001',  # manual
                        'cm1e_stv0_057_fp3__001',  # manual
                        'cm1e_stv0_055_fp3__001',  # graphvis, pid 960810
                    ]
                },
                'fod': {
                    "errList": [
                        'cm1e_stv0_044_fod__001',  # graphvis, pid 960810
                    ]
                },
                'fo5': {
                    "errList": [
                        'cm1e_stv0_103_fo5__001',  # HH-LL
                        'cm1e_stv0_089_fo5__001',  # HH-LL
                    ]
                }
            },
            'stv03': {
                'fo5': {
                    "errList": [
                        'cm1e_stv03_002_fo5__001',
                        'cm1e_stv03_164_fo5__001',  # HH-LL
                        'cm1e_stv03_138_fo5__001',  # graph vis, all releases 961411,961211,
                        'cm1e_stv03_149_fo5__001',  # DOE fingerprint, negative IE
                    ]
                },
                'fp3': {
                    "errList": [
                        'cm1e_stv03_001_fp3__001',  # DOE fingerprint, negative IE
                        'cm1e_stv03_064_fp3__001',  # high ti, graph vis, all releases 961411
                        # 'cm1e_stv03_317_fp3__001',  # eraly termination
                    ]
                },
                'fod': {
                    "errList": [
                        # 'cm1e_stv03_253_fod__001',  # missing result to cross check logbook
                        # 'cm1e_stv03_274_fod__001',  # missing result to cross check logbook
                        # 'cm1e_stv03_179_fod__001',  # missing result to cross check logbook
                        # 'cm1e_stv03_279_fod__001',  # missing result to cross check logbook
                        # 'cm1e_stv03_002_fod__001',  # early termination
                    ]
                }

            },
            'm1': {
                'fo5': {
                    "errList": [
                        'cm1e_m1_007_fo5__001',  # HH-LL
                        'cm1e_m1_050_fo5__001',  # HH-LL
                        'cm1e_m1_043_fo5__001',  # graph vis, all releases 961411,961211
                        'cm1e_m1_050_fo5__001',  # graph vis, all releases 961411,961211
                    ]
                },
                'fp3': {
                    "errList": [
                        # 'cm1e_m1_049_fp3__001',  # high ti
                        # 'cm1e_m1_067_fp3__001',  # zero ti
                        # 'cm1e_m1_068_fp3__001',  # zero ti
                        # 'cm1e_m1_069_fp3__001',  # zero ti
                        'cm1e_m1_061_fp3__001',  # high ti
                        # 'cm1e_m1_073_fp3__001',  # high tn
                        # 'cm1e_m1_021_fp3__001',  # err termination, low tn
                    ]
                },
                'fod': {
                    "errList": [
                    ]
                }

            }
        }}
        self.lc_pos = 3
        self.rls_pos = 1

    def NISSAN(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 1
        self.rls_pos = 0

    def PEDPRO(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 0
        self.rls_pos = 0

    def YARIS(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 0
        self.rls_pos = 0

    def ENVS(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 0
        self.rls_pos = 0

    def TL2PID(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 0
        self.rls_pos = -2

    def envs(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 0
        self.rls_pos = 0

    def YARIS_BUMPER(self):
        self.unit = {"IE": 1e-3, "t": 1000}  # from Nmm, s to kNmm,ms
        self.err = {"release": {
            '': {
                '': {
                    "errList": [
                        '121147.draco', '121374.draco', '120954.draco', '121043.draco', '120959.draco', '121166.draco', '121397.draco', '120955.draco', '121053.draco', '121391.draco', '121178.draco', '121248.draco', '121377.draco', '121132.draco', '121562.draco', '121382.draco', '121360.draco', '121363.draco', '121366.draco', '121393.draco', '121256.draco', '121143.draco', '121242.draco', '121372.draco', '120957.draco', '121231.draco', '121395.draco', '121549.draco', '121138.draco', '121380.draco', '121140.draco', '121131.draco', '121134.draco', '121535.draco', '121534.draco', '121135.draco', '121047.draco', '121232.draco', '121546.draco', '121558.draco', '121215.draco', '121381.draco', '121359.draco', '121365.draco', '121559.draco', '121388.draco', '121383.draco', '121139.draco', '121092.draco', '121540.draco', '121081.draco', '121551.draco'
                    ]
                }
            }
        }}
        self.lc_pos = 0
        self.rls_pos = 0

    def cypher(self):
        return oems.query(self)

    class query:

        def __init__(self, oem):
            self.oem = oem
            self.oem.set_data_path()
            self.oem.backend_server()
            self.driver = self.oem.driver

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
            import numpy as np
            import ld_data.functions as KG

            def nrg_normalized(nameReg, fts, norm_opt):
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

            sims_nrg = []
            sims_pid = []
            sims = []
            for s in self.simList:
                sim = KG.Sim.nodes.get_or_none(sim_name=s)
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
            import numpy as np
            import pandas as pd
            import re

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
            # ['.*'], limit=ns)

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
            # c_lc = get_color(len(lcU))
            c_rls = rlsU
            # c_rls = get_color(len(rlsU))

            try:
                pidU = np.sort(np.unique(sims_pid))
                ntPid = sims_pid.shape[1]
            except:
                sims_pid_flat = []
                for s in sims_pid:
                    sims_pid_flat += list(s)
                pidU = np.sort(np.unique(np.array(sims_pid_flat)))
                ntPid = min([s.shape for s in sims_pid])[0]
            # pidU, countPid = self.make_pidList(regs, pidU.tolist()) doens't work when more loadceses are loaded as regs and should be edited to work with regs_list
            pidU, countPid = self.make_pidList('.*', pidU.tolist())
            pidU = np.array(pidU)
            pidU = np.delete(pidU, np.argwhere(pidU == 0))

            c_pid = get_color(len(pidU))
            def ordinal(n): return "%d%s" % (
                n, "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10::4])
            c_ord = np.repeat(
                np.array([[ordinal(x)
                           for x in range(1, ntPid + 1)]], dtype=object),
                # [get_color(sims_pid.shape[1])],
                len(sims), axis=0)
            c_ord[:, nOrd:] = '{0} to {1}'.format(
                ordinal(nOrd + 1), ordinal(ntPid))
            # c = ['b', 'r', 'g', 'c', 'm', 'y', 'k', '#4AFFB5', '#593F50']

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

                relPath = '/'.join(self.oem.save_path.split('/')[-2:])

                dfi['pic'] = ['./{0}/{1}_{2}_iso.png'.format(
                    relPath, s, int(pi)) for s in sim]

                # for missing data in cevt
                # for pici in dfi['pic']:
                #     if glob.glob(pici) == []:
                #         makePath = (
                #             # self.oem.save_path +
                #             self.oem.save_path +
                #             '/*_{1}_*{0}*_iso.png'.format(int(pi), dfi['c_rls'][0]))
                #         pic_pid = glob.glob(makePath)
                #         try:
                #             pic = pic_pid[0].split('/')[-1]
                #             dfi['pic'] = [
                #                 './{0}/{1}'.format(relPath, pic) for s in sim]
                #         except IndexError:
                #             dfi['pic'] = ['' for s in sim]
                #             pass

                df = df.append(dfi)
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


# oem = oems('YARIS')
# oem = oems('TL2PID')
# # oem = oems('ENVS')
# oem = oems('NISSAN')
# oem.set_data_path()

# oem.set_data_path(
#     inPath='/home/apakiman/leo1/Projects/carGraph/runs/YARIS/full_front/TL2PID_12_0[56]*/*TL2PID*')
# #     inPath='/home/ndv/stud/data/LOAD_PATH_DTCT/env_0/episode_*/')

# print(glob.glob(oem.data_path))
# oem.metapost()

# oem = oems('CEVT', 'stv03', 'fp3')
# oem.cypher().out_dataframe(ns=100, nPID=5, nOrd=5, regs='.*stv03.*fp3.*')
# oem.metapost()

if __name__ == '__main__':

    import sys

    print(sys.argv[1:])
