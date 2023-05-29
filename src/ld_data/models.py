# from django.db import models
import neomodel
import numpy as np


class WgtRel(neomodel.StructuredRel):

    '''
    Defining weighted relations for edges. 
    the array size is the same to relay the key value
    relation of each property. 

        w_e_value   array type describing a list of values.
        w_e_key     array type describing a list of keys.
    '''

    # weight = neomodel.FloatProperty()

    w_e_value = neomodel.ArrayProperty()
    w_e_key = neomodel.ArrayProperty()


class GrpFts(neomodel.StructuredRel):

    '''
    contain the properties of the grouped parts 
    related to the simulation, the _e_ in naming is 
    referring to the edge property

        type    describes the method
                box_mrg
        key     list of names of the variable
                nrg_max
        value   list of values
    '''

    grp_e_type = neomodel.StringProperty()
    grp_e_value = neomodel.ArrayProperty()
    grp_e_key = neomodel.ArrayProperty()


class Fts(neomodel.StructuredRel):

    '''
    features value for an FE entity connected to its method
    that the keys are in the Ftsthe _e_ in naming is 
    referring to the edge property

        value   list of values, the keys are in the target
                node, Mthd
    '''

    value = neomodel.ArrayProperty()


class Sim(neomodel.StructuredNode):

    """the simulation nodes that hold the global features of the simulation e.g. 
    total energy 

    # Generic properties

        uid                 the unique id for simulaiton
        sim_name            simulation name should be a unique property
        sim_oem             to which car factory the result belongs to
        sim_abb             simulation name abbreviation, for better visualization of the name
        sim_lc              load case of the analysis e.g pedestrian head impact
        sim_rel             the development stage (release) of the product
        sim_dspln           the analysis discipline e.g. safety, durability
        dim_description     the specific labelling of the data e.g. optimization 
        sim_path_pre        path to the file for input simulation data
        sim_path_post       path to the file for output simulation data

    # Crash-specific properties
    # Energy features, initial time(t_i), absorption end time (t_n)

        sim_ie_tot_max      max of the total internal energy of the simulation    
        sim_ie_prt_max      max of the internal energy of the parts in a simulation
        sim_ke_tot_max      max of the total kinetic energy of the simulation 
        sim_ie_ti_ll        simulation t_i with the lower limit method (ll)                        
        sim_ie_ti_grad      simulation t_i with the gradient method (grad)
        sim_ie_tn_pct       simulation t_n with the percentage method (pct)
        sim_ie_tn_max       simulation t_n with the maximum method (max)
        sim_tn_pct_max      max of the parts for tn_pct value
        sim_tn_max_max      max of the parts for tn_max value
        sim_ti_ll_pp        ranges of the values for ti_ll
        sim_ti_grad_pp      ranges of the values for ti_grad
        sim_t_max           max of t_n


    Returns:
        boolean: None
    """

    uid = neomodel.UniqueIdProperty()
    sim_name = neomodel.StringProperty(unique_index=True)
    sim_oem = neomodel.StringProperty()
    sim_abb = neomodel.StringProperty()
    sim_lc = neomodel.StringProperty()
    sim_rel = neomodel.StringProperty()
    sim_dspln = neomodel.StringProperty()
    sim_description = neomodel.StringProperty()
    sim_path_pre = neomodel.StringProperty()
    sim_path_post = neomodel.StringProperty()

    # relations
    sim_nrg_part = neomodel.Relationship('Part', 'NRG_PART', model=WgtRel)
    sim_node = neomodel.RelationshipTo('Node', 'OUT_NODE')
    sim_elmnt = neomodel.RelationshipTo('Elmnt', 'OUT_ELMNT')
    sim_model = neomodel.RelationshipTo('Model', 'SIM_MODEL')
    sim_imp = neomodel.RelationshipTo('Imp', 'SIM_IMP')
    sim_barr = neomodel.RelationshipTo('Imp', 'SIM_BARR')
    sim_des = neomodel.RelationshipTo('Des', 'SIM_DES', model=WgtRel)

    # energy features
    sim_ie_tot_max = neomodel.FloatProperty()
    sim_ie_prt_max = neomodel.FloatProperty()
    sim_ke_tot_max = neomodel.FloatProperty()
    sim_ie_ti_ll = neomodel.FloatProperty()
    sim_ie_ti_grad = neomodel.FloatProperty()
    sim_ie_tn_pct = neomodel.FloatProperty()
    sim_ie_tn_max = neomodel.FloatProperty()
    sim_tn_pct_max = neomodel.FloatProperty()
    sim_tn_max_max = neomodel.FloatProperty()
    sim_ti_ll_pp = neomodel.FloatProperty()
    sim_ti_grad_pp = neomodel.FloatProperty()
    sim_t_max = neomodel.FloatProperty()

    def get_childNodes(self, prop, node):
        '''
        query to get the analysis depending on the current analysis.

        return Nodes
        '''

        cypherTxt = '''
            match (p:{2})-[*1]-(n:Sim)
            where n.sim_name='{0}'
            return p.{1}
            '''.format(self.sim_name, prop, node)
        idList, _ = self.cypher(cypherTxt)

        idList = [x[0] for x in idList]
        return(idList)

    def get_child(self):
        '''
        query to get the analysis depending on the current analysis.

        return list of strings, simulation names
        '''

        childList, _ = self.cypher('''
            match (n1)-[r:sim_based_on]->(n2)
            where n1.sim_name='{0}'
            return n2.sim_name
            '''.format(self.sim_name))
        try:
            child = [str(i[0]) for i in childList]
        except IndexError:
            child = []
        return child

    def forward_length(self, maxLen):
        '''
        return the number of analysis done after current analysis
        '''

        c_len_list = {}
        child_list = self.get_child()
        for c in child_list:
            cL, _ = self.cypher('''
                match p=(n1:Sim)-[r:sim_based_on*1..{}]->(n2:Sim)
                where n1.sim_name = '{}'
                return length(p) order by length(p) desc limit 1
                '''.format(maxLen, c))
            try:
                c_len_list.update([(c, cL[0][0])])
                print(c)
            except IndexError:
                if cL == []:
                    c_len_list.update([(c, 0)])
        return(c_len_list)

    def get_reference(self):
        '''
        What are the references of the current analysises
        '''

        refList, _ = self.cypher('''
            match (n1)-[r:sim_based_on]->(n2)
            where n2.sim_name='{0}'
            return n1.sim_name
            '''.format(self.sim_name))
        try:
            # reference = ', '.join(str(i[0]) for i in refList)
            reference = [str(i[0]) for i in refList]
        except IndexError:
            reference = []
            reference = ''

        return reference

    def get_model(self):
        '''
        What is the input FE-model into the current analyseses
        '''

        model_id, _ = self.cypher('''
            match (n1:Sim)-[]-(n2:Model)
            where n1.sim_name='{}'
            return n2.model_id
        '''.format(self.sim_name))

        return model_id[0][0]

    def get_parts(self, relp=''):
        '''
        what are the important parts of the current analyses
        '''

        cypherTxt = '''
            match (n:Sim)-[r{1}]-(p:Part)
            where n.sim_name='{0}'
            return p.part_id order by r.ec_order
            '''.format(self.sim_name, relp)

        partList, _ = self.cypher(cypherTxt)

        part = [x[0] for x in partList]

        return(part)

    def get_childByID(self, id, node):
        '''
        find the analysis depending on the current analysis with node input.
        '''

        prop = node.lower() + '_id'
        cypherTxt = '''
            match (n:Sim)-[*1..3]-(p:{2})
            where n.sim_name='{0}' and  p.{3}={1}
            return p.uid
            '''.format(self.sim_name, id, node, prop)
        child, _ = self.cypher(cypherTxt)

        if not child == []:
            child = child[0][0]

        return(child)

    def embed_nrg(self, ft, err=[]):
        '''
        normalize enery features based on options

        opt1 = ti_ll, tn_pct
        opt2 = ti_grad
        '''
        cypherTxt = '''
        match (s:Sim)-[r:NRG_PART]-(p)
        where s.sim_name='{0}'
        return p.{1}, p.{2}, p.{3}, p.part_id
        order by p.{1} desc
        '''.format(self.sim_name, ft[0], ft[1], ft[2])
        # print(cypherTxt)
        a = self.cypher(cypherTxt)
        a = np.asarray(a[:-1][0])
        return(a)

    def embed_nrg_sel(self, ft, pid_list, err=[]):
        '''
        normalize enery features based on options

            opt1 = ti_ll, tn_pct
            opt2 = ti_grad
        '''

        cypherTxt = '''
        match (s:Sim)-[r:NRG_PART]-(p)
        where s.sim_name='{0}' and p.part_id in {4}
        return p.{1}, p.{2}, p.{3}, p.part_id
        order by p.{1} desc
        '''.format(self.sim_name, ft[0], ft[1], ft[2], pid_list)
        # print(cypherTxt)
        a = self.cypher(cypherTxt)
        a = np.asarray(a[:-1][0])
        return(a)

    def get_behavs(self):
        '''
        get the behaviour nodes belonging to an analysis.
        '''

        cypherTxt = '''
            match (e:Behav)-[:PART_BEHAV]-(p:Part)-[:NRG_PART]-(s:Sim)
            where s.uid = '{0}'
            return e.uid
            '''.format(self.uid)
        # print(cypherTxt)
        eUids = self.cypher(cypherTxt)
        eUids = [x[0] for x in eUids[0]]

        return(eUids)

    def get_deses(self):
        '''
        get the designs nodes belonging to an analysis.
        '''

        cypherTxt = '''
            match (m:Des)-[:PART_DES]-(p:Part)-[:NRG_PART]-(s:Sim)
            where s.uid = '{0}'
            return m.uid
            '''.format(self.uid)
        # print(cypherTxt)
        mUids = self.cypher(cypherTxt)
        mUids = [x[0] for x in mUids[0]]

        return(mUids)


class Part(neomodel.StructuredNode):
    """The part nodes. The main data from simulations are stored in this node 
    and later modelled on the remaining introduces semantics, e.g. edge weights.

    # Generic properties

        :parameter uid:           the unique id for simulaiton
        :parameter part_id:       the identification number of part in the 
                                  simulation
        :parameter part_name:     the name of the part from the simulation
        :parameter part_sim_name: the simulation name of the part
        :parameter part_sim_abb:  the simulation abbreviation name 

    # Part as a box
        part_cog            the centre of the part if located in a box, x, y, z
        part_min            the min x,y,z of the box holding the part
        part_max            the max x,y,z of the box holding the part


    Args:
        neomodel (neomodel.StructuredNode)

    Returns:
        boolean: None"""

    uid = neomodel.UniqueIdProperty()
    part_id = neomodel.IntegerProperty()
    part_name = neomodel.StringProperty()
    part_sim_name = neomodel.StringProperty()
    part_sim_abb = neomodel.StringProperty()

    part_cog = neomodel.ArrayProperty()
    part_min = neomodel.ArrayProperty()
    part_max = neomodel.ArrayProperty()

    part_des = neomodel.Relationship('Des', 'PART_DES', model=WgtRel)
    part_behav = neomodel.Relationship('Behav', 'PART_BEHAV')
    Ipart_pltf = neomodel.Relationship('Pltf', 'BELONGS_TO')
    part_ubdy = neomodel.Relationship('Ubdy',  'BELONGS_TO')
    part_fts = neomodel.Relationship('Mthd', 'PART_FTS', model=Fts)

    def part_nrg(self, fts):
        '''
        Returning the specified energy feature of a part.
        '''

        cypherTxt = '''
            match (p:Part)
            where p.uid='{0}'
            return p.{1}
            order by p.{1} desc
        '''.format(self.uid, fts)

        a = self.cypher(cypherTxt)
        a = np.array(a[0][0])
        return(a)

    def part_power(self):

        cypherTxt = '''
            match (p:Part)
            where p.uid='{0}'
            return p.nrg_max, p.tn_pct, p.ti_grad
            '''.format(self.uid)
        # print(cypherTxt)

        a = self.cypher(cypherTxt)
        IE, tn, ti = np.array(a[0][0])
        power = IE / (tn - ti)
        return(power)

    def part_behav_con(self, uid):

        cypherTxt = '''
            match (p:Part)-[r:PART_BEHAV]-(e:Behav)
            where p.uid = "{0}"
            and e.uid = "{1}"
            return r
        '''.format(self.uid, uid)

        a = self.cypher(cypherTxt)
        if not a[0] == []:
            return(True)
        else:
            return(False)

    def get_sim_name(self):

        cypherTxt = '''
            match (s:Sim)-[:INCL_PART | :NRG_PART]-(p:Part)
            where p.uid='{0}'
            return distinct s.sim_name, s.sim_abb
            '''.format(self.uid)

        simName, _ = self.cypher(cypherTxt)
        return(simName[0])

    def get_sim(self):

        cypherTxt = '''
            match (s:Sim)-[:NRG_PART]-(p:Part)
            where p.uid = '{0}'
            return s.uid
            '''.format(self.uid)
        # print(cypherTxt)
        sUids = self.cypher(cypherTxt)
        sUids = [x[0] for x in sUids[0]]

        return(sUids[0])

    def get_des(self, typ):

        cypherTxt = '''
            match (m:Des)-[:PART_DES]-(p:Part)
            where p.uid = '{0}' and m.des_type='{1}'
            return m.uid
            '''.format(self.uid, typ)
        # print(cypherTxt)
        mUids = self.cypher(cypherTxt)
        mUids = [x[0] for x in mUids[0]]

        return(mUids[0])


class Node(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    node_id = neomodel.IntegerProperty()
    node_name = neomodel.StringProperty()
    node_type = neomodel.StringProperty()

    node_xyz = neomodel.ArrayProperty()
    node_max_disp_xyz = neomodel.ArrayProperty()

    node_part = neomodel.Relationship('Part', 'IN_PART')


class Elmnt(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    elmnt_id = neomodel.IntegerProperty()
    elmnt_name = neomodel.StringProperty()
    elmnt_type = neomodel.StringProperty()

    elmnt_xyz = neomodel.ArrayProperty()
    elmnt_max_disp_xyz = neomodel.ArrayProperty()

    elmnt_part = neomodel.Relationship('Part', 'IN_PART')


class Des(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    des_type = neomodel.StringProperty()

    des_pid = neomodel.IntegerProperty()

    # des_behav = neomodel.RelationshipTo('Behav', 'DES_BEHAV')

    def get_byType(self):

        cypherTxt = '''
            match (p:Part)-[r:PART_DES]-(m:Des)
            where r.type = {0}
            return m.uid
            '''.format(type)

        # uid = self.cypher(cypherTxt)

        # idList = [x[0] for x in idList]
        return('test')

    def get_part_nrg(self, ft):

        cypherTxt = '''
            match (s:Sim)-[:NRG_PART]-(p:Part)-[]-(m:Des)
            where m.des_pid = {0}
            return p.{1}, p.{2}, p.{3}
            order by p.{1} desc
            '''.format(self.des_pid, ft[0], ft[1], ft[2])
        a = self.cypher(cypherTxt)
        print(cypherTxt)
        a = np.asarray(a[:-1][0])
        return(a)

    def get_behav(self):

        cypherTxt = '''
            match (e:Behav)-[:PART_BEHAV]-(p:Part)-[:PART_DES]-(m:Des)
            where m.uid = '{0}'
            return e.uid
            '''.format(self.uid)
        # print(cypherTxt)
        eUids = self.cypher(cypherTxt)
        eUids = [x[0] for x in eUids[0]]

        return(eUids)

    def get_sims(self):

        cypherTxt = '''
            match (m:Des)-[:PART_DES]-(p:Part)-[:NRG_PART]-(s:Sim)
            where m.uid = '{0}'
            return s.uid
        '''.format(self.uid)
        # print(cypherTxt)
        sUids = self.cypher(cypherTxt)
        sUids = [x[0] for x in sUids[0]]

        return(sUids)


class Behav(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    behav_type = neomodel.StringProperty()

    behav_embd = neomodel.FloatProperty(unique_index=True)
    behav_id = neomodel.StringProperty()

    behav_method = neomodel.StringProperty()

    def get_sims(self):

        cypherTxt = '''
            match (e:Behav)-[:PART_BEHAV]-(p:Part)-[:NRG_PART]-(s:Sim)
            where e.uid = '{0}'
            return s.uid
            '''.format(self.uid)
        # print(cypherTxt)
        sUids = self.cypher(cypherTxt)
        sUids = [x[0] for x in sUids[0]]

        return(sUids)


class Model(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    model_name = neomodel.StringProperty(unique_index=True)
    model_name = neomodel.StringProperty()

    # connection between models with same parent
    model_ref = neomodel.RelationshipTo('Model', 'MODEL_REF')
    # based on status , model status
    model_base_status = neomodel.RelationshipTo('Model', 'MODEL_STATUS')

    model_veh = neomodel.Relationship('Veh', 'MODEL_VEH')
    model_attr = neomodel.Relationship('Attr', 'MODEL_ATTR')

    def get_media(self):

        media, _ = self.cypher('''
            match (m:Model)-[]-(n:Media)
            where m.model_name='{0}'
            return n.media_path
                '''.format(self.model_name))
        mediaList = [x[0] for x in media]
        return mediaList

    def get_baseStatusModel(self):

        baseList, _ = self.cypher('''
            match (m1:Model)-[r:MODEL_STATUS]->(m2:Model)
            where m1.model_name='{0}'
            return m2.model_name
            '''.format(self.model_name))

        baseModels = [i[0] for i in baseList]
        return baseModels

    def get_sims(self):

        simOut, _ = self.cypher('''
            match (m:Model)-[]-(n:Sim)
            where m.model_name='{0}'
            return n 
            '''.format(self.model_name))

        sims = [r[0] for r in simOut]

        return(sims)

    def get_base_sims(self):

        simOut, _ = self.cypher('''
            match (m1:Model)-[]-(n1:Sim)
            where m1.model_name='{0}'
            with collect(n1) as currentSims
            unwind currentSims as sim
            match (m1:Model)-[r:MODEL_STATUS]->(m2:Model)-[]-(n2:Sim)
            where m1.model_name='{0}' and n2.sim_row = sim.sim_row and n2.sim_col = sim.sim_col 
            with collect(n2.uid) as updatedSims
            match (m1:Model)-[r:MODEL_STATUS]->(m2:Model)-[]-(n3:Sim)
            where m1.model_name='{0}'  and NOT n3.uid in updatedSims
            with collect(n3) as baseSims

            return baseSims

            '''.format(self.model_name))

        sims = [r[0] for r in simOut]

        return(sims)


class Imp(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    imp_test_code = neomodel.StringProperty()  # test code
    # imp_name = neomodel.StringProperty()  # impact
    imp_loadcase = neomodel.StringProperty()  # loadcase
    imp_discipline = neomodel.StringProperty()  # passive safety
    imp_velocity_unit = neomodel.StringProperty()
    imp_angle_unit = neomodel.StringProperty()

    # specific properties
    imp_name = neomodel.StringProperty()
    imp_velocity = neomodel.FloatProperty()
    imp_angle = neomodel.FloatProperty()


class Barr(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    barr_id = neomodel.IntegerProperty(unique_index=True)
    barr_name = neomodel.StringProperty()


class Grp(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    grp_type = neomodel.StringProperty()

    grp_cog = neomodel.ArrayProperty(
        unique_index=True)  # center of the box, if
    grp_rng = neomodel.ArrayProperty(unique_index=True)

    grp_part = neomodel.RelationshipTo('Part', 'CONT_PART')
    grp_sim = neomodel.RelationshipTo('Sim', 'GRP_FTS', model=GrpFts)


class Veh(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    veh_name = neomodel.StringProperty()
    veh_abb = neomodel.StringProperty()
    veh_spec = neomodel.JSONProperty()

    veh_ubdy = neomodel.Relationship('Ubody', 'VEH_UBDY')
    veh_pltf = neomodel.Relationship('Pltf', 'VEH_PLTF')


class Pltf(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    pltf_name = neomodel.StringProperty()
    pltf_spec = neomodel.JSONProperty()


class Ubdy(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    ubdy_name = neomodel.StringProperty()
    ubdy_spec = neomodel.JSONProperty()


class Attr(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    attr_name = neomodel.StringProperty()
    attr_spec = neomodel.JSONProperty()

    attr_prtcl = neomodel.Relationship('Prtcl', 'ATTR_PRTCL')


class Prtcl(neomodel.StructuredNode):

    uid = neomodel.UniqueIdProperty()
    prtcl_name = neomodel.StringProperty()
    prtcl_spec = neomodel.JSONProperty()


class Mthd(neomodel.StructuredNode):

    '''
    contain the properties of the method for feature extraction
    that can be connected to part, node and element. that reduces their
    properties in time to some features. 

        name          the method name
                      e.g. nrg_parts, nrg_parts_grid  

        cnfg_key      list of name of the configuration variable
                      e.g. n_grid, nrg_pctMax

        cnfg_value    list of configuration values

        fts_keys      list of names of features that are linked to this node 
    '''
    uid = neomodel.UniqueIdProperty()
    mthd_name = neomodel.StringProperty()
    mthd_cnfg_keys = neomodel.ArrayProperty()
    mthd_cnfg_vals = neomodel.ArrayProperty()
    mthd_keys = neomodel.ArrayProperty()

    # ------------------------------------------
    # should implement from part to Mthd node and Fts edge
    # nrg_max = neomodel.FloatProperty()
    # ti_ll = neomodel.FloatProperty()
    # ti_grad = neomodel.FloatProperty()
    # tn_pct = neomodel.FloatProperty()
    # tn_max = neomodel.FloatProperty()
    # ------------------------------------------
