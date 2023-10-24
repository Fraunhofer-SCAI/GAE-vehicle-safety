# from django.db import models
import neomodel
import numpy as np


#----------------
# Edges

class WgtRel(neomodel.StructuredRel):
    """
    Weighted Relations edge.

    Defining weighted relations for edges. The array size is the same to relay the key value relation of each property. 

    :param w_e_value:   
        Describing a list of values.
    :type w_e_value:    
        array
    :param w_e_key:     
        Array type describing a list of keys.
    :type w_e_key:      
        array
    """

    w_e_value = neomodel.ArrayProperty()
    w_e_key = neomodel.ArrayProperty()


class GrpFts(neomodel.StructuredRel):
    """
    Grouped Features edge.

    Contain the properties of the grouped parts related to the simulation, the _e_ in naming refers to the edge property.

    :param grp_e_type:
        Describes the method for merging features, box_mrg or nrg_max.               
    :type grp_e_type:
        string          
    :param grp_e_key:
        List of names of the variable.               
    :type grp_e_key: 
        array   
    :param grp_e_value:
       List of values of the variables.
    :type grp_e_value:
        array
    """

    grp_e_type = neomodel.StringProperty()
    grp_e_value = neomodel.ArrayProperty()
    grp_e_key = neomodel.ArrayProperty()


class Fts(neomodel.StructuredRel):
    """
    Features edge.

    Relation defining the method features. Feature values for an FE entity connected to its method that the keys are in the Fts the \_e\_ in naming is referring to the edge property.

    :param value:
       List of values, the keys are in the target node, Mthd.
    :type value: array
    """

    value = neomodel.ArrayProperty()


class Attr(neomodel.StructuredNode):
    """
    Attribute of the analyses.

    The analysis discipline, e.g. safety and durability.

    :param uid:
        The unique ID for Attribute.
    :type uid: neomodel.UniqueIdProperty
    :param attr_name:
        Name of the attribute, e.g. safety.
    :type attr_name: string
    :param attr_spec:
        The attribute specifications, including keys and values in JSON format.
    :type attr_spec:
        json
    :param attr_prtcl:
        The edge connects the attribute to its protocols, "ATTR_PRTCL".
    :type attr_prtcl:
        neomodel.Relationship
    """

    uid = neomodel.UniqueIdProperty()
    attr_name = neomodel.StringProperty()
    attr_spec = neomodel.JSONProperty()

    attr_prtcl = neomodel.Relationship('Prtcl', 'ATTR_PRTCL')


#----------------
# Nodes

class Barr(neomodel.StructuredNode):
    """
    Barrier node.

    Describing the feature of the barrier in the analysis. Specific for crash analysis.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param barr_id: 
        Identifying the barrier id, unique_index=True.
    :type barr_id: int
    :param barr_name:
        Barrier name.
    :type barr_name: string
    """

    uid = neomodel.UniqueIdProperty()
    barr_id = neomodel.IntegerProperty(unique_index=True)
    barr_name = neomodel.StringProperty()


class Behav(neomodel.StructuredNode):
    """
    Behaviour node.

    Bucketing the FE-simulation output features.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param behav_type: 
        Then name of the type of the behaviour.
    :type behav_type: string
    :param behav_embd:
        The single embedding value of the behaviour.
    :type behav_embd: float
    :param behav_id:
        The user identifier of the behaviour.
    :type behav_id: string
    :param behav_method:
        The method name that is used for extracting the behaviour embedding.
    :type behav_method: string
    """

    uid = neomodel.UniqueIdProperty()
    behav_type = neomodel.StringProperty()

    behav_embd = neomodel.FloatProperty(unique_index=True)
    behav_id = neomodel.StringProperty()

    behav_method = neomodel.StringProperty()

    def get_sims(self):
        """
        Get simulations of a specific behaviour.

        :returns: List of unique IDs, of simulations.
        :rtype: list
        """

        cypherTxt = """
            match (e:Behav)-[:PART_BEHAV]-(p:Part)-[:NRG_PART]-(s:Sim)
            where e.uid = '{0}'
            return s.uid
            """.format(self.uid)
        # print(cypherTxt)
        sUids = self.cypher(cypherTxt)
        sUids = [x[0] for x in sUids[0]]

        return(sUids)


class Des(neomodel.StructuredNode):
    """
    Design node.

    Introduced class to connect parts of FE-models based on the similarity of the input data.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param des_type: 
        Then name of the type of the design.
    :type behav_type: string
    :param des_pid:
        The property ID (PID), of the design.
    :type behav_embd: int
    """

    uid = neomodel.UniqueIdProperty()
    des_type = neomodel.StringProperty()
    des_pid = neomodel.IntegerProperty()


    def get_byType(self):
        """
        Get parts belonging to a design, considering the design type.

        :returns: List of unique IDs, of parts.
        :rtype: list
        """

        cypherTxt = """
            match (p:Part)-[r:PART_DES]-(m:Des)
            where r.type = {0}
            return m.uid
            """.format(type)

        result = self.cypher(cypherTxt)
        idList = [x[0] for x in result]

        return(idList)

    def get_part_nrg(self, ft):
        """
        Get energetic features of a specific design, ordered by the first feature.

        :param ft:
            List of feature names to be returned for each part.
        :type ft: list

        :returns: List of dictionaries with parts and their energy features.
        :rtype: np.asarray
        """

        cypherTxt = """
            match (s:Sim)-[:NRG_PART]-(p:Part)-[]-(m:Des)
            where m.des_pid = {0}
            return p.{1}, p.{2}, p.{3}
            order by p.{1} desc
            """.format(self.des_pid, ft[0], ft[1], ft[2])
        
        result = self.cypher(cypherTxt)
        a = np.asarray(result[:-1][0])
        
        return(a)

    def get_behav(self):
        """
        Get unique IDs of behaviours of a design.

        :returns: List of behaviours unique IDs.
        :rtype: list
        """

        cypherTxt = """
            match (b:Behav)-[:PART_BEHAV]-(p:Part)-[:PART_DES]-(m:Des)
            where m.uid = '{0}'
            return b.uid
            """.format(self.uid)
        
        result = self.cypher(cypherTxt)
        bUids = [x[0] for x in result[0]]

        return(bUids)

    def get_sims(self):
        """
        Get unique IDs of simulations of a design.

        :returns: List of simulations unique IDs.
        :rtype: list
        """

        cypherTxt = """
            match (m:Des)-[:PART_DES]-(p:Part)-[:NRG_PART]-(s:Sim)
            where m.uid = '{0}'
            return s.uid
        """.format(self.uid)

        result = self.cypher(cypherTxt)
        sUids = [x[0] for x in result[0]]

        return(sUids)


class Elmnt(neomodel.StructuredNode):
    """
    Element node.

    Describing the feature of an element in the analysis, e.g. sensor elements or section forces.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param elmnt_id: 
        The ID of the elment in the simulation.
    :type barr_id: int
    :param elmnt_name:
        Element name.
    :type elmnt_name: string
    :param elmnt_type:
        Element type.
    :type elmnt_type: string
    :param elmnt_xyz: 
        The x, y, z, C.O.G coordinates of the element. 
    :type elmnt_xyz: array
    :param elmnt_max_disp_xyz:
        Maximum displacement of the element during the simulation in x, y, and z.
    :type elmnt_max_disp_xyz: array
    :param elmnt_part:
        Relation connecting the element to its belonging part, "IN_PART".
    :type elmnt_part:
        neomodel.Relationship
    """

    uid = neomodel.UniqueIdProperty()
    elmnt_id = neomodel.IntegerProperty()
    elmnt_name = neomodel.StringProperty()
    elmnt_type = neomodel.StringProperty()

    elmnt_xyz = neomodel.ArrayProperty()
    elmnt_max_disp_xyz = neomodel.ArrayProperty()

    elmnt_part = neomodel.Relationship('Part', 'IN_PART')


class Grp(neomodel.StructuredNode):
    """
    Grouping node.

    Grouping the parts and their feature to be analysed for a desired functionality.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty

    :param grp_type:
        The type of grouping, e.g. box based or PID.
    :param grp_cog: string
    :param grp_cog:
        The centre of geometry of the box, unique_index=True.
    :type grp_cog: string
    :param grp_rng:
        The range of the box in x, y, and z direction that describes the box dimension, unique_index=True.
    :type grp_rng: array
    :param grp_part:
        The relationship connecting the parts that are grouped with the Grp node, "CONT_PART".
    :type grp_part: neomodel.RelationshipTo
    :param grp_sim:
        The relationship connecting the simulations that are grouped with the Grp node, "GRP_FTS".
    :type grp_sim: neomodel.RelationshipTo
    """

    uid = neomodel.UniqueIdProperty()
    grp_type = neomodel.StringProperty()

    grp_cog = neomodel.ArrayProperty(
        unique_index=True)  
    grp_rng = neomodel.ArrayProperty(unique_index=True)

    grp_part = neomodel.RelationshipTo('Part', 'CONT_PART')
    grp_sim = neomodel.RelationshipTo('Sim', 'GRP_FTS', model=GrpFts)


class Imp(neomodel.StructuredNode):
    """
    Impactor node.

    The FE-model of an impactor that hit several positions of a vehicle in several simulations, e.g. head and lower leg.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param imp_name:
        The name of the impactor.
    :type imp_name: string
    :param imp_loadcase:
        The name of the load case the impactor belongs to.
    :type imp_loadcase: string
    :param imp_discipline:
        The discipline name of the impactor, e.g. pedestrian.
    :type imp_discipline: string
    :param imp_test_code:
        The test code of the impact for summarization of the reports.
    :type imp_test_code: string
    :param imp_velocity:
        The velocity value of the impactor.
    :type imp_velocity: float
    :param imp_velocity_unit:
        The velocity unit.
    :type imp_velocity_unit: string
    :param imp_angle:
        The impact angle value.
    :type imp_angle: float
    :param imp_angle_unit:
        The angle of the impact.
    :type imp_angle_unit: string
        The measurement unit of the impact angle.
        
    """

    uid = neomodel.UniqueIdProperty()

    imp_name = neomodel.StringProperty()
    imp_loadcase = neomodel.StringProperty()  
    imp_discipline = neomodel.StringProperty()  
    imp_test_code = neomodel.StringProperty()  

    imp_velocity = neomodel.FloatProperty()
    imp_velocity_unit = neomodel.StringProperty()
    imp_angle = neomodel.FloatProperty()
    imp_angle_unit = neomodel.StringProperty()


class Model(neomodel.StructuredNode):
    """
    FE-Model node.

    FE-model of the vehicle disregarding the load-case.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param model_name:
        The model name, unique_index=True.
    :type model_name: string
    

    :param model_ref:
          Connecting the model to its reference analyses, 'MODEL_REF'.
    :type model_ref: neomodel.RelationshipTo
    :param model_base_status:
          Connecting the model to its base status model to predict the missing analysis, 'MODEL_STATUS'.
    :type model_base_status: neomodel.RelationshipTo
    :param model_veh:
          Connecting the model to its vehicle, 'MODEL_VEH'.
    :type model_veh: neomodel.RelationshipTo
    :param model_attr:
          Connecting the model to its attribute, 'MODEL_ATTR'.
    :type model_attr: neomodel.RelationshipTo
    """

    uid = neomodel.UniqueIdProperty()
    model_name = neomodel.StringProperty(unique_index=True)

    # connection between models with the same parent
    model_ref = neomodel.RelationshipTo('Model', 'MODEL_REF')
    # Based on status, model status
    model_base_status = neomodel.RelationshipTo('Model', 'MODEL_STATUS')

    model_veh = neomodel.Relationship('Veh', 'MODEL_VEH')
    model_attr = neomodel.Relationship('Attr', 'MODEL_ATTR')


    def get_baseStatusModel(self):
        """
        Get the base status models of the model.

        :returns: List of model names.
        :rtype: list
        """

        baseList, _ = self.cypher(
            """
            match (m1:Model)-[r:MODEL_STATUS]->(m2:Model)
            where m1.model_name='{0}'
            return m2.model_name
            """.format(self.model_name))

        baseModels = [i[0] for i in baseList]
        return baseModels

    def get_sims(self):
        """
        Get FE simulations that are analysed for the model.

        :returns: List of simulation nodes.
        :rtype: list
        """

        simOut, _ = self.cypher(
            """
            match (m:Model)-[]-(n:Sim)
            where m.model_name='{0}'
            return n 
            """.format(self.model_name))

        sims = [r[0] for r in simOut]

        return(sims)

    def get_base_sims(self):
        """
        Get base simulations for pedestrian points that results are missing for the model.

        :returns: List of simulations.
        :rtype: list
        """

        simOut, _ = self.cypher(
            """
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

            """.format(self.model_name))

        sims = [r[0] for r in simOut]

        return(sims)


class Mthd(neomodel.StructuredNode):

    """
    Method node.

    Contains the properties of the method for feature extraction that can be connected to part, node and element. that reduces their properties in time to some features. 

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param mthd_name:
        the method name e.g. nrg_parts, nrg_parts_grid  
    :type mthd_name: string
    :param mthd_cnfg_key:
        list of name of the configuration variable e.g. n_grid, nrg_pctMax
    :type mthd_cnfg_key: array
    :param mthd_cnfg_value:
        list of configuration values
    :type mthd_cnfg_value: array
    :param mthd_fts_keys:
        list of names of features that are linked to this node 
    :type mthd_fts_keys: array
    """

    uid = neomodel.UniqueIdProperty()
    mthd_name = neomodel.StringProperty()
    mthd_cnfg_keys = neomodel.ArrayProperty()
    mthd_cnfg_vals = neomodel.ArrayProperty()
    mthd_keys = neomodel.ArrayProperty()


class Node(neomodel.StructuredNode):
    """
    FE-model node node.

    Specific nodes of the FE-model that are for time history output.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param node_id:
        The analysis identifier ID of the node.
    :type node_id: int 
    :param node_name:
        The node name in the analyses.
    :type node_name: string 
    :param node_type:
        The node type specifies the format of time series data.
    :type node_type: string 
    :param node_xyz:
        The coordinates of the node.
    :type node_xyz: array 
    :param node_max_disp_xyz:
        Max displacement of the node in x, y, and z.
    :type node_max_disp_xyz: array 
    :param node_part:
        Connecting the node to the part node it is belonging to, "IN_PART"
    :type node_part: neomodel.Relationship
    """

    uid = neomodel.UniqueIdProperty()
    node_id = neomodel.IntegerProperty()
    node_name = neomodel.StringProperty()
    node_type = neomodel.StringProperty()

    node_xyz = neomodel.ArrayProperty()
    node_max_disp_xyz = neomodel.ArrayProperty()

    node_part = neomodel.Relationship('Part', 'IN_PART')


class Part(neomodel.StructuredNode):
    """
    Finite Element Part node.

    The part nodes. The main data from simulations are stored in this node and later modelled on the remaining introduced semantics, e.g. edge weights.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty

    :param part_id:       
        The identification number of the part in the simulation.
    :type part_id: int
    :param part_name:     
        The name of the part from the simulation.
    :type part_name: string
    :param part_sim_name: 
        The simulation name of the part.
    :type part_sim_name: string
    :param part_sim_abb:  
        The simulation abbreviation name .
    :type part_sim_abb: string


    :param part_cog:
        The centre of the part if located in a box, x, y, z.
    :type part_cog: array
    :param part_min:
        The min x,y,z of the box holding the part.
    :type part_min: array
    :param part_max:
        The max x,y,z of the box holding the part.
    :type part_max: array


    :param part_des:
       Connecting part to design node, 'PART_DES', model=WgtRel
    :type part_des: neomodel.Relationship('Des', 'PART_DES', model=WgtRel)
    :param part_behav:
       Connecting part to behaviour node, 'PART_BEHAV'
    :type part_behav: neomodel.Relationship('Behav', 'PART_BEHAV')
    :param part_pltf:
       Connecting part to its platform, 'BELONGS_TO'
    :type Ipart_pltf: neomodel.Relationship('Pltf', 'BELONGS_TO')
    :param part_ubdy:
       Connecting part to its upperbody,  'BELONGS_TO'
    :type part_ubdy: neomodel.Relationship('Ubdy',  'BELONGS_TO')
    :param part_fts:
       Connecting part to its features, 'PART_FTS', model=Fts
    :type part_fts: neomodel.Relationship('Mthd', 'PART_FTS', model=Fts)
    """

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
    part_pltf = neomodel.Relationship('Pltf', 'BELONGS_TO')
    part_ubdy = neomodel.Relationship('Ubdy',  'BELONGS_TO')
    part_fts = neomodel.Relationship('Mthd', 'PART_FTS', model=Fts)


    def get_des(self, type):
        """
        Get the design, based on the specified type, nodes belonging to a part, filtered by part uid.

        :param type: The type of the design nodes to be selected.
        :type type: string

        :returns: A list of unique identifiers of the Des nodes.
        :rtype: list
        """

        cypherTxt = """
            match (d:Des)-[:PART_DES]-(p:Part)
            where p.uid = '{0}' and d.des_type='{1}'
            return d.uid
            """.format(self.uid, type)
        
        result = self.cypher(cypherTxt)
        mUids = [x[0] for x in result[0]]

        return(mUids[0])

    def part_nrg(self, fts):
        """
        Returning the specified energy feature of a part, filtered with part UID.

        :param fts: specifier of the property to return.
        :type fts: string

        :returns: An array of part features ordered descending.
        :rtype: np.array
        """

        cypherTxt = """
            match (p:Part)
            where p.uid='{0}'
            return p.{1}
            order by p.{1} desc
        """.format(self.uid, fts)

        result = self.cypher(cypherTxt)
        a = np.array(result[0][0])
        return(a)

    def part_power(self):
        """
        Calculating the energy absorptionpower of a part, filtered with part UID.

        :returns: An array of part absorption powers.
        :rtype: np.array
        """

        cypherTxt = """
            match (p:Part)
            where p.uid='{0}'
            return p.nrg_max, p.tn_pct, p.ti_grad
            """.format(self.uid)

        result = self.cypher(cypherTxt)
        IE, tn, ti = np.array(result[0][0])
        power = IE / (tn - ti)
        return(power)

    def part_behav_con(self, uid):
        """
        Checking if a behaviour is connected to a part.

        :param uid: The Behav UID.
        :type uid: string

        :returns: If connected returns True if not False.
        :rtype: boolean
        """

        cypherTxt = """
            match (p:Part)-[r:PART_BEHAV]-(e:Behav)
            where p.uid = "{0}"
            and e.uid = "{1}"
            return r
        """.format(self.uid, uid)

        result = self.cypher(cypherTxt)
        if not result[0] == []:
            return(True)
        else:
            return(False)

    def get_sim_name(self):
        """
        Get the part's simulation name.

        :returns: The name of the simulation.
        :rtype: string
        """

        cypherTxt = """
            match (s:Sim)-[:INCL_PART | :NRG_PART]-(p:Part)
            where p.uid='{0}'
            return distinct s.sim_name, s.sim_abb
            """.format(self.uid)

        simName, _ = self.cypher(cypherTxt)
        return(simName[0])

    def get_sim(self):
        """
        Get the part's simulation node.

        :returns: The simulation node UID.
        :rtype: string
        """

        cypherTxt = """
            match (s:Sim)-[:NRG_PART]-(p:Part)
            where p.uid = '{0}'
            return s.uid
            """.format(self.uid)
        
        sUids = self.cypher(cypherTxt)
        sUids = [x[0] for x in sUids[0]]

        return(sUids[0])


class Pltf(neomodel.StructuredNode):
    """
    Platform node.

    A relatively large set of product components that are physically connected as a stable sub-assembly and are common for different final vehicles

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param pltf_name:
        Platform name.
    :type pltf_name: string
    :param pltf_spec:
        A json specifying the platform metadata.
    :type pltf_spec: json
    """

    uid = neomodel.UniqueIdProperty()
    pltf_name = neomodel.StringProperty()
    pltf_spec = neomodel.JSONProperty()


class Prtcl(neomodel.StructuredNode):
    """
    Portocol node.

    Vehicle Portocol specification.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param prtcl_name:
        Portocol name.
    :type prtcl_name: string
    :param prtcl_spec:
        A json specifying the Portocol metadata.
    :type prtcl_spec: json
    """

    uid = neomodel.UniqueIdProperty()
    prtcl_name = neomodel.StringProperty()
    prtcl_spec = neomodel.JSONProperty()


class Sim(neomodel.StructuredNode):
    """
    Simulation nodes.
    
    The simulation nodes that hold the global features of the simulation e.g. total energy.


    :param uid:                 
        The unique id for simulaiton.
    :type uid: neomodel.UniqueIdProperty
    :param sim_name:            
        The simulation name should be a unique property.
    :type sim_name:            
        string, unique_index=True.
    :param sim_oem:             
        The OEM name to which the simulation belongs.
    :type sim_oem:             
        string.
    :param sim_abb:             
        Simulation name abbreviation, for better visualization of the name.
    :type sim_abb:             
        string.
    :param sim_lc:              
        Load case of the analysis e.g. pedestrian head impact.
    :type sim_lc:              
        string.
    :param sim_rel:             
        The development stage (release) of the product.
    :type sim_rel:             
        string.
    :param sim_dspln:           
        The analysis discipline e.g. safety, durability.
    :type sim_dspln:           
        string.
    :param dim_description:     
        The specific labelling of the data e.g. optimization.
    :type dim_description:     
        string.
    :param sim_path_pre:        
        Path to the file for input simulation data.
    :type sim_path_pre:        
        string.
    :param sim_path_post:       
        Path to the file for output simulation data.
    :type sim_path_post:       
        string.

    :param sim_nrg_part:
        Connecting simulation to its energetic parts, 'NRG_PART', model=WgtRel.
    :type sim_nrg_part: neomodel.RelationshipTo
    :param sim_node:
        Connecting simulation to its selective time history nodes, 'OUT_NODE'.
    :type sim_node: neomodel.RelationshipTo
    :param sim_elmnt:
        Connecting simulation to its selective time history elements, 'OUT_ELMNT'.
    :type sim_elmnt: neomodel.RelationshipTo
    :param sim_model:
        Connecting simulation to its input FE model, 'SIM_MODEL'.
    :type sim_model: neomodel.RelationshipTo
    :param sim_imp:
        Connecting simulation to its impactor if any, 'SIM_IMP'.
    :type sim_imp: neomodel.RelationshipTo
    :param sim_barr:
        Connecting simulation to its barrier if any, 'SIM_BARR'.
    :type sim_barr: neomodel.RelationshipTo
    :param sim_des:
        Connecting simulation to its design nodes, 'SIM_DES', model=WgtRel.
    :type sim_des: neomodel.RelationshipTo
    
    Crash-specific properties:

    - initial time: (t_i)
    - absorption end time: (t_n)

    :param sim_ie_tot_max:
        Max of the total internal energy of the simulation. 
    :type sim_ie_tot_max:
        float.
    :param sim_ie_prt_max:
        Max of the internal energy of the parts in a simulation.
    :type sim_ie_prt_max:
        float.
    :param sim_ke_tot_max:
        Max of the total kinetic energy of the simulation. 
    :type sim_ke_tot_max:
        float.
    :param sim_ie_ti_ll:
          Simulation t_i with the lower limit method (ll). 
    :type sim_ie_ti_ll:  
        float.
    :param sim_ie_ti_grad:
        Simulation t_i with the gradient method (grad).
    :type sim_ie_ti_grad:
        float.
    :param sim_ie_tn_pct:
         Simulation t_n with the percentage method (pct).
    :type sim_ie_tn_pct: 
        float.
    :param sim_ie_tn_max:
         Simulation t_n with the maximum method (max).
    :type sim_ie_tn_max: 
        float.
    :param sim_tn_pct_max:
        Max of the parts for tn_pct value.
    :type sim_tn_pct_max:
        float.
    :param sim_tn_max_max:
        Max of the parts for tn_max value.
    :type sim_tn_max_max:
        float.
    :param sim_ti_ll_pp:
          Ranges of the values for ti_ll.
    :type sim_ti_ll_pp:  
        float.
    :param sim_ti_grad_pp:
        Ranges of the values for ti_grad.
    :type sim_ti_grad_pp:
        float.
    :param sim_t_max:
             Max of t_n.
    :type sim_t_max:     
        float.
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
        """
        Returning node properties of nodes connected to a simulation. Simulation is selected by its name. 

        :param prop:
            The node property key is to be returned.
        :type prop: string
        :param node:
            The node label that is searched based on connectivity to the simulation.

        :returns: List of properties, prop, of a node (n:node).
        :rtype: list
        """

        cypherTxt = """
            match (p:{2})-[*1]-(n:Sim)
            where n.sim_name='{0}'
            return p.{1}
            """.format(self.sim_name, prop, node)
        idList, _ = self.cypher(cypherTxt)

        idList = [x[0] for x in idList]
        return(idList)

    def get_child(self):
        """
        Query to get the analysis depending on the current analysis.

        :returns: List of strings, simulation names.
        :rtype: list
        """

        childList, _ = self.cypher(
            """
            match (n1)-[r:sim_based_on]->(n2)
            where n1.sim_name='{0}'
            return n2.sim_name
            """.format(self.sim_name))
        try:
            child = [str(i[0]) for i in childList]
        except IndexError:
            child = []
        return child

    def forward_length(self, maxLen):
        """
        Return the number of analyses done after the current analysis

        :returns: List of properties, prop, of a node (n:node).
        :rtype: list
        """

        c_len_list = {}
        child_list = self.get_child()
        for c in child_list:
            cL, _ = self.cypher("""
                match p=(n1:Sim)-[r:sim_based_on*1..{}]->(n2:Sim)
                where n1.sim_name = '{}'
                return length(p) order by length(p) desc limit 1
                """.format(maxLen, c))
            try:
                c_len_list.update([(c, cL[0][0])])
                print(c)
            except IndexError:
                if cL == []:
                    c_len_list.update([(c, 0)])
        return(c_len_list)

    def get_reference(self):
        """
       Get the reference simulations of the simulation.

        :returns: List of simulations, that are referenced in the development tree.
        :rtype: list
        """

        refList, _ = self.cypher(
            """
            match (n1)-[r:sim_based_on]->(n2)
            where n2.sim_name='{0}'
            return n1.sim_name
            """.format(self.sim_name))
        try:
            reference = [str(i[0]) for i in refList]
        except IndexError:
            reference = []
            reference = ''

        return reference

    def get_model(self):
        """
        Return the input FE-model of the simulation.

        :returns: The model unique identifier of the simulation.
        :rtype: int
        """

        model_id, _ = self.cypher(
            """
            match (n1:Sim)-[]-(n2:Model)
            where n1.sim_name='{}'
            return n2.model_id
        """.format(self.sim_name))

        return model_id[0][0]

    def get_parts(self, relp=''):
        """
        Return the part nodes of the simulation with specific relationship.

        :param relp:
            Relationship label to select the part nodes.
        :type relp:
            string

        :returns: The list of parts belonging to the simulations.
        :rtype: list
        """

        cypherTxt = """
            match (n:Sim)-[r{1}]-(p:Part)
            where n.sim_name='{0}'
            return p.part_id order by r.ec_order
            """.format(self.sim_name, relp)

        partList, _ = self.cypher(cypherTxt)

        part = [x[0] for x in partList]

        return(part)

    def get_childByID(self, id, node):
        """
        Find the analysis depending on the current analysis with node input, with a maximum walking distance of 3.

        :param id:
            Specifying the uid of the node to be selected.
        :type id: int
        :param node:
            Node label to be selected.

        :returns: The list of nodes uid.
        :rtype: list
        """

        prop = node.lower() + '_id'
        cypherTxt = """
            match (n:Sim)-[*1..3]-(p:{2})
            where n.sim_name='{0}' and  p.{3}={1}
            return p.uid
            """.format(self.sim_name, id, node, prop)
        child, _ = self.cypher(cypherTxt)

        if not child == []:
            child = child[0][0]

        return(child)

    def embed_nrg(self, ft):
        """
        Select energy features based on specified options for energetic parts of the simulation, in order descending by first feature.

        :param ft: List of string specifying the energy features to be selected.
        :type ft: list

        :returns: A list of dictionaries with part id and features of the parts of the simulation.
        :rtype: list
        """
        cypherTxt = """
        match (s:Sim)-[r:NRG_PART]-(p)
        where s.sim_name='{0}'
        return p.{1}, p.{2}, p.{3}, p.part_id
        order by p.{1} desc
        """.format(self.sim_name, ft[0], ft[1], ft[2])
        
        result = self.cypher(cypherTxt)
        parts_fts = np.asarray(result[:-1][0])
        return(parts_fts)

    def embed_nrg_sel(self, ft, pid_list):
        """
        Select energy features, filtered by a list of part ids, based on specified options for energetic parts of the simulation, in order descending by first feature.

        :param ft: List of string specifying the energy features to be selected.
        :type ft: list
        :param pid_list:
            List of pid in string, identifying the parts result to be returned.
        :type pid_list: list

        :returns: A list of dictionaries with part id and features of the parts of the simulation.
        :rtype: list
        """

        cypherTxt = """
            match (s:Sim)-[r:NRG_PART]-(p)
            where s.sim_name='{0}' and p.part_id in {4}
            return p.{1}, p.{2}, p.{3}, p.part_id
            order by p.{1} desc
        """.format(self.sim_name, ft[0], ft[1], ft[2], pid_list)
        
        resutl = self.cypher(cypherTxt)
        parts_fts_slct = np.asarray(resutl[:-1][0])
        return(parts_fts_slct)

    def get_behavs(self):
        """
        Get the behaviour nodes belonging to an analysis.

        :returns: A list of unique identifiers of the Behav nodes related to the simulation.
        :rtype: list
        """

        cypherTxt = """
            match (b:Behav)-[:PART_BEHAV]-(p:Part)-[:NRG_PART]-(s:Sim)
            where s.uid = '{0}'
            return b.uid
            """.format(self.uid)
        
        result = self.cypher(cypherTxt)
        bUids = [x[0] for x in result[0]]

        return(bUids)

    def get_deses(self):
        """
        Get the design nodes belonging to an analysis, filtered by simulation uid.

        :returns: A list of unique identifiers of the Des nodes related to the simulation.
        :rtype: list
        """

        cypherTxt = """
            match (d:Des)-[:PART_DES]-(p:Part)-[:NRG_PART]-(s:Sim)
            where s.uid = '{0}'
            return d.uid
            """.format(self.uid)
        
        result = self.cypher(cypherTxt)
        dUids = [x[0] for x in result[0]]

        return(dUids)


class Ubdy(neomodel.StructuredNode):
    """
    Upper body node.

    What deviates between the vehicles in simultaneous developments, e.g. sedan versus a minivan.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param ubdy_name:
        Upper body name.
    :type ubdy_name: string
    :param ubdy_spec:
        A json specifying the upper body metadata.
    :type ubdy_spec: json
    """

    uid = neomodel.UniqueIdProperty()
    ubdy_name = neomodel.StringProperty()
    ubdy_spec = neomodel.JSONProperty()


class Veh(neomodel.StructuredNode):
    """
    Vehicle node.

    The subject of the CAE analysis.

    :param uid:
        Unique identifier of the database.
    :type uid: neomodel.UniqueIdProperty
    :param veh_name:
        Vehicle name.
    :param veh_abb:
        Vehicle abbreviation for shortening in reporting.
    :type veh_abb: string
    :param veh_spec:
        A json specifying the vehicle metadata.
    :type veh_spec: json
    :param veh_ubdy:
        Connecting the vehicle to its upper body node, 'VEH_UBDY'
    :type veh_ubdy: neomodel.Relationship
    :param veh_pltf:
        Connecting the vehicle to its platform node, 'VEH_PLTF'
    :type veh_pltf: neomodel.Relationship      
    """

    uid = neomodel.UniqueIdProperty()
    veh_name = neomodel.StringProperty()
    veh_abb = neomodel.StringProperty()
    veh_spec = neomodel.JSONProperty()

    veh_ubdy = neomodel.Relationship('Ubody', 'VEH_UBDY')
    veh_pltf = neomodel.Relationship('Pltf', 'VEH_PLTF')
