# PYTHON script
import os
# import constants

from dotenv import load_dotenv
from neomodel import config
from neo4j import GraphDatabase
load_dotenv()



class oems:

    def __init__(self, OEM_NAME, IE_UNIT, TIME_UNIT, DATA_PATH, SIM_PATTERN, rls='', lc=''):
        self.name = OEM_NAME
        self.rls = rls
        self.lc = lc

        self.unit = {"IE": IE_UNIT, "t": TIME_UNIT}  # from Nmm, s to kNmm,ms
        self.err = {}
        self.lc_pos = 0
        self.rls_pos = 0

        self.data_path = DATA_PATH
        self.data_path += SIM_PATTERN


    def backend_server(self):
        
        neo_user = os.getenv("NEO4J_USE")
        neo_pass = os.getenv("NEO4J_PASS")
        host = os.getenv("NEO4J_HOST")
        svr_id = os.getenv("NEO4J_PORT")
        bolt = os.getenv("NEO4J_BOLT")

        def neo4j_bolt():

            config.DATABASE_URL = os.environ.get(
                'NEO4J_BOLT_URL', f'bolt://{neo_user}:{neo_pass}@{host}:{svr_id}')


            uri = f"neo4j://{host}:{bolt}"
            self.driver = GraphDatabase.driver(uri, auth=(neo_user, neo_pass))
            neo4j_bolt()



