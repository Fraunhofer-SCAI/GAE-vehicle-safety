# GAE-vehicle-safety

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="##Run-Database-Server">Run Database Servery</a></li>
    <li><a href="##Databases">Databases</a></li>
        
  </ol>
</details>

<!-- PDF Extraction -->

<br />
<br />

# Graph Assisted Engineering in Vehicle Safety

GAE is a graph modeling for CAE analysis in automotive R&D development processes. in connection with its design requirement.
The first release of GAE considers vehicle safety with [EuroNCAP](https://www.euroncap.com/en) safety requirements.
We connect CAE data to the protocols that are used to assess vehicle safety performances.
The R\&D process includes CAD engineering and safety attributes, with a focus on multidisciplinary problem-solving.
For morinformation of graph modeling,nk](https://arxiv.org/abs/2209.14910).
<br />
<br />

### Run Database Server

Currently the database is availabledump`file that is that is developed with`neo4j-community-4.2.4`dump file can be loaded into `neo4j-community-4.2.4`. We provide this version as it is no longer availble on neo4j webpage. you can extract it and then run the server with

```
tar -xzvf neo4j-community-4.2.4.tar.gz
neo4j-community-4.2.4/bin/neo4j start
```

Then you can load one od available `.dump` file databases to your database.

```
./neo4j-community-4.2.4/bin/neo4j-admin load --from=../database/FILE.dump --database=neo4j --force
```

the `--force` will take care of the upgrade if you use a more recent version of neo4j, just remember to edit the config file in `conf\neo4j.conf` and set ``dbms.allow_upgrade=true`, [more info](https://neo4j.com/docs/operations-manual/current/configuration/neo4j-conf/)
<br />
<br />

### Databases

Here we have 5 dump files in `.\database` that are:

| Databases                 | Info                                                                                                                                                                                                                                                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 00_GAE_v1.0.dump          | Merged version of all 4 databases                                                                                                                                                                                                                                                                                  |
| 01_modelCompare_v1.0.dump | FE-model changes, graph for [ModelCompare](https://www.scai.fraunhofer.de/de/geschaeftsfelder/numerische-datenbasierte-vorhersage/produkte/modelcompare.html) tool                                                                                                                                                 |
| 02_simKnowledge_v1.0.dump | F-simulation as a graph with energy features, "Knowledge discovery assistants for crash simulations with graph algorithms and energy absorption features", "Simrank++ Update for Crash Simulation Similarity Precidtion with Energy Absorption Features, [link](https://ins.uni-bonn.de/staff/garcke#publications) |
| 03_euroNcap_v1.0.dump     | Graph modeling [EuroNCAP](https://www.euroncap.com/en) safety requirements with web scrawling.                                                                                                                                                                                                                     |
| 04_safetyLoad_v1.0.dump   | load-case specification that is used in [CAEWebVis](https://caewebvis.scai.fraunhofer.de/) visualization                                                                                                                                                                                                           |
