#!/bin/bash


cd ..

#--------------------------------------------------------------------------------------------
# In repo
#--------------------------------------------------------------------------------------------
# Set env
cp GAE-vehicle-safety/src/.env-sample GAE-vehicle-safety/src/.env

#--------------------------------------------------------------------------------------------
# Python 3.8
#--------------------------------------------------------------------------------------------
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh
bash Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -b -f -s -p "py-3.8"
# Activate installed python
eval "$("py-3.8/bin/conda" shell.bash hook)"
python -m pip install --upgrade pip
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# Cannot replace sklearn with scikit-learn as lasso package also uses it !
pip install -r GAE-vehicle-safety/requirements.txt



#--------------------------------------------------------------------------------------------
# JDK 11
#--------------------------------------------------------------------------------------------
# From here : https://jdk.java.net/java-se-ri/11-MR2
# https://download.java.net/openjdk/jdk11.0.0.1/ri/openjdk-11.0.0.1_linux-x64_bin.tar.gz
wget https://download.java.net/openjdk/jdk11.0.0.1/ri/openjdk-11.0.0.1_linux-x64_bin.tar.gz
tar -xvf openjdk-11.0.0.1_linux-x64_bin.tar.gz
rm openjdk-11.0.0.1_linux-x64_bin.tar.gz
# Folder "jdk-11.0.0.1" is created 
export PATH=jdk-11.0.0.1/bin:$PATH
export LD_LIBRARY_PATH=jdk-11.0.0.1/lib:$LD_LIBRARY_PATH


#--------------------------------------------------------------------------------------------
# Neo4j-4.2.4
#--------------------------------------------------------------------------------------------
# Set database and set initial password
wget https://dist.neo4j.org/neo4j-community-4.2.4-unix.tar.gz
tar -xvf neo4j-community-4.2.4-unix.tar.gz
# Append to config file
echo "dbms.allow_upgrade=true" >> neo4j-community-4.2.4/conf/neo4j.conf
# Add this database
./neo4j-community-4.2.4/bin/neo4j-admin load --from=GAE-vehicle-safety/data/02_simKnowledge_v1.0.dump --database=neo4j --force
# Set password
./neo4j-community-4.2.4/bin/neo4j-admin set-initial-password NEO4J

