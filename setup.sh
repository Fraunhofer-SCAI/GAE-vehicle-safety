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
# Check https://www.python.org/downloads/ for other versions
# Get python 3.8.20 src code and compile it and then install it locally.
pypath=$(pwd)/"py-3.8"
wget https://www.python.org/ftp/python/3.8.20/Python-3.8.20.tar.xz
tar -xvf Python-3.8.20.tar.xz
cd Python-3.8.20
./configure --enable-optimizations --prefix="$pypath"
make -j$(nproc)
make install
cd ..
rm -rf Python-3.8.20
rm Python-3.8.20.tar.xz
# Make python and pip softlinks
cd py-3.8/bin
ln -s python3.8 python
ln -s pip3 pip
cd ../..
# Activate installed python
export PATH="$pypath/bin":$PATH
python -m pip install --upgrade pip
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
# Cannot replace sklearn with scikit-learn as lasso package also uses it !
pip install -r GAE-vehicle-safety/requirements.txt



#--------------------------------------------------------------------------------------------
# JDK 11
#--------------------------------------------------------------------------------------------
# From Eclipse Temurin (https://adoptium.net)
# Link page https://adoptium.net/temurin/releases?version=11&os=linux&arch=x64&mode=filter
# Direct file link https://release-assets.githubusercontent.com/github-production-release-asset/372924883/ef97605a-fdd8-4a47-b1d6-90d796c8a8bb
wget https://release-assets.githubusercontent.com/github-production-release-asset/372924883/ef97605a-fdd8-4a47-b1d6-90d796c8a8bb
mv ef97605a-fdd8-4a47-b1d6-90d796c8a8bb OpenJDK11U-jdk_x64_linux_hotspot_11.0.28_6.tar.gz
tar -xvf OpenJDK11U-jdk_x64_linux_hotspot_11.0.28_6.tar.gz
# Folder "jdk-11.0.28+6" is created
export PATH=jdk-11.0.28+6/bin:$PATH
export LD_LIBRARY_PATH=jdk-11.0.28+6/lib:$LD_LIBRARY_PATH
rm OpenJDK11U-jdk_x64_linux_hotspot_11.0.28_6.tar.gz


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
rm neo4j-community-4.2.4-unix.tar.gz

