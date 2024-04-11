#!/bin/bash

cd ..

export PATH=jdk-11.0.0.1/bin:$PATH
export LD_LIBRARY_PATH=jdk-11.0.0.1/lib:$LD_LIBRARY_PATH

./neo4j-community-4.2.4/bin/neo4j console

