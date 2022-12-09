#!/bin/bash
# usage:
# install_env.sh <environment name> <environment definition file (conda yaml)>

# set script to stop after first fail
set -e

# Install our dependencies,
conda env create -f $2

# Install conda-pack,
conda install -c conda-forge conda-pack
# Use conda-pack to create a standalone environment in /venv
conda-pack -n $1 -o /tmp/env.tar
mkdir /venv
cd /venv
tar xf /tmp/env.tar
rm /tmp/env.tar
# Finally, when venv in a new location, fix up paths
/venv/bin/conda-unpack