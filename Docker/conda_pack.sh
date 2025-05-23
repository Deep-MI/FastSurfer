#!/bin/bash
# usage:
# conda_pack.sh <environment name>
#
# packs the environment into /venv

# set script to stop after first fail
set -e

# Install conda-pack
mamba install -c conda-forge conda-pack
# Use conda-pack to create a standalone environment in /venv
conda-pack -n "$1" -o /tmp/env.tar
mkdir /venv
cd /venv
tar xf /tmp/env.tar
rm /tmp/env.tar
# Finally, when venv in a new location, fix up paths
/venv/bin/conda-unpack
