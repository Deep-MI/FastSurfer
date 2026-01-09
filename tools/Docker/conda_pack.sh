#!/bin/bash
# usage:
# conda_pack.sh <environment name>
#
# packs the environment into /venv

# set script to stop after first fail
set -e

# Install conda-pack
mamba env create -n pack_env -c conda-forge conda-pack "setuptools<81"

# make sure setuptools is <81 for conda-pack 0.8.1 https://github.com/conda/conda-pack/issues/391
setuptools_major=$(mamba list -n pack_env setuptools -e | sed '/^#/d' | grep -oE '=[^.]+\.')
if [[ "${setuptools_major:1:-1}" -lt 81 ]] ; then mamba install -n pack_env -c conda-forge "setuptools<81" ; fi

# Use conda-pack to create a standalone environment in /venv
mamba run -n pack_env conda-pack -n "$1" -o /tmp/env.tar
mkdir /venv
cd /venv
tar xf /tmp/env.tar
rm /tmp/env.tar

# Finally, when venv in a new location, fix up paths
/venv/bin/conda-unpack
