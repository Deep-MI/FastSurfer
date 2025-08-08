#!/bin/bash
# usage:
# conda_pack.sh <environment name>
#
# packs the environment into /venv

# set script to stop after first fail
set -e

# Install conda-pack
mamba install -c conda-forge conda-pack
# make sure setuptools is <81 for conda-pack 0.8.1 https://github.com/conda/conda-pack/issues/391
setuptools_major=$(mamba list setuptools -e | sed '/^#/d' | grep -oE '=[^.]+\.')
if [[ "${setuptools_major:1:-1}" -lt 81 ]] ; then mamba install -c conda-forge "setuptools<81" ; fi
# Use conda-pack to create a standalone environment in /venv
conda-pack -n "$1" -o /tmp/env.tar
mkdir /venv
cd /venv
tar xf /tmp/env.tar
rm /tmp/env.tar
# Finally, when venv in a new location, fix up paths
/venv/bin/conda-unpack
