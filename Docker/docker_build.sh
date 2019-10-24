#!/bin/bash
cd ..
docker build -t fastsurfer:gpu -f ./Docker/Dockerfile .
