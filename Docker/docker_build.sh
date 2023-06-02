#!/bin/bash
cd ..
echo "2.1-$(git rev-parse --short HEAD) ($(git branch --show-current))" > VERSION
echo "git status:" >> VERSION
git status -s -b | grep -v __pycache__ >> VERSION
echo "Version info added to the docker image:"
cat VERSION
docker build --rm=true -t fastsurfer:gpu -f ./Docker/Dockerfile .
rm VERSION