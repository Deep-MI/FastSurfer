#!/bin/bash

image_name="fastsurfer"
image_tag="latest"

cd ..
pip install tomli
version=$(python -c "import tomli
with open('fastsurfer.toml', 'rb') as f:
  print(tomli.load(f)['fastsurfer']['version'])")
git_hash=$(git rev-parse --short HEAD)
echo "${version}-${git_hash} ($(git branch --show-current))" > VERSION
echo "git status:" >> VERSION
git status -s -b | grep -v __pycache__ >> VERSION
echo "Version info added to the docker image:"
cat VERSION
docker build --rm=true -t $image_name:$version-$git_hash -f ./Docker/Dockerfile .
if [ "$(docker image list "$image_name:$image_tag" | wc -l)" != "1" ]; then
  docker image rm $image_name:$image_tag
fi 
docker tag $image_name:$image_tag $image_name:$version-$git_hash
rm VERSION
