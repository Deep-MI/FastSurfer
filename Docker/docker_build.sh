#!/bin/bash

image_name="fastsurfer"
image_tag="dev"

pushd || exit

FASTSURFER_HOME="$(cd "$(dirname "$(dirname "$THIS_SCRIPT")")" &> /dev/null && pwd)"
export FASTSURFER_HOME
version_tag="$(bash "$FASTSURFER_HOME/run_fastsurfer.sh" --version | cut -d" " -f 1)"
bash "$FASTSURFER_HOME/run_fastsurfer.sh" --version long > "$FASTSURFER_HOME/VERSION.txt"
echo "Version info added to the docker image:"
cat "$FASTSURFER_HOME/VERSION.txt"
docker build --rm=true -t "$image_name:$version_tag" -f ./Docker/Dockerfile .
if [ "$(docker image list "$image_name:$image_tag" | wc -l)" != "1" ]; then
  docker image rm $image_name:$image_tag
fi
docker tag "$image_name:$image_tag" "$image_name:$version_tag"
rm "$FASTSURFER_HOME/VERSION.txt"

popd || exit