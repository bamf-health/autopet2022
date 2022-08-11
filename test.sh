#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut --delimiter=' ' --fields=1)
MEM_LIMIT="30g"  # Maximum is currently 30g, configurable in your algorithm image settings on grand challenge

docker volume create autopet-output-$VOLUME_SUFFIX

#gpus='"device=1"' uses a T4
# Do not change any of the parameters to docker run, these are fixed
docker run --rm \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test/input/:/input/ \
        -v autopet-output-$VOLUME_SUFFIX:/output/ \
        --gpus='"device=1"' \
        autopet

# docker run --rm \
#         -v autopet-output-$VOLUME_SUFFIX:/output/ \
#         python:3.9-slim cat /output/results.json | python -m json.tool


# docker run --rm \
#         -v autopet-output-$VOLUME_SUFFIX:/output/ \
#         -v $SCRIPTPATH/test/input/:/input/ \
#         python:3.9-slim python -c "import json, sys; f1 = json.load(open('/output/results.json')); f2 = json.load(open('/input/expected_output.json')); sys.exit(f1 != f2);"

mkdir -p $SCRIPTPATH/test/output/

# move the output file out of the docker volume
docker run --rm \
        -v autopet-output-$VOLUME_SUFFIX:/output/ \
        -v $SCRIPTPATH/test/output:/output_local/ \
        alpine:latest cp /output/images/automated-petct-lesion-segmentation/001.mha /output_local/001.mha

if [ $? -eq 0 ]; then
    echo "Tests successfully passed..."
else
    echo "Expected output was not found..."
fi

docker volume rm autopet-output-$VOLUME_SUFFIX
