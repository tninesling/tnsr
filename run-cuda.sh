#!/bin/sh

docker run -it --rm --gpus all -v "$PWD":/app -w /app cuda12 /bin/bash
