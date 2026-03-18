#!/bin/bash

sudo CUDA_HOME=/usr/local/cuda-12.2 /usr/local/cuda-12.2/bin/ncu \
  --set roofline \
  --launch-skip 4 \
  --launch-count 4 \
  -o profiles/profile_5_best \
  -f .venv/bin/python3 kernels/kernel_5.py