#!/bin/bash

sudo CUDA_HOME=/usr/local/cuda-12.2 /usr/local/cuda-12.2/bin/ncu \
  --set roofline \
  --launch-skip 2 \
  --launch-count 1 \
  -o profiles/profile_128_32_32_4_2 \
  -f .venv/bin/python3 kernels/kernel_1.py