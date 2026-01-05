#!/usr/bin/env python
import sys
import os
import torch.utils.cpp_extension

# Monkey patch the CUDA version check to always pass
original_check = torch.utils.cpp_extension._check_cuda_version
def patched_check(*args, **kwargs):
    pass

torch.utils.cpp_extension._check_cuda_version = patched_check

# Add the allow-unsupported-compiler flag to NVCC
os.environ['TORCH_NVCC_FLAGS'] = os.environ.get('TORCH_NVCC_FLAGS', '') + ' -allow-unsupported-compiler'

# Now run the original setup
exec(open('setup.py').read())
