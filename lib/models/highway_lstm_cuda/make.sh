#!/usr/bin/env bash

#CUDA_PATH=/usr/local/cuda/

CUDA_ARCH="-gencode arch=compute_30,code=sm_30 \
           -gencode arch=compute_35,code=sm_35 \
           -gencode arch=compute_50,code=sm_50 \
           -gencode arch=compute_52,code=sm_52 \
           -gencode arch=compute_60,code=sm_60 \
           -gencode arch=compute_61,code=sm_61 "
#          -gencode arch=compute_70,code=sm_70 "

echo "Building kernel for following target architectures: "
echo $CUDA_ARCH

cd src
echo "Compiling kernel"
nvcc -c -o highway_lstm_kernel.cu.o highway_lstm_kernel.cu --compiler-options -fPIC $CUDA_ARCH

# F-Net style
#nvcc -c -o highway_lstm_kernel.cu.o highway_lstm_kernel.cu -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../
python build.py
