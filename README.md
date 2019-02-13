## Installation instructions

1) Make sure CUDA code can run on your machine. To install CUDA Toolkit follow 
    the instruction at https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html.

1) Clone this repo 
    
    `git clone https://github.com/alessiosarullo/KBVRD.git`
    
    `cd KBVRD`
    
1) Clone PyTorch version of Detectron

    `git clone https://github.com/roytseng-tw/mask-rcnn.pytorch.git pydetectron`
    
1) Install the following packages. I recommend using Conda. Code has been developed 
    on Python 3 with PyTorch 0.4.1:
        
    ```
    conda install pytorch=0.4.1 -c pytorch
    conda install torchvision -c pytorch
    conda install cython matplotlib numpy scipy opencv pyyaml packaging
    pip install pycocotools
    ```
        
    Note: you need GCC for pycocotools.
    
1) Compile Detectron

    ```
    cd pydetectron/lib
    ./make.sh
    ```
    