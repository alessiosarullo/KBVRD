## Installation instructions

1) Make sure CUDA code can run on your machine. To install CUDA Toolkit follow 
    the instruction at https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html.

1) Clone this repo 
    
    `git clone https://github.com/alessiosarullo/KBVRD.git`
    
    `cd KBVRD`
    
1) Download data and pretrained models:

    - [HICO-DET](http://www-personal.umich.edu/~ywchao/hico/). Place the content in `data/HICO-DET`.
    - The model you plan on using from [Detectron's model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md).
    The default is [this](https://dl.fbaipublicfiles.com/detectron/35858828/12_2017_baselines/e2e_mask_rcnn_R-50-C4_2x.yaml.01_46_47.HBThTerB/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl).
    Place it in `data/pretrained_model`.
    
1) Clone PyTorch version of Detectron

    `git clone https://github.com/roytseng-tw/mask-rcnn.pytorch.git pydetectron`
    
1) Install the following packages. I recommend using Conda. Code has been developed 
    on Python 3 with PyTorch 0.4.1:
        
    ```
    conda install pytorch=0.4.1 torchvision cuda90 -c pytorch
    conda install cython matplotlib numpy scipy opencv pyyaml packaging pandas
    pip install pycocotools
    ```
        
    Note: you need GCC for pycocotools and you might need to change 
    `cuda90`  depending on your CUDA version.
    
1) Compile Detectron

    ```
    cd pydetectron/lib
    ./make.sh
    ```
    Note that you might have to modify `make.sh`  and change CUDA_PATH and CUDA_ARCH to fit
    your system's configuration (for example enabling compute capability 70 for Volta GPUs).
    
    Warning: there is currently a compatibility issue between Python 3.7 
    and Cython which might cause trouble in compiling the C code. 
    If GCC fails try with Python 3.6.

1) Substitute `rpn_heads.py` and `model_builder.py`, add `generate_proposals_torch.py`.
    
1) Compile HighwayLSTM