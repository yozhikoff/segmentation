# segmentation

## Installation
1) Clone this repository
2) Download and extract [this](https://www.dropbox.com/s/qvtgbz0bnskn9wu/dsb2018_topcoders.zip?dl=0)
3) Install `CUDA 10` and `cuDNN 7` (the latest versions work aswell)
4) Create new conda env
``` 
conda create -n new_env python=3.6.9 -y
conda activate new_env
```
5) Install packages via conda and pip, simply (inside your conda env!)

**Read note below first!**
```
sh ./install.sh
```
6) Copy `data_test` to `dsb2018_topcoders` and run `predict_test.sh`
```
cp -r data_test dsb2018_topcoders
./predict_test.sh
```

This works only on the GPU machine.

### Tensorflow note
Pre-compilated versions of TF don't support modern processor instructions and work properly only with some particular versions of CUDA and cuDNN, so it's a realy good idea to build it from sources instead of conda installation. 

