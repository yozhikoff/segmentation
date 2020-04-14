# Segmentation of nuclei using DSB-2018 top-1 neural network model

For comparison of Data Science Bowl 2018 best segmentation models see [Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl, Juan C. Caicedo et al](https://www.nature.com/articles/s41592-019-0612-7).  

## Installation
1. Clone this repository

```
git clone https://github.com/yozhikoff/segmentation.git
```

2. Download [this](https://www.dropbox.com/s/qvtgbz0bnskn9wu/dsb2018_topcoders.zip?dl=0) and extract it to the
segmentation folder, replace all existing files using `Ay` keys when unzip asks about it. Note that you need to export to `/repo/segmentation/dsb2018_topcoders` withing the repo.

```
wget https://www.dropbox.com/s/qvtgbz0bnskn9wu/dsb2018_topcoders.zip?dl=1 dsb2018_topcoders.zip # note dl=1
unzip /path/to/zip/dsb2018_topcoders.zip -d /path/to/repo/segmentation/dsb2018_topcoders 
```

3. Go to the segmentation folder and reset git files

```shell script
cd /path/to/repo/segmentation
git reset --hard
```

4. Create new conda env
``` 
conda create -n seg python=3.6.9 -y
conda activate seg
``` 
5) Install packages via conda and pip, simply (inside your conda env!)

```
sh ./install.sh
```
6) Test your installation using
```
python run_test.py
```

You can also try `example_notebook.ipynb` if you want to see usage details.
