## DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark

> "Even with these dark eyes, a gift of the dark night, I go to seek the shining light."   --Gu Cheng 1956-1993

<p align="center">
    <img src="cmu_ri_logo.png" alt="Logo" width="40%"">   
    <img src="NOAA_logo_mobile.svg" alt="Logo" width="25%">
  </a>
</p>

### Novel-view rendering: Simulating a light cone and re-illuminating the environment.
Please check our [videos](https://www.linkedin.com/posts/tianyi-zhang-396b0a186_darkgs-building-3d-gaussians-with-a-torch-activity-7197672371393019905-iY2-?utm_source=share&utm_medium=member_desktop) ([Bilibili](https://www.bilibili.com/video/BV1Euu4eqEtN/?vd_source=ccc6b1a36055375ca812070948900795#reply222119730496)).
<p align="center">
    <img src="darkgs.gif" alt="Logo" width="100%">
  </a>
</p>

### Sister Repo for Camera-Light calibration
The sister repo Neural Light Simulator for light-camera calibration is [here](https://github.com/tyz1030/neuralight). 

## Install
Installation generally follows vanilla Gaussian Splatting installation.
```
git clone git@github.com:tyz1030/darkgs.git --recursive
```
or
```
git clone https://github.com/tyz1030/darkgs.git --recursive
```
Conda environment setup
```
conda env create --file environment.yml
conda activate darkgs
```
Also need to install lietorch
```
pip install git+https://github.com/princeton-vl/lietorch.git
```


## Data
Please find our example data on [Google Drive](https://drive.google.com/drive/folders/1EzhrEBCEHCSF3jtRwMXQpqF9wgh4KlPD?usp=drive_link) and [DropBox](https://www.dropbox.com/scl/fo/nc61inva76a40u934iit0/AAywA7NXF1adODJRnJT2gJI?rlkey=bw4p3ut569ngiml6x5o286zp5&st=jgv98hvj&dl=0).
#### Make your own data
Please put your RAW images subfolder named "raw". To make COLMAP less struggle, I gamma-curved/manually increased the brightness of the raw images for feature extraction and matching. These corrected images are put in "input" subfolder. We only use "raw" images to build DarkGS.
```
python3 convert.py -s <path to your own dataset>
```
## Light Calibration
If you are using your own light-camera setup, please calibrate your system using [neural light simulator](https://github.com/tyz1030/neuralight). Then put model_parameters.pth in the root directory. 

## Quick Start
Train
```
python train.py -s <path to example dataset>
```
Visualize with SIRB viewer:
```
./SIBR_remoteGaussian_app
```
Then you will be able to steer your light cone by pressing "JKLI" on the keyboard.\
#### Visualize (a checkpoint) after training:
SIBR_gaussianViewer_app is not compatible with this repo. Please try the following:
```
python3 viz_chkpt.py -s data/lab1/ -m output/<xxxxxx-xxx> --start_checkpoint output/<xxxxxx-xxx>/chkpnt30000.pth
```
then in another terminal
```
./SIBR_remoteGaussian_app
```

#### Relighting (I'm working on the release)
Meanwhile, there is no one-true-solution to relighting.\
One quick hack through is when running viz_chkpt.py, uncomment line 129 in scene/lighting.py. And you will also need to brighten, white balance and gamma correct the final render results to make it look good otherwise it is in blue-greenish RAW format.



## Cite
This work is picked up by IROS 2024 as oral presentation!
[Arxiv](https://arxiv.org/abs/2403.10814)
```
@INPROCEEDINGS{zhang2024darkgs,
  author={Tianyi Zhang and Kaining Huang and Weiming Zhi and Matthew Johnson-Roberson},
  booktitle={2024 International Conference on Intelligent Robots and Systems (IROS)}, 
  title={DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark}, 
  year={2024}}
```

## Acknowledgement
* This work is supported by NOAA.
* Copyright 2024 Tianyi Zhang, Carnegie Mellon University. All rights reserved.
