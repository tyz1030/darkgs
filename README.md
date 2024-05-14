## DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark

> "Even with these dark eyes, a gift of the dark night, I go to seek the shining light."   --Gu Cheng 1956-1993

<p align="center">
    <img src="cmu_ri_logo.png" alt="Logo" width="40%"">   
    <img src="NOAA_logo_mobile.svg" alt="Logo" width="25%">
  </a>
</p>
Novel-view rendering: Simulating a light cone and re-illuminating the environment.
<p align="center">
    <img src="darkgs.gif" alt="Logo" width="100%">
  </a>
</p>


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
Please find our example data on [Google Drive](https://drive.google.com/drive/folders/1EzhrEBCEHCSF3jtRwMXQpqF9wgh4KlPD?usp=drive_link).

## Quick Start
Train
```
python train.py -s <path to example dataset>
```
Visualize with SIRB viewer:
```
./SIBR_remoteGaussian_app
```
Then you will be able to steer your light cone by pressing "JKLI" on the keyboard.

## Cite
[Arxiv](https://arxiv.org/abs/2403.10814)
```
@misc{zhang2024darkgs,
      title={DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark}, 
      author={Tianyi Zhang and Kaining Huang and Weiming Zhi and Matthew Johnson-Roberson},
      year={2024},
      eprint={2403.10814},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
* This work is supported by NOAA.
* Copyright 2024 Kaining Huang and Tianyi Zhang, Carnegie Mellon University. All rights reserved.
