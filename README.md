# DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark

Novel-view rendering: Simulating a light cone and re-illuminating the environment.
<p align="center">
  <a href="">
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
Train
```
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```
