# Simple Pipelining scripts using OpenCV + TF 2.0+ Video Feeds
The repository is a simple implementation using TF 2.0, OpenCV2 and video feeds such as the one provided from an android phone webcam. 

![CV Pipeline](assets/as_group_image.png)

The pipeline uses a start-of-the-art arquitecture in computer vision: yolov3 (originally implemented from https://github.com/zzh8829/yolov3-tf2 and forked on https://github.com/DanielLSM/yolov3-tf2). 


# Instalation

##  Ubuntu 18.04 

### Requirements:
- Anaconda 3

##  Instructions 

0) Open a terminal
1) Clone the repository 
```
cd ~
git clone https://github.com/DanielLSM/cv-android-script
```
2) Move to the repository in your system
```
cd cv-android-script/
```
3) Install the anaconda environment
```
conda env create -f wasp_cv.yml
```
4) Load the anaconda environment
```
conda activate wasp_cv
```
5) Install this package on the environment
```
pip install -e .
```
