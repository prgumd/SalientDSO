# SalientDSO

Bringing attention to Direct Sparse Odometry

## 1. What?
This is the accompanying source code for the paper **[SalientDSO: Bringing attention to Direct Sparse Odometry](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8671715)** published in IEEE Transactions in Autonmation Science and Engineering. Visit our [project page](http://prg.cs.umd.edu/SalientDSO) for more details.

**The dataset will be published soon.**


## 2. Video
[![ SalientDSO: Bringing attention to Direct Sparse Odometry](assets/Video.PNG)](https://www.youtube.com/watch?v=JQnF_yYk0wI)

## 3. Citation
If you use this for your research please cite:

```
@ARTICLE{salientdso, 
author={H. {Liang} and N. J. {Sanket} and C. {Fermüller} and Y. {Aloimonos}}, 
journal={IEEE Transactions on Automation Science and Engineering}, 
title={SalientDSO: Bringing Attention to Direct Sparse Odometry}, 
year={2019}, 
volume={}, 
number={}, 
pages={1-8}, 
keywords={Visualization;Semantics;Optimization;Task analysis;Cameras;Computer vision;Direct sparse odometry (DSO);scene parsing;SLAM;visual saliency.}, 
doi={10.1109/TASE.2019.2900980}, 
ISSN={}, 
month={},}
```

Also, citing the original DSO paper is appreciated.

```
@ARTICLE{dso, 
author={J. {Engel} and V. {Koltun} and D. {Cremers}}, 
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={Direct Sparse Odometry}, 
year={2018}, 
volume={40}, 
number={3}, 
pages={611-625}, 
keywords={distance measurement;image motion analysis;image sampling;optimisation;probability;direct sparse odometry;visual odometry method;motion formulation;fully direct probabilistic model;consistent optimization;joint optimization;reference frame;camera motion;sample pixels;smooth intensity variations;Cameras;Geometry;Three-dimensional displays;Optimization;Robustness;Computational modeling;Visualization;Visual odometry, SLAM, 3D reconstruction, structure from motion}, 
doi={10.1109/TPAMI.2017.2658577}, 
ISSN={}, 
month={March},}
```

## 4. Setup Instructions

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### 4.1. Prerequisites

You need to have the following packages on your machine.

* [Eigen3](http://eigen.tuxfamily.org/index.php?title=Main_Page) - Eigen
* [OpenCV](https://opencv.org) - OpenCV
* [Pangolin](https://github.com/stevenlovegrove/Pangolin) - Pangolin

### 4.2. Installing

```
git clone git@github.com:huaijenliang/SalientDSO-pub.git
```

#### 4.2.1. Required Dependencies
- suitesparse and eigen3.

Install using

```
sudo apt-get install libsuitesparse-dev libeigen3-dev libboost-all-dev
```

- OpenCV 

Install using

```
sudo apt-get install libopencv-dev

``` 

- Pangolin

Install from [https://github.com/stevenlovegrove/Pangolin](https://github.com/stevenlovegrove/Pangolin)

- ziplib


```
sudo apt-get install zlib1g-dev
cd dso/thirdparty
tar -zxvf libzip-1.1.1.tar.gz
cd libzip-1.1.1/
./configure
make
sudo make install
sudo cp lib/zipconf.h /usr/local/include/zipconf.h   # (no idea why that is needed).
```

- sse2neon (For ARM builds only)
After cloning, just run ```git submodule update --init``` to include this. It translates Intel-native SSE functions to ARM-native NEON functions during the compilation process.


#### 4.2.2. Building
Test if the original DSO setting works first. Compile using ```USE_SALIENCY_SAMPLING=false``` in ```setting.h``` and build using the following instructions.

```
cd dso
mkdir build
cd build
cmake ..
make -j4

```



this will compile a library ```libdso.a```, which can be linked from external projects. It will also build a binary dso\_dataset, to run SalientDSO on datasets. Once you've tested this you can re-compile using ```USE_SALIENCY_SAMPLING=true``` in ```setting.h``` to use SalientDSO functionality.


### 5. Usage


#### 5.1. Test DSO Setting
Run on a dataset from [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset) using

```
DATASETPATH=Path to your dataset
build/bin/dso_dataset files=$DATASETPATH/images.zip calib=$DATASETPATH/camera.txt gamma=$DATASETPATH/pcalib.txt vignette=$DATASETPATH/vignette.png preset=0 mode=0
```

#### 5.2. Test Saliency Setting
Recompile using ```USE_SALIENCY_SAMPLING=true``` in ```setting.h``` to use this functionality. First install SalGAN from [here](https://github.com/imatge-upc/salgan) and run it on your dataset. Store the saliency images in ```.png``` format in a folder. **These instructions will be posted soon.**


Run on a dataset from [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset) using (this assumes that the saliency files are stored in the dataset in ```DATASETFOLDER/saliency/``` folder).


```
DATASETPATH=Path to your dataset
build/bin/dso_dataset files=$DATASETPATH/images.zip calib=$DATASETPATH/camera.txt gamma=$DATASETPATH/pcalib.txt vignette=$DATASETPATH/vignette.png saliency=$DATASETPATH/saliency/ preset=0 mode=0 smoothing=1

```


#### 5.3. Test Semantic Filtered Saliency Setting (SalientDSO's proposed approach)
Recompile using ```USE_SALIENCY_SAMPLING=true``` in ```setting.h``` to use this functionality. First install SalGAN from [here](https://github.com/imatge-upc/salgan) and run it on your dataset. Store the saliency images in ```.png``` format in a folder. Also, run scene parsing using PSPNet from [here](https://github.com/hellochick/PSPNet-tensorflow) and run it on your dataset. Store the saliency images in ```.png``` format in a folder. **These instructions will be posted soon.** 

Run on a dataset from [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset) using (this assumes that the saliency files are stored in the dataset folder in ```DATASETFOLDER/saliency/``` folder and scene parsing is stored in the dataset folder ```DATASETFOLDER/segmentation```).


```
DATASETPATH=Path to your dataset
build/bin/dso_dataset files=DATASETPATH/images.zip calib=DATASETPATH/camera.txt gamma=DATASETPATH/pcalib.txt vignette=DATASETPATH/vignette.png saliency=DATASETPATH/saliency/ segmentation=DATASETPATH/segmentation/ preset=0 mode=0 smoothing=2

```

#### 5.4. Running on CVL-UMD dataset
Download the dataset from [here](). Recompile using ```USE_SALIENCY_SAMPLING=true``` in ```setting.h``` to use this functionality. Run using the following instructions. 

```
DATASETPATH=Path to your dataset
build/bin/dso_dataset files=DATASETPATH/images calib=DATASETPATH/camera.txt saliency=DATASETPATH/saliency segmentation=DATASETPATH/segmentations/ calib_no_rect=DATASETPATH/camera.txt result=cvl_02_seg_0.txt preset=0 mode=1 quiet=1 saliencyAdd=0 saliencyMeanWeight=60.0 immatureDensity=1000 smoothing=2 patchSize=8 sampleoutput=1 points=pointclouds/pts_cvl_02_seg_0.txt

```

### Additional Dataset Format
The format assumed is that of [https://vision.in.tum.de/mono-dataset](https://vision.in.tum.de/mono-dataset).
However, it should be easy to adapt it to your needs, if required. The binary is run with:

- `saliency=XXX` where XXX is either a folder or .zip archive containing saliency images. They are sorted *alphabetically*. for .zip to work, need to comiple with ziplib support.

- `segmentation=XXX` where XXX is either a folder or .zip archive containing saliency images. They are sorted *alphabetically*. for .zip to work, need to comiple with ziplib support.

- `calib_no_rect=XXX` where XXX is a geometric camera calibration file. It is used if saliency images and segmentations is already undistorted.

### Commandline Options
there are many command line options available, see `main_dso_pangolin.cpp`. some examples include
- `result=XXX` : set the path to the output trajectory file
- `smoothing=X` : 
  -`smoothing=0` : set saliency smoothing term to a constant
  -`smoothing=1` : set saliency smoothing term to the average of the saliency / saliencyMeanWeight
  -`smoothing=2` : apply saliency filtering and set saliency smoothing term to a constant
- `saliencyAdd=X` : set constant term for saliency smoothing to X
- `immatureDensity=X` : set the number of candidate points to X
- `saliencyMeanWeight=X` : set saliencyMeanWeight for saliency smoothing to X
- `patchSize=X` : set the size KxK of sampling patch to XxX
- `points=XXX` : set the path to the output 3D point cloud(sampleoutput has to be set to 1)


## Authors

- **Huai-Jen Liang**


## Acknowledgments

We would like to thank the authors of [DSO](https://github.com/JakobEngel/dso) on which this code is based on.
