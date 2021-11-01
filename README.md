# 1.About This Project  
**Our Official Website**:  www.houmo.ai  
**Who We Are:** We are Houmo - A Great AI  Company.  
We wish to change the world with unlimited computing power,   
We will subvert the AI chip with in memory computing.  

This Project we created is the first one that migrate tensorrt inference engine into [Google Mediapipe](https://github.com/google/mediapipe).  
The purpose of this project is to apply mediapipe to more AI chips.    

# 2.Our Build Environment  
## 2.1Hardware: AGX Xavier System Information  
 - NVIDIA Jetson AGX Xavier [16GB]
   * Jetpack 4.6 [L4T 32.6.1]
   * NV Power Mode: MAXN - Type: 0
   * jetson_stats.service: active
 - Board info:
   * Type: AGX Xavier [16GB]
   * SOC Family: tegra194 - ID:25
   * Module: P28xx-00xx - Board: P28xx-00xx
   * Code Name: galen
   * CUDA GPU architecture (ARCH_BIN): 7.2
   * Serial Number: *****
 - Libraries:
   * CUDA: 10.2.300
   * cuDNN: 8.2.1.32
   * TensorRT: 8.0.1.6
   * Visionworks: 1.6.0.501
   * OpenCV: 3.4.15-dev compiled CUDA: NO
   * VPI: ii libnvvpi1 1.1.12 arm64 NVIDIA Vision Programming Interface library
   * Vulkan: 1.2.70
 - jetson-stats:
   * Version 3.1.1
   * Works on Python 2.7.17

## 2.2Build-Essential  
**a)gcc and g++ version 8.4.0 (Ubuntu/Linaro 8.4.0-1ubuntu1~18.04)**   
install command:  
```
$sudo apt install gcc-8 g++-8
$sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
```
check gcc g++ version  
```
$gcc -v
$g++ -v
```
**b)Mediapipe Dependencies**   
Please refer to [Mediapipe Installation](https://google.github.io/mediapipe/getting_started/install.html) and install all required software packages and  runtime libraries.  

**c)Cuda Related**  
We created two soft links pointing to cuda headers. Change to your own paths if needed.   
```
**/mediapipe_plus/third_party/cuda_trt$ tree
.
├── BUILD
├── usr_local_cuda
│   └── include -> /usr/local/cuda/include
└── usr_local_cuda-10.2
    └── include -> /usr/local/cuda-10.2/targets/aarch64-linux/include/
```

# 3. Build and Run This Project On Nvidia AGX Xavier
**Please follow the instructions below to compile and run our demo.**  
## 3.1 Upgrade Your AGX Xavier IF Jetpack Version Smaller than 4.6 or TensorRT Version Smaller Than 8.0  
**Warning:Before Upgrading, Please BackUp Your AGX Xavier To Prevent Data Loss**  
Refer to xavier official website:[Over-the-Air Update](https://docs.nvidia.com/jetson/l4t/index.html#page/Tegra%20Linux%20Driver%20Package%20Development%20Guide/updating_jetson_and_host.html#wwpID0E0PB0HA):  
Section: Updating a Jetson Device -> To update to a new minor release  

## 3.2 Clone and Build The Project
**Clone**  
```
$git clone ***
$cd **
```
**Build the Demo**  
```
$bazel build //calculators/tensorrt:trt_inference_calculator_test
```
**Run**  
```
GLOG_logtostderr=1   ./bazel-bin/calculators/tensorrt/trt_inference_calculator_test   --input_video_path=./test1.mp4   --remote_run=true
```

**Expected Output**  
  Under current folder, there will be a video file named **"trt_infer.mp4"** be generated.  
Each frame has detected facial boxes and facial points.Like This:  
![face_detection_trt](images/demo1.gif)   


## 3.3 About The Demo  
We created several calculators under directory "./calculators/"  to build a TensorRT Engine from onnx .    
And the target:trt_inference_calculator_test is a face detection demo to show how to use these calculators.   
Face Detection Demo is an ultrafast face detection solution that comes with 6 landmarks and multi-face support.     
It is based on BlazeFace, a lightweight and well-performing face detector.   
![image](https://user-images.githubusercontent.com/50320677/135061243-fb20b79f-f902-4a5b-92eb-0e563d101090.png)  
 

# TODO  
We left several TODOs which will be done in next version .  
[] Use vpi interfaces to accelerate pre and post process such as color space convertion、 resize 、 saving images etc.  
[] Reuse  mediapipe official released  post process calculators.  
