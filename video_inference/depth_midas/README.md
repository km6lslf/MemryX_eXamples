# Depth Estimation Using MiDaS

The **Depth Estimation** example demonstrates real-time depth inference using the pre-trained MiDaS v2 Small model on MemryX accelerators. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/depth.png" alt="Depth Estimation Example" width="35%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [MiDaS v2 Small](https://arxiv.org/pdf/1907.01341) 🔗
| **Model Type**       | Depth Estimation                                                        |
| **Framework**        | [TensorFlow](https://www.tensorflow.org/) 🔗
| **Model Source**     | [Download from TensorFlow Hub](https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1) 🔗
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/example_files/depth_estimation_using_midas.zip)
| **Input**            | 256x256 (default)
| **Output**           | Depth map (matches input resolution)
| **OS**               | Linux, Windows
| **License**          | [MIT](LICENSE.md)

## Requirements (Linux)

Before running the application, ensure that **OpenCV** and **curl** are installed, especially for the Python implementation. You can install OpenCV and curl using the following commands:

```bash
pip install opencv-python
sudo apt install curl
```

## Running the Application (Linux)

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/example_files/depth_estimation_using_midas.zip
mkdir -p models
unzip depth_estimation_using_midas.zip -d models
```

<details> 
<summary> (Optional) Download and compile the model yourself </summary>
If you prefer, you can download and compile the model rather than using the precompiled model. Download the pre-trained MiDaS v2 Small model from TensorFlow Hub:

```bash
wget https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1 -O midas_v2_small.tflite
```

You can now use the MemryX Neural Compiler to compile the model and generate the DFP file required by the accelerator:

```bash
mx_nc -m midas_v2_small.tflite
```

</details>


### Step 2: Run the Script/Program

With the compiled model, you can now run real-time inference. Below are the examples of how to do this using Python and C++.

#### Python

To run the Python example for real-time depth estimation using MX3, simply execute the following command:

```bash
python src/python/run_depth_estimate.py
```
You can specify the model path and DFP (Compiled Model) path with the following options:

* `-m` or `--model`: Path to the model file (default is midas_2_small.tflite)
* `-d` or `--dfp`: Path to the compiled DFP file (default is ../assets/midas_v2_small.dfp)

For example, to run with a specific model and DFP file, use:

```bash
python src/python/run_depth_estimate.py -m <model_path> -d <dfp_path>
```

If no arguments are provided, the script will use the default model and DFP paths.

#### C++

To run the C++ example for real-time depth estimation using MX3, follow these steps:

1. Build the project using CMake. From the project directory, execute:

```bash
mkdir build
cd build
cmake ..
make
```

2. Run the application. By default, the camera will be used as the input. You can also provide a video file for input and specify a DFP file if needed.

* To run using the default DFP file and a camera as input

```bash
./depthEstimation
```

* To run with a video file as input:

```bash
./depthEstimation --video <path_to_video_file>
```

* To specify a custom DFP file, use the `-d` option:


```bash
./depthEstimation -d <path_to_dfp_file>
```


## Running the Application (Windows)

[Download](https://developer.memryx.com/example_files/depthEstimation_windows.zip) the compiled C++ executable version, and extract the zip.

Then just double-click `depthestimation.exe` to launch using the first available webcam.

Alternatively, you can use commandline arguments such as `--video` if launching the exe within Command Prompt or PowerShell.


## Tutorial

A more detailed tutorial with a complete code explanation is available on the [MemryX Developer Hub](https://developer.memryx.com). You can find it [here](https://developer.memryx.com/docs/tutorials/realtime_inf/realtime_depth.html)


## Third-Party Licenses

This project uses third-party software, models, and libraries. Below are the details of the licenses for these dependencies:

- **Model**: [MiDaS v2 Small (TF Lite) from kaggle](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗 
  - License: [MIT](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗

- **Code and Pre/Post-Processing**: Some code components, including pre/post-processing, were sourced from the MiDaS v2 Small model provided on [Kaggle](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1)  
  - License: [MIT](https://www.kaggle.com/models/intel/midas/tfLite/v2-1-small-lite/1) 🔗

- **Preview**: ["Two Baseball Players Talking to Each Other" on Pexels](https://www.pexels.com/video/two-baseball-players-talking-to-each-other-5182642/)  
  - License: [Pexels License](https://www.pexels.com/license/)

## Summary

This guide offers a quick and easy way to run depth estimation using the MiDaS model on MemryX accelerators. You can use either the Python or C++ implementation to perform real-time inference. Download the full code and the pre-compiled DFP file to get started immediately.
