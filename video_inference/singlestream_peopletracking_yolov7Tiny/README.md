# People Tracking Using Yolov7 Tiny

The **People Tracking** example demonstrates real-time people tracking on a single input stream using the pre-trained yolov7 tiny model on MemryX accelerators. It uses [Kalman filters](https://www.mathworks.com/help/vision/ug/using-kalman-filter-for-object-tracking.html) to identify the "same" person across frames, and assigns unique IDs upon detecting a "new" person. This guide provides setup instructions, model details, and necessary code snippets to help you quickly get started.

<p align="center">
  <img src="assets/people_counting.gif" alt="People Tracking Example" width="30%" />
</p>

## Overview

| Property             | Details                                                                 |
|----------------------|-------------------------------------------------------------------------|
| **Model**            | [Yolov7 Tiny](https://arxiv.org/pdf/2207.02696)                         |
| **Model Type**       | Object Detection                                                        |
| **Framework**        | [ONNX](https://onnx.ai/)                                                |
| **Model Source**     | [Download](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt) and [export](https://github.com/WongKinYiu/yolov7/blob/main/export.py) to onnx |
| **Pre-compiled DFP** | [Download here](https://developer.memryx.com/docs/model_explorer/YOLO-v7-tiny_416_416_3.zip) |
| **Model Dataset**    | [COCO](https://docs.ultralytics.com/datasets/detect/coco/) |
| **Model Resolution** | 416x416                                                    |
| **Output**           | Bounding box coordinates & object probabilities            |
| **OS**           | Linux            |
| **License**          | [GPL](LICENSE.md)                                          |

## Requirements

Before running the application, ensure that OpenCV Python is installed. You can install using the following command in your Python virtual env:

```bash
# For application
pip install opencv-python
```

If you want to export and compile the DFP yourself, also install yolov7 dependencies:

```bash
# For exporting the source yolov7 model to onnx
pip install seaborn pyyaml pandas
```

## Running the Application

### Step 1: Download Pre-compiled DFP

To download and unzip the precompiled DFPs, use the following commands:
```bash
wget https://developer.memryx.com/docs/model_explorer/YOLO-v7-tiny_416_416_3.zip
mkdir -p assets
unzip YOLO-v7-tiny_416_416_3.zip -d assets
```

<details> 
<summary> (Optional) Export and compile the model yourself </summary>

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt -O yolov7tiny.pt

python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 416 416 --max-wh 416
```

The export script will generate a yolov7-tiny onnx file, which can be compiled to DFP with the NeuralCompiler:

```bash
 mx_nc -v -m yolov7-tiny.onnx -v --autocrop
```

The compiler will generate the DFP and a post-processing file which can be passed as inputs to the application. Put these files in the `assets/` folder with the names `yolov7-tiny_416.dfp` and `yolov7-tiny_416.post.onnx`.

</details>

### Step 2: Run the Program

To run the Python example for people tracking with yolov7-tiny using MX3, simply execute the following command:

```bash
# ensure a camera device is connected, as the default video input is a cam
python run_yolov7tiny_singlestream_peopletracking.py 
```

You can specify the input video path with the following option:

* `--video_path` : Path to video file as inputs (default is /dev/video0, which is the first camera connected to the system)

For example, to run with a video file as input, run:

```bash
python run_yolov7tiny_singlestream_peopletracking.py --video_path ~/Videos/test_video.mp4
```

If no arguments are provided, the script will use the default (first camera).


## Third-Party Licences

This project uses third-party software and models. Below are the details of the licenses for these dependencies:

- **Model** Copyright (c) 2023 Chien-Yao Wang, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. [GPLv3 License](https://github.com/WongKinYiu/yolov7/blob/main/LICENSE.md)

