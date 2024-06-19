# AIXI Grain Detection

## Introduction

This is an implementation of Faster R-CNN in PyTorch for the detection of aluminium grains from videos of Al-Cu alloys solidifying from fully liquid imaged via X-ray radiography at synchrotron sources.

Trained weights and dataset are not included, contact rohit.abraham@ccc.ox.ac.uk for further information.

## Setup

To set up the project environment, use the provided `environment.yml` file with a conda package manager to install the appropriate packages.

```sh
conda env create -f environment.yml
conda activate aixi_grain_detection
```


## Training

- A short example of the dataset has been included in `train/example_data` with train and validation sets in order to validate set-up.
- Annotations file is provided separately; contact the repo owner.
- `dataset.py` takes annotations in CSV format with the following headers: `filename,xmin,ymin,xmax,ymax,class`.
- If `AIXI_PUBLIC` is not the root working directory, change file paths in `train.py` before running.


## Evalution

To evaluate a trained model on a test set, a COCO style prediction file can be produced using `gen_coco_predictions.py`. The outputted file can be used with the fork of [Review Object Detection Metrics](https://github.com/rohit5-2/review_object_detection_metrics_AIXI) to generate various metrics and PR curve data.


## Inference

To generate predictions on an experiment take the following steps:

- Collect frames belonging to a particular experiment in a folder with some form of sequential naming convention
- Use `generate_predictions.py` to generate a CSV of detections
- (optional) Use `generate_video.py` to generate a visualisation of an experiment

An example set of frames belonging to an experiment has been included. 
