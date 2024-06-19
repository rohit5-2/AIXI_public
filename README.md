# AIXI Grain Detection Repo

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
- If `AIXI_PUBLIC` is not the root directory, change file paths in `train.py` before running.


