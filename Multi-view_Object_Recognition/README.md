# VDDIB-SR for multi-view shape recognition

## Preliminary

* We adopt the MVCNN model for shape recognition on the ModelNet40 dataset. Some codes are borrowed from [this GitHub repository](https://github.com/jongchyisu/mvcnn_pytorch).

* We use the `ModelNet40` dataset in this experiment, which contains CAD models from 40 categories. In particular, we consider the multi-view object recognition task and generate 12 views from every CAD model. Download the multi-view dataset from [this link](http://supermoe.cs.umass.edu/shape_recog/shaded_images.tar.gz) and put it under folder `./modelnet40_images_new_12x`. For more details about the dataset, please visit the [homepage](https://modelnet.cs.princeton.edu/).


## Overview

* Consider there are twelve devices and one server in an edge inference system. Each device perceives a view of CAD models.

* For simplicity, the parameter $T$ (i.e., the maximum number of transmission attempts) of VDDIB-SR is set to 2, which means that there are two communication rounds in edge inference.

* We construct single-view model `SVCNN_IB` and multi-view model `MVCNN_DIB` for shape recognition.

* Run `phase1_train_SVCNN_IB.py` to train the feature extractor based on the information bottleneck (IB) principle.

* Run `phase2_train_MVCNN_DIB.py` to train the distributed feature coding scheme to reduce the communication overhead based on the distributed information bottleneck (DIB) framework.

* Test the performance of the model by calling `inference.py`.

## How to run
### An example

### Step 1

Using the command `python phase1_train_SVCNN_IB.py ` to train an `SVCNN_IB` model.

### Step 2

With the well-trained `SVCNN_IB` model, the next step is to train an `MVCNN_DIB` model by `python phase2_train_MVCNN_DIB.py -SVCNN_model_path ./PATH/TO/THE/SVCNN_IB/MODEL -hid_dim1 8 -hid_dim2 12 -bits 1`.

The parameter `-SVCNN_model_path` is used to load the `SVCNN_IB` model trained in Step 1. The parameters `-hid_dim1 `, `-hid_dim2 `, and `-bits` correspond to the communication overhead in edge inference. The costs of each device in the first round transmission and second round transmission are `hid_dim1 * bits` bits and `hid_dim2 * bits` bits, respectively.

### Step 3

After Step 2, we test the `MVCNN_DIB` model by running `python3 test_MVCNN.py -MVCNN_model_path ./PATH/TO/THE/MVCNN_DIB/MODEL -hid_dim1 8 -hid_dim2 12 -bits 1`.



## Others

* GPU and Cuda are required to run this code. The GPU memory consumption is around 18.5 GB.
