# VDDIB-SR for multi-view image classification


## Overview

* We consider a two-view image classification task on the `MNIST` dataset. Assume that there are two devices and one server in an edge inference system. Device 1 can perceive the upper half of an MNIST digit, and device 2 can observe the lower half of the same MNIST digit.

* For simplicity, the parameter $T$ (i.e., the maximum number of transmission attempts) of VDDIB-SR is set to 2, which means that there are two communication rounds in edge inference.

* The proposed VDDIB-SR method will first train feature extractors. To reduce the complexity of the training process, the feature extractors of device 1 and device 2 have the same parameters. The function `train_VIB()` is used to train the feature extractors. 

* After the feature extraction step, the function `train_VDDIB_SR()` is called to train the distributed feature encoding scheme for communication overhead reduction.

* After the above two steps, calling the function `fine_tune()` for model fine-tuning could further improve the performance.

* In the inference phase, we select a `threshold` to determine *if the current received features are sufficient for the inference*. Increasing the `threshold` value encourages devices to transmit more features to the server to improve the inference performance. But this leads to high communication overhead. We provide a `threshold_list` to show the communication-performance tradeoff.




## How to run
### An example
`python main_MNIST.py --dim1 2  --dim2 4 --bit 1 --epochs 30 `

The parameters `--dim1`, `--dim2`, and `--bits` correspond to the communication overhead in edge inference. The communication overhead of each device in the first round transmission and second round transmission is `dim1 * bits` bits and `dim2 * bits`, respectively.


## Others

* `torchvision` package will automatically download the `MNIST` dataset when running the script `main_MNIST.py`.

* The `attention module` has not been applied in this task since the performance gain is marginal when the number of views is small.
