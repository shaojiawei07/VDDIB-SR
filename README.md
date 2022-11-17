# VDDIB-SR

This repository contains the codes to reproduce the main experimental results in the [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9837474) "Task-Oriented Communication for Multi-Device Cooperative Edge Inference", which is accepted to IEEE Transaction on Wireless Communication. The content continues updating.



## Logs

**[Aug-16-2022]** I have released the code for the MNIST dataset. 

**[Jul-23-2022]** Our paper has been accepted, and I plan to release the codes before Aug-16-2022.


## Dependencies
### Packages
```
Pytorch 1.11.0
Torchvision 0.12.0
```
### Datasets
```
MNIST
```


## How to run
### Train the VDDIB-SR method on the MNIST dataset
`python main_VDDIB-SR_MNIST.py --dim1 2  --dim2 4 --bit 1 --epochs 30 `


The parameters `--dim1`, `--dim2`, and `--bits` correspond to the communication overhead in the transmission.
The costs of each device in the first round transmission and second round transmission are `dim1 * bit` bits and `dim2 * bit` bits, respectively.




## Citation

```
@ARTICLE{9837474,
  author={Shao, Jiawei and Mao, Yuyi and Zhang, Jun},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Task-Oriented Communication for Multi-Device Cooperative Edge Inference}, 
  year={2022},
  doi={10.1109/TWC.2022.3191118}}
```

## Others

The `attention module` has not been applied for the two-view MNIST dataset since the performance gain is marginal when the number of views is small.
