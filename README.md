# VDDIB-SR

This repository aims at introducing the VDDIB-SR method, which can effectively reduce the communication overhead in multi-device edge inference. This method is proposed in our [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9837474) "Task-Oriented Communication for Multi-Device Cooperative Edge Inference", which is accepted to IEEE Transaction on Wireless Communication. 

We release the code to reproduce some experimental results. Particularly, we focus on the multi-view image classification task on the MNIST dataset and the multi-view object recognition task on the ModelNet40 dataset. The codes for these two tasks are in the folder `./Multi-view_Image_Classification` and folder `./Multi-view_Object_Recognition`, respectively.





## Dependencies
### Packages
```
Pytorch 1.11.0
Torchvision 0.12.0
scikit-image 0.19.3
```
### Datasets

```
MNIST
ModelNet40
```


## Citation

```
@ARTICLE{9837474,
  author={Shao, Jiawei and Mao, Yuyi and Zhang, Jun},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Task-Oriented Communication for Multidevice Cooperative Edge Inference}, 
  year={2023},
  volume={22},
  number={1},
  pages={73-87},
  doi={10.1109/TWC.2022.3191118}}

```
