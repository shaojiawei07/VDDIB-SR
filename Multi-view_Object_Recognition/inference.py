import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import MultipleViewTrainer
from tools.ImgDataset import MultiviewImgDataset
from models.MVCNN import SVCNN_IB, MVCNN_DIB

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-bits", type=int, default=1)
parser.add_argument("-hid_dim1", type=int, default=8)
parser.add_argument("-hid_dim2", type=int, default=12)
parser.add_argument("-val_path", type=str, default="./modelnet40_images_new_12x/*/test")
parser.add_argument("-MVCNN_model_path", type=str)
parser.set_defaults(train=False)

if __name__ == '__main__':
    args = parser.parse_args()
    cnet = SVCNN_IB(args.name, nclasses=40)
    cnet_2 = MVCNN_DIB(args.name, cnet, nclasses=40, num_views=args.num_views, hid_dim1 = args.hid_dim1, hid_dim2 = args.hid_dim2, bits = args.bits)
    cnet_2.load_state_dict(torch.load(args.MVCNN_model_path))
    del cnet
    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    trainer = MultipleViewTrainer(cnet_2, None, val_loader, None, nn.CrossEntropyLoss(),num_views=args.num_views, args = args)

    threshold_list = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975,0.99]

    print("\nThreshold_list",threshold_list)

    for i in range(len(threshold_list)):
        trainer.cascade_inference(threshold_list[i])


