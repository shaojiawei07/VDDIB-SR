import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import os,shutil,json
import argparse

from tools.Trainer import SingleViewTrainer
from tools.ImgDataset import SingleImgDataset
from models.MVCNN import SVCNN_IB

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="SVCNN")
parser.add_argument("-single_batch",type = int, default = 128)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="./modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="./modelnet40_images_new_12x/*/test")
parser.add_argument("-epoch", type=int, default=50)
parser.add_argument("-gamma", type=float, default=1e-4)

parser.set_defaults(train=False)


def create_folder(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        ...

if __name__ == '__main__':
    args = parser.parse_args()
    pretraining = not args.no_pretraining
    create_folder("saved_model")
    cnet = SVCNN_IB(args.name, nclasses=40, pretraining=pretraining)
    optimizer = optim.Adam(cnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = SingleImgDataset(args.train_path, scale_aug=False, rot_aug=False, test_mode=False, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.single_batch, shuffle=True, num_workers=0)
    val_dataset = SingleImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.single_batch, shuffle=False, num_workers=0)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths)))
    trainer = SingleViewTrainer(cnet, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), args = args)
    trainer.train_IB(args.epoch)


