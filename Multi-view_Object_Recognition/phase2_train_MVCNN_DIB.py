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
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=16)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-2)
parser.add_argument("-SVCNN_model_path", type=str, help="load the pretrained model after phase1")
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.001)
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-bits", type=int, default=1)
parser.add_argument("-hid_dim1", type=int, default=8)
parser.add_argument("-hid_dim2", type=int, default=12)
parser.add_argument("-train_path", type=str, default="./modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="./modelnet40_images_new_12x/*/test")
parser.add_argument("-epoch", type=int, default=50)
parser.add_argument("-beta", type=float, default=1e-1,help="Controls DIB")
parser.add_argument("-view_number_constraint", type=float, default=11)

#parser.set_defaults(train=False)


def create_folder(log_dir):
    # make summary folder
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    else:
        print('WARNING: summary folder already exists!! It will be overwritten!!')
        shutil.rmtree(log_dir)
        os.mkdir(log_dir)

if __name__ == '__main__':
    args = parser.parse_args()

    cnet = SVCNN_IB(args.name, nclasses=40)
    try:
        cnet.load_state_dict(torch.load(args.SVCNN_model_path))
    except:
        print("Fail to load the pretrained model")

    cnet_2 = MVCNN_DIB(args.name, cnet, nclasses=40, num_views=args.num_views, hid_dim1 = args.hid_dim1, hid_dim2 = args.hid_dim2, bits = args.bits)
    del cnet

    optimizer = optim.SGD(cnet_2.parameters(), lr=args.lr)
    
    train_dataset = MultiviewImgDataset(args.train_path, scale_aug=False, rot_aug=False, test_mode=False, num_views=args.num_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=8) # shuffle needs to be false! it's done within the trainer

    val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, test_mode=True, num_views=args.num_views)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=8)
    print('num_train_files: '+str(len(train_dataset.filepaths)))
    print('num_val_files: '+str(len(val_dataset.filepaths))+'\n')
    trainer = MultipleViewTrainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), num_views=args.num_views, args = args)
    trainer.train_DIB(n_epochs = 1)
    trainer.fine_tune(args.epoch)


