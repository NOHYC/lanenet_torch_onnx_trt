import os
import torch
from model.lanenet.train_lanenet import train_model
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import transforms

from model.eval_function import Eval_Score

import numpy as np
import pandas as pd
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",required=True, help="Dataset path")
    parser.add_argument("--save", required=False, help="Directory to save model", default="./save")
    parser.add_argument("--epochs", required=False, type=int, help="Training epochs", default=25)
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--batch", required=False, type=int, help="Batch size", default=4)
    parser.add_argument("--val", required=False, type=bool, help="Use validation", default=False)
    parser.add_argument("--lr", required=False, type=float, help="Learning rate", default=0.0001)
    return parser.parse_args()


DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def data_loader(data_set,data_type,resize_height,resize_width,batch):
    
    dataset_file = os.path.join(data_set, data_type + ".txt")
    data_trans = transforms.Compose([transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ])
    target_transforms = transforms.Compose([Rescale((resize_width, resize_height))])
    dataset = TusimpleSet(dataset_file, transform=data_trans, target_transform=target_transforms)
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return train_loader

def save_log(model, log, save_dir):

    df=pd.DataFrame({'epoch':[],'training_loss':[],'val_loss':[]})
    df['epoch'] = log['epoch']
    df['training_loss'] = log['training_loss']
    df['val_loss'] = log['val_loss']

    train_log_save_filename = os.path.join(save_dir, 'training_log.csv')
    df.to_csv(train_log_save_filename, columns=['epoch','training_loss','val_loss'], header=True,index=False,encoding='utf-8')
    print("training log is saved: {}".format(train_log_save_filename))
    
    model_save_filename = os.path.join(save_dir, 'model.pth')
    torch.save(model.state_dict(), model_save_filename)
    print("model is saved: {}".format(model_save_filename))


def train():
    args = parse_args()

    train_loader = data_loader(args.dataset,"train",args.height,args.width,args.batch)
    val_loader = data_loader(args.dataset,"val",args.height,args.width,args.batch)
    dataloaders = {'train' : train_loader, 'val' : val_loader}
    dataset_sizes = {'train': len(train_loader.dataset), 'val' : len(val_loader.dataset)}

    model = LaneNet()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model, log = train_model(model, optimizer, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, num_epochs=args.epochs)
    save_log(model, log, args.save)



if __name__ == '__main__':
    train()
