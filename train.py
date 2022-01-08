import os
import torch
from model.lanenet.train_lanenet import TrainModel
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torch.autograd import Variable
import json
from torchvision import transforms
import numpy as np
import pandas as pd
import cv2
import argparse

def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", required=False,help="cfg Directory", default="./cfg/cfg.json" )
    parser.add_argument("--save", required=False, help="Directory to save model", default="./save")
    return parser.parse_args()


def OpenCFG(cfg_dir):
    with open(cfg_dir) as f:
        data_cfg = json.load(f)
        if data_cfg is not None:
            print("cfg load success")
            return data_cfg["train"]
        else :
            raise Exception("not load cfg file")
    

def TrainDataLoader(data_type,cfg_json):
    if data_type == "val" and cfg_json["validation"] == False:
        return None
    normalize, saturation, brightness, contrast, hue = cfg_json["transform"]["Normalize"], cfg_json["transform"]["saturation"], cfg_json["transform"]["brightness"], cfg_json["transform"]["contrast"], cfg_json["transform"]["hue"]
    dataset_file = os.path.join(cfg_json["data_set"], data_type + ".txt")
    data_trans = transforms.Compose([transforms.Resize((cfg_json["height"], cfg_json["width"])),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),transforms.ToTensor(),
            transforms.Normalize(normalize[0], normalize[1])])
    target_transforms = transforms.Compose([Rescale((cfg_json["width"], cfg_json["height"]))])
    dataset = TusimpleSet(dataset_file, transform=data_trans, target_transform=target_transforms)
    train_loader = DataLoader(dataset, batch_size=cfg_json["batch"], shuffle=True)
    return train_loader

def SaveLog(model, log, save_dir):

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

def Train():
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ParseArgs()
    cfg_json= OpenCFG(args.cfg_dir)
    dataloaders = {}
    dataset_sizes = {}
    for train_type in ["train", "val"]:
        dataloaders[train_type] = TrainDataLoader(train_type, cfg_json)
        dataset_sizes[train_type] = len(dataloaders[train_type].dataset)
    model = LaneNet()
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_json.learning_rate)

    model, log = TrainModel(model, optimizer, dataloaders=dataloaders, dataset_sizes=dataset_sizes, device=DEVICE, num_epochs=cfg_json["epochs"])
    SaveLog(model, log, args.save)

if __name__ == '__main__':
    Train()
