
import os
import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model.eval_function import EvalScore
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", required=False,help="cfg Directory", default="./cfg/cfg.json" )
    parser.add_argument("--save", help="Directory to save output", default="./test")
    return parser.parse_args()

def OpenCFG(cfg_dir):
    with open(cfg_dir) as f:
        data_cfg = json.load(f)
        if data_cfg is not None:
            print("cfg load success")
            return data_cfg["train"]
        else :
            raise Exception("not load cfg file")


def EvalDataLoader(data_type,cfg_json):
    normalize, saturation, brightness, contrast, hue = cfg_json["transform"]["Normalize"], cfg_json["transform"]["saturation"], cfg_json["transform"]["brightness"], cfg_json["transform"]["contrast"], cfg_json["transform"]["hue"]
    dataset_file = os.path.join(cfg_json["data_set"], data_type + ".txt")
    data_trans = transforms.Compose([transforms.Resize((cfg_json["height"], cfg_json["width"])),
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),transforms.ToTensor(),
            transforms.Normalize(normalize[0], normalize[1])])
    target_transforms = transforms.Compose([Rescale((cfg_json["width"], cfg_json["height"]))])
    dataset = TusimpleSet(dataset_file, transform=data_trans, target_transform=target_transforms)
    train_loader = DataLoader(dataset, batch_size=cfg_json["batch"], shuffle=True)
    return train_loader


def Evaluation():
    args = parse_args()
    cfg_json = OpenCFG(args.cfg_dir)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    eval_dataloader = EvalDataLoader("test",cfg_json)
    model_path = cfg_json["model"]
    model = LaneNet()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(DEVICE)

    iou, dice = 0, 0
    with torch.no_grad():
        for x, target, _ in eval_dataloader:
            y = model(x.to(DEVICE))
            y_pred = torch.squeeze(y['binary_seg_pred'].to('cpu')).numpy()
            y_true = torch.squeeze(target).numpy()
            Score = EvalScore(y_pred, y_true)
            dice += Score.Dice()
            iou += Score.IoU()
    
    print('Final_IoU: %s'% str(iou/len(eval_dataloader.dataset)))
    print('Final_F1: %s'% str(dice/len(eval_dataloader.dataset)))


if __name__ == "__main__":
    Evaluation()