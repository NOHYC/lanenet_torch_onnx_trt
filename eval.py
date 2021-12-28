
import os
import torch
from dataloader.data_loaders import TusimpleSet
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torchvision import transforms
from model.eval_function import Eval_Score
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Dataset path")
    parser.add_argument("--model", help="Model path", default='./save/model.pth')
    parser.add_argument("--width", required=False, type=int, help="Resize width", default=512)
    parser.add_argument("--height", required=False, type=int, help="Resize height", default=256)
    parser.add_argument("--save", help="Directory to save output", default="./test")
    return parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_loader(data_type,data_set,resize_height,resize_width,batch):
    
    dataset_file = os.path.join(data_set, data_type + ".txt")
    transforms.Compose([transforms.Resize((resize_height, resize_width)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  ])
    target_transforms = transforms.Compose([Rescale((resize_width, resize_height))])
    dataset = TusimpleSet(dataset_file, transform=transforms, target_transform=target_transforms)
    train_loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return train_loader


def evaluation():
    args = parse_args()
    eval_dataloader = data_loader(args.dataset,"test",args.height,args.width,args.batch)
    model_path = args.model
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
            Score = Eval_Score(y_pred, y_true)
            dice += Score.Dice()
            iou += Score.IoU()
    
    print('Final_IoU: %s'% str(iou/len(eval_dataloader.dataset)))
    print('Final_F1: %s'% str(dice/len(eval_dataloader.dataset)))


if __name__ == "__main__":
    evaluation()