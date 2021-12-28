import time
import os
import sys
import torch
from dataloader.transformers import Rescale
from model.lanenet.LaneNet import LaneNet
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2
import timeit
import argparse

resize_height = 256
resize_width = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="video path")
    parser.add_argument("--model", help="Model path", default='./save/model.pth')
    return parser.parse_args()

def infer():
    args = parse_args()
    video_path = args.video
    model = LaneNet(arch="ENET")
    model.load_state_dict(torch.load(args.model))
    model.eval()
    model.to(DEVICE)
    data_transform = transforms.Compose([
        transforms.Resize((resize_height,  resize_width)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cap = cv2.VideoCapture(video_path)
    while(cap.isOpened()):

        ret, frame = cap.read()
        if ret:
            dummy_input = Image.fromarray(frame)
            dummy_input = data_transform(dummy_input)
            dummy_input = torch.unsqueeze(dummy_input, dim=0).to(DEVICE)

            start_t = timeit.default_timer()
            outputs = model(dummy_input)
            terminate_t = timeit.default_timer()
            
            FPS = int(1./(terminate_t - start_t ))
            print("FPS : ",FPS)

            instance_pred = torch.squeeze(outputs['instance_seg_logits'].detach().to('cpu')).numpy() * 255
            #binary_pred = torch.squeeze(outputs['binary_seg_pred']).to('cpu').numpy() * 255
            #binary_pred = np.expand_dims(binary_pred, axis = 0)
            cv2.imshow("tex", instance_pred.transpose((1, 2, 0)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    infer()