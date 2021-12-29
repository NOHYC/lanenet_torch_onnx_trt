import numpy as np
import argparse
import torch
import torch.onnx
from model.lanenet import LaneNet

parser = argparse.ArgumentParser(description = "pythorch lanenet")
parser.add_argument("--torch_dir",required=True, help = "need torch lanenet_model")
parser.add_argument("--onnx_dir",required=True, help = "save onnx")
args = parser.parse_args()


model = LaneNet.LaneNet()

map_location = lambda storage, loc : storage
if torch.cuda.is_available():
    map_location = None

model.load_state_dict( torch.load(args.torch_dir, map_location = map_location) )

model.eval()
batch_size = 1

x = torch.randn(batch_size,3, 256, 512 ,requires_grad = True).cuda()

torch_out = model(x)

torch.onnx.export(model, x,args.onnx_dir+"/Lanenet.onnx", export_params = True,
 opset_version = 9, input_names = ['input'], output_names = ['output1','output2','output3'])
