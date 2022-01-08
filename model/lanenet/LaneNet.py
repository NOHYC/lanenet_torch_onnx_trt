
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.lanenet.backbone.ENet import ENet_Encoder, ENet_Decoder

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LaneNet(nn.Module):
    def __init__(self, in_ch = 3):
        super(LaneNet, self).__init__()
        self.no_of_instances = 3  
        self._encoder = ENet_Encoder(in_ch)
        self._encoder.to(DEVICE)

        self._decoder_binary = ENet_Decoder(2)
        self._decoder_instance = ENet_Decoder(self.no_of_instances)
        self._decoder_binary.to(DEVICE)
        self._decoder_instance.to(DEVICE)
        self.relu = nn.ReLU().to(DEVICE)
        self.sigmoid = nn.Sigmoid().to(DEVICE)

    def forward(self, input_tensor):
        c = self._encoder(input_tensor)
        binary = self._decoder_binary(c)
        instance = self._decoder_instance(c)
        binary_seg_ret = torch.argmax(F.softmax(binary, dim=1), dim=1, keepdim=True)
        pix_embedding = self.sigmoid(instance)
        return {
            'instance_seg_logits': pix_embedding,
            'binary_seg_pred': binary_seg_ret,
            'binary_seg_logits': binary
        }
