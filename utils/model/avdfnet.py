import torch
import torch.nn as nn
from .avnet import AVNet
from typing import OrderedDict

class AVDFNet(nn.Module):
    
    def __init__(self):
        super(AVDFNet,self).__init__()
        self.extractor = AVNet(18, False, 10)
        # self.extractor.load_state_dict(torch.load('weights/avnet18-data-36-acc0.9979296066252588.pt'))
        self.extractor.classfier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(128, 64)),
            ('bn1', nn.BatchNorm1d(64, momentum=0.01)),
            ('drop2', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(64, 2))
        ]))

    def forward(self, images, audio):
        return self.extractor(images, audio)