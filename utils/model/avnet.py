import torch
import torch.nn as nn
from torchvision import models
from typing import OrderedDict

class AVNet(nn.Module):
    
    def __init__(self, type=18, pretrained=False, nums=10):
        super(AVNet,self).__init__()
        if type == 18:
            self.visual = models.resnet18(pretrained=pretrained)
        elif type == 50:
            self.visual = models.resnet50(pretrained=pretrained)
        self.visual.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.visual.fc.in_features, 512)),
            ('bn1', nn.BatchNorm1d(512, momentum=0.01)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(512, 256))
        ]))
        self.gru = nn.GRU(input_size=256, hidden_size=128, num_layers=3, batch_first=True)
        self.classfier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(128, 64)),
            ('bn1', nn.BatchNorm1d(64, momentum=0.01)),
            ('drop2', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(64, nums))
        ]))
        # self.classfier2 = nn.Sequential(OrderedDict([
        #     ('drop1', nn.Dropout(p=0.5)),
        #     ('fc1', nn.Linear(128, 64)),
        #     ('bn1', nn.BatchNorm1d(64, momentum=0.01)),
        #     ('drop2', nn.Dropout(p=0.5)),
        #     ('fc2', nn.Linear(64, 2))
        # ]))
        self.softmax = nn.Softmax(-1)

    def forward(self, images, audio):
        
        visual_features = []
        for t in range(images.size(1)):
            x = self.visual(images[:,t,:,:,:])
            visual_features.append(x)
        v0 = torch.stack(visual_features, dim=0).transpose_(0, 1)
        v1 = self.gru(v0)[0]
        v2 = v1[:,-1,:]
        f = 0.25 * v2 + 0.75 * audio

        x = self.classfier(f)

        return v2, audio, x