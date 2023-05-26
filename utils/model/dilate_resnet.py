import torch
import torch.nn as nn

class DilateResNet(nn.Module):

    def __init__(self):
        super(DilateResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layer1 = BasicBlock(64, 64)
        self.layer2 = BasicBlock(64, 128)
        self.layer3 = BasicBlock(128, 256)
        self.layer4 = BasicBlock(256, 512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 2)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return self.softmax(out)



class BasicBlock(nn.Module):
    
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        
        self.conv = None
        if inplanes != planes:
            self.conv = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, 3, 2, 5, 5, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2,2)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.conv is not None:
            x = self.conv(x)
        out += x
        out = self.relu(out)

        identity = out
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.pool(identity)
        out = self.relu(out)

        return out
        