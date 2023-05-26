import torch
import torch.nn as nn

class GramResNet(nn.Module):
    
    def __init__(self):
        super(GramResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = BasicBlock(64, 64, [])
        self.layer2 = BasicBlock(64, 128, [2])
        self.layer3 = BasicBlock(128, 256, [2, 2, 2])
        self.layer4 = BasicBlock(256, 512, [2, 2, 2, 2, 2])
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
    
    def __init__(self, inplanes, planes, strides):
        super(BasicBlock, self).__init__()
        
        self.conv = nn.Conv2d(inplanes, planes, 1, bias=False)

        self.conv1 = nn.Conv2d(inplanes, planes, 3, 2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, 3, 1, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv4 = nn.Conv2d(planes, planes, 3, 1, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.gram = Gram(planes, strides)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool2d(2,2)

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.conv(identity)
        identity = self.pool(identity)
        out += identity
        out = self.relu(out)

        identity = out

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        gram = self.gram(x)
        
        out = self.conv4(out)
        out = self.bn4(out)
        out += identity
        out += gram
        out = self.relu(out)

        return out
    
class Gram(nn.Module):
    def __init__(self, planes, strides):
        super(Gram, self).__init__()
        
        convs = []
        for stride in strides:
            convs.append(nn.Conv2d(1, 1, 3, stride, 1, 1, bias=False))
        convs.append(nn.Conv2d(1, planes, 1, bias=False))
        
        self.conv = nn.Sequential(*convs)

    def forward(self, x):
        b, c, h, w = x.size()
        identity1 = x.view(b, c, h*w)
        identity2 = identity1.transpose(1,2)
        gram = identity1.bmm(identity2) / c / h / w
        gram = gram.unsqueeze(1)
        gram = self.conv(gram)
        
        return gram
        