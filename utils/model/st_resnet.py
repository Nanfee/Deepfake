import torch
import torch.nn as nn
import torchvision.models as models
from typing import OrderedDict

class STResNet(nn.Module):

    def __init__(self, type, hidden_dim, hidden_layers):

        super(STResNet, self).__init__()


        if type == 18:
            self.down = DownSample(128, 512)
            self.s_net = models.resnet18(pretrained=True)
        elif type == 50:
            self.down = DownSample(512, 2048)
            self.s_net = models.resnet50(pretrained=True)

        feature_size = self.s_net.fc.in_features

        for param in self.s_net.parameters():
            param.requires_grad = False

        self.s_net.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(feature_size, 1024)),
            ('bn1', nn.BatchNorm1d(1024, momentum=0.01)),
            ('relu', nn.ReLU()),
            ('drop', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(1024, 256))
        ]))

        self.t_net = nn.GRU(input_size=256+136, hidden_size=hidden_dim, num_layers=hidden_layers, batch_first=True)

        self.classifier = nn.Sequential(OrderedDict([
            ('drop1', nn.Dropout(p=0.5)),
            ('fc1', nn.Linear(hidden_dim, 128)),
            ('bn1', nn.BatchNorm1d(128, momentum=0.01)),
            ('drop2', nn.Dropout(p=0.5)),
            ('fc2', nn.Linear(128, 2))
        ]))

        self.softmax = nn.Softmax(-1)

    
    def forward(self, x_3d, landmark):
        
        # spatial encoding
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):

            x = self.s_net.conv1(x_3d[:,t,:,:,:])
            x = self.s_net.bn1(x)
            x = self.s_net.relu(x)
            x = self.s_net.maxpool(x)

            x = self.s_net.layer1(x)
            x = self.s_net.layer2(x)
            y = x
            x = self.s_net.layer3(x)
            x = self.s_net.layer4(x)
            # y = self.down(y)
            # x = x + 2*y

            x = self.s_net.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.s_net.fc(x)

            cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        cnn_embed_seq = torch.cat((cnn_embed_seq, landmark), dim=2)
        
        # temporal encoding
        x = self.t_net(cnn_embed_seq)[0]
        x = x[:,-1,:]
        # features = x
        x = self.classifier(x)
        return self.softmax(x)#, features
    

class DownSample(nn.Module):
    def __init__(self, in_planes, planes):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.down = nn.AvgPool2d(4, 4)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.down(x)
        return x

