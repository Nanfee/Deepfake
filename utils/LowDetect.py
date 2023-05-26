import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from .model.dilate_resnet import DilateResNet
from .model.spd_resnet import SPDResNet
from .model.gram_resnet import GramResNet

trans = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def detect(fun, faceFolder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if fun == 'dilate':
        model = DilateResNet()
        model.load_state_dict(torch.load('utils/weights/DilateResNet-data-93-acc0.9981191973644804.pt'), strict=False)
    elif fun == 'spd':
        model = SPDResNet()
        model.load_state_dict(torch.load('utils/weights/SPDResNet-data-131-acc0.9994609164420485.pt'), strict=False)
    else:
        model = GramResNet()
        model.load_state_dict(torch.load('utils/weights/GramResNet-data-128-acc0.9982150344414495.pt'), strict=False)
    
    model = model.to(device)
    model.eval()

    ans = []
    for imgName in tqdm(os.listdir(faceFolder)):
        img = Image.open(os.path.join(faceFolder, imgName))
        img = img.convert('RGB')
        img = trans(img).unsqueeze(0)
        img = img.cuda()
        output = model(img)
        print(output)
        # output = F.softmax(model(img)[0], 1)
        ans.append({
            'name': imgName,
            'output': output.cpu().detach().numpy().tolist()[0]
        })
        #real: [1, 0], fake: [0, 1]
        # _, predict = torch.max(output, 1)
        # predict = int(predict)
    return ans
