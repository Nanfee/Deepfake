import os
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

trans = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def detect(modelPath, faceFolder):
    model = torch.load(modelPath).cuda()
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    ans = []
    for imgName in tqdm(os.listdir(faceFolder)):
        img = Image.open(os.path.join(faceFolder, imgName))
        img = trans(img).unsqueeze(0)
        img = img.cuda()
        output = model(img)
        output = F.softmax(model(img)[0], 1)
        ans.append({
            'name': imgName,
            'output': output.cpu().detach().numpy().tolist()[0]
        })
        #real: [1, 0], fake: [0, 1]
        # _, predict = torch.max(output, 1)
        # predict = int(predict)
    return ans
