import os
import torch
import cv2
import dlib
from imutils import face_utils
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from .model.st_resnet import STResNet

trans = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def detect(faceFolder, pre_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STResNet(type=18, hidden_dim=128, hidden_layers=3)
    model.load_state_dict(torch.load('utils/weights/resnet18-DeeperForensics_Advanced-100-acc0.9867899603698811.pt'), strict=False)
    
    model = model.to(device)
    model.eval()

    ans = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pre_path)

    videos = os.listdir(faceFolder)
    for videoName in tqdm(videos):
        # 抽帧并裁减人脸
        path = os.path.join(faceFolder, 'images')
        if not os.path.exists(path):
            os.makedirs(path)
        vidcap = cv2.VideoCapture(os.path.join(faceFolder, videoName))
        frames = []
        landmarks = []
        count = 0
        while True:
            success, frame = vidcap.read()
            if success:
                frames.append(frame)
                count += 1
                if count >= 30:
                    break
            else:
                print('fail')
                break
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = detector(gray, 0)
            if len(face) == 0:
                print(path + 'failed!')
                return
            face = face[0]
            width = face.bottom() - face.top()
            landmark = predictor(frame, face)
            landmark = face_utils.shape_to_np(landmark)
            landmark = (landmark - np.array([face.left(), face.top()])) / width - 0.5
            landmarks.append(landmark.ravel().tolist())
            # 保存裁减图片
            img = frame[face.top():face.bottom(), face.left():face.right()]
            if width < 224:
                inter_para = cv2.INTER_CUBIC
            else:
                inter_para = cv2.INTER_AREA
            face_norm = cv2.resize(img, (224, 224), interpolation=inter_para)
            cv2.imwrite(os.path.join(path, str(i) + '.png'), face_norm)
        landmarks = np.array(landmarks)

        vidcap.release()

        images = []
        for imgName in os.listdir(path):
            img = Image.open(os.path.join(path, imgName))
            img = trans(img)
            images.append(img)

        images = torch.stack(images, dim=0).unsqueeze(0)
        landmarks = torch.Tensor(landmarks).unsqueeze(0)
        print(images.shape)
        print(landmarks.shape)
        images, landmarks = images.to(device), landmarks.to(device)
        
        output = model(images, landmarks)
        print(output)
        # output = F.softmax(model(img)[0], 1)
        ans.append({
            'name': videoName,
            'output': output.cpu().detach().numpy().tolist()[0]
        })
        #real: [1, 0], fake: [0, 1]
        # _, predict = torch.max(output, 1)
        # predict = int(predict)
    return ans
