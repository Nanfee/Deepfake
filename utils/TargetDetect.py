import os
import torch
import cv2
import dlib
import math
from imutils import face_utils
from moviepy.editor import *
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from .model.avdfnet import AVDFNet
from towhee import pipeline 

trans = transforms.Compose([
    transforms.Resize([256, 256]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

def detect(faceFolder, pre_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AVDFNet()
    model.load_state_dict(torch.load('utils/weights/avdfnet-data-92-acc0.9926854754440961.pt'), strict=False)
    
    model = model.to(device)
    model.eval()

    ans = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pre_path)

    videos = os.listdir(faceFolder)
    print('dddddddd--------')
    for videoName in tqdm(videos):
        # 抽帧并裁减人脸
        path = os.path.join(faceFolder, 'images')
        apath = os.path.join(faceFolder, 'audio')
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(apath):
            os.makedirs(apath)
        vidcap = cv2.VideoCapture(os.path.join(faceFolder, videoName))
        fps = math.ceil(vidcap.get(cv2.CAP_PROP_FPS))
        gap = fps // 15
        frames = []
        landmarks = []
        idx = 0
            
        # 抽帧
        while True:
            if len(frames) == 15:
                break
            success, frame = vidcap.read()
            idx += 1
            if success:
                if idx % gap == 0:
                    frames.append(frame)
            else:
                print('fail')
                break
            
        # 裁剪人脸
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = detector(gray, 0)
            if len(face) != 0:
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
            vidcap.release()

        # 音频处理
        audio = np.loadtxt(os.path.join('/home/spendar/Deepfake/audio', videoName, 'audio.csv'), delimiter=',', dtype=float)
        audio = audio.astype(np.float32)
        audio = torch.from_numpy(audio)

        images = []
        for imgName in os.listdir(path):
            img = Image.open(os.path.join(path, imgName))
            img = trans(img)
            images.append(img)

        images = torch.stack(images, dim=0).unsqueeze(0)
        audio = audio.unsqueeze(0)
        print(images.shape)
        print(audio.shape)
        images, audio = images.to(device), audio.to(device)
        
        _, _, output = model(images, audio)
        print(output)
        output = F.softmax(output, 1)
        ans.append({
            'name': videoName,
            'output': output.cpu().detach().numpy().tolist()[0]
        })
        #real: [1, 0], fake: [0, 1]
        # _, predict = torch.max(output, 1)
        # predict = int(predict)
    return ans
