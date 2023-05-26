import os
import shutil
import math
import json
from PIL import Image
import numpy as np
from Deepfake.settings import BASE_DIR
from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
def process(request):
    return render(request, 'process/process.html')

def noise(request):
    return render(request, 'process/noise.html')

def noise_process(request):
    print('----------')
    # 提取token，检测用户
    token = request.COOKIES.get('token')
    print(token)
    username = token.split('_')[0]
    # 确定用户文件夹
    userPath = os.path.join(BASE_DIR, 'temp', username)
    prePath = os.path.join(BASE_DIR, 'utils', 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(userPath):
        os.mkdir(userPath)
    # 写入视频
    count = int(request.POST.get('count'))
    for i in range(count):
        img = request.FILES['files'+str(i)]
        with open(os.path.join(userPath, img.name), 'wb+') as f:
            for line in img.chunks():
                f.write(line)
    snr = 1
    res = []
    for imageName in os.listdir(userPath):
        image = Image.open(os.path.join(userPath, imageName))
        image = np.array(image)
        h, w, c = image.shape
        noise = np.random.randn(h, w, c)
        noise = math.sqrt((np.sum(image**2)/math.pow(10., (snr/10)))/(np.sum(noise*noise))) * noise
        image += noise.astype(np.uint8)
        # print(10*math.log10(np.sum(image**2)/np.sum(noise**2)))
        image = Image.fromarray(np.uint8(image))
        path = os.path.join(BASE_DIR, 'static', 'images', imageName)
        image.save(path)
        image.close()
        res.append('images/'+imageName)
    # 获取结果并返回
    shutil.rmtree(userPath)
    print(json.dumps(res))
    return HttpResponse(json.dumps(res), content_type="application/json")

def texture(request):
    return render(request, 'process/texture.html')

def scale(request):
    return render(request, 'process/scale.html')