import json
import shutil

from django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import Paper, Author
from Deepfake.settings import BASE_DIR
import os

from utils.TwoStreamDetect import detect as two_detect


# Create your views here.

def low_resolution(request):
    return render(request, 'detections/low-resolution.html')


def cross_gan(request):
    return render(request, 'detections/cross-gan.html')


def detections(request):
    return render(request, 'detections/detections.html')


def two_stream(request):
    paper = Paper.objects.get(paper_title='双分支神经网络检测')
    return render(request, 'detections/two_stream.html', {'paper': paper})


def two_stream_detect(request):
    print('----------')
    # 提取token，检测用户
    token = request.COOKIES.get('token')
    print(token)
    username = token.split('_')[0]
    # 确定用户文件夹
    userPath = os.path.join(BASE_DIR, 'temp', username)
    if not os.path.exists(userPath):
        os.mkdir(userPath)
    # 写入图像
    count = int(request.POST.get('count'))
    for i in range(count):
        img = request.FILES['files'+str(i)]
        with open(os.path.join(userPath, img.name), 'wb+') as f:
            for line in img.chunks():
                f.write(line)
    # 获取结果并返回
    modelPath = os.path.join(BASE_DIR, 'utils', 'lcnn.pt')
    res = two_detect(modelPath, userPath)
    shutil.rmtree(userPath)
    print(json.dumps(res))
    return HttpResponse(json.dumps(res), content_type="application/json")


def attention(request):
    paper = Paper.objects.get(paper_title='注意力机制')
    return render(request, 'detections/attention.html', {'paper': paper})


def attention_detect(request):
    pass
