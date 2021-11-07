import json

from django.http import HttpResponse
from django.shortcuts import render, redirect

from .models import User
import redis
import hashlib
import os


def info(request):
    data = {}
    if (request.method == 'POST'):
        token = request.COOKIES.get('token')
        name = token.split('_')[0]
        try:
            user = User.objects.get(name=name)
            data['name'] = user.name
            # data['img'] = user.img
        except:
            data['msg'] = '用户名不存在'
    return HttpResponse(json.dumps(data), content_type="application/json")


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username', None)
        password = request.POST.get('password', None)
        print(username, '正在登录')
        message = '所有字段都需要被填写'
        if username and password:
            try:
                user = User.objects.get(name=username)
                if user.password == password:
                    # 将状态保存到缓存
                    token = hashlib.sha1(os.urandom(24)).hexdigest()
                    token = user.name + '_' + token
                    pool = redis.ConnectionPool(host='127.0.0.1', port=6379, max_connections=10)
                    conn = redis.Redis(connection_pool=pool, decode_responses=True)
                    conn.set(user.name, token, ex=86400)
                    target = redirect('/home/')
                    target.set_cookie('token', token, expires=86400)
                    return target
                else:
                    message = '密码不正确'
            except:
                message = '用户名不存在'
        return render(request, 'users/login.html', {'message': message})
    return render(request, 'users/login.html')


def logout(request):
    target = redirect('/user/login/')
    try:
        pool = redis.ConnectionPool(host='127.0.0.1', port=6379, max_connections=10)
        conn = redis.Redis(connection_pool=pool, decode_responses=True)
        token = request.COOKIES['token']
        username = token.split('_')[0]
        token1 = conn.get(username)
        if token == token1:
            conn.hdel(username, token)
        target.delete_cookie('token')
    except Exception:
        print(Exception)
    return target


def register(request):
    if request.method == 'POST':
        username = request.POST.get('username', None)
        password = request.POST.get('password', None)
        password2 = request.POST.get('password2', None)
        telephone = request.POST.get('telephone', None)
        message = '所有字段都需要被填写'
        print(username, password, password2, telephone)
        if username and password and password2 and telephone:
            try:
                user = User.objects.get(name=username)
                message = '用户名已存在'
            except:
                if password == password2:
                    user = User(name=username, password=password, phone=telephone)
                    user.save()
                    token = hashlib.sha1(os.urandom(24)).hexdigest()
                    token = user.name + '_' + token
                    pool = redis.ConnectionPool(host='127.0.0.1', port=6379, max_connections=10)
                    conn = redis.Redis(connection_pool=pool, decode_responses=True)
                    conn.set(user.name, token, ex=86400)
                    target = redirect('/home/')
                    target.set_cookie('token', token, expires=86400)
                    return target
                else:
                    message = '两次输入密码不一致'
        print(message)
        return render(request, 'users/register.html', {'message': message})

    return render(request, 'users/register.html')
