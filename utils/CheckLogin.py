import redis
from django.shortcuts import redirect, render
from django.utils.deprecation import MiddlewareMixin


class CheckLogin(MiddlewareMixin):
    def process_request(self, request):
        if request.method == 'POST' and not request.path.startswith('/admin/') and not request.path.startswith('/user/'):
            token = request.COOKIES.get('token')
            if not token:
                return redirect('/user/login/')
            else:
                index = token.find('_')
                name = token[0:index]
                conn = redis.StrictRedis(host='127.0.0.1', port=6379, decode_responses=True)
                token1 = conn.get(name)
                print(token1)
                print(token)
                print(token != token1)
                if token1 == '' or token != token1:
                    return redirect('/user/login/')
