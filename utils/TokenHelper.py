from itsdangerous import SignatureExpired,  BadData, BadSignature
from itsdangerous.url_safe import URLSafeTimedSerializer


class TokenException(Exception):
    def __init__(self,obj):
        super().__init__()
        self.status = {}
        if isinstance(obj,SignatureExpired):
            self.status['code'] = -1
            self.status['msg'] = "签名过期"
        elif isinstance(obj,BadSignature):
            self.status['code'] = -2
            self.status['msg'] = "签名失效"
        elif isinstance(obj,BadData):
            self.status['code'] = -3
            self.status['msg'] = "数据异常"

    @property
    def code(self):
        return self.status['code']
    @property
    def message(self):
        return self.status['msg']

    def __str__(self):
        return self.status['msg']

class TokenManager:
    def __init__(self,secret_key,salt='hello world'):
        """secret_key 加密密钥"""
        self.secret_key = secret_key
        self.salt = salt

    def generate_token(self,playload):
        """
        :param playload: 负载，也就是你要序列化的数据，不要用关键数据（如密码等）做playload
        :return: token字符串
        """
        serializer = URLSafeTimedSerializer(self.secret_key, self.salt)
        return serializer.dumps(playload)

    def confirm_token(self,token,expired=86400):
        """
        验证token
        :param token: generate_validate_token产生的字符串
        :param expired: 过期时间，以秒为单位
        :return: 成功返回负载数据，失败返回错误码
        """
        serializer = URLSafeTimedSerializer(self.secret_key, self.salt)
        try:
            data = serializer.loads(token,max_age=expired)
        except BadData as e:
           return TokenException(e)
        # 签名验证通过，返回原始数据
        return data