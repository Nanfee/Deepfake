from django.urls import path

from . import views

app_name='users'
urlpatterns = [
    path('info/', views.info, name='indo'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('register/', views.register, name='register'),
]