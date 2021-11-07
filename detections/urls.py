from django.urls import path

from . import views

app_name = 'detections'
urlpatterns = [
    path('', views.detections, name='detections'),
    path('low_resolution/', views.low_resolution, name='low_resolution'),
    path('cross_gan/', views.cross_gan, name='cross_gan'),
    path('two_stream_detect/', views.two_stream_detect, name='two_stream_detect'),
    path('two_stream/', views.two_stream, name='two_stream'),
    path('attention/', views.attention, name='attention'),
    path('attention_detect/', views.attention_detect, name='attention_detect'),
]
