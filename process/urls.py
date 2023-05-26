from django.urls import path

from . import views

app_name='process'
urlpatterns = [
    path('', views.process, name='process'),
    path('noise/', views.noise, name='noise'),
    path('noise_process/', views.noise_process, name='noise_process'),
    path('texture/', views.texture, name='texture'),
    path('scale/', views.scale, name='scale')
]