from django.urls import path

from . import views

app_name = 'common'
urlpatterns = [
    path('header/', views.header, name='header'),
]
