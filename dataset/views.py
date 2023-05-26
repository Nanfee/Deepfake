from django.shortcuts import render

def dataset(request):
    return render(request, 'dataset/datasets.html')
