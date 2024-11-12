from django.shortcuts import render
from model_handling.processing import *


def index(request):
    return render(request, 'index.html')


def nextpage(request, id=None):
    data_pred = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    if id == 1:
        if request.method == 'POST':
            text = request.POST.get('text')
            data = model1.preprocessing(text)
            result = model1.give_pred(data)
            return render(request, 'models.html', {'id': 1, 'result': data_pred.get(result)})
        else:
            return render(request, 'models.html', {'id': 1, 'result': ''})
    else:
        return render(request, 'models.html', {'id': 2, 'result': ''})
