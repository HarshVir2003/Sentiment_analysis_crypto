from django.core.exceptions import PermissionDenied
from django.shortcuts import render
from model_handling.processing import *


def index(request):
    return render(request, 'index.html')


def nextpage(request, id=None):
    data_pred = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    models = {1: model1, 2: model2}
    if id not in models:
        raise PermissionDenied()
    if request.method == 'POST':
        text = request.POST.get('text')
        data = models.get(id).preprocessing(text)
        result = models.get(id).give_pred(data)
        return render(request, 'models.html', {'id': id, 'result': data_pred.get(result)})
    else:
        return render(request, 'models.html', {'id': id, 'result': ''})
