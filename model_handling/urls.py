from django.urls import path
from model_handling.views import *

urlpatterns = [
    path('', index, name='Home'),
    path('model/<int:id>', nextpage, name='model'),
]