from django.urls import include,path

import dotenv
from . import views
from django.contrib import admin

urlpatterns = [path("", views.index, name="index"),
               path("", dotenv.setVar, name="setVar")
               ]