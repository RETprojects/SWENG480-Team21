from django.urls import path
from . import views

urlpatterns = [
    path("jonathan/", views.jonathan),
]
