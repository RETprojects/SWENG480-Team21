from django.urls import path
from . import views

urlpatterns = [
    path("home/", views.home, name="home"),
    path("browsepattern/", views.browse_pattern, name="browse-pattern"),
    path("recommendpattern/", views.recommend_pattern, name="recommend-pattern"),
    path("submitpattern/", views.submit_pattern, name="submit-pattern"),
    path("collectpattern/", views.collect_pattern, name="collect-pattern"),
]
