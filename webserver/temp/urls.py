from django.urls import path
from . import views

urlpatterns = [
    path("home/", views.home, name="home"),
    path("browsepattern/", views.browse_pattern, name="browse-pattern"),
    path("recommendpattern/", views.recommend_pattern, name="recommend-pattern"),
    path("submitpattern/", views.submit_pattern, name="submit-pattern"),
    path("collectpattern/", views.collect_pattern, name="collect-pattern"),
path("collectpatternmanage/", views.collect_pattern_manage, name="collect-pattern-manage"),
    path("login/", views.loginView, name="login"),
    path("logout/", views.logoutView, name="logout"),
]
