from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path('browsepattern/', views.browse_pattern, name='browse-pattern'),
    path('recommendpattern/', views.recommend_pattern, name='recommend-pattern'),
]
