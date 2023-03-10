import os
import sys

from django.forms import Form

sys.path.append('C:\\Users\\Benjamin\\PycharmProjects\\SWENG480-Team21')
#import crawler.tutorial.tutorial.spiders.automated_scraping

from django.core import serializers
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import PatternCategory, PatternCatalog, Pattern
from .forms import SubmitPatternForm


# Create your views here.

def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


def browse_pattern(request):
    patternList = Pattern.objects.all()
    patternListJSON = serializers.get_serializer("json")().serialize(patternList)
    patternCategoryList = PatternCategory.objects.all()
    patternCategoryListJSON = serializers.get_serializer("json")().serialize(patternCategoryList)
    patternCatalogList = PatternCatalog.objects.all()
    patternCatalogListJSON = serializers.get_serializer("json")().serialize(patternCatalogList)
    return render(request, 'browsepattern.html',
                  {'patternList': patternList, 'patternCategoryList': patternCategoryList,
                   'patternCatalogList': patternCatalogList, 'patternListJSON': patternListJSON,
                   'patternCategoryListJSON': patternCategoryListJSON,
                   'patternCatalogListJSON': patternCatalogListJSON})
  

def recommend_pattern(request):
    template = loader.get_template('recommendpattern.html')
    return HttpResponse(template.render())


def submit_pattern(request):
    context = {
        "form": SubmitPatternForm
    }
    return render(request, 'submitpattern.html', context=context)


def collect_pattern(request):
    if request.method == 'POST' and 'run_script' in request.POST:
        #print(sys.path)
        from crawler.tutorial.tutorial.spiders.automated_scraping import run
        run(1, 2)
        #print(Form(request.POST).cleaned_data['textarea1'])
    return render(request, 'collectpattern.html')
