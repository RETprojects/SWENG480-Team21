import json

from django.forms import model_to_dict
from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from .models import PatternCategory, PatternCatalog, Pattern


# Create your views here.

def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


def browse_pattern(request):
    patternList = Pattern.objects.all()
    patternCategoryList = PatternCategory.objects.all()
    patternCatalogList = PatternCatalog.objects.all()
    return render(request, 'browsepattern.html', {'patternList': patternList, 'patternCategoryList': json.dumps([model_to_dict(x) for x in patternCategoryList]), 'patternCatalogList': json.dumps([model_to_dict(x) for x in patternCatalogList])})

def recommend_pattern(request):
    template = loader.get_template('recommendpattern.html')
    return HttpResponse(template.render())
