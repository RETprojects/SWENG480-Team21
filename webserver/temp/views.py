import json

from django.core import serializers
from django.core.serializers.json import DjangoJSONEncoder
from django.forms import model_to_dict
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
from .models import PatternCategory, PatternCatalog, Pattern


# Create your views here.

def home(request):
    template = loader.get_template('home.html')
    return HttpResponse(template.render())


def browse_pattern(request):
    patternList = list(Pattern.objects.all())
    patternListJSON = serializers.get_serializer("json")().serialize(Pattern.objects.all())
    patternCategoryList = PatternCategory.objects.all()
    patternCatalogList = PatternCatalog.objects.all()
    return render(request, 'browsepattern.html', {'patternList': patternList, 'patternCategoryList': patternCategoryList, 'patternCatalogList': patternCatalogList, 'patternListJSON': patternListJSON})
def recommend_pattern(request):
    template = loader.get_template('recommendpattern.html')
    return HttpResponse(template.render())
