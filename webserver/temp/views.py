import os
import sys

from django.core import serializers
from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader

from .forms import CollectPatternForm, SubmitPatternForm
from .models import Pattern, PatternCatalog, PatternCategory

sys.path.append(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0])


def home(request):
    template = loader.get_template("home.html")
    return HttpResponse(template.render())


def browse_pattern(request):
    patternList = Pattern.objects.all()
    patternListJSON = serializers.get_serializer("json")().serialize(patternList)
    patternCategoryList = PatternCategory.objects.all()
    patternCategoryListJSON = serializers.get_serializer("json")().serialize(
        patternCategoryList
    )
    patternCatalogList = PatternCatalog.objects.all()
    patternCatalogListJSON = serializers.get_serializer("json")().serialize(
        patternCatalogList
    )
    return render(
        request,
        "browsepattern.html",
        {
            "patternList": patternList,
            "patternCategoryList": patternCategoryList,
            "patternCatalogList": patternCatalogList,
            "patternListJSON": patternListJSON,
            "patternCategoryListJSON": patternCategoryListJSON,
            "patternCatalogListJSON": patternCatalogListJSON,
        },
    )


def recommend_pattern(request):
    template = loader.get_template("recommendpattern.html")
    return HttpResponse(template.render())


def submit_pattern(request):
    form = SubmitPatternForm()
    if request.method == "POST" and "run_script" in request.POST:
        form = SubmitPatternForm(request.POST)
        if form.is_valid():
            from ml.predict import main

            temp = main(form.cleaned_data["content"])
            return render(request, "submitpattern.html", {"form": form, "data": temp})
    return render(request, "submitpattern.html", {"form": form})


def collect_pattern(request):
    form = CollectPatternForm()
    if request.method == "POST" and "run_script" in request.POST:
        form = CollectPatternForm(request.POST)
        if form.is_valid():
            from crawler.tutorial.tutorial.spiders.automated_scraping import run
            run(form.cleaned_data["urlContent"], form.cleaned_data["sectionContent"])
            os.chdir(os.path.split(os.path.dirname(__file__))[0])
            return render(request, "collectpattern.html", {"form": form})

    return render(request, "collectpattern.html", {"form": form})
