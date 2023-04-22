import os
import sys

from django.contrib import messages
from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.core import serializers
from django.shortcuts import render, redirect

import predict

from .forms import CollectPatternForm, SubmitPatternForm, ModifyCollectPatternForm
from .models import Pattern, PatternCatalog, PatternCategory

sys.path.append(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0])


def home(request):
    return render(request, "home.html")


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
    temp = request.session.get("temp")
    algorithmList = []
    categoryList = []
    algorithmPatternList = []
    singlePatternList = []
    for i in enumerate(temp):
        temp[i[0]] = " ".join(temp[i[0]].split())
        if i[1].find("--------------------------------------------") == 0:
            algorithmList.append(temp[i[0] - 1])
            categoryList.append(temp[i[0] + 1].split()[-1])
            copyPatternList = singlePatternList[:]
            algorithmPatternList.append(copyPatternList)
            singlePatternList.clear()
        if i[1].find("pattern") > 0:
            singlePatternList.append(temp[i[0]][temp[i[0]].find("pattern") + 9 :])
    copyPatternList = singlePatternList[:]
    algorithmPatternList.append(copyPatternList)
    singlePatternList.clear()
    if len(algorithmPatternList) != 0:
        algorithmPatternList.pop(0)
    # print(algorithmPatternList[0])
    # print(algorithmList)
    listList = determinePattern(algorithmPatternList)
    return render(
        request,
        "recommendpattern.html",
        {
            "data": temp,
            "algorithmList": algorithmList,
            "categoryList": categoryList,
            "algorithmPatternList": algorithmPatternList,
            "listList": listList,
            "category": determineCategory(categoryList),
        },
    )


def submit_pattern(request):
    form = SubmitPatternForm()
    if request.method == "POST" and "run_script" in request.POST:
        form = SubmitPatternForm(request.POST)
        if form.is_valid():
            content = form.cleaned_data["content"]
            content = content.replace("\t", "").replace("\r", "").replace("\n", " ")
            temp = predict.main(content)
            request.session["temp"] = temp
            return redirect("/temp/recommendpattern/")
            # return render(request, "submitpattern.html", {"form": form, "data": temp})
    return render(request, "submitpattern.html", {"form": form})


@login_required(login_url="/admin/login/")
def collect_pattern(request):
    form = CollectPatternForm()
    if request.method == "POST" and "run_script" in request.POST:
        form = CollectPatternForm(request.POST)
        if form.is_valid():
            from crawler.tutorial.tutorial.spiders.automated_scraping import run

            run(form.cleaned_data["urlContent"], form.cleaned_data["sectionContent"])
            os.chdir(os.path.split(os.path.dirname(__file__))[0])
            # print(os.path.join(os.path.split(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0])[0], "crawler", "tutorial", "tutorial", "spiders", "automated_scraping_output.txt"))
            path = os.path.join(
                os.path.split(
                    os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
                )[0],
                "crawler",
                "tutorial",
                "tutorial",
                "spiders",
                "automated_scraping_output.txt",
            )
            file = open(path, "r")
            fileData = file.read()
            if fileData == "":
                return render(
                    request,
                    "collectpattern.html",
                    {"form": form, "message": "Error crawling URL"},
                )
            print(fileData)

            temp = predict.main(fileData)
            categoryList = []
            for i in enumerate(temp):
                temp[i[0]] = " ".join(temp[i[0]].split())
                if i[1].find("--------------------------------------------") == 0:
                    categoryList.append(temp[i[0] + 1])
            request.session["category"] = determineCategory(categoryList)
            request.session["fileData"] = fileData
            request.session["link"] = form.cleaned_data["urlContent"]
            return redirect("/temp/collectpatternmanage/")
            # return render(request, "collectpattern.html", {"form": form})

    return render(request, "collectpattern.html", {"form": form})


@login_required(login_url="/admin/login/")
def collect_pattern_manage(request):
    form = ModifyCollectPatternForm()
    category = request.session.get("category")
    fileData = request.session.get("fileData")
    link = request.session.get("link")
    if request.method == "POST" and "run_script" in request.POST:
        form = ModifyCollectPatternForm(request.POST)
        if form.is_valid():
            pattern = Pattern()
            pattern.name = form.cleaned_data["nameContent"]
            pattern.category = form.cleaned_data["categoryContent"]
            pattern.miscellaneous = form.cleaned_data["descriptionContent"]
            pattern.link = link
            pattern.save()
            return redirect("/temp/home/")
    return render(
        request,
        "collectpatternmanage.html",
        {"form": form, "category": category, "fileData": fileData},
    )


@login_required(login_url="/admin/login/")
def loginView(request):
    response = redirect("/temp/home/")
    return response


def logoutView(request):
    logout(request)
    response = redirect("/temp/home/")
    return response


def determineCategory(categoryList):
    categoryList[:] = [i.split()[-1] for i in categoryList]
    return max(set(categoryList), key=categoryList.count)


def determinePattern(algorithmPatternList):
    newList = []
    newList2 = []
    newList3 = []
    for i in algorithmPatternList:
        for j in i:
            newList.append(j)
            newList2.append(j.split()[-1].replace("%", ""))
            newList3.append(j.split(j.split()[-1])[0])
    setList = set(newList3)
    setListValues = [0] * len(setList)
    setListOccurrence = [0] * len(setList)
    setListDiv = [0] * len(setList)
    for i in enumerate(newList3):
        for j in enumerate(setList):
            if i[1] == j[1]:
                setListValues[j[0]] += int(newList2[int(i[0])])
                setListOccurrence[j[0]] += 1
    for i in enumerate(setList):
        setListDiv[i[0]] = setListValues[i[0]] / setListOccurrence[i[0]]
    # print(setList)
    # print(setListValues)
    # print(setListOccurrence)
    # print(setListDiv)
    listList = [list(setList), setListValues, setListOccurrence, setListDiv]
    return listList
