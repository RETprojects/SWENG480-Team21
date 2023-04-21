from django import forms
from .models import PatternCategory


class SubmitPatternForm(forms.Form):
    content = forms.CharField(label="",
                              widget=forms.Textarea(attrs={'style': 'resize: none; width: 99.65%; height: 375px'}))


class CollectPatternForm(forms.Form):
    urlContent = forms.CharField(label="", widget=forms.TextInput(attrs={'style': 'width: 50%;'}))
    sectionContent = forms.CharField(label="", widget=forms.TextInput(attrs={'style': 'width: 50%;'}))


class ModifyCollectPatternForm(forms.Form):
    categoryContent = forms.ModelChoiceField(queryset=PatternCategory.objects.all())
    nameContent = forms.CharField(label="", widget=forms.TextInput(attrs={'style': 'width: 20%;'}))
    descriptionContent = forms.CharField(label="", widget=forms.Textarea(
        attrs={'style': 'resize: none; width: 50%; height: 200px'}))
