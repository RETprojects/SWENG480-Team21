from django import forms
from django.forms import ModelForm
from .models import PatternCatalog


class PatternCatalogForm(ModelForm):
    class Meta:
        model = PatternCatalog
        fields = ('name',)
