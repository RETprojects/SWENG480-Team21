from django.contrib import admin
from .models import Pattern, PatternCatalog, PatternCategory, Problem

# Register your models here.

admin.site.register(Pattern)
admin.site.register(PatternCatalog)
admin.site.register(PatternCategory)
admin.site.register(Problem)


