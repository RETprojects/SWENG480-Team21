from django.contrib import admin
from .models import Pattern, PatternCatalog, PatternCategory

# Register your models here.

admin.site.register(Pattern)
admin.site.register(PatternCatalog)
admin.site.register(PatternCategory)
