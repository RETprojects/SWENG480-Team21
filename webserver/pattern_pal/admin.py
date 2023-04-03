from django.contrib import admin


from .models import MyPattern, MyProblem

admin.site.register(MyPattern)
admin.site.register(MyProblem)
