from django.db import models


class MyPattern(models.Model):
    name = models.TextField()
    correct_category = models.IntegerField()
    overview = models.TextField()


class MyProblem(models.Model):
    design_problem = models.TextField()
    correct_pattern = models.TextField()
