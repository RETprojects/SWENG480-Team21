# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.
from django.db import models


class Pattern(models.Model):
    category = models.ForeignKey('PatternCategory', models.DO_NOTHING)
    name = models.CharField(max_length=150)
    intent = models.CharField(max_length=150)
    problem = models.CharField(max_length=150)
    discussion = models.CharField(max_length=150)
    structure = models.TextField()
    miscellaneous = models.CharField(max_length=1500)
    link = models.CharField(max_length=150)

    def __str__(self):
        return self.name

    class Meta:
        managed = False
        db_table = 'pattern'


class PatternCatalog(models.Model):
    name = models.CharField(max_length=150)
    description = models.CharField(max_length=300, blank=True, null=True)
    url = models.CharField(max_length=2000)

    def __str__(self):
        return self.name

    def get_attname(self):
        return self.name

    class Meta:
        managed = False
        db_table = 'pattern_catalog'


class PatternCategory(models.Model):
    catalog = models.ForeignKey(PatternCatalog, models.DO_NOTHING)
    name = models.CharField(max_length=150)
    description = models.CharField(max_length=300)

    def __str__(self):
        return self.name

    def get_attname(self):
        return self.name

    class Meta:
        managed = False
        db_table = 'pattern_category'


class PatternMl(models.Model):
    category = models.ForeignKey(PatternCategory, models.DO_NOTHING)
    name = models.CharField(max_length=150)
    intent = models.CharField(max_length=1000)
    problem = models.CharField(max_length=1000)
    discussion = models.CharField(max_length=1000)
    structure = models.CharField(max_length=1000)

    def __str__(self):
        return self.name

    class Meta:
        managed = False
        db_table = 'pattern_ml'


class Problem(models.Model):
    category = models.ForeignKey(PatternCategory, models.DO_NOTHING)
    description = models.CharField(max_length=150)

    def __str__(self):
        return self.category.name

    class Meta:
        managed = False
        db_table = 'problem'


class ProblemPatternMatch(models.Model):
    problem = models.ForeignKey(Problem, models.DO_NOTHING)
    pattern = models.ForeignKey(Pattern, models.DO_NOTHING)
    similarity_score = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.similarity_score

    class Meta:
        managed = False
        db_table = 'problem_pattern_match'
