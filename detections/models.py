import uuid

from django.db import models
from utils import UUIDTools


class Paper(models.Model):
    paper_id = models.UUIDField(primary_key=True, auto_created=True, default=uuid.uuid4, editable=False)
    paper_title = models.CharField(max_length=200)
    paper_title_en = models.CharField(max_length=200)
    paper_intro = models.CharField(max_length=500)
    paper_authors = models.ManyToManyField('Author')
    paper_summary = models.CharField(max_length=500)

    def __str__(self):
        return '%s' % (self.paper_title)

    class Meta:
        db_table = 'papers'


class Author(models.Model):
    author_id = models.UUIDField(primary_key=True, auto_created=True, default=uuid.uuid4, editable=False)
    author_name = models.CharField(max_length=100)
    author_department = models.CharField(max_length=200)

    def __str__(self):
        return '%s - %s' % (self.author_name, self.author_department)

    class Meta:
        db_table = 'authors'
