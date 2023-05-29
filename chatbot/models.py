from django.db import models

# Create your models here.
class Question(models.Model):
  question_text = models.CharField(max_length=200)
  reply = models.CharField(max_length=300)
  def __str__(self):
        return self.question_text


