# Generated by Django 3.2.4 on 2022-04-20 15:23

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diabetes', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Ans',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('ans', models.CharField(max_length=10)),
            ],
        ),
    ]