# Generated by Django 2.2.5 on 2022-05-09 05:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_auto_20220507_1148'),
    ]

    operations = [
        migrations.AddField(
            model_name='image',
            name='ph',
            field=models.FloatField(default=1),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='image',
            name='temparature',
            field=models.FloatField(default=0),
            preserve_default=False,
        ),
    ]
