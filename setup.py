#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch1.0keras2.2
from distutils.core import setup
import setuptools

setup(
    name='speech_hackathon',
    version='1.0',
    install_requires=[
        'python-Levenshtein',
        'librosa',
        'visdom'
    ]
)
