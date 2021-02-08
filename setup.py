from distutils.core import setup
from setuptools import find_packages

setup(name='DeepLearning',
      version='1.0',
      description='Deep learning environment',
      author='Jaden Fiotto-Kaufman',
      author_email='jadenfk@outlook.com',
      packages=find_packages(include=['deeplearning*']))