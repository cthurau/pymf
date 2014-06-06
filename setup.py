import os
from setuptools import setup

setup(name='PyMF',
      version='0.2',
      description='Python Matrix Factorization Module',
      author='Christian Thurau',
      author_email='cthurau@googlemail.com',
      url='https://github.com/cthurau/pymf/',
      packages = ['pymf'],    
      #package_dir = {'pymf': 'pymf'},   
      scripts=['scripts/testpymf.py',],      
      license = 'BSD 3 Clause',
      install_requires=['cvxopt', 'numpy', 'scipy'],
      long_description=open('README.md').read(),
      )     
