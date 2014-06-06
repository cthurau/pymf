import os
#from distutils.core import setup
from setuptools import setup

setup(name='PyMF',
      version='0.1.9',
      description='Python Matrix Factorization Module',
      author='Christian Thurau',
      author_email='cthurau@googlemail.com',
      url='http://code.google.com/p/pymf/',
      packages = ['pymf'],    
      package_dir = {'pymf': './lib/pymf'},   
      scripts=['scripts/testpymf.py',],      
      license = 'OSI Approved :: GNU General Public License (GPL)',
      install_requires=['cvxopt'],
      long_description=open('README.txt').read(),
      )     
