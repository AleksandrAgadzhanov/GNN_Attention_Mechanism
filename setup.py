import sys

from os import path
from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, only Python >= 3.6 is supported")
here = path.abspath(path.dirname(__file__))

setup(
    name='GNN_Attention_Mechanism',
    version='0.0.1',
    description='Graph Neural Networks for Finding Adversarial Example',
    author='Aleksandr Agadzhanov, University of Oxford',
    packages=find_packages(),
    install_requires=['matplotlib', 'numpy', 'torch', 'gurobipy', 'pandas', 'mlogger']
)
