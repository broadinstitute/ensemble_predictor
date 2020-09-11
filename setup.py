import os
from setuptools import setup, find_packages

setup(
    name='ensemble_predictor',
    version='1.0',
    author="Joshua Dempster",
    description="Tools for building an ensemble of sueprvised regressors that filter features",
    long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'scipy', 'h5py', 'sklearn']
)
