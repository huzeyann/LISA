from setuptools import setup, find_packages

# read the requirements file
import os
requirement_file = 'requirements.txt'
with open(requirement_file) as f:
    required = f.read().splitlines()
    
setup(
    name="lisa",
    version="0.0.1",
    # packages=['lisa'],
    packages=find_packages(),
    install_requires=required,
)