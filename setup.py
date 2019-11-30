from setuptools import setup
from setuptools import find_packages

setup(
    name='map2map',
    version='0.0',
    description='Neural network emulators to transform field data',
    author='Yin Li',
    author_email='eelregit@gmail.com',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scipy',
        'tensorboard',
    ],
    scripts=[
        'scripts/m2m.py',
    ]
)
