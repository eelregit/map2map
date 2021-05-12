from setuptools import setup, find_packages

setup(
    name='map2map',
    version='0.0',
    description='Neural network emulators to transform field data',
    author='Yin Li et al.',
    author_email='eelregit@gmail.com',
    packages=find_packages(),
    scripts=[
        'm2m.py',
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.2',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorboard',
    ],
)
