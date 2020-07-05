from setuptools import setup
from setuptools import find_packages

setup(
    name="map2map",
    version="0.0",
    description="Neural network emulators to transform field data",
    author="Yin Li et al.",
    author_email="eelregit@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["torch", "numpy", "scipy",],
    extras_require={"visualization": ["tensorboard"],},
    scripts=["scripts/m2m.py",],
)
