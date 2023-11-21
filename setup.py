"""
Setup for carla_navigation
"""
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(packages=['carla_navigation'], package_dir={'': 'src'})

setup(**d)
