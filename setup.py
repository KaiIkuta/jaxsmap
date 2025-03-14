#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
from setuptools import setup, find_packages

sys.path.insert(0, "jaxsmap")
from version import __version__


setup(
    name='jaxsmap',
    version=__version__,
    author='Kai Ikuta',
    packages=find_packages(),
    author_email='kaiikuta.astron@gmail.com',
    packages=[
        'jaxsmap',
        ],
    include_package_data=True,
    url='https://github.com/KaiIkuta/jaxsmap',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    description='Sparse mapping for stellar surface with JAX',
    package_data={'': ['README.md', 'LICENSE']},
    install_requires=install_requires,
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        ],
    )
