# MIT License

# Copyright (c) 2021 Joshua Goings

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from setuptools import setup, find_packages

# Parse requirements from file
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

short_description = \
    """
    A library for exact and approximate simulation of 
    quantum algorithms for X-ray absorption.
    """

setup(
    name='pade',
    version='0.0.1',
    author='Joshua Goings',
    description=short_description,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/jjgoings/pade',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "examples": [
            "matplotlib>=3.7.2", "jupyter"
        ],
    },  
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research"
        "License :: OSI Approved ::  MIT",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)