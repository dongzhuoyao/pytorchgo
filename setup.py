# Author: Tao Hu <taohu620@gmail.com>

import os
from setuptools import setup,find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

#python setup.py build_ext --inplace    TODO  in seg evaluation

setup(
    name = "pytorchgo",
    version = "0.0.2",
    author = "Tao Hu",
    author_email = "taohu620@gmail.com",
    description = ("a simple scaffold of Pytorch, easy to customize, extremely boosting your research."),
    license = "BSD",
    keywords = "pytorch deeplearning",
    url = "http://packages.python.org/an_example_pypi_project",
    packages=find_packages(),
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)