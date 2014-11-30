import os
from setuptools import setup

PACKAGE_NAME = 'redditnlp'
VERSION = '0.11'


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# Fix PyPI install
description = 'A tool to perform natural language processing of reddit content.'
try:
    long_description = read('README.md')
except IOError:
    long_description = description

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author='Jai Juneja',
    author_email='jai.juneja@gmail.com',
    description=description,
    license='BSD',
    keywords='reddit, natural language processing, machine learning',
    url='https://github.com/jaijuneja/reddit-nlp',
    packages=[PACKAGE_NAME,],
    long_description=long_description,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python'
    ],
    install_requires=[
        'praw>=2.1.19',
        'nltk>=3.0.0',
        'numpy>=1.8.0',
        'scikit-learn>=0.15.2',
    ],
    package_data={PACKAGE_NAME: ['words/*.txt'],
                  '': ['README.md']},
    test_suite='tests'
)