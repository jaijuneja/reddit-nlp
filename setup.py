import ez_setup
ez_setup.use_setuptools(version='5.4.2')

from setuptools import setup
import os

PACKAGE_NAME = 'redditnlp'
VERSION = '0.1.2'


def read(filename):
    filepath = os.path.join(os.path.dirname(__file__), filename)
    try:
        # Convert GitHub markdown to restructured text (needed for upload to PyPI)
        from pypandoc import convert
        return convert(filepath, 'rst')
    except ImportError:
        return open(filepath).read()

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
    include_package_data=True,
    package_data={PACKAGE_NAME: ['words/*.txt'],
                  '': ['README.md', 'ez_setup.py', 'example.py']},
    test_suite='tests'
)