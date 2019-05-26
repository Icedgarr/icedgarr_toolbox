import os
import re
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


SHORT = 'icedgarr_toolbox'


__author__ = 'Roger'


def get_version():
    version_file = os.path.join('icedgarr_toolbox', '_version.py')
    initfile_lines = open(version_file, 'rt').readlines()
    version_regex = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(version_regex, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (version_file,))


def get_requirements():
    with open('requirements.txt') as fp:
        return [x.strip() for x in fp.read().split('\n') if not x.startswith('#')]


setup(
    name='icedgarr_toolbox',
    version=get_version(),
    packages=find_packages(),
    install_requires=get_requirements(),
    url='https://github.com/Icedgarr/icedgarr_toolbox/',
    classifiers=(
        'Programming Language :: Python :: 3.6'
    ),
    description=SHORT,
    long_description=SHORT,
    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'mockredispy']
)
