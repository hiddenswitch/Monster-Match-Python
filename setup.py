from setuptools import setup, find_packages
import platform
import os


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='monstermatch',
    version='1.0.0',
    readme=readme(),
    packages=find_packages(),
    setup_requires=["pytest-runner"],
    tests_require=["pytest", "more-itertools<6.0.0"],
    entry_points={
        'console_scripts': [
            'monstermatch=monstermatch.bin.monstermatch:main']
    },
    install_requires=["numpy",
                      "scipy",
                      "pandas",
                      "Cython",
                      "scikit-learn @ git+https://github.com/TomDLT/scikit-learn.git#nmf_missing",
                      "click"],
    url='http://monstermatch.hiddenswitch.com',
    license='AGPLv4',
    author='bberman',
    author_email='ben@hiddenswitch.com',
    description='Creative code for the Monster Match game'
)
