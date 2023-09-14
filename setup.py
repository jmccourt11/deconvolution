#https://pythonhosted.org/an_example_pypi_project/setuptools.html

import os
from setuptools import setup
from setuptools import find_packages


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "deconvolution_JMM",
    version = "0.0.1",
    author = "Joseph McCourt",
    author_email = "jmccourt@anl.gov",
    description = ("A collection of functions to perform deconvolution "
                                   "of ptychography/SAXS diffraction data."),
    license = read('LICENSE'),
    keywords = "deconvolution ptychoSAXS",
    url = "https://github.com/jmccourt11/deconvolution",
    packages=['deconvolution_JMM', 'tests'],
    # packages=find_packages('src'),
    # package_dir={'': 'src'},
    long_description=read('README.md'),
    # classifiers=[
    #     "Development Status :: 3 - Alpha",
    #     "Topic :: Utilities",
    #     "License :: OSI Approved :: BSD License",
    # ],
)