from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('example', ['example.pyx'], include_dirs=[numpy.get_include()])
setup(ext_modules=cythonize([package]))