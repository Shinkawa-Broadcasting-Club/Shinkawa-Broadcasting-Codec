from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('test', ['test.pyx'], include_dirs=[numpy.get_include()], extra_compile_args=['/O2', '/Ob3', '/Oi', '/Ot', '/GL', '/fp:fast', '/arch:SSE4.2'], extra_link_args=['/LTCG'])
setup(ext_modules=cythonize([package]))
