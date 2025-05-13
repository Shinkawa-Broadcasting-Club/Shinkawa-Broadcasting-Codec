from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

package = Extension('array', ['array.pyx'], include_dirs=[numpy.get_include()], extra_compile_args=['/O2', '/openmp', '/fp:fast', '/arch:AVX2', '/favor:AMD64', '/GL', '/Gy', '/Qpar', '/Qvec-report:2', '/Oi', '/EHsc'], extra_link_args=['/OPT:REF', '/OPT:ICF', '/LTCG'])
setup(ext_modules=cythonize([package]))