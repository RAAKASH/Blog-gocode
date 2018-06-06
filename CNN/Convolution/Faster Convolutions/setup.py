from distutils.core import setup
from Cython.Build import cythonize
import numpy
from distutils.extension import Extension

extensions = [Extension("*", ["*.pyx"])]

setup(
    ext_modules = cythonize(extensions),
    include_dirs=[numpy.get_include()]
)