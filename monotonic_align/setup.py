from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name="monotonic_align",
    ext_modules=cythonize("/content/MB-iSTFT-VITS2/monotonic_align/core.pyx"), #/app/monotonic_align/core.pyx if docker
    include_dirs=[numpy.get_include()],
)
