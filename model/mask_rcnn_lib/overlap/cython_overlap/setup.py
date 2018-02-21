from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize


import numpy as np


#  /opt/anaconda3/bin/python3 setup.py build_ext --inplace


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


ext_modules = [
    Extension(
        "cython_box_overlap",
        ["cython_box_overlap.pyx"],
        extra_compile_args=["-Wno-cpp", "-Wno-unused-function"],
        include_dirs = [numpy_include]
    ),
]


setup(    name='mask_rcnn',
          cmdclass={"build_ext": build_ext},
          ext_modules=cythonize(ext_modules))

