#
# Copyright (C) 2020 by TODO - All rights reserved.
#

from setuptools import Extension
from Cython.Build import cythonize

cpp_ext = Extension(
    name="piquasso.gaussian.state",
    sources=["piquasso/gaussian/state.pyx"],
)

EXTENSIONS = [cpp_ext]


def build(setup_kwargs):
    setup_kwargs.update(
        {
            "ext_modules": cythonize(EXTENSIONS, language_level=3),
        }
    )
