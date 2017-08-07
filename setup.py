from distutils.core import setup, Extension
from distutils.command.clean import clean
from distutils.command.install import install
import os
import numpy as np


class RsInstall(install):
    def run(self):
        install.run(self)
        c = clean(self.distribution)
        c.all = True
        c.finalize_options()
        c.run()

NUMPY_PATH = np.get_include()
DIR_PATH = os.path.dirname(os.path.abspath(__file__))
LOCAL_INCLUDE = os.path.join(DIR_PATH, 'include')

setup(
    name='_pyrs',
    version='1.0',
    py_modules=['pyrs'],
    ext_modules=[
        Extension(
            '_pyrs', [os.path.join(DIR_PATH, 'cpyrs', 'pyrs.cpp')],
            libraries=['realsense'],
            include_dirs=[NUMPY_PATH, '/usr/local/include', LOCAL_INCLUDE],
            library_dirs=['/usr/local/lib'],
            extra_compile_args=['-std=gnu++11'],
            language='c++'
        ),
    ],
    cmdclass={'install': RsInstall},
)