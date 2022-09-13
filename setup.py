import shutil
from distutils.core import setup
from distutils.extension import Extension

import numpy
from Cython.Distutils import build_ext

print('Build extension modules...')
print('==============================================')

ext_modules = [Extension('core',
                         ['src/core/core.pyx',
                          'src/core/RLMF.cpp'],
                         language='c++',
                         include_dirs=[numpy.get_include()],
                         extra_compile_args=["-O2"]
                         )]

setup(
    name='Extended Cython module',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
# need to be modified in accordance with the running environment
shutil.move('core.cp38-win_amd64.pyd', 'src/core.pyd')
print('==============================================')
print('Build done.\n')
