import sys
import setuptools
from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import numpy as np  # Note: The "pyproject.toml" file ensures that numpy has been installed when we reach here

# MSVC requires explicit C11 mode for the C99 features used in the C source files
# (mixed declarations and code, int in for-loop initializers)
_extra_compile_args = ['/std:c11'] if sys.platform == 'win32' else []

# Ensure that we overwrite any existing *.so files and print a warning message when the compilation fails instead of
# silently continuing (because of optional=True in the Extension class below)
class BuildExtForcedWithWarning(_build_ext):
    def finalize_options(self):
        super().finalize_options()
        self.force = True

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            # Write to stderr: pip captures and discards stdout from a "successful" build backend
            # subprocess, so a print() warning would be silently dropped. stderr is always shown.
            import sys
            sys.stderr.write(f"\n*** WARNING: C extension '{ext.name}' failed to compile. ***\n"
                             f"    NeoRadium will use pure-Python fallbacks (slower).\n"
                             f"    Error: {e}\n\n")
            sys.stderr.flush()

# Define the C extensions
_nrext = Extension('neoradium.nrext._ext',
                   sources=['neoradium/nrext/_ext.c',
                            'neoradium/nrext/crc.c',
                            'neoradium/nrext/LdpcLBP.c'],
                   include_dirs=[np.get_include()],
                   extra_compile_args=_extra_compile_args,
                   optional=True)

# Get version from "neoradium/__init__.py":
with open("neoradium/__init__.py") as f: lines = f.read().split('\n')
nrVersion = '0.0.0'
for line in lines:
    if line[:11]=="__version__":
        nrVersion = line.split("'")[1]
        break

installedPackages = [ 'numpy>=1.24.0',   # Make sure this matches the one in "pyproject.toml"
                      'matplotlib',
                      'jupyterlab',
                      'scipy',
                      'Pillow' ]

setuptools.setup(name="neoradium",
                 cmdclass={'build_ext': BuildExtForcedWithWarning},
                 version = nrVersion,
                 author = "Shahab Hamidi-Rad",
                 author_email = "shahab.hamidi-rad@interdigital.com",
                 description = "NeoRadium 3GPP 5G NR wireless communication python library",
                 long_description = open("README.md", "r", encoding="utf-8").read(),
                 long_description_content_type = 'text/markdown',
                 license = 'InterDigital Limited Software Evaluation License',
                 url = 'https://github.com/InterDigitalInc/NeoRadium',
                 project_urls = {
                     'Source':        'https://github.com/InterDigitalInc/NeoRadium',
                     'Documentation': 'https://interdigitalinc.github.io/NeoRadium/',
                     'Bug Tracker':   'https://github.com/InterDigitalInc/NeoRadium/issues',
                 },
                 packages = ['neoradium', 'neoradium.nrext'],
                 package_data = {'neoradium': ['data/*.json']},
                 ext_modules=[_nrext],
                 classifiers=[ 'Development Status :: 5 - Production/Stable',
                               'Intended Audience :: Science/Research',
                               'Topic :: Scientific/Engineering :: Information Analysis',
                               'Programming Language :: Python :: 3.10',
                               'Programming Language :: Python :: 3.11',
                               'Programming Language :: Python :: 3.12',
                               'Programming Language :: Python :: 3.13'],
                 python_requires='>=3.10, <4',
                 install_requires=installedPackages)
