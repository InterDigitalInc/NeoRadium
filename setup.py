import sys
import setuptools

# Get version from "neoradium/__init__.py":
with open("neoradium/__init__.py") as f: lines = f.read().split('\n')
nrVersion = '0.0.0'
for line in lines:
    if line[:11]=="__version__":
        nrVersion = line.split("'")[1]
        break

installedPackages = [ 'numpy>=1.24.0',
                      'matplotlib',
                      'jupyterlab',
                      'scipy' ]

setuptools.setup(name="neoradium",
                 version = nrVersion,
                 author = "Shahab Hamidi-Rad",
                 author_email = "shahab.hamidi-rad@interdigital.com",
                 description = "NeoRadium 3GPP 5G NR wireless communication python library",
                 long_description = open("README.md", "r", encoding="utf-8").read(),
                 license = open("LICENSE", "r", encoding="utf-8").read(),
                 url = "http://www.interdigital.com",
                 packages = ['neoradium'],
                 classifiers=[ 'Development Status :: 5 - Production/Stable',
                               'Intended Audience :: Researchers',
                               'Topic :: Software Development',
                               'Programming Language :: Python :: 3.9',
                               'Programming Language :: Python :: 3.10',
                               'Programming Language :: Python :: 3.11',
                               'Programming Language :: Python :: 3.12'],
                 python_requires='>=3.9, <4',
                 install_requires=installedPackages)
