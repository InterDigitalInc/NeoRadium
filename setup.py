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
                      'scipy',
                      'Pillow' ]

setuptools.setup(name="neoradium",
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
                 classifiers=[ 'Development Status :: 5 - Production/Stable',
                               'Intended Audience :: Science/Research',
                               'Topic :: Scientific/Engineering :: Information Analysis',
                               'License :: Other/Proprietary License',
                               'Programming Language :: Python :: 3.10',
                               'Programming Language :: Python :: 3.11',
                               'Programming Language :: Python :: 3.12'],
                 python_requires='>=3.10, <4',
                 install_requires=installedPackages)
