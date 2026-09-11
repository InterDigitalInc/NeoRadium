Installation
============

**NeoRadium** requires Python 3.10 or later.

----

Install from PyPI
-----------------

Create a virtual environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python3 -m venv ve
    source ve/bin/activate
    pip install --upgrade pip setuptools

Install **NeoRadium**
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install neoradium

Get the Playground examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Playground notebooks are not included in the ``pip`` package. To access them, clone the GitHub repository:

.. code-block:: bash

    git clone https://github.com/InterDigitalInc/NeoRadium.git
    cd NeoRadium
    jupyter lab Playground/

----

.. _troubleshooting:

Troubleshooting
---------------

If you see this warning after installation:

.. code-block:: text

    RuntimeWarning: C extension not available; using slower pure-Python fallback.

it means **NeoRadium**'s C extension (which accelerates CRC and LDPC decoding) could not be
loaded. This happens when no prebuilt wheel matched your platform and ``pip`` compiled from
source without a working C compiler.

**NeoRadium** still works correctly without it — only CRC and LDPC operations are slower.

To fix this, install a C compiler for your platform:

* **macOS** — Install Xcode Command Line Tools:

  .. code-block:: bash

      xcode-select --install

* **Linux (Debian / Ubuntu)**:

  .. code-block:: bash

      sudo apt-get install gcc python3-dev

* **Linux (RHEL / Rocky / AlmaLinux / Fedora)**:

  .. code-block:: bash

      sudo dnf install gcc python3-devel

* **Windows** — Install `Visual Studio Build Tools <https://visualstudio.microsoft.com/visual-cpp-build-tools/>`_
  with the **"Desktop development with C++"** workload.

After installing the compiler, reinstall **NeoRadium** (e.g. using ``--force-reinstall``) to trigger recompilation.
