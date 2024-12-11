.. neoradium documentation master file, created by
   sphinx-quickstart on Mon May 24
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: NeoRadiumRect.png
    :width: 300
    :align: center

What is NeoRadium?
==================
**NeoRadium** is a Python library designed to simplify the simulation of physical layer communication pipeline based on the latest **3GPP 5G NR** standard. Its object-oriented architecture effectively hides the complexities involved in different stages of the communication pipeline, enabling you to swiftly develop and run end-to-end simulations on any standard computer.

Wireless communication research often is focused on one particular block or module of the end-to-end communication pipeline (e.g. equalization). However, implementing the entire 3GPP-compliant pipeline just to test one block can be time-consuming and cumbersome. This is where **NeoRadium** comes in. It provides the end-to-end communication pipeline functionality based on 3GPP standard while allowing the researchers to customize, study, and evaluate performance of their implementation. It achieves all these capabilities without high-end hardware, complex setup, or costly GPUs. As long as your computer runs Python 3.8+ with a basic setup, you're good to go!

**NeoRadium** includes a comprehensive :doc:`source/Playground/Playground`, where you can experiment with numerous examples. These examples take the form of `Jupyter Notebooks <https://jupyter.org>`_ and explain API details and their usage in practical contexts.


Key features
============
**NeoRadium** offers a versatile suite of functionalities designed to streamline 5G NR physical layer research and development. Here is a summary of what is available in current version. More features are continually added to expand its capabilities.

    * Channel Coding: Efficient transport block encoding and decoding using Polar and LDPC coding algorithms based on TS 38.212 specifications.
    * Carriers and Bandwidth Parts: Precise timing calculations for Cyclic Prefix, OFDM symbols, slots, subframes, and frames.
    * Reference signal generation including DMRS, PT-RS, and CSI-RS as per TS 38.211, TS 38.212, and TS 38.214.
    * Resource Grid functionality including resource mapping, OFDM modulation/demodulation, and precoding, aligned with TS 38.101, TS 38.104, and TS 38.211.
    * Channel Estimation, Noise Estimation, Equalization
    * Resource Grid Visualization: Gain valuable insights through visualized resource grid contents.
    * PDSCH Communication Pipeline: Simulate the complete PDSCH end-to-end communication pipeline, including modulation/demodulation, mapping/de-mapping, interleaving/de-interleaving, scrambling/descrambling, and transport block size calculations, as specified in TS 38.211 and TS 38.214.
    * Antenna array implementation and simulation based on TR 38.901
    * Antenna Field Analysis: Calculate antenna field power and directivity and create 2D/3D visualization.
    * Channel modeling: Apply CDL or TDL channel models to time-domain or frequency-domain signals.

.. toctree::
   :hidden:

   self

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Getting Started

   source/installation
   source/Playground/Playground

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: API
   
   source/API/Carrier
   source/API/Grid
   source/API/Waveform
   source/API/Modulation
   source/API/RefSig
   source/API/PhyChannels
   source/API/ChanCode
   source/API/Antenna
   source/API/Channels
   source/API/Random

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
