Playground
==========

This section contains example notebooks that demonstrate how **NeoRadium** APIs are used in practice and highlight the library’s main capabilities. The source code for these examples is available in the ``Playground`` directory, and each notebook can be opened and explored interactively in Jupyter.


Antenna
-------
    .. toctree::
        :maxdepth: 1

        Notebooks/Antenna/AntennaElement.ipynb
        Notebooks/Antenna/AntennaPanel.ipynb
        Notebooks/Antenna/AntennaArray.ipynb
        Notebooks/Antenna/BeamSweeping.ipynb

Stochastic Channel Models
-------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/Channels/ChannelMatrix.ipynb
        Notebooks/Channels/cdlTiming.ipynb
        Notebooks/Channels/TdlChannel.ipynb
        Notebooks/Channels/CdlChannelDataset.ipynb
        Notebooks/Channels/CustomCdl.ipynb
        Notebooks/Channels/CdlBearingAngles.ipynb

Trajectory-based Channel Model
------------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/RayTracing/DeepMimo.ipynb
        Notebooks/RayTracing/TrajChannel.ipynb
        Notebooks/RayTracing/TrajEndToEnd.ipynb
        Notebooks/RayTracing/TrajChannelAnim.ipynb
        Notebooks/RayTracing/ChannelGeneration.ipynb
        Notebooks/RayTracing/ChannelSequences.ipynb
        Notebooks/RayTracing/AnimatedBER.ipynb
        Notebooks/RayTracing/AnimatedCN.ipynb
        Notebooks/RayTracing/TrjBearingAngles.ipynb
        Notebooks/RayTracing/BeamSweeping.ipynb
        Notebooks/RayTracing/BeamSweepingTraj.ipynb
        Notebooks/RayTracing/LinkAdaptation.ipynb

Channel State Information Reference Signals (CSI-RS)
----------------------------------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/CSI-RS/CSI-RS.ipynb
        Notebooks/CSI-RS/CSI-RS-Time.ipynb
        Notebooks/CSI-RS/CSI-RS-Beams.ipynb

CSI-Feedback
----------------------------------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/CSI-Feedback/CSI-Feedback1.ipynb
        Notebooks/CSI-Feedback/CSI-Feedback2.ipynb
        Notebooks/CSI-Feedback/CSI-Feedback3.ipynb

DM-RS and PT-RS
---------------
    .. toctree::
        :maxdepth: 1

        Notebooks/DMRS/DMRS.ipynb
        Notebooks/DMRS/CDMsWithNoData.ipynb
        Notebooks/DMRS/PTRS.ipynb

Physical Downlink Shared Channel (PDSCH)
----------------------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/PDSCH/PDSCH-endToEnd.ipynb
        Notebooks/PDSCH/PDSCH-BER.ipynb
        Notebooks/PDSCH/PDSCH-BLER.ipynb
        Notebooks/PDSCH/ChannelNoiseEst.ipynb
        Notebooks/PDSCH/PDSCH-Throughput.ipynb
        Notebooks/PDSCH/AdvancedPDSCH.ipynb

Channel Coding
--------------
    .. toctree::
        :maxdepth: 1

        Notebooks/ChanCode/LDPC.ipynb
        Notebooks/ChanCode/NumIter.ipynb
        Notebooks/ChanCode/Polar.ipynb

Hybrid Automatic Repeat reQuest (HARQ)
--------------------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/HARQ/Harq.ipynb
        Notebooks/HARQ/HarqEventCallback.ipynb
        Notebooks/HARQ/TransportChannelWithHarq.ipynb

A Deep Learning Case Study
--------------------------
    .. toctree::
        :maxdepth: 1

        Notebooks/MLChEst/MLChestDataGen.ipynb
        Notebooks/MLChEst/MLChestTrainTorch.ipynb
        Notebooks/MLChEst/MLChestEvaluateTorch.ipynb

Research Papers
---------------

`A Self-Refining Multi-Layer Receiver Pipeline <https://ieeexplore.ieee.org/abstract/document/11443343>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    .. toctree::
        :maxdepth: 1

        readme.md <Notebooks/Research/SelfRefining/readme>
        Notebooks/Research/SelfRefining/MLChEstDataGen.ipynb
        Notebooks/Research/SelfRefining/MLChEstTrain.ipynb
        Notebooks/Research/SelfRefining/MLChEstEvaluateNMSE.ipynb
        Notebooks/Research/SelfRefining/MLChEstEvaluateBLER.ipynb
        Notebooks/Research/SelfRefining/MLChEstEvaluateHARQ.ipynb


`LWM-Spectro: A Foundation Model for Wireless Baseband Signal Spectrograms <https://arxiv.org/abs/2601.08780>`_
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


    .. toctree::
        :maxdepth: 1
            
        View the code <https://huggingface.co/wi-lab/lwm-spectro>
        

Other Examples
--------------

    .. toctree::
        :maxdepth: 1

        Notebooks/Others/SnrCalculations.ipynb

Comparing with MATLAB
---------------------

Antenna
"""""""

    .. toctree::
        :maxdepth: 1

        Notebooks/CompareWithMatlab/Antenna/AntennaElement.ipynb
        Notebooks/CompareWithMatlab/Antenna/AntennaPanel.ipynb
        Notebooks/CompareWithMatlab/Antenna/AntennaArray.ipynb

CDL Channel Model
"""""""""""""""""

    .. toctree::
        :maxdepth: 1

        Notebooks/CompareWithMatlab/CDL/CDL-Matlab.ipynb
        Notebooks/CompareWithMatlab/CDL-SISO/SisoCdl.ipynb

CSI-RS
""""""

    .. toctree::
        :maxdepth: 1

        Notebooks/CompareWithMatlab/CSI-RS/CSI-RS-Matlab.ipynb

PDSCH
"""""

    .. toctree::
        :maxdepth: 1

        Notebooks/CompareWithMatlab/PDSCH/PDSCH-waveform.ipynb


LDPC and Polar Coding
"""""""""""""""""""""

    .. toctree::
        :maxdepth: 1

        Notebooks/CompareWithMatlab/LDPC/LDPC-Matlab.ipynb
        Notebooks/CompareWithMatlab/Polar/PolarMatlab.ipynb

