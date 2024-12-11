# Copyright (c) 2024 InterDigital AI Lab
"""
This module implements the base class for channel models. Currently CDL
and TDL channel models are available in **NeoRadium** both of which are
derived from the :py:class:`ChannelBase` class in this module. For more
details about these subclasses please refer to
:py:class:`~neoradium.cdl.CdlChannel` or :py:class:`~neoradium.tdl.TdlChannel`.
"""
# ****************************************************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 11/22/2023    Shahab Hamidi-Rad       Completed the documentation
# ****************************************************************************************************************************************************

import numpy as np
import scipy.io, time
from scipy.signal import lfilter, resample_poly
from scipy.interpolate import interp1d

from .grid import Grid
from .waveform import Waveform
from .utils import polarInterpolate, interpolate, getMultiLineStr
from .random import random
from .carrier import SAMPLE_RATE, 𝜅

# ****************************************************************************************************************************************************
class ChannelBase:
    r"""
    This is the base channel model class that handles higher level processing
    such as application to time-domain waveforms (:py:class:`~neoradium.waveform.Waveforms`) or
    frequency-domain resource grids (:py:class:`~neoradium.grid.Grid`).
    
    This class also provides the :py:meth:`getChannelMatrix` function that calculates and
    returns the "Channel Matrix" for the channel model.
    
    Almost all of interactions with channel models are done using the methods
    of this class. The derived classes mostly implement the lower level processes
    such as how the channel multi-path information are generated or how MIMO
    antenna arrays are processed in connection with the channel multi-path
    properties.
    """
    # ************************************************************************************************************************************************
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments.

                :delaySpread: The delay spread in **nanoseconds**. The default is
                    30 ns. It can also be a string containing one of the values in
                    following table (See **3GPP TR 38.901 table 7.7.3-1**)
                    
                    ======================  ==============
                    Delay Spread str        Delay spread
                    ======================  ==============
                    'VeryShort'             10 ns
                    'Short'                 30 ns
                    'Nominal'               100 ns
                    'Long'                  300 ns
                    'VeryLong'              1000 ns
                    ======================  ==============

                :dopplerShift: The maximum doppler shift in Hz. The default is 40 Hz (corresponding
                    to a speed of about 10 km/h). A value of zero makes the channel model static.
                :carrierFreq: The carrier frequency of the channel model in Hz. The default is 3.5 GHz.
                :normalizeGains: A boolean flag. The default is ``True``. If ``True``, the path
                    gains are normalized before being applied to the signals.
                :normalizeOutput: A boolean flag. The default is ``True``. If ``True``, the gains
                    are normalized based on the number of receive antenna.
                :txDir: The transmission direction. The default is "Downlink". This is a string
                    and must be one of "Downlink" or "Uplink".
                :kFactor: The K-Factor (in dB) used for scaling. The default is ``None``. If not
                    specified (``kFactor=None``), K-factor scaling is disabled.
                :timing: A text string that specifies the way channel values are interpolated
                    in time. It can be one of the following values.
                    
                    * **'polar'**: The channel values are converted to polar coordinates before being linearly interpolated.
                    * **'linear'**: The channel values are interpolated linearly.
                    * **'nearest'**: A nearest neighbor interpolation is used.
                    * **'matlab'**: Use this only when comparing the results with Matlab's 5G toolkit for debugging and diagnostics purposes.

                :filterDelay: The delay used by the channel filter class
                    (:py:class:`ChannelFilter`). The default is 7 sample.
                :stopBandAtten: The Stop-band attenuation value (in dB) used by the channel
                    filter class (:py:class:`ChannelFilter`). The default is 80 dB.
                :seed: The seed used for by the random functions in the channel model. Set
                    this to a fixed value so that the channel model creates repeatable results.
                    The default is ``None`` which means this channel model uses the
                    **NeoRadium**'s :doc:`global random generator <./Random>`.
                    

        **Other Properties:**
        
        All of the parameters mentioned above are directly available. Here is a list
        of additional properties.
        
            :coherenceTime: The `Coherence time <https://en.wikipedia.org/wiki/Coherence_time_(communications_systems)>`_
                of the channel model in seconds.
            :channelFilter: The (:py:class:`ChannelFilter`) object used by this channel
                model.
            :sampleRate: The sample rate used by this channel model. For 3GPP standard,
                this is set to 30,720,000 samples per second.
            :curTime: The current time for this channel. This starts at zero when the
                channel is created and is increased each time the :py:meth:`goNext`
                function is called.

        """
        self.sampleRate = SAMPLE_RATE      # Fixed for 5G: 30,720,000 Hz
        self.sampleDensity = 𝜅             # Fixed for 5G: 64

        self.delaySpread = kwargs.get('delaySpread', 30)                # Default: 30ns
        if type(self.delaySpread)==str:
            # See TR38.901 - Table 7.7.3-1
            strToDelaySpread = {"VeryShort": 10, "Short": 30, "Nominal": 100, "Long": 300, "VeryLong": 1000}
            if self.delaySpread not in strToDelaySpread:
                raise ValueError("'delaySpread' must be a number or one of 'VeryShort', 'Short', 'Nominal', 'Long', or 'VeryLong'")
            self.delaySpread = strToDelaySpread[self.delaySpread]
            
        self.dopplerShift = kwargs.get('dopplerShift', 40)              # Default: 40Hz (about 10 km/h)
        self.carrierFreq = kwargs.get('carrierFreq', 3.5e9)             # Default: 3.5 GHz

        self.normalizeGains = kwargs.get('normalizeGains', True)        # True means normalize the path gains
        self.normalizeOutput = kwargs.get('normalizeOutput', True )     # Normalize output gains based on the number of receive antenna

        self.txDir = kwargs.get('txDir', 'Downlink')
        if self.txDir not in ['Downlink', 'Uplink']:
            raise ValueError("Unsupported 'txDir' (%s). It must be one of 'Downlink' or 'Uplink'."%(self.txDir))

        self.kFactor = kwargs.get('kFactor', None)                              # Desired K-factor for scaling in dB. None means K-factor scaling is disabled
        self.timing = kwargs.get('timing', 'polar')                             # The method used to upsample channel gains (Polyphase, Interpolate, or Step)
        if self.timing.lower() not in ['polar', 'linear', 'nearest', 'matlab']: # Nearest is the fastest and least accurate. Polar is the slowest and most accurate
            raise ValueError("The 'timing' must be one of 'polar', 'linear', 'nearest', or 'matlab'.")

        self.filterDelay = kwargs.get('filterDelay', 7 )
        self.stopBandAtten = kwargs.get('stopBandAtten', 80 )
        
        self.seed = kwargs.get('seed', None)
        if self.seed is None:   self.rangen = random                            # Use the NeoRadium's global random generator
        else:                   self.rangen = random.getGenerator(self.seed)    # Use a new dedicated random generator for this channel

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent, title, getStr):
        r"""
        Prints the properties of this channel model object.

        Parameters
        ----------
        indent : int (default: 0)
            The number of indentation characters.
            
        title : str or None (default: None)
            If specified, it is used as a title for the printed information.

        getStr : Boolean (default: False)
            If ``True``, it returns the information in a text string instead
            of printing the information.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns
            the information in a text string. Otherwise, nothing is returned.
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  delaySpread: %s ns\n"%(str(self.delaySpread))
        repStr += indent*' ' + "  dopplerShift: %s Hz\n"%(str(self.dopplerShift))
        repStr += indent*' ' + "  carrierFreq: %s Hz\n"%(str(self.carrierFreq))
        repStr += indent*' ' + "  normalizeGains: %s\n"%(str(self.normalizeGains))
        repStr += indent*' ' + "  normalizeOutput: %s\n"%(str(self.normalizeOutput))
        repStr += indent*' ' + "  txDir: %s\n"%(self.txDir)
        repStr += indent*' ' + "  timing method: %s\n"%(self.timing)
        repStr += indent*' ' + "  coherenceTime: %f (Sec.)\n"%(self.coherenceTime)
        if self.kFactor is not None:
            repStr += indent*' ' + "  kFactor: %s db\n"%(str(self.kFactor))
        if getStr: return repStr
        print(repStr)
        
    # ************************************************************************************************************************************************
    def makeFilter(self):   # Not documented
        # Creates a channel filter objects that is used by this channel model.
        return ChannelFilter(self.pathDelays, self.nrNt[1], self.stopBandAtten, self.filterDelay)

    # ************************************************************************************************************************************************
    def restart(self, restartRanGen=False):
        r"""
        Resets the state of this channel model to the initial state. Sets
        current time to zero and resets the channel filter.

        Parameters
        ----------
        restartRanGen : Boolean (default: False)
            If a ``seed`` was not provided to this channel model, this parameter
            is ignored. Otherwise, if ``restartRanGen`` is set to ``True``, this
            channel model's random generator is reset and if ``restartRanGen`` is
            ``False`` (default), the random generator is not reset. This means
            if ``restartRanGen`` is ``False``, calling this function starts a new
            sequence of channel instances which are different from the sequence when
            the channel was instantiated.
        """
        if (self.seed is not None) and restartRanGen:
            self.rangen = random.getGenerator(self.seed)
            
        self.curTime = 0
        self.channelFilter.resetStates()

    # ************************************************************************************************************************************************
    def goNext(self):
        r"""
        This method is called after each application of the channel to a signal. It
        advances the current time value and updates the channel filter state making
        the channel model ready for the next application in time.
        """
        self.curTime = self.nextTime
        self.channelFilter.goNext()

    # ************************************************************************************************************************************************
    def getMaxDelay(self):
        r"""
        Calculates and returns the maximum delay of this channel model in
        time-domain samples.

        Returns
        -------
        int
            The maximum delay of this channel model in number of
            time-domain samples.
        """
        return int(np.ceil(max(self.pathDelays)*self.sampleRate/1e9 + self.filterDelay))

    # ************************************************************************************************************************************************
    def applyKFactorScaling(self):  # Not documented
        # This function applies the K-Factor Scaling. This should be called only for profiles with
        # LOS paths and only when the K-Factor scaling is enabled (kFactor is not None).
        # See TR 38.901 - Sec. 7.7.6 K-factor for LOS channel models
        assert self.hasLos
        assert self.kFactor is not None
        
        # Linear Power Values
        powers = 10**(self.pathPowers/10.0)

        kModel = self.pathPowers[0] - 10*np.log10(powers[1:].sum())             # TR 38.901 - Eq. 7.7.6-2
        self.pathPowers[1:] = self.pathPowers[1:] - self.kFactor + kModel       # TR 38.901 - Eq. 7.7.6-1

        # Now we need to re-normalize the delay spreads
        # Calculate weighted RMS
        powerDelay = powers * self.pathDelays
        sumP = powers.sum()
        rms =  np.sqrt( np.square(powerDelay).sum()/sumP - np.square( powerDelay.sum()/sumP ) )
        self.pathDelays /= rms
    
    # ************************************************************************************************************************************************
    @property   # This property is already documented above in the __init__ function.
    def coherenceTime(self):
        # https://en.wikipedia.org/wiki/Coherence_time_(communications_systems)
        return np.sqrt(9/(16*np.pi))/self.dopplerShift

    # ************************************************************************************************************************************************
    def getChannelTimes(self, ns):                                              # Not documented
        inputDuration = ns/self.sampleRate                                      # Duration of input signal in seconds

        if self.timing.lower()=='matlab':
            chanGenRate = max(self.dopplerShift * 2 * self.sampleDensity, 1)    # Get the rate at which channel gains are generated based on doppler shift and sample density
            numChannels = chanGenRate * inputDuration                           # Channels needed for the duration of input signal
            numChannels = int( max( np.ceil(numChannels), 2) )+1                # Make sure we have at least 2 channels. Also create 2 additional channels for the transition.
            chanTimes = np.arange(numChannels)/chanGenRate
            
        else:
            dt = self.coherenceTime/self.sampleDensity                          # The time difference in seconds between 2 channel gain evaluations

            numGainEvals = inputDuration/dt                                     # The number of channel gain evaluations
            if numGainEvals<3:                                                  # We want at least 3 channel gain evaluations
                numGainEvals=3
                dt = inputDuration/2
                chanTimes = np.float64([0, dt, inputDuration])
            else:
                numGainEvals = int(np.ceil(numGainEvals))
                dt = inputDuration/numGainEvals
                chanTimes = np.arange(numGainEvals+1)*dt
                chanTimes[-1] = inputDuration
        
        self.nextTime = self.curTime + inputDuration
        
        return self.curTime + chanTimes

    # ************************************************************************************************************************************************
    def upSample(self, channelGains, channelTimes, upSampledTimeInfo):  # Not documented
        # channelGains is an nc x nr x nt x np tensor.
        # channelTimes is not 0 based. The curTime was added to it in the function "getChannelTimes" above.

        if np.isscalar(upSampledTimeInfo):  newTimes = np.arange(upSampledTimeInfo)/self.sampleRate     # upSampledTimeInfo is the total number of samples on the output
        else:                               newTimes = upSampledTimeInfo                                # upSampledTimeInfo is the list of times for the output gains

        shiftedTimes = channelTimes-channelTimes[0]         # Make channel times 0-based
        if self.timing.lower()=='polar':        gains = polarInterpolate(shiftedTimes, channelGains, newTimes, "linear")
        elif self.timing.lower()=='linear':     gains = interpolate(shiftedTimes, channelGains, newTimes, "linear")
        elif self.timing.lower()=='nearest':    gains = interpolate(shiftedTimes, channelGains, newTimes, "nearest")
        elif self.timing.lower()=='matlab':     gains = interpolate(shiftedTimes, channelGains, newTimes, "nearest")  # Same as "nearest" case
        else:                           assert False, "Unsupported interpolation type!"
        return gains                    # Shape: ns x nr x nt x numPaths

    # ************************************************************************************************************************************************
    def applyToGrid(self, grid):
        r"""
        Applies this channel model to the transmitted resource grid object specified
        by ``grid`` in frequency domain and returns the received resource grid in
        a new :py:class:`~neoradium.grid.Grid` object. This function first calls the
        :py:meth:`~ChannelBase.getChannelMatrix` function to get the channel
        matrix and then calls the :py:meth:`~neoradium.grid.Grid.applyChannel` method of
        the :py:class:`~neoradium.grid.Grid` class to get the received resource grid.

        Parameters
        ----------
        grid : :py:class:`~neoradium.grid.Grid`
            The transmitted resource grid. It is an ``Nt x L x K``
            :py:class:`~neoradium.grid.Grid` object where ``Nt`` is the number
            of transmit antenna, ``L`` is the number OFDM symbols, and
            ``K`` is the number of subcarriers in the resource grid.

        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A resource grid containing the received signal information. It is
            an ``Nr x L x K`` :py:class:`~neoradium.grid.Grid` object where ``Nr`` is the number
            of receive antenna, ``L`` is the number OFDM symbols, and ``K`` is the
            number of subcarriers in the resource grid.
        """
        channelMatrix = self.getChannelMatrix(grid.bwp)
        return grid.applyChannel(channelMatrix)
        
    # ************************************************************************************************************************************************
    def applyToSignal(self, inputSignal, keepNt=False):
        r"""
        Applies this channel model to the time-domain :py:class:`~neoradium.waveform.Waveform`
        specified by ``inputSignal`` and returns another :py:class:`~neoradium.waveform.Waveform`
        object containing the received signal in time domain.

        Parameters
        ----------
        inputSignal : :py:class:`~neoradium.waveform.Waveform`
            The transmitted time-domain waveform. It is an ``Nt x Ns``
            :py:class:`~neoradium.waveform.Waveform` object where
            ``Nt`` is the number of transmit antenna and ``Ns`` is
            the number of time samples in the transmitted waveform.
            
        keepNt : Boolean (default: False)
            This should always be set to ``False``. It is only set to
            ``True`` when it is used internally to calculate a channel
            matrix.

        Returns
        -------
        :py:class:`~neoradium.waveform.Waveform`
            A :py:class:`~neoradium.waveform.Waveform` object containing the
            received signal. It is an ``Nr x Ns`` :py:class:`~neoradium.waveform.Waveform`
            object where ``Nr`` is the number of receive antenna and ``Ns`` is
            the number of time samples in the received waveform.
        """
        nr, nt = self.nrNt
        ns = inputSignal.shape[1]   # Number of time-domain samples in the input signal
        if inputSignal.shape[0]!=nt:
            raise ValueError("Invalid number of input streams (%d) in the inputSignal! (must be %d)"%(inputSignal.shape[0], nt))
    
        channelTimes = self.getChannelTimes(ns)                             # shape: nc
        channelGains = self.getPathGains(channelTimes)                      # Shape: nc x nr x nt x np
        gains = self.upSample(channelGains, channelTimes, ns)               # Shape: ns x nr x nt x np
        
        filterOutput = self.channelFilter.applyToSignal(inputSignal)        # Shape: ns x nt x np
        filterOutput = filterOutput.reshape(ns,1,nt,-1)                     # Shape: ns x 1  x nt x np
        if keepNt:                                                          # This is used when creating a channelMatrix
            return Waveform((gains * filterOutput).sum(3).reshape(ns,-1).T) # Sum over np only, then reshape and transpose => Shape: nr*nt x ns
        
        output = (gains * filterOutput).sum((2,3))                          # Sum over nt and np. => Shape: ns x nr
        return Waveform(output.T)                                           # Shape: nr x ns

    # ************************************************************************************************************************************************
    def getChannelMatrixTDExp(self, bwp, numSlots=1, windowing="STD"):  # Not documented
        # This experimental method calculates the channel matrix by applying this channel to a time-domain waveform.
        # We first create a fake grid, then OFDM-modulate it, and apply the channel to the waveform. We then OFDM demodulate
        # the waveform to get the channel matrix.
        # Note that this usually takes longer than other channel matrix calculations.
        ll, kk = numSlots*bwp.symbolsPerSlot, 12*bwp.numRbs
        nr, nt = self.nrNt
        nFFT = bwp.nFFT
        
        l0 = bwp.slotNoInSubFrame * bwp.symbolsPerSlot                  # Number of symbols from start of this subframe
        maxL = bwp.symbolsPerSubFrame - l0                              # Max number of remaining symbols in this subframe from l0
        if ll > maxL:
            raise ValueError("Cannot get channel matrix crossing a subframe boundary! (At most %d symbols)"%(maxL))

        numPad = ((nFFT-kk+1)//2,(nFFT-kk)//2)                # Number of zeros to pad to be beginning and end of subcarriers
        paddedGrid = np.pad(np.ones((nt,ll,kk)), ((0,0),(0,0),numPad))  # Shape: nt, ll, nFFT
        shiftedPaddedGrid = np.fft.ifftshift(paddedGrid, axes=2)        # Shifted for IFFT
        txWaveForm = np.fft.ifft(shiftedPaddedGrid, axis=2)             # Time-Domain waveforms:  Shape: ll, nFFT

        symLens = bwp.symbolLens[l0:l0+ll]                              # Symbol lengths in samples for each symbol in the next numSlots
        cpLens = symLens-nFFT                                           # CP lengths in samples for each symbol in the next numSlots
        maxSymLen = symLens.max()
        indexes = (np.arange(maxSymLen) - cpLens[:,None])%nFFT          # Indexes used to insert the CP-Len elements from the end of symbol waveforms to the beginning.

        txWaveFormWithCPs = np.zeros((nt, ll, maxSymLen), dtype=np.complex128)      # Shape: ll, maxSymLen
        for l in range(ll): txWaveFormWithCPs[:,l,:] = txWaveForm[:,l,indexes[l]]   # Insert the CP-Len elements from the end of symbol waveforms to the beginning

        # Upconversion. See 3GPP TS 38.211 V17.0.0 (2021-12), Section 5.4
        n0 = bwp.symbolLens[:l0].sum()                                  # Number of samples from start of current subframe
        startIndexes = np.cumsum(np.append(n0,symLens[:-1]))            # Start sample index of each symbol in the next numSlots from the start of current subframe
        phaseFactors = np.exp( 2j * np.pi * self.carrierFreq * (-startIndexes-cpLens)/bwp.sampleRate )   # ll values
        txWaveFormWithCPs *= phaseFactors[None,:,None]                       # Upconversion

        # Now stitch the symbol waveforms back to back keeping only the first (symLens[l]) samples for each symbol 'l'
        txWaveForm = Waveform(np.concatenate([txWaveFormWithCPs[:,l,:symLen] for l,symLen in enumerate(symLens)],
                                             axis=1))
        if windowing.upper()!='NONE':
            txWaveForm = txWaveForm.applyWindowing(cpLens, windowing, bwp)

        maxDelay = self.getMaxDelay()
        txWaveForm = txWaveForm.pad(maxDelay)                           # Use the same signal for all Tx Antenna => Shape: nt x ns
        
        rxWaveForm = self.applyToSignal(txWaveForm, keepNt=True)        # nr*nt x ns
        offset = self.getTimingOffset(rxWaveForm.shape[1])              # channel delay value (Not the same as self.channelDelay)
        rxWaveFormDelayed = rxWaveForm.sync(offset)
        
        channelMatrix = rxWaveFormDelayed.ofdmDemodulate(bwp, f0=self.carrierFreq).grid     # nr*nt x ll x kk
        channelMatrix = np.transpose(channelMatrix,(1,2,0)).reshape(ll, kk, nr, nt)         # Shape: ll x kk x nr x nt
        return channelMatrix

    # ************************************************************************************************************************************************
    def getChannelMatrixTD(self, bwp, numSlots=1, interpolateTime=True):        # Not documented
        # This method calculates the channel matrix in time domain. It first calculates
        # the CIR, then makes a dummy received signal using the CIR, and applies OFDM demodulation
        # to the dummy received signal to calculate the Channel matrix.
        nr, nt = self.nrNt

        symbolLens = bwp.getSymLensForNextSlots(numSlots)               # Length of each symbol (including CP) for the next 'numSlots'
        totalSamples = sum(symbolLens)                                  # Duration in number of samples for the specified 'numSlots'
        totalDuration = totalSamples/bwp.sampleRate                     # Duration in seconds for the specified 'numSlots'
        symbolLens[0] -= bwp.nFFT                                       # A symbol starts just after its CP. So, the first one starts at the cpLen[0]
        symStartSamples = np.cumsum(symbolLens)                         # Symbol start sample index
        symbolTimes = (symStartSamples+self.filterDelay)/bwp.sampleRate # Symbol start times. We want a channel gain instance at each one of these
        
        if interpolateTime:
            # Calculate the channel times based on totalSamples. This will require an interpolation (see the call to upSample below)
            # to get the channel values at the OFDM symbol times.
            channelTimes = self.getChannelTimes(totalSamples)
        else:
            # This is usually faster because we het the channel gain values directly at the OFDM symbol times. No interpolation
            # is needed in this case.
            self.nextTime = self.curTime + totalDuration
            channelTimes = self.curTime + symbolTimes
        
        channelGains = self.getPathGains(channelTimes)                          # Shape: numChannels x nr x nt x numPaths

        # Calculate channel offset:
        meanGains = channelGains.mean(axis=0)
        hMean = meanGains.dot(self.channelFilter.coeffsMatrix)                      # (nr x nt x np) dot (np x nf) -> nr x nt x nf
        offset = np.abs(hMean.sum(axis=1)).sum(axis=0).argmax()

        if interpolateTime:
            # upsample channel gains
            channelGains = self.upSample(channelGains, channelTimes, symbolTimes)

        # Calculate Channel Impulse Response
        h = channelGains.dot(self.channelFilter.coeffsMatrix)                       # (ns x nr x nt x np) dot (np x nf) -> ns x nr x nt x nf

        # Create a fake received signal and populate it with the CIR values
        idxes = symStartSamples[:,None] - offset + np.arange(h.shape[-1])           # ns x nf
        idxes = idxes.flatten()

        rx = np.zeros((totalSamples, nr*nt), dtype=np.complex128)                   # totalSamples x nr*nt
        rx[idxes] = np.transpose(h,(0,3,1,2)).reshape(-1,nr*nt)
        channelMatrix = Grid.ofdmDemodulate(bwp, rx.T).grid                         # nr*nt x ns x numSubcarriers
        _, ns, k = channelMatrix.shape
        channelMatrix = np.transpose(channelMatrix,(1,2,0)).reshape(ns, k, nr, nt)  # ns x numSubcarriers x nr x nt
        return channelMatrix

    # ************************************************************************************************************************************************
    def getChannelMatrixFD(self, bwp, numSlots=1):                      # Not documented
        # This method calculate the channel matrix using the multi-path formula. This is expected to be more accurate than the
        # getChannelMatrixTD method. But the getChannelMatrixTD is closer to what happens in time domain when the channel is applied to an
        # OFDM waveform.
        
        # First find the times at which we want to calculate the channel gains. These are the times at the middle of each symbol.
        ll, kk = numSlots*bwp.symbolsPerSlot, 12*bwp.numRbs
        nr, nt = self.nrNt
        
        l0 = bwp.slotNoInSubFrame * bwp.symbolsPerSlot          # Number of symbols from start of this subframe
        symLens = bwp.symbolLens[l0:l0+ll]                      # Symbol lengths in samples for each symbol in the next numSlots
        startIndexes = np.cumsum(np.append(0,symLens[:-1]))     # Start sample index of symbols in next numSlots from the start of current subframe
        middleIndexes = startIndexes + symLens//2               # The sample index of middle of each symbol
        middleTimes = middleIndexes/bwp.sampleRate              # The time at the middle of each symbol

        channelGains = self.getPathGains(middleTimes)           # Channel gains. Shape: ll x nr x nt x pp

        # Calculate channel delay:
        hh = channelGains.mean(0).dot(self.channelFilter.coeffsMatrix)  # Get average of gains over time and multiply by filter coefficient. Shape: nr x nt x nf
        hh = hh.sum(1)                                                  # Then sum over all transmit antenna. Shape: nr x nf
        filterDelSamples = np.abs(hh).sum(0).argmax()                   # Sum the magnitudes over receive antenna and pick the sample index of maximum magnitude
        filterDelSamples -= self.filterDelay                            # This is the filter delay in number of samples
        filterDelay = filterDelSamples/bwp.sampleRate                   # Filter delay in seconds

        delays = self.pathDelays*1e-9                                   # Path delays in seconds (from ns)
        delays -= filterDelay                                           # Subtract the filter delay
        kDeltaF = (np.arange(kk)-kk//2)*bwp.spacing*1000                # k.∆f  (∆f = subcarrier spacing)
        
        # Now calculate the channel matrix h
        # Shapes: ll,nr,nt,pp,1                       1,1,1,1,kk                       1,1,1,pp,1    =>   ll,nr,nt,pp,kk
        h = channelGains[...,None]*np.exp(-2j*np.pi * kDeltaF[None,None,None,None,:] * delays[None,None,None,:,None])
        h = h.sum(3).transpose((0,3,1,2))                               # Sum over all paths, and transpose to shape ll,kk,nr,nt
        return h                                                        # Shape: ll,kk,nr,nt

    # ************************************************************************************************************************************************
    def getChannelMatrix(self, bwp, numSlots=1, timeDomain=True, interpolateTime=False):
        r"""
        Calculates and returns the channel matrix of this channel model. The channel
        matrix is a 4-D ``L x K x Nr x Nt`` complex numpy array where ``L`` is the
        number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the
        number of receive antenna, and ``Nt`` is the number of transmit antenna.

        Parameters
        ----------
        bwp : :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part object used to get information about the shape of
            the channel matrix, the timing information about OFDM symbols, and
            the FFT size.
            
        numSlots : int
            The number slots. This is used to determine the number of OFDM symbols
            ``L``. This function can create channel matrixes at slot boundaries. In
            other words ``L`` is always a multiple of ``bwp.symbolsPerSlot``.
        
        timeDomain : Boolean (default: True)
            Calculate the channel matrix in time domain. NeoRadium can calculate the
            channel matrix in both time domain and frequency domain.
            
            The time-domain calculation provides results that are closer to the
            effect of applying the channel in time domain to an OFDM-modulated
            waveform.
            
            The frequency domain method calculates the channel matrix directly
            using the multi-path information.
            
        interpolateTime : Boolean (default: False)
            This parameter is only used when ``timeDomain=True``.
            
            If set to ``True`` the path gains are first calculated at the sampling
            times based on the current ``dopplerShift`` parameter of this channel
            model. These gains are then upsampled (Interpolated) using the method
            specified by the ``timing`` property of this channel model. This is usually
            slower and results in a less accurate channel matrix.
            
            If set to ``False`` (the default) the path gains are calculated
            directly at the OFDM symbol times and there is no need for interpolation.

        Returns
        -------
        4-D complex numpy array
            A 4-D ``L x K x Nr x Nt`` complex numpy array where ``L`` is the
            number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the
            number of receive antenna, and ``Nt`` is the number of transmit antenna.
        """
        # Note:
        # The most accurate approach is when using time domain without interpolation (The default)
        # This is closest to the effect of applying channel to the time domain waveform when "Polar" interpolation is used.
        if self.timing.lower()=="matlab":   return self.getChannelMatrixTD(bwp, numSlots, True)   # Always use time-domain with interpolation for Matlab case
        if timeDomain:                      return self.getChannelMatrixTD(bwp, numSlots, interpolateTime)
        return self.getChannelMatrixFD(bwp, numSlots)

    # ************************************************************************************************************************************************
    def getTimingOffset(self, numSamples=3072):
        r"""
        This function calculates the timing offset used in synchronization of a
        received time-domain signal. It first calculates the Channel Impulse
        Response (CIR) and then uses it to find the index of the maximum power.

        Parameters
        ----------
        numSamples : int (default: 3072)
            The number of time-domain samples to consider when calculating the
            CIR. The default value of ``3072`` is equivalent to the duration of one
            sub-frame (1 millisecond) which works for most situations. When a
            time-domain signal is available, the actual number of samples could
            be used for this parameter.
            
        Returns
        -------
        int
            The time-domain offset value used for synchronization of a received
            waveform. This return value can be directly passed to the
            :py:meth:`~Waveform.sync` function to synchronize a received waveform.
        """
        channelTimes = self.getChannelTimes(numSamples)
        channelGains = self.getChannelGains(channelTimes)               # numChannels x nr x nt x np
        meanGains = channelGains.mean(axis=0)                           # nr x nt x np

        # Calculate Channel Impulse Response using the mean gains
        h = meanGains.dot(self.getPathFilters())                        # (nr x nt x np) dot (np x nf) -> nr x nt x nf

        # Sum over transmit antenna
        h = h.sum(axis=1)                                               # nr x nf

        # Find the index of maximum magnitude
        offset = np.abs(h).sum(axis=0).argmax()
        return offset

    # ************************************************************************************************************************************************
    def getPathGains(self, channelTimes=None):
        r"""
        This function calculates the path gains of this channel at the times
        specified by the ``channelTimes``. The returned value is a 4-D tensor
        of shape (``len(channelTimes) x nr x nt x np``). The path gains are
        normalized based on the ``normalizeOutput`` and ``normalizeGains``
        values (See :py:class:`ChannelBase`)

        Parameters
        ----------
        channelTimes : numpy array or None (default: None)
            This is a 1-D list of the times at which the path gains are
            calculated. If this is None, then the path gains are calculated
            for current time (See ``curTime`` in :py:class:`ChannelBase`).
            
        Returns
        -------
        4-D complex numpy array
            The path gains at the specified times. The shape of returned tensor
            is (``len(channelTimes) x nr x nt x np``).
        """
        if channelTimes is None:    # If channel times is not given, then return channel gains at current time
            channelTimes = np.float64([self.curTime])
        
        pathGains = self.getChannelGains(channelTimes)                                              # len(channelTimes) x nr x nt x np
        if self.normalizeOutput:    pathGains /= np.sqrt(self.nrNt[0])                              # Divide by sqrt(nr)
        if self.normalizeGains:     pathGains /= np.sqrt((10.0**(self.pathPowers/10.0)).sum())      # Divide by sqrt(sum(clusterPowers))
        return pathGains                                                                            # len(channelTimes) x nr x nt x np
 
    # ************************************************************************************************************************************************
    def getPathFilters(self):
        r"""
        This function returns the channel filter coefficients for each
        path. It can be used with the path gains obtained by the
        :py:meth:`getPathGains` function to calculate Channel Impulse
        Response (CIR).
                    
        Returns
        -------
        2-D numpy array
            The p'th row of the returned matrix contains the filter
            coefficients of the p'th path. Please note that different
            paths may have different number of filter coefficients.
            The number of columns in the returned matrix is equal to
            maximum number of filter coefficients for all the paths.
            The rows are zero-padded at the end if the actual number
            of filter coefficients is less than the number of columns.
        """
        return self.channelFilter.coeffsMatrix

# ****************************************************************************************************************************************************
class ChannelFilter:
    r"""
    This class implements a channel filter that is used by all channel
    models. Since the path delays are not always a multiple of sampling
    period, this filter is used to apply the required (sometimes
    fractional) shifts to each path's waveform.
    This class calculates the filter coefficients for each path/tap. This
    is done by making a "Windowed Sinc Low Pass Filter". We first
    create a |Kaiser_window| and then apply it to the |Sinc_function|.
    This gives us the FIR coefficients that can be applied to any signal
    using the |lfilter| function.
        
    For more information please see |How_to_Create_a_Configurable_Filter_Using_a_Kaiser_Window|.
        
    In our case, ``stopBandAtten`` (See :py:class:`ChannelBase`) is equal to
    :math:`A=-20\log_{10} \delta`. Where :math:`\delta` is the "ripple" value
    of the filter. The default value of :math:`A=80db` results in
    :math:`\delta = 10^{-A/20} = 0.0001` and assuming a window size
    of 801, the transition bandwidth also known as "rolloff" is calculated
    as:
    
        .. math::

            b = \frac {A-8} {2.285*2*\pi*800} = 0.00626868

    Since this class is used only internally by the channel model classes, its
    functions are not included in the **NeoRadium**'s API documentation.
    Other types of filters could be designed and used.

    .. *********** The following links open in a new tab ****************

    .. |How_to_Create_a_Configurable_Filter_Using_a_Kaiser_Window| raw:: html

        <a href="https://tomroelandts.com/articles/how-to-create-a-configurable-filter-using-a-kaiser-window"
           target="_blank">How to Create a Configurable Filter Using a Kaiser Window</a>

    .. |lfilter| raw:: html

        <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lfilter.html"
           target="_blank">lfilter</a>
         
    .. |Kaiser_window| raw:: html

        <a href="https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html"
           target="_blank">Kaiser window</a>

    .. |Sinc_function| raw:: html

        <a href="https://numpy.org/doc/stable/reference/generated/numpy.sinc.html"
           target="_blank">Sinc function</a>
    """
    # ************************************************************************************************************************************************
    def __init__(self, pathDelays, numTxAntenna, stopBandAtten, filterDelay=None, **kwargs):
        self.sampleRate = SAMPLE_RATE
        self.pathDelays = pathDelays
        self.numTxAntenna = numTxAntenna
        self.filterDelay = filterDelay
        self.stopBandAtten = stopBandAtten

        self.filterLen = kwargs.get('filterLen', 16)
        self.numInterpol = kwargs.get('numInterpol', 50)
        self.normalize = kwargs.get('normalize', True)
        self.numPaths = len(pathDelays)
        
        # The problem is that the delays are not always an integer number of time-domain samples. We need to
        # be able to apply fractional delays to the signal samples in time domain.
        
        # First convert delays from seconds to number of samples:
        delaysInSamples = self.pathDelays * 1e-9 * self.sampleRate      # Note that pathDelays are in nanoseconds
        delayFractions = delaysInSamples - np.int32(delaysInSamples)    # Fractional Parts of delays
        
        intGrid = np.arange(self.numInterpol,-1,-1)/self.numInterpol                # [[50/50, 49/50, 48/50, .., 1/50, 0/50]] (shape: 1x51)
        phaseIndexes = np.abs(delayFractions.reshape(-1,1)-intGrid).argmin(axis=1)  # Idx of closest value from above to each fraction value (Shape:p)

        # If the delay is (almost) an integer number of samples, then interpolation is not required. This is the case when the phase index is 0 or 50
        self.interpolationRequired = (phaseIndexes!=0)*(phaseIndexes!=self.numInterpol)     # 0 or 1 for each path delay value.  (Shape: p)

        # For the case when the phase index is 0, we need to increment the integer value of the delay samples
        intDelays = np.int32(delaysInSamples) + 1*(phaseIndexes==0)                         # integer part of path delays        (Shape: p)

        # The following is the range [-7..8]  (filterLen=16) shifted by the integer number of delay samples
        intRanges = intDelays.reshape(-1,1) + np.int32([[1-self.filterLen//2, self.filterLen//2]]) * self.interpolationRequired.reshape(-1,1)  # Shape: p,2
        # The following is the union of all the ranges for different paths
        rangeIndexes = list(range(intRanges[:,0].min(), intRanges[:,1].max()+1))

        # The filter delay needs to be at least from the min of the range to 0 (which is equal to -rangeIndexes[0]. Note that rangeIndexes[0] is a negative value)
        # The value -rangeIndexes[0] is called "Causal latency"
        if self.filterDelay is None:    self.filterDelay = -rangeIndexes[0]
        else:                           assert self.filterDelay >= -rangeIndexes[0]

        self.integerDelays = intRanges[:,0] + self.filterDelay  # The integer delay for each path.

        if np.any(self.interpolationRequired):
            interpolationMatrix = []
            for i, idx in enumerate(phaseIndexes):
                if self.interpolationRequired[i]:
                    fir = self.getWindowedSincFIR(self.stopBandAtten)
                    interpolationMatrix += [ fir[idx::self.numInterpol].tolist() ]
                else:
                    interpolationMatrix += [ self.filterLen*[0] ]
        else:
            interpolationMatrix = 1

        fracDelayCoeffs = np.float64(interpolationMatrix)

        self.filterCoeffs = []
        maxLen = 0
        for p,fracDelay in enumerate(fracDelayCoeffs):
            if self.interpolationRequired[p]:   coeffs = self.integerDelays[p]*[0] + fracDelay.tolist()
            else:                               coeffs = self.integerDelays[p]*[0] + [1]
            maxLen = max( maxLen, len(coeffs) )
            self.filterCoeffs += [ np.float64(coeffs) ]

        # Also save a copy of the filter Coeffs in a matrix (shorter rows are zero-padded)
        self.coeffsMatrix = np.float64( [list(xx) + (maxLen-len(xx))*[0] for xx in self.filterCoeffs])  # np x nf
        self.resetStates()

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)      # Not documented
    def print(self, indent=0, title=None, getStr=False):        # Not documented
        if title is None:   title = "Channel Filter Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  filterDelay (samples): %s\n"%(str(self.filterDelay))
        repStr += indent*' ' + "  numTxAntenna: %s\n"%(str(self.numTxAntenna))
        repStr += indent*' ' + "  numPaths: %s\n"%(str(self.numPaths))
        repStr += getMultiLineStr("pathDelays (ns)", self.pathDelays, indent, "%6f", 6, numPerLine=10)
        repStr += indent*' ' + "  filterLen: %s\n"%(str(self.filterLen))
        repStr += indent*' ' + "  numInterpol: %s\n"%(str(self.numInterpol))
        repStr += indent*' ' + "  normalize: %s\n"%(str(self.normalize))
        repStr += indent*' ' + "  stopBandAtten: %s\n"%(str(self.stopBandAtten))

        if getStr: return repStr
        print(repStr)

    # ************************************************************************************************************************************************
    def applyToSignal(self, signal):        # Not documented
        # This function applies the channel filter to the given signal. The
        # "signal" can be a "Waveform" object or a numpy array.
        # Signal shape is: nt x ns
        filterOut = []
        signalWaveform = signal if type(signal)==np.ndarray else signal.waveform
        self.nextFilterStates = []
        for p, pathFilterCoeffs in enumerate(self.filterCoeffs):
            pathOut, pathNewState = lfilter(pathFilterCoeffs, 1, signalWaveform.T, 0, self.filterStates[p])
            filterOut += [pathOut]          # The shape of pathOut is: ns x nt
            self.nextFilterStates += [ pathNewState ]

        filterOut = np.stack(filterOut,2)
        return filterOut                    # Shape: ns x nt x np
        
    # ************************************************************************************************************************************************
    def resetStates(self):                  # Not documented
        # This function resets the filter state to all zeros.
        self.filterStates = [ np.float64( (len(self.filterCoeffs[i])-1)*[self.numTxAntenna*[0]]) for i in range(self.numPaths) ]
        self.nextFilterStates = None

    # ************************************************************************************************************************************************
    def goNext(self):                       # Not documented
        # This is called by the ChannelBase.goNext. Current state of the filter will be
        # used as initial state for the next call to the applyToSignal method.
        self.filterStates = self.nextFilterStates

    # ************************************************************************************************************************************************
    def getWindowedSincFIR(self, stopBandAtten):    # Not documented
        # We want to create a "Windowed Sinc Low Pass Filter" and return the FIR values.
        # To understand what is happening here, see "How to Create a Simple Low-Pass Filter" at:
        #   https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        # We do this by making a "windowed sinc filter". So, we first need to make a "window".
        # See the following page for info about how to make a kaiser window:
        #   https://tomroelandts.com/articles/how-to-create-a-configurable-filter-using-a-kaiser-window
        # In our case, stopBandAtten is the same as 𝐴=−20log10(𝛿). Where the 𝛿 is the "ripple".
        # The default value of A=80db with window size of 801=50*16+1 results in 𝛿=0.0001 and b=0.00626868. Where b is the transition bandwidth (AKA rolloff).
        # 𝛿 = 10**(-A/20)
        # b = (A-8)/(2.285*2*𝛑*800)
        
        # Note: in the first case below, I am using "8.861" instead of the "8.7" in the original literature to make it continuous at 50.
        if stopBandAtten > 50:      beta = 0.1102*(stopBandAtten-8.861)
        elif stopBandAtten < 21:    beta = 0;
        else:                       beta = (0.5842*((stopBandAtten-21)**0.4)) +  0.07886*(stopBandAtten-21);
    
        # Kaiser window is used with the beta value calculated based on the above heuristic approach.
        nn = self.numInterpol * self.filterLen
        kaiserWindow = np.kaiser(nn+1, beta)        # Shape: (nn+1,)   (i.e.: 801)
        
        # Now make a "windowed sinc filter":
        m = np.arange(-nn//2,nn//2+1,1)/self.numInterpol
        fir = kaiserWindow * np.sinc(m)

        # At multiples of "self.numInterpol", the fir values are close to zero. We force them to exactly zero, except the one
        # at the center which is close to 1 and here forced to 1
        fir[0 : nn+1 : self.numInterpol] = 0
        fir[nn//2] = 1

        fir = fir[:-1]                      # drop the last value => Shape: (nn,)
        return fir
