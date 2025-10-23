# Copyright (c) 2024 InterDigital AI Lab
"""
This module serves as the foundation for channel models. Currently, **NeoRadium** supports three types of channel 
models: CDL, TDL, and Trajectory-based channel models. Each model is derived from the :py:class:`ChannelModel` 
class defined in this module. For more information about these subclasses, please refer to 
:py:class:`~neoradium.cdl.CdlChannel`, :py:class:`~neoradium.tdl.TdlChannel`, 
or :py:class:`~neoradium.trjchan.TrjChannel`.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 03/27/2025    Shahab Hamidi-Rad       First version of the file. This file replaces the older "channel.py" module. It
#                                       has been redesigned to work with more general cases arising with ray-tracing
#                                       channel models.
# 05/07/2025    Shahab                  Completed the documentation.
# **********************************************************************************************************************
import numpy as np
from scipy.signal import lfilter

from .grid import Grid
from .waveform import Waveform
from .utils import interpolate, freqStr, toLinear, toDb
from .random import random
from .carrier import SAMPLE_RATE, 𝜅

# **********************************************************************************************************************
class ChannelModel:
    r"""
    This is the base channel model class that handles higher level processing, such as creating Channel Impulse Response
    (CIR) and channel matrices and applying the channel to a time-domain :py:class:`~neoradium.waveform.Waveform` 
    or a frequency-domain resource :py:class:`~neoradium.grid.Grid`.
    
    Almost all interactions with channel models are done using the methods of this class. The derived classes mostly
    implement the lower level processes such as how the channel multipath information is obtained or how MIMO antenna
    arrays are processed in connection with the channel multipath properties.
    """
    # ******************************************************************************************************************
    def __init__(self, bwp, **kwargs):
        r"""
        Parameters
        ----------
        bwp : :py:class:`~neoradium.carrier.BandwidthPart` 
            The bandwidth part object used by the channel model to create channel matrices.

        kwargs : dict
            A set of optional arguments.

                :normalizeGains: A boolean flag. The default value is `True`, indicating that the path gains 
                    are normalized before they are applied to the signals.
                    
                :normalizeOutput: A boolean flag. The default value is `True`, indicating that the gains are 
                    normalized based on the number of receive antennas.
                    
                :txDir: A string that represents the transmission direction, which can be either “Downlink” or 
                    “Uplink”. By default, it is set to “Downlink”.
                    
                :filterLen: The length of the channel filter. The default is 16 samples.
                
                :delayQuantSize: The size of delay fraction quantization for the channel filter. The default is 64.
                
                :stopBandAtten: The Stop-band attenuation value (in dB) used by the channel filter. The default is 80dB.
                
                :seed: The seed used by the random functions in the channel model. Setting this to a fixed value ensures
                    that the channel model generates repeatable results. The default value is `None`, indicating 
                    that this channel model uses the **NeoRadium**’s :doc:`global random number generator <./Random>`.
                    
                :dopplerShift: The maximum Doppler shift in Hertz. The default value is 40 Hertz, which corresponds to
                    a speed of approximately 10 kilometers per hour. A value of zero makes the channel model static. 
                    For trajectory-based channel models, this value is automatically assigned based on the maximum 
                    trajectory speed.
                    
                :carrierFreq: The carrier frequency of the channel model in hertz. The default is 3.5 GHz.
                    

        **Other Properties:**
        
        All of the parameters mentioned above are directly available. Here is a list of additional properties:
        
            :coherenceTime: The `Coherence time <https://en.wikipedia.org/wiki/Coherence_time_(communications_systems)>`_
                of the channel model in seconds. This is calculated based on the ``dopplerShift`` parameter.
            :sampleRate: The sample rate used by this channel model. For 3GPP standard, this is set to 30,720,000 
                samples per second.
        """
        if bwp is None: raise ValueError("The bandwidth part cannot be 'None'!")
        self.sampleRate = bwp.sampleRate                            # Fixed for 5G: 30,720,000 Hz
        self.bwp = bwp                                              # The bandwidth part

        self.dopplerShift = kwargs.get('dopplerShift', 40)          # Default: 40Hz (about 10 km/h)
        self.carrierFreq = kwargs.get('carrierFreq', 3.5e9)         # Default: 3.5 GHz

        self.normalizeGains = kwargs.get('normalizeGains', True)    # Normalize the path gains
        self.normalizeOutput = kwargs.get('normalizeOutput', True ) # Normalize gains based on the No. of RX antennas
        
        self.txDir = kwargs.get('txDir', 'Downlink')                # Currently used only by TDL.
        if self.txDir not in ['Downlink', 'Uplink']:
            raise ValueError("Unsupported 'txDir' (%s). It must be one of 'Downlink' or 'Uplink'."%(self.txDir))

        self.filterLen = kwargs.get('filterLen', 16 )
        self.stopBandAtten = kwargs.get('stopBandAtten', 80 )
        self.delayQuantSize = kwargs.get('delayQuantSize', 64 ) # Delay fraction quantization size (See getCoeffMatrix)
        
        self.seed = kwargs.get('seed', None)
        if self.seed is None:
            self.rangen = random                            # Use the NeoRadium's global random number generator
        else:
            self.rangen = random.getGenerator(self.seed)    # Use a new dedicated random number generator
        
        self.allFirs = self.buildFirs()
        # Note: Make sure the derived class calls restart at the end of its constructor
                
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this channel model object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If `None` (default), the text
            "Channel Model Properties:" is used for the title.

        getStr : Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Channel Model Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  carrierFreq:     {freqStr(self.carrierFreq)}\n"
        repStr += indent*' ' + f"  normalizeGains:  {str(self.normalizeGains)}\n"
        repStr += indent*' ' + f"  normalizeOutput: {str(self.normalizeOutput)}\n"
        repStr += indent*' ' + f"  txDir:           {self.txDir}\n"
        repStr += indent*' ' + f"  filterLen:       {self.filterLen} samples\n"
        repStr += indent*' ' + f"  delayQuantSize:  {self.delayQuantSize}\n"
        repStr += indent*' ' + f"  stopBandAtten:   {self.stopBandAtten} dB\n"
        repStr += indent*' ' + f"  dopplerShift:    {freqStr(self.dopplerShift)}\n"
        repStr += indent*' ' + f"  coherenceTime:   {self.coherenceTime} sec\n"
        if getStr: return repStr
        print(repStr)
        
    # ******************************************************************************************************************
    def restart(self, restartRanGen=False, applyToBwp=True):
        r"""
        Resets the state of this channel model to the initial state.

        Parameters
        ----------
        restartRanGen : Boolean
            If a ``seed`` was not provided to this channel model, this parameter is ignored. Otherwise, if 
            ``restartRanGen`` is set to `True`, the random number generator of this channel model is reset. If 
            ``restartRanGen`` is `False` (the default), the random number generator is not reset. This means that 
            if ``restartRanGen`` is `False`, for stochastic channel models, calling this function starts a new 
            sequence of channel instances, which differs from the sequence when the channel was instantiated.
            
        applyToBwp : Boolean
            If set to `True` (the default), this function restarts the :py:class:`~neoradium.carrier.BandwidthPart` 
            associated with this channel model. Otherwise, the :py:class:`~neoradium.carrier.BandwidthPart` state 
            remains unchanged.
        """
        # Note: The random number generator must be reset in the derived classes. Resetting it here is not appropriate
        #       because this function is typically called at the end of the derived class restart function, and the
        #       derived classes may need random values before this call.
        if applyToBwp: self.bwp.restart()
        self.filterDelays = np.array([(self.filterLen-1)//2])   # The default filter delay(s)
        self.curSlotStart = 0                                   # Start of current slot in samples
        self.nextSlotStart = 0                                  # Start of next slot in samples
        self.prepareForNextSlot()

    # ******************************************************************************************************************
    def goNext(self, applyToBwp=True):
        r"""
        This method is called after each application of the channel to a signal to move to the next slot. It advances 
        the channel model’s internal variable that keeps track of the current time.

        Parameters
        ----------
        applyToBwp : Boolean
            If set to `True` (the default), this function advances the timing state of the 
            :py:class:`~neoradium.carrier.BandwidthPart` associated with this channel model. Otherwise, the 
            :py:class:`~neoradium.carrier.BandwidthPart` state remains unchanged.
        """
        self.curSlotStart = self.nextSlotStart
        if applyToBwp: self.bwp.goNext()

    # ******************************************************************************************************************
    def getMaxDelay(self):
        r"""
        Calculates and returns the maximum delay of this channel model in time-domain samples for the current slot.

        Returns
        -------
        int
            The maximum delay of this channel model in number of time-domain samples for the current slot.
        """
        return int(np.ceil(self.pathDelays.max()*self.sampleRate/1e9 + self.filterDelays.max()))

    # ******************************************************************************************************************
    @property   # This read-only property is already documented above in the __init__ function.
    def coherenceTime(self):
        # https://en.wikipedia.org/wiki/Coherence_time_(communications_systems)
        return np.sqrt(9/(16*np.pi))/self.dopplerShift

    # ******************************************************************************************************************
    @property
    def nrNt(self):
        raise NotImplementedError("The derived channel model classes must implement the `nrNt` property!")

    # ******************************************************************************************************************
    def getPathGains(self):
        raise NotImplementedError("The derived channel model classes must implement the `getPathGains` method!")

    # ******************************************************************************************************************
    def applyToGrid(self, grid):
        r"""
        This function applies the channel model to the transmitted resource grid object specified by ``grid`` in 
        the frequency domain. It then returns the received resource grid in a new :py:class:`~neoradium.grid.Grid` 
        object. The function first calls the :py:meth:`~ChannelModel.getChannelMatrix` method to obtain the channel 
        matrix. Subsequently, it invokes the :py:meth:`~neoradium.grid.Grid.applyChannel` method of the 
        :py:class:`~neoradium.grid.Grid` class to calculate the received resource grid.

        Parameters
        ----------
        grid : :py:class:`~neoradium.grid.Grid`
            The transmitted resource grid. An ``Nt x L x K`` :py:class:`~neoradium.grid.Grid` object, where 
            ``Nt`` represents the number of transmit antennas, ``L`` denotes the number of OFDM symbols, and ``K`` is
            the number of subcarriers in the resource grid.

        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A resource grid containing the received signal. An ``Nr x L x K`` :py:class:`~neoradium.grid.Grid` object,
            where ``Nr`` represents the number of receive antennas, ``L`` denotes the number of OFDM symbols, and 
            ``K`` is the number of subcarriers in the resource grid.
        """
        channelMatrix = self.getChannelMatrix()
        return grid.applyChannel(channelMatrix)

    # ******************************************************************************************************************
    def buildFirs(self):                    # Not documented
        # We want to create a "Windowed Sinc Low Pass Filter" and return the FIR values.
        # To understand what is happening here, see "How to Create a Simple Low-Pass Filter" at:
        #   https://tomroelandts.com/articles/how-to-create-a-simple-low-pass-filter
        # We do this by making a "windowed sinc filter". So, we first need to make a "window".
        # See the following page for info about how to make a kaiser window:
        #   https://tomroelandts.com/articles/how-to-create-a-configurable-filter-using-a-kaiser-window
        # In our case, stopBandAtten is the same as 𝐴=−20log10(𝛿). Where the 𝛿 is the "ripple".
        # The default value of A=80 dB with window size of N=1025=64*16+1 results in 𝛿=0.0001 and b≈0.005. Where
        # b is the transition bandwidth (AKA rolloff).
        # 𝛿 = 10**(-A/20)
        # b = (A-8)/(2.285*2*𝛑*(N-1))
        #
        # See Also the "Filter Experiments" files in the "OtherExperiments" folder in the playgrounds (Not published)
        
        # Calculate beta:
        if self.stopBandAtten > 50:   beta=0.1102*(self.stopBandAtten-8.7)
        elif self.stopBandAtten < 21: beta=0
        else:                         beta=(0.5842*((self.stopBandAtten-21)**0.4)) +  0.07886*(self.stopBandAtten-21)

        nn = self.delayQuantSize * self.filterLen
        kaiserWindow = np.kaiser(nn+1, beta)                                # Shape: (nn+1,)
        
        m = np.arange(-nn//2,nn//2+1,1)/self.delayQuantSize
        fir = kaiserWindow * np.sinc(m)                                     # Shape: (nn+1,)
        
        # At multiples of "self.delayQuantSize", the fir values are close to zero. We force them to exactly zero,
        # except the one at the center which is close to 1 and here forced to 1
        fir[0 : nn+1 : self.delayQuantSize] = 0
        fir[nn//2] = 1
        
        # Another way to create the FIR (Remarked out for now):
        # TODO: Check to see if there is a significant performance difference
        # fir = firwin(nn+1, 1/self.delayQuantSize, window=('kaiser',beta))
        # fir/=fir.max()
        
        allFirs = fir[:-1].reshape(self.filterLen, self.delayQuantSize).T   # Shape: delayQuantSize x filterLen
        
        # Note that we have delayQuantSize+1 total filters. The last filter is the shifted version of the
        # first filter:
        return np.concatenate([allFirs,np.roll(allFirs[:1],-1)])            # Shape: (delayQuantSize+1) x filterLen

    # ******************************************************************************************************************
    def getCoeffMatrix(self):               # Not documented
        # 'self.pathDelays' is set by the derived classes. It is an array of length numPaths containing
        # delay values in nanoseconds.
        numPaths = len(self.pathDelays)
        delaysInSamples = self.pathDelays * 1e-9 * self.sampleRate      # Note that pathDelays are in nanoseconds
        intDelays = np.int32(delaysInSamples)                           # Integer parts of delays
        delayFractions = delaysInSamples - intDelays                    # Fractional parts of delays

        # Calculate Filter Delay. The filter delay is additional delay to ensure the filter's "Causal latency".
        # The lowest delay cannot be less than (filterLen-1)//2.
        self.filterDelay = np.clip( self.filterLen//2 - 1 - intDelays.min(), 0, None)
        intDelays += self.filterDelay  # Update the integer delay. Now each delay is at least (filterLen-1)//2

        # Now quantizing the fractions
        # Indexes of closest value in [1, (q-1)/q,(q-2)/q..,1/q,0] to each fraction value (q=delayQuantSize)
        quantIndexes = np.int32(np.round(self.delayQuantSize*(1-delayFractions)))   # Shape: numPaths
        fracCoeffs = self.allFirs[quantIndexes]                                     # Shape: numPaths x filterLen

        # Now calculating the Coefficient Matrix:
        coeffLen = intDelays.max() + self.filterLen//2 + 1  # It must includes the second half of the filter
        
        coeffMatrix = np.zeros(numPaths*coeffLen)
        indexes = intDelays[:,None] + np.arange(self.filterLen) + \
                  np.arange(numPaths)[:,None]*coeffLen - self.filterLen//2 + 1
                  
        coeffMatrix[indexes] = fracCoeffs
        return coeffMatrix.reshape((numPaths,coeffLen))                             # Shape: numPaths x coeffLen

    # ******************************************************************************************************************
    def prepareForNextSlot(self):
        # Shape values:
        # nc: Number of symbols in this slot
        # pp: Max number of paths
        # cl: Max length channel filters
        if self.nextSlotStart > self.curSlotStart: return   # The parameters for this slot have already been initialized
        
        symLens = self.bwp.getSymLens()                 # lengths of the next (nc+1) symbols
        slotLen = symLens[:-1].sum()                    # Length of this slot in samples
        symLens[0] -= self.bwp.nFFT                     # A symbol starts just after its CP. (symLen = cpLen + nFFT)
        symStarts = np.cumsum(symLens)                  # Symbol start samples for the next (nc+1)
        symLens[0] += self.bwp.nFFT                     # Fix the symLens

        self.chanGainSamples = self.curSlotStart+symStarts              # Calculate gains at these (nc+1) samples
        chanGains1 = self.getChannelGains()                             # Shape: (nc+1) x nr x nt x pp
        chanGains = chanGains1[:-1]                                     # Shape: nc x nr x nt x pp

        # Save data for next slot
        self.nextSlotStart = self.curSlotStart + slotLen                # Advance the nextSlotStart

        coeffMatrix = self.getCoeffMatrix()                             # Shape: pp x cl

        # This is the discrete-time channel impulse response: h={h[0], h[1], ..., h[cl-1]}
        nc,nr,nt,pp = chanGains.shape
        cir = np.matmul(chanGains.reshape(nc,-1,pp), coeffMatrix[None,:,:]).reshape(nc,nr,nt,-1) # nc x nr x nt x cl
        chanOffset = np.abs(cir.sum((0,2))).sum(0).argmax() # Sum on (nc,nt) -> ABS -> sum on nr -> argmax
        
        # Saving the channel gains and coefficient matrix for this slot:
        self.symLens = symLens                          # Shape: (nc+1,)
        self.chanGains = chanGains                      # Shape: nc x nr x nt x pp
        self.chanGains1 = chanGains1                    # Shape: (nc+1) x nr x nt x pp
        self.coeffMatrix = coeffMatrix                  # Shape: pp x cl
        self.cir = cir                                  # Shape: nc x nr x nt x cl
        self.chanOffset = chanOffset                    # A number

    # ******************************************************************************************************************
    def getTimingOffset(self):
        self.prepareForNextSlot()   # Making sure the Channel Offset is ready for this slot.
        return self.chanOffset
        
    # ******************************************************************************************************************
    def getChannelMatrix(self):
        r"""
        This method calculates and returns the channel matrix at the current time. The channel matrix is a 4-D complex
        NumPy array with dimensions ``L x K x Nr x Nt``, where ``L`` represents the number of OFDM symbols, ``K`` 
        denotes the number of subcarriers, ``Nr`` is the number of receive antennas, and ``Nt`` indicates the number 
        of transmit antennas. Please refer to the notebook :doc:`../Playground/Notebooks/Channels/ChannelMatrix` for 
        an example of using this function.

        Returns
        -------
        4-D complex NumPy array
            A 4-D complex NumPy array with dimensions ``L x K x Nr x Nt``, where ``L`` represents the number of 
            OFDM symbols, ``K`` denotes the number of subcarriers, ``Nr`` is the number of receive antennas, and 
            ``Nt`` indicates the number of transmit antennas.
        """
        # For better understanding of what is going on here, see the slide "Calculating Channel Matrix" in the
        # implementation notes.
        self.prepareForNextSlot()   # Making sure the Channel Gains, CIR, and Channel Offset are ready for this slot.
        
        # Creating channel matrix using the CIR
        nc,nr,nt,cl = self.cir.shape
        fftWaveform = np.zeros((nc, self.bwp.nFFT, nr*nt), dtype=np.complex128)
        idx = np.append(np.arange(-self.chanOffset,0), np.arange(cl-self.chanOffset))
        if len(idx) > self.bwp.nFFT:
            print(f"WARNING: The delay spread is larger than FFT size! Ignoring larger delays!")
            idx = idx[:self.bwp.nFFT]
            cl = self.bwp.nFFT
            fftWaveform[:,idx,:] = np.transpose(self.cir[...,:cl].reshape(nc, -1, cl), (0,2,1)) # Shape: nc x cl x nr*nt
        else:
            fftWaveform[:,idx,:] = np.transpose(self.cir.reshape(nc, -1, cl), (0,2,1))          # Shape: nc x cl x nr*nt

        chanData = np.fft.fft(fftWaveform, axis=1)                                  # Shape: nc x nFFT x nr*nt

        kk = 12*self.bwp.numRbs
        idx = np.append(np.arange(kk//2)+self.bwp.nFFT-kk//2, np.arange(kk//2))

        channelMatrix = chanData[:,idx,:].reshape(nc, kk, nr, nt) # Shape: nc x kk x nr*nt -> nc x kk x nr x nt

        return channelMatrix

    # ******************************************************************************************************************
    def applyToSignal(self, inputSignal):
        r"""
        This method applies the channel model to the time-domain waveform specified by ``inputSignal`` and returns 
        another :py:class:`~neoradium.waveform.Waveform` object containing the received signal in time domain.

        Parameters
        ----------
        inputSignal : :py:class:`~neoradium.waveform.Waveform`
            The transmitted time-domain waveform. An ``Nt x Ns`` :py:class:`~neoradium.waveform.Waveform` object, 
            where ``Nt`` represents the number of transmit antennas and ``Ns`` denotes the number of time samples in 
            the transmitted waveform.

        Returns
        -------
        :py:class:`~neoradium.waveform.Waveform`
            A :py:class:`~neoradium.waveform.Waveform` object containing the received signal. It is an ``Nr x Ns`` 
            :py:class:`~neoradium.waveform.Waveform` object where ``Nr`` denotes the number of receive antennas and 
            ``Ns`` represents the number of time samples in the received waveform.
        """
        self.prepareForNextSlot()   # Making sure the Channel Gains, CIR, and Channel Offset are ready for this slot.

        slotLen = self.symLens[:-1].sum()       # Note that symLens has the length for the next nc+1 symbols
        ntSig, ns = inputSignal.shape           # Number of TX antennas and input samples from the input signal
        nc, nr, nt, pp = self.chanGains.shape

        if ntSig != nt:
            raise ValueError(f"The number of transmit antennas in the signal does not match the channel.")

        if ns < slotLen:
            raise ValueError(f"The inputSignal is too short. It must be at least {slotLen} samples.")

        # Apply the delays to the signal
        filterOut = []
        signalWaveform = inputSignal if isinstance(inputSignal, np.ndarray) else inputSignal.waveform
        
        for p, pathFilterCoeffs in enumerate(self.coeffMatrix):
            filterOut += [ lfilter(pathFilterCoeffs, 1, signalWaveform.T, 0) ]  # Shape of each pathOut: ns x nt
        filterOutput = np.stack(filterOut,2)                                    # Shape: ns x nt x pp
    
        # Getting gains for all ns samples
        idx = np.concatenate([self.symLens[i]*[i] for i in range(len(self.symLens))])[:ns]  # Shape: ns
        if ns>self.symLens.sum():
            print(f"WARNING: The delays are larger than symbol size! Extending gains to match delays!")
            idx = np.append(idx, (ns-self.symLens.sum())*[len(self.symLens)-1])
        output = (self.chanGains1[idx] * filterOutput[:,None,:,:]).sum((2,3))   # Sum over nt and pp => Shape: ns x nr
        return Waveform(output.T)                                               # Shape: nr x ns

    # ******************************************************************************************************************
    def getChannelGains(self):
        r"""
        This function calculates the path gains for the current slot at the beginning of each symbol. The result 
        is a 4-D tensor of shape ``L x Nr x Nt x Np``, where ``L`` represents the number of symbols per slot, 
        ``Nr`` and ``Nt`` indicate the number of receiver and transmitter antennas, respectively, and ``Np`` denotes
        the number of paths. The path gains are normalized based on the ``normalizeOutput`` and ``normalizeGains`` 
        values.

        Returns
        -------
        4-D complex NumPy array
            The path gains as a NumPy array of shape ``L x Nr x Nt x Np``.
        """
        pathGains = self.getPathGains()                                 # nc x nr x nt x pp
        if self.normalizeOutput:
            pathGains /= np.sqrt(self.nrNt[0])                          # Divide by sqrt(nr)
        if self.normalizeGains:
            pathGains /= np.sqrt(toLinear(self.pathPowers).sum())       # Divide by sqrt(sum(clusterPowers))
        return pathGains                                                # nc x nr x nt x pp
 
    # ******************************************************************************************************************
    def applyKFactorScaling(self):  # Not documented
        # This function applies the K-Factor Scaling. This should be called only for profiles with
        # LOS paths and only when the K-Factor scaling is enabled (kFactor is not None).
        # See TR 38.901 - Sec. 7.7.6 K-factor for LOS channel models
        assert self.hasLos
        assert self.kFactor is not None
        
        # Linear Power Values
        powers = toLinear(self.pathPowers)
        kModel = toDb(powers[0]/powers[1:].sum())                               # TR 38.901 - Eq. 7.7.6-2
        
        # Note that we change the power only for Laplacian Clusters (CDL) or Rayleigh fading taps (TDL)
        self.pathPowers[1:] = self.pathPowers[1:] - self.kFactor + kModel       # TR 38.901 - Eq. 7.7.6-1

        # Now we need to re-normalize the delay spreads
        # Calculate weighted RMS
        powerDelay = powers * self.pathDelays
        sumP = powers.sum()
        rms =  np.sqrt( np.square(powerDelay).sum()/sumP - np.square( powerDelay.sum()/sumP ) )
        self.pathDelays /= rms
