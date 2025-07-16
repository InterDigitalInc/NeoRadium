# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``waveform.py`` implements the :py:class:`Waveform` class which encapsulates a time-domain signal
transmitted from a set of transmitter antennas or received by a set of receiver antennas. A waveform object is
usually created by applying OFDM modulation to a resource grid object. See :py:meth:`~neoradium.grid.Grid.ofdmModulate`
method of the :py:class:`~neoradium.grid.Grid` class for more information.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 12/23/2023    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************

import numpy as np
from .random import random
from .utils import toLinear

# **********************************************************************************************************************
class Waveform:
    r"""
    This class encapsulates a set of sequences of complex values representing the time-domain signals as transmitted
    by each transmitter antenna or as received by each receiver antenna. A Waveform object is usually created by
    applying OFDM modulation to a resource grid.
    
    Once you have a Waveform object, you can apply a channel model to it, add AWGN noise to it, or apply other signal
    processing tasks such as *windowing*. All of these processes result in new ``Waveform`` objects.
    
    At the receiver the received signals are usually converted back to the frequency domain by applying OFDM 
    demodulation, which results in a :py:class:`~neoradium.grid.Grid` object representing the received resource grid.
    """
    # ******************************************************************************************************************
    def __init__(self, waveform, noiseVar=0):
        r"""
        Parameters
        ----------
        waveform : 2D complex numpy array
            A ``P x Ns`` 2D complex numpy array representing a set of time-domain signals of length ``Ns`` for each
            one of ``P`` antenna elements. The value ``P`` is equal to ``Nt``, the number of transmitter antennas when
            this is a transmitted signal, and equal to ``Nr``, the number of receiver antennas when this is a received 
            signal.
            
        noiseVar : float
            The variance of the noise applied to the time-domain signals in this Waveform object. This is usually
            initialized to zero. When an AWGN noise is applied to the waveform using the :py:meth:`addNoise` function, 
            the variance of the noise is saved in the Waveform object.


        **Other Read-Only Properties:**
        
            :shape: Returns the shape of the 2-dimensional waveform numpy array.
            :numPorts: The number of transmitter or receiver antennas (``P``) for this waveform.
            :length: The length of the time-domain signal in number of samples (``Ns``).
        """
        self.waveform = waveform
        self.noiseVar = noiseVar

    # ******************************************************************************************************************
    @property
    def shape(self):            return self.waveform.shape
    @property
    def numPorts(self):         return self.waveform.shape[0]
    @property
    def length(self):           return self.waveform.shape[1]

    # ******************************************************************************************************************
    def __getitem__(self, key): return self.waveform[key]

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this Waveform object.

        Parameters
        ----------
        indent: int
            The number of indentation characters.
            
        title: str
            If specified, it is used as a title for the printed information.

        getStr: Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Waveform Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  Number of Ports: %d\n"%(self.numPorts)
        repStr += indent*' ' + "  Length: %d\n"%(self.length)
        if self.noiseVar>0:
            repStr += indent*' ' + "  Noise Var.: %s\n"%(str(self.noiseVar))
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def addNoise(self, **kwargs):
        r"""
        Adds Additive White Gaussian Noise (AWGN) to this waveform based on the given noise properties. The *noisy*
        waveform is then returned in a new :py:class:`Waveform` object.
        
        This is to some extent similar to the :py:meth:`~neoradium.grid.Grid.addNoise` method of the 
        :py:class:`~neoradium.grid.Grid` class which applies noise in the frequency domain.
        
        Parameters
        ----------
        kwargs: dict
            One of the following parameters, which specify how the noise signal is generated, **must** be specified.
            
            :noise: A numpy array with the same shape as this :py:class:`Waveform` object containing the noise
                information. If the noise information is provided by ``noise``, it is added directly to the waveform. In
                this case all other parameters are ignored.
            
            :noiseStd: The standard deviation of the noise. An AWGN complex noise signal is generated with zero mean
                and the specified standard deviation. If ``noiseStd`` is specified, ``noiseVar``, ``snrDb``, and 
                ``nFFT`` values below are ignored.

            :noiseVar: The variance of the noise. An AWGN complex noise signal is generated with zero mean and the
                specified variance. If ``noiseVar`` is specified, the values of ``snrDb`` and ``nFFT`` are ignored.

            :snrDb: The signal to noise ratio in dB. First the noise variance is calculated using the given SNR 
                value and the ``nFFT`` value. Then an AWGN complex noise signal is generated with zero mean and
                the calculated variance. Please note that if an SNR value is used to specify the amount of noise, the
                value of ``nFFT`` should also be provided.
                
            :nFFT: This is only used if ``snrDb`` is specified. It is the length of Fast Fourier Transform that is 
                applied to the signals for time/frequency domain conversion. This value is usually available from the
                :py:class:`~neoradium.carrier.BandwidthPart` object being used for the transmission. This function 
                uses the following formula to calculate the noise variance :math:`\sigma^2_{AWGN}` from :math:`snrDb`
                and :math:`nFFT` values:
                
                .. math::

                    \sigma^2_{AWGN} = \frac 1 {N_r.nFFT.10^{\frac {snrDb} {10}}}

                where :math:`N_r` is the number of receiver antennas.

            :ranGen: If provided, it is used as the random generator
                for the AWGN generation. Otherwise, if this is not specified, **NeoRadium**'s :doc:`global random
                generator <./Random>` is used.

        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object containing the *noisy* version of this waveform.
        """
        noise = kwargs.get('noise', None)
        if noise is not None:
            if self.shape != noise.shape:
                raise ValueError("Shape mismatch: Waveform: %s vs Noise: %s"%(str(self.shape), str(noise.shape)))
            return Waveform(self.waveform+noise, noise.var())
        
        ranGen = kwargs.get('ranGen', random)       # The Random Generator
        noiseStd = kwargs.get('noiseStd', None)
        if noiseStd is not None:
            noise = ranGen.awgn(self.shape, noiseStd)
            return Waveform(self.waveform+noise, noiseStd*noiseStd)

        noiseVar = kwargs.get('noiseVar', None)
        if noiseVar is not None:
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        # NOTE: To add noise to a waveform using SNR, you need to specify the nFFT value used for OFDM modulation.
        snrDb = kwargs.get('snrDb', None)
        if snrDb is not None:
            # SNR is the average SNR per RE per RX antenna
            snr = toLinear(snrDb)
            nFFT = kwargs.get('nFFT', None)
            if nFFT is None:
                raise ValueError("When using SNR, you must also specify the FFT size!")
            noiseVar = 1/(snr * self.numPorts * nFFT)  # Note: It is assumed that numPorts is the number of RX antennas
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        raise ValueError("You must specify the noise power using 'snrDb', 'noiseVar', or 'noiseStd'!")

    # ******************************************************************************************************************
    def pad(self, numPad):
        r"""
        Appends a sequence of ``numPad`` zeros to the end of time-domain signals in this :py:class:`Waveform` object.
        
        To make sure a signal is received in its entirety when it goes through a channel model, we usually need to 
        pad zeros to the end of the time-domain signal. The number of these zeros usually depends on the maximum 
        channel delay. The function :py:meth:`~neoradium.channelmodel.ChannelModel.getMaxDelay` of the channel model 
        can be used to get the number of padded zeros.
        
        Parameters
        ----------
        numPad: int
            The number of time-domain zero samples to be appended to the end of this waveform.

        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which is ``numPad`` samples longer than the original waveform.
        """
        return Waveform( np.concatenate((self.waveform, np.zeros((self.numPorts, numPad))), axis=1), self.noiseVar )

    # ******************************************************************************************************************
    def sync(self, timingOffset):
        r"""
        Removes ``timingOffset`` values from the beginning of the time-domain signals in this :py:class:`Waveform` 
        object. This effectively shifts the signal in time domain by ``timingOffset`` samples.
        
        When a time-domain signal goes through a channel model, it is delayed in time because of the propagation delay.
        Different transmission paths may be affected by different propagation delays. The channel's ``chanOffset`` 
        member can be used to obtain the ``timingOffset``. In practice, this value is calculated by finding the 
        time-domain sample index where the correlation between the received signal and a set of reference signals is 
        at its maximum. See for example the function :py:meth:`~neoradium.grid.Grid.estimateTimingOffset` of the
        :py:class:`~neoradium.grid.Grid` class.
        
        Parameters
        ----------
        timingOffset: int
            The number of time-domain samples that are removed from the beginning of the time-domain signals in this
            :py:class:`Waveform` object.


        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which is ``timingOffset`` samples shorter than the original waveform.
        """
        return Waveform(self.waveform[:,timingOffset:], self.noiseVar)

    # ******************************************************************************************************************
    def applyChannel(self, channel):
        r"""
        Applies the channel model ``channel`` to this Waveform object and returns a new Waveform object representing
        the received signal. This function internally calls the 
        :py:meth:`~neoradium.channelmodel.ChannelModel.applyToSignal` method of the channel model passing in this 
        waveform object as the ``inputSignal``.
        
        Parameters
        ----------
        channel: :py:class:`~neoradium.channelmodel.ChannelModel`
            The channel model that is applied to this time-domain waveform.

        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which represents the received time-domain
            waveform.
        """
        return channel.applyToSignal(self)

    # ******************************************************************************************************************
    @classmethod
    def getWindowingSize(cls, cpLen, bwp):          # Not documented
        # This function is based on tables in section F.5 in both TS 38.101-1 and TS 38.101-2 V18.4.0 (2023-12)
        # In all the related tables for Normal CP the window size is half the CP length.
        if bwp.cpType == 'normal':  return (cpLen+1)//2

        # For extended, we need to use Table F.5.4-1 in TS 38.101-1 and TS 38.101-2
        # We only need the table when the ratio is not 85.9%. The dict below is CP-Len -> W from tables
        # above for cases where the ratio is not 85.9%
        winTable = { 64: 54, 96:80, 128:106, 192:164 }
        if cpLen in winTable:       return winTable[cpLen]
        # If not in the table, we assume ratio of 85.9%
        return int(np.round(cpLen*0.859))

    # ******************************************************************************************************************
    def applyWindowing(self, cpLens, windowing, bwp):
        r"""
        This is a helper function that is used to apply *windowing* to the OFDM waveform obtained from OFDM modulation
        of a resource grid.

        This method supports several different windowing approaches including the ones specified in **3GPP TS 38.104,
        Sections B.5.2 and C.5.2**.
        
        You usually do not need to call this function directly. It is called internally at the end of the OFDM
        modulation process when the function :py:meth:`~neoradium.grid.Grid.ofdmModulate` of the
        :py:class:`~neoradium.grid.Grid` class is called.
        
        Parameters
        ----------
        cpLens: list
            A list of integer values each representing the length of cyclic prefix part at the beginning of each OFDM
            symbol in number of time-domain samples. This list can be obtained from the 
            :py:class:`~neoradium.carrier.BandwidthPart` object.
            
        windowing: str
            A text string specifying how the window length is obtained. It can be one of the following:
            
            :"STD": The windowing size is determined based on **3GPP TS 38.104, Sections B.5.2 and C.5.2**.
            :Ratio as percentage: A windowing ratio can be specified as a percentage value. For example, the text 
                string "%25" represents a windowing ratio of ``0.25``. The window length is calculated as the minimum 
                value of ``cpLens`` multiplied by the windowing ratio, and rounded to the nearest integer value.
            :Ratio: A windowing ratio (between 0 and 1) can be specified as a number.
                For example, the text string "0.125" represents a windowing ratio of ``0.125``. The window length is
                calculated as the minimum value of ``cpLens`` multiplied by the windowing ratio and rounded to the
                nearest integer value.
            :Window Length: The actual window length can also be specified as an integer value. For example, the text 
                string "164" represents a window length equal to ``164``.

        bwp: :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part used for the communication.
            
        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which represents the waveform after applying the windowing.
        """
        # This is a helper class method that is used to apply windowing to the OFDM waveform after OFDM modulation
        if "%" in windowing:                # Percentage is given
            windowingRatio = np.float64( windowing.replace('%','') )/100.0
            windowLen = min([int(.5 + windowingRatio*cpLen) for cpLen in cpLens])
        elif "." in windowing:              # Ratio is given (Must be between 0 and 1)
            windowingRatio = np.float64( windowing )
            if windowingRatio<0 or windowingRatio>1: raise ValueError("The windowing ratio must be between 0 and 1")
            windowLen = min([int(.5 + windowingRatio*cpLen) for cpLen in cpLens])
        elif windowing.upper() == "STD":
            # Use standard, can have different windowing lengths for different symbols based on cpLens
            windowLen = min([Waveform.getWindowingSize(cpLen, bwp) for cpLen in cpLens])
        else:
            # A fixed integer value is given (no '.'s included in the windowing text)
            windowLen = int(windowing)
            if windowLen >= min(cpLens):            raise ValueError("The windowing size must be smaller than CP size")
        
        # Note: We want to keep the waveform length unchanged. Therefore we are applying an 'overlapping' windowing
        outputWaveForm = np.zeros_like(self.waveform)

        # This is a sequence of 'windowLen' values monotonically increasing from 0 to 1
        raisedCosine = (.5*(1-np.sin(np.pi*np.arange(windowLen-1,-windowLen,-2)/(2*windowLen))))[None,:] # 1 x windowLen

        symStart = 0
        ll = len(cpLens)
        for s, cpLen in enumerate(cpLens):
            symLen = cpLen + bwp.nFFT
            symWaveForm = self.waveform[:,symStart:symStart+symLen]
                        
            # Extend symbol waveForm by inserting a copy of 'windowLen' samples from near the end (before the already
            # copied CP) to the start. shape: pp x (symLen+windowLen)
            symWaveFormEx = np.concatenate( (symWaveForm[:,bwp.nFFT-windowLen:bwp.nFFT], symWaveForm), axis=1)
            
            # Apply the 'raisedCosine' to the first 'windowLen' samples of this symbol
            symWaveFormEx[:,:windowLen] *= raisedCosine
            
            # Apply the 'raisedCosine' in reverse order to the last 'windowLen' samples of this symbol
            symWaveFormEx[:,-windowLen:] *= raisedCosine[:,::-1]

            if s<(ll-1):
                symLenEx = symLen + windowLen
                outputWaveForm[:, symStart:symStart+symLenEx] += symWaveFormEx
            else:
                # We are at the last symbol of the grid
                outputWaveForm[:, symStart:symStart+symLen] += symWaveFormEx[:,:symLen]
                outputWaveForm[:, :windowLen] += symWaveFormEx[:,-windowLen:]
            
            symStart += symLen
            
        outputWaveForm = np.roll(outputWaveForm, -windowLen, axis=1)
        return Waveform(outputWaveForm)

    # ******************************************************************************************************************
    def ofdmDemodulate(self, bwp, f0=0, cpOffsetRatio=0.5):
        r"""
        Applies OFDM demodulation to the waveform which results in a frequency-domain resource grid returned as a
        :py:class:`~neoradium.grid.Grid` object.

        If an AWGN noise was applied to the waveform using the :py:meth:`addNoise` method, then the amount of noise
        is transferred to the :py:class:`~neoradium.grid.Grid` object that is created. The noise variance of
        the returned resource grid is equal to the waveform's noise variance times ``nFFT``.
       
        Parameters
        ----------
        bwp: :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part used for the communication.
            
        f0: float
            The carrier frequency of the waveform. If it is 0 (default), then a baseband waveform is assumed. This
            should match the value originally used when applying OFDM modulation at the transmitter side. See the
            :py:meth:`~neoradium.grid.Grid.ofdmModulate` method of the :py:class:`~neoradium.grid.Grid` class.

        cpOffsetRatio: float
            This value determines where, in the cyclic prefix (as a ratio from the beginning of the CP), the FFT 
            should be applied. The default value of ``0.5`` means that the FFT is applied at the midpoint of the 
            cyclic prefix."
            
        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A :py:class:`~neoradium.grid.Grid` object representing the received resource grid.
        """
        # Grid is only needed in this method. Importing it here helps avoid "Circular import" with Grid<->Waveform
        from .grid import Grid
        
        symLens = bwp.getSymLens()[:-1]                         # getSymLens returns symbolsPerSlot+1 values
        cpLens = symLens-bwp.nFFT                               # symbolsPerSlot values
        cpStarts = np.cumsum(np.append(0,symLens[:-1]))         # symbolsPerSlot values
        fftStarts = np.int32(np.round(cpLens * cpOffsetRatio))  # FFT start offset from the cpStarts
        idx = (cpLens[:,None] - fftStarts[:,None] + np.arange(bwp.nFFT))%bwp.nFFT + fftStarts[:,None] + cpStarts[:,None]
        fftWaveform = self.waveform[ :, idx ]
    
        gridData = np.fft.fft(fftWaveform, axis=2)          # Shape: nr x ll x nFFT
        gridData = np.fft.fftshift(gridData, axes=2)        # Shape: nr x ll x nFFT

        # Get the kk values in the middle of nFFT sample resulted from FFT
        kk = 12*bwp.numRbs
        k0 = bwp.nFFT//2 - kk//2
        idx = range(k0, k0+kk)  # The 'kk' indices from the total 'nFFT' samples that we are interested in

        grid = Grid(bwp, numPlanes=self.shape[0])
        grid.grid = gridData[:,:,idx]                       # Shape: nr x ll x kk
        grid.reTypeIds = np.ones(grid.shape, dtype=np.uint8)*grid.retNameToId["RX_DATA"]
        grid.noiseVar = self.noiseVar * bwp.nFFT            # Convert the noise variance from time to frequency domain

        symStarts = cpStarts + cpLens
        if f0>0: grid.grid *= np.exp(2j*np.pi*f0*symStarts/bwp.sampleRate ).reshape(1,-1,1)
        return grid

