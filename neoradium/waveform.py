# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``waveform.py`` implements the :py:class:`Waveform` class which encapsulates
a time-domain signal transmitted from a set of transmitter antenna or received by a set
of receiver antenna. A waveform object is usually created by applying OFDM modulation to
a resource grid object. See :py:meth:`~neoradium.grid.Grid.ofdmModulate` method of
the :py:class:`~neoradium.grid.Grid` class for more information.
"""
# ****************************************************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 12/23/2023    Shahab Hamidi-Rad       Completed the documentation
# ****************************************************************************************************************************************************

import numpy as np

from .random import random

# ****************************************************************************************************************************************************
windowingTables = { # For PDSCH, the UE is the receiver, we should look at TS 38.101-1 and TS 38.101-2
               '15N': [ #  TS 38.101-1 V17.0.0 (2020-12), Table F.5.3-1
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [5,         512,       36,        18],
                          [10,        1024,      72,        36],
                          [15,        1536,      108,       54],
                          [20,        2048,      144,       72],
                          [25,        2048,      144,       72],
                          [30,        3072,      216,       108],
                          [40,        4096,      288,       144],
                          [50,        4096,      288,       144]] ],

               '30N': [ #  TS 38.101-1 V17.0.0 (2020-12), Table F.5.3-2 in TS 38.101-1
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [5,         256,       18,        9],
                          [10,        512,       36,        18],
                          [15,        768,       54,        27],
                          [20,        1024,      72,        36],
                          [25,        1024,      72,        36],
                          [30,        1536,      108,       54],
                          [40,        2048,      144,       72],
                          [50,        2048,      144,       72],
                          [60,        3072,      216,       108],
                          [70,        3072,      216,       108],
                          [80,        4096,      288,       144],
                          [90,        4096,      288,       144],
                          [100,       4096,      288,       144]] ],

               '60N': [ #  TS 38.101-1 V17.0.0 (2020-12), Table F.5.3-3
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [10,        256,       18,        9],
                          [15,        384,       27,        14],
                          [20,        512,       36,        18],
                          [25,        512,       36,        18],
                          [30,        768,       54,        27],
                          [40,        1024,      72,        36],
                          [50,        1024,      72,        36],
                          [60,        1536,      108,       54],
                          [70,        1536,      108,       54],
                          [80,        2048,      144,       72],
                          [90,        2048,      144,       72],
                          [100,       2048,      144,       72]],
                        #  TS 38.101-2 V17.0.0 (2020-12), Table F.5.3-1
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [50,        1024,      72,        36],
                          [100,       2048,      144,       72],
                          [200,       4096,      288,       144]] ],

               '60E': [ #  TS 38.101-1 V17.0.0 (2020-12), Table F.5.4-1
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [10,        256,       64,        54],
                          [15,        384,       96,        80],
                          [20,        512,       128,       106],
                          [25,        512,       128,       110],
                          [30,        768,       192,       164],
                          [40,        1024,      256,       220],
                          [50,        1024,      256,       220],
                          [60,        1536,      384,       330],
                          [70,        1536,      384,       330],
                          [80,        2048,      512,       440],
                          [90,        2048,      512,       440],
                          [100,       2048,      512,       440]],
                        #  TS 38.101-2 V17.0.0 (2020-12), Table F.5.4-1
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [50,        1024,      256,       220],
                          [100,       2048,      512,       440],
                          [200,       4096,      1024,      880]] ],

               '120N':[ #  TS 38.101-2 V17.0.0 (2020-12), Table F.5.3-2
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [50,        512,       36,        18],
                          [100,       1024,      72,        36],
                          [200,       2048,      144,       72],
                          [400,       4096,      288,       144]] ],
                          
               '240N':[ #  Could not find anything for this in the standards!
                        #  Bandwidth  FFT Size   CP-LEN     W
                        [ [100,        512,      36,        18],
                          [200,       1024,      72,        36],
                          [400,       2048,      144,       72],
                          [800,       4096,      288,       144]] ] }


# ****************************************************************************************************************************************************
class Waveform:
    r"""
    This class encapsulates a set of sequences of complex values representing
    the time-domain signals as transmitted by each transmitter antenna or as
    received by each receiver antenna. A Waveform object is usually created by
    applying OFDM modulation to a resource grid.
    
    Once you have a Waveform object, you can apply a channel model to it, add AWGN
    noise to it, or apply other signal processing tasks such as *windowing*. All of
    these processes result in new Waveform objects.
    
    At the receiver the received signals are usually converted back to the frequency
    domain by applying OFDM demodulation, which results in a :py:class:`~neoradium.grid.Grid`
    object representing the received resource grid.
    """
    # ************************************************************************************************************************************************
    def __init__(self, waveform, noiseVar=0):
        r"""
        Parameters
        ----------
        waveform : 2D complex numpy array
            An ``P x Ns`` 2D complex numpy array representing a set of time-domain signals
            of length ``Ns`` for each one of ``P`` antenna elements. The value ``P`` is
            equal to ``Nt`` the number of transmitter antenna when this is a transmitted
            signal and equal to ``Nr`` the number of receiver antenna when this is a
            received signal.
            
        noiseVar : float (default: 0)
            The variance of the noise applied to the time-domain signals in this Waveform
            object. This is usually initialized to zero. When an AWGN noise is applied to
            the waveform using the :py:meth:`addNoise` function, the variance of the noise
            is saved in the Waveform object.


        **Other Read-Only Properties:**
        
            :shape: Returns the shape of the 2-dimensional waveform numpy array.

            :numPorts: The number of transmitter or receiver antenna (``P``) for this waveform.

            :length: The length of the time-domain signal in number of samples (``Ns``).
        """
        self.waveform = waveform
        self.noiseVar = noiseVar

    # ************************************************************************************************************************************************
    @property
    def shape(self):            return self.waveform.shape
    @property
    def numPorts(self):         return self.waveform.shape[0]
    @property
    def length(self):           return self.waveform.shape[1]

    # ************************************************************************************************************************************************
    def __getitem__(self, key): return self.waveform[key]

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this Waveform object.

        Parameters
        ----------
        indent: int (default: 0)
            The number of indentation characters.
            
        title: str or None (default: None)
            If specified, it is used as a title for the printed information.

        getStr: Boolean (default: False)
            If ``True``, it returns the information in a text string instead
            of printing the information.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns
            the information in a text string. Otherwise, nothing is returned.
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

    # ************************************************************************************************************************************************
    def addNoise(self, **kwargs):
        r"""
        Adds Additive White Gaussian Noise (AWGN) to this waveform based on
        the given noise properties. The *noisy* waveform is then returned in a new
        :py:class:`Waveform` object.
        
        This is to some extent similar to the :py:meth:`~neoradium.grid.Grid.addNoise`
        method of the :py:class:`~neoradium.grid.Grid` class which applies noise
        in frequency domain.
        
        Parameters
        ----------
        kwargs: dict
            One of the following parameters **must** be specified. They specify
            how the noise signal is generated.
            
            :noise: A numpy array with the same shape as this :py:class:`Waveform`
                object containing the noise information. If the noise information
                is provided by ``noise`` it is added directly to the waveform. In
                this case all other parameters are ignored.
            
            :noiseStd: The standard deviation of the noise. An AWGN complex noise
                signal is generated with zero-mean and the specified standard
                deviation. If ``noiseStd`` is specified, ``noiseVar``, ``snrDb``,
                and ``nFFT`` values below are ignored.

            :noiseVar: The variance of the noise. An AWGN complex noise
                signal is generated with zero-mean and the specified variance. If
                ``noiseVar`` is specified, the values of ``snrDb`` and ``nFFT``
                are ignored.

            :snrDb: The signal to noise ratio in dB. First the noise variance
                is calculated using the given SNR value and the ``nFFT`` value.
                Then an AWGN complex noise signal is generated with zero-mean and
                the calculated variance. Please note that if an SNR value is used
                to specify the amount of noise, the value of ``nFFT`` should also
                be provided.
                
            :nFFT: This is only used if ``snrDb`` is specified. It is the length of
                Fast Fourier Transform that is applied to the signals for time/frequency
                domain conversion. This value is usually available from the
                :py:class:`~neoradium.carrier.BandwidthPart` object being used for
                the transmission. This function uses the following formula to calculate
                the noise variance :math:`\sigma^2_{AWGN}` from :math:`snrDb` and
                :math:`nFFT` values:
                
                .. math::

                    \sigma^2_{AWGN} = \frac 1 {N_r.nFFT.10^{\frac {snrDb} {10}}}

                where :math:`N_r` is the number of receiver antenna.

            :ranGen: If provided, it is used as the random generator
                for the AWGN generation. Otherwise, if this is not specified,
                **NeoRadium**'s :doc:`global random generator <./Random>` is
                used.

        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object containing the *noisy* version of this waveform.
        """
        noise = kwargs.get('noise', None)
        if noise is not None:
            if self.shape != noise.shape:
                raise ValueError("The waveform shape %s does not match the noise shape %s!"%(str(self.shape), str(noise.shape)))
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
            snr = 10**(snrDb/10)
            nFFT = kwargs.get('nFFT', None)
            if nFFT is None:
                raise ValueError("When using SNR, you must also specify the FFT size!")
            noiseVar = 1/(snr * self.numPorts * nFFT)  # Note: It is assumed that numPorts is the number of RX antenna
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        raise ValueError("You must specify the noise power using 'snrDb', 'noiseVar', or 'noiseStd'!")

    # ************************************************************************************************************************************************
    def pad(self, numPad):
        r"""
        Appends a sequence of ``numPad`` zeros to the end of time-domain signals
        in this :py:class:`Waveform` object.
        
        To make sure a signal is received in its entirety when it goes through a
        channel model, we usually need to pad zeros to the end of the time-domain
        signal. The number of these zeros usually depend on the maximum channel
        delay. The function :py:meth:`~neoradium.channel.ChannelBase.getMaxDelay`
        of the channel model can be used to get the number of padded zeros.
        
        Parameters
        ----------
        numPad: int
            The number of time-domain zero samples to be appended to the end of this
            waveform.

        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which is ``numPad`` samples longer than the original
            waveform.
        """
        return Waveform( np.concatenate((self.waveform, np.zeros((self.numPorts, numPad))), axis=1), self.noiseVar )

    # ************************************************************************************************************************************************
    def sync(self, timingOffset):
        r"""
        Removes ``timingOffset`` values from the beginning of the time-domain signals
        in this :py:class:`Waveform` object. This effectively shifts the signal in
        time domain by ``timingOffset`` samples.
        
        When a time-domain signal goes through a channel model, it is delayed in time
        because of the propagation delay. Different transmission paths may be affected
        by different propagation delays. The function
        :py:meth:`~neoradium.channel.ChannelBase.getTimingOffset` can be used to
        obtain the overall propagation delay given by ``timingOffset``. In practice
        this value is calculated by finding the time-domain sample index where the
        correlation between the received signal and a set of reference signals is
        at its maximum. See for example the function
        :py:meth:`~neoradium.grid.Grid.estimateTimingOffset` of the
        :py:class:`~neoradium.grid.Grid` class.
        
        Parameters
        ----------
        timingOffset: int
            The number of time-domain samples that are removed from the beginning of
            the time-domain signals in this :py:class:`Waveform` object.


        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which is ``timingOffset`` samples shorter than the
            original waveform.
        """
        return Waveform(self.waveform[:,timingOffset:], self.noiseVar)

    # ************************************************************************************************************************************************
    def applyChannel(self, channel):
        r"""
        Applies the channel model ``channel`` to this Waveform object and returns a
        new Waveform object representing the received signal. This function internally
        calls the :py:meth:`~neoradium.channel.ChannelBase.applyToSignal` method
        of the channel model passing in this waveform object as the ``inputSignal``.
        
        Parameters
        ----------
        channel: :py:class:`~neoradium.channel.ChannelBase`
            The channel model that is applied to this time-domain waveform.

        Returns
        -------
        :py:class:`Waveform`
            A new Waveform object which represents the received time-domain
            waveform.
        """
        return channel.applyToSignal(self)

    # ************************************************************************************************************************************************
    @classmethod
    def getWindowingSize(cls, cpLen, bwp):          # Not documented
        # This is a helper class method that finds the windowing size for the OFDM
        # modulation using the tables in TS 38.104 V17.4.0 (2021-12), Sections B.5.2
        # and C.5.2.
        tablesKey = "%d%s"%(bwp.spacing, bwp.cpType[0].upper())
        bw = bwp.bandwidth
        
        # In first attempt, we first find all the tables matching the tableKey, then
        # find the all the Ws in the rows matching 'cpLen' and 'bw'. Then we pick the
        # smallest 'W' in the list of found values.
        windowingValues = []
        for table in windowingTables[tablesKey]:
            rowIndexes = [ i for i in range(len(table)) if table[i][2]==cpLen ]

            if len(rowIndexes)==0:  continue        # No match in this table.
                
            if len(rowIndexes)==1:                  # Only one match
                windowingValues += [ table[ rowIndexes[0] ][3] ]
                continue
                
            # We have more than one match for 'cpLen'. Pick the one closest to our bandwidth
            minDiffIdx = np.argmin([ np.abs(table[i][0]*1e6-bw) for i in rowIndexes ])
            windowingValues += [ table[ rowIndexes[minDiffIdx] ][3] ]
            
        if len(windowingValues)>0:  return min(windowingValues)
        
        # If no exact match was found in our first attempt, we try to find the
        # closest the 'W' with closest 'cpLen' and 'bw'
        for table in windowingTables[tablesKey]:
            minDiffIdx = np.argmin([ np.abs(table[i][2]-cpLen) for i in range(len(table)) ])
            closestCp = table[minDiffIdx][2]
            rowIndexes = [ i for i in range(len(table)) if table[i][2]==closestCp ]
            if len(rowIndexes)==1:
                windowingValues += [ int( 0.5 + table[ rowIndexes[0] ][3]*cpLen/closestCp) ]
                continue

            # We have more than one match for the closest 'cpLen'. Pick the one closest to our bandwidth
            minDiffIdx = np.argmin([ np.abs(table[i][0]*1e6-bw) for i in rowIndexes ])
            windowingValues += [ int( 0.5 + table[ rowIndexes[minDiffIdx] ][3]*cpLen/closestCp) ]

        return min(windowingValues)

    # ************************************************************************************************************************************************
    def applyWindowing(self, cpLens, windowing, bwp):
        r"""
        This is a helper function that is used to apply *windowing* to the OFDM
        waveform obtained from OFDM modulation of a resource grid.

        This method supports several different windowing approaches including the
        ones specified in **3GPP TS 38.104, Sections B.5.2 and C.5.2**.
        
        You usually do not need to call this function directly. It is called
        internally at the end of the OFDM modulation process when the function
        :py:meth:`~neoradium.grid.Grid.ofdmModulate` of the
        :py:class:`~neoradium.grid.Grid` class is called.
        
        Parameters
        ----------
        cpLens: list
            A list of integer values each representing the length of cyclic prefix
            part at the beginning of each OFDM symbol in number of time-domain samples.
            This list can be obtained from the :py:class:`~neoradium.carrier.BandwidthPart`
            object.
            
        windowing: str
            A text string specifying how the window length is obtained. It can be one
            the following:
            
            :"STD": The windowing size is determined based on **3GPP TS 38.104, Sections
                B.5.2 and C.5.2**.
            :Ratio as percentage: A windowing ratio can be specified as a percentage value.
                For example the text string "%25" represents a windowing ratio of ``0.25``.
                The window length is calculated as the minimum value of ``cpLens`` multiplied
                by the windowing ratio and rounded to the nearest integer value.
            :Ratio: A windowing ratio (between 0 and 1) can be specified as a number.
                For example the text string "0.125" represents a windowing ratio of ``0.125``.
                The window length is calculated as the minimum value of ``cpLens`` multiplied
                by the windowing ratio and rounded to the nearest integer value.
            :Window Length: The actual window length can also be specified as an integer value.
                For example the text string "164" represents a window length equal to ``164``.

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
            if windowingRatio<0 or windowingRatio>1:    raise ValueError("The windowing ratio must be between 0 and 1")
            windowLen = min([int(.5 + windowingRatio*cpLen) for cpLen in cpLens])
        elif windowing.upper() == "STD":    # Use standard, can have different windowing lengths for different symbols based on cpLens
            windowLen = min([Waveform.getWindowingSize(cpLen, bwp) for cpLen in cpLens])
        else:                               # A fixed integer value is given (no '.'s included in the windowing text)
            windowLen = int(windowing)
            if windowLen >= min(cpLens):                raise ValueError("The windowing size must be smaller than CP size")
        
        # Note: We want to keep the waveform length unchanged. Therefore we are applying an 'overlapping' windowing
        outputWaveForm = np.zeros_like(self.waveform)

        # This is a sequence of 'windowLen' values monotonically increasing from 0 to 1
        raiseCosine = (.5*(1-np.sin(np.pi*np.arange(windowLen-1,-windowLen,-2)/(2*windowLen)))).reshape(1,-1)   # Shape: 1 x windowLen

        symStart = 0
        ll = len(cpLens)
        for s, cpLen in enumerate(cpLens):
            symLen = cpLen + bwp.nFFT
            symWaveForm = self.waveform[:,symStart:symStart+symLen]
                        
            # Extend symbol waveForm by inserting a copy of 'windowLen' samples from near the end (before the already copied CP) to the start
            symWaveFormEx = np.concatenate( (symWaveForm[:,bwp.nFFT-windowLen:bwp.nFFT], symWaveForm), axis=1)  # shape: pp x (symLen+windowLen)
            symWaveFormEx[:,:windowLen] *= raiseCosine              # Apply the 'raiseCosine' to the first 'windowLen' samples of this symbol
            symWaveFormEx[:,-windowLen:] *= raiseCosine[:,::-1]     # Apply the 'raiseCosine' in reverse order to the last 'windowLen' samples of this symbol

            if s<(ll-1):
                symLenEx = symLen + windowLen
                outputWaveForm[:, symStart:symStart+symLenEx] += symWaveFormEx
            else:   # We are at the last symbol of the grid
                outputWaveForm[:, symStart:symStart+symLen] += symWaveFormEx[:,:symLen]
                outputWaveForm[:, :windowLen] += symWaveFormEx[:,-windowLen:]
            
            symStart += symLen
            
        outputWaveForm = np.roll(outputWaveForm, -windowLen, axis=1)
        return Waveform(outputWaveForm)

    # ************************************************************************************************************************************************
    def ofdmDemodulate(self, bwp, f0=0, cpOffsetRatio=0.5):
        r"""
        Applies OFDM demodulation to the waveform which results in a frequency-domain
        resource grid returned as a :py:class:`~neoradium.grid.Grid` object.

        If an AWGN noise was applied to the waveform using the :py:meth:`addNoise`
        method, then the amount of noise is transferred to the
        :py:class:`~neoradium.grid.Grid` object that is created. The noise variance of
        the returned resource grid is equal to the waveform's noise variance times
        ``nFFT``.
       
        Parameters
        ----------
        bwp: :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part used for the communication.
            
        f0: float (default: 0)
            The carrier frequency of the waveform. If it is 0 (default), then a baseband
            waveform is assumed. This should match the value originally used when applying
            OFDM modulation at the transmitter side. See the
            :py:meth:`~neoradium.grid.Grid.ofdmModulate` method of the :py:class:`~neoradium.grid.Grid`
            class.

        cpOffsetRatio: float (default: 0.5)
            This value determines where in the cyclic prefix (Ratio from the beginning of CP),
            should we start applying FFT. The default value of ``0.5`` means the starting sample
            for applying FFT is the mid point in the cyclic prefix.
            
        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A :py:class:`~neoradium.grid.Grid` object representing the received resource grid.
        """
        from .grid import Grid  # Grid is only needed in this method. Importing it here helps avoid "Circular import" with Grid<->Waveform
        nr, ns = self.shape
        numSlots = bwp.getNumSlotsForSamples(ns)
        ll =  numSlots * bwp.symbolsPerSlot
        kk = 12*bwp.numRbs
        nFFT = bwp.nFFT
        
        if (cpOffsetRatio<=0) or (cpOffsetRatio>=1.0):      raise ValueError("\"cpOffsetRatio\" must be between 0 and 1!")
        if f0<0:                                            raise ValueError("\"f0\" must be a non-negative value!")

        l0 = bwp.slotNoInSubFrame * bwp.symbolsPerSlot      # Number of symbols from start of this subframe
        maxL = bwp.symbolsPerSubFrame - l0
        if ll > maxL:                                       raise ValueError("Cannot modulate across subframe boundary! (At most %d symbols)"%(maxL))
            
        lRange = np.arange(ll) + l0                         # The grid's symbol indexes from the start of current subframe
        cpLens = [bwp.getCpLen(l) for l in lRange ]         # Number of samples in Cyclic Prefix for each symbol
        
        # Removing CPs:
        symWaveforms = np.zeros((nr, ll, nFFT)) + 0j
        sampleIdx = 0
        for l,cpLen in enumerate(cpLens):
            symLen = cpLen + nFFT
            symOffset = int(cpLen*cpOffsetRatio + .5)
            symIndexes = (np.arange(nFFT) + cpLen - symOffset) % nFFT + symOffset + sampleIdx
            symWaveforms[:,l,:] = self.waveform[ :, symIndexes ]
            sampleIdx += symLen
            
        gridData = np.fft.fft(symWaveforms, axis=2)         # Shape: nr x ll x nFFT
        gridData = np.fft.fftshift(gridData, axes=2)        # Shape: nr x ll x nFFT

        # Calculating the shift related to carrier frequency (f0)
        startTimes = np.cumsum( [0] + [cpLen+nFFT for cpLen in cpLens[:-1]] ) / bwp.sampleRate
        cpTimes = np.array(cpLens) / bwp.sampleRate
        symF0PhaseFactors = np.exp( -2j * np.pi * f0 * (-startTimes-cpTimes) ).reshape(1,-1,1) if f0>0 else 1   # ll values

        k0 = nFFT//2 - kk//2
        idx = range(k0, k0+kk)  # These are the 'kk' indexes of the samples from the total 'nFFT' samples that we are interested in

        grid = Grid(bwp, numPlanes=nr, numSlots=numSlots)
        grid.grid = gridData[:,:,idx] * symF0PhaseFactors           # Shape: nr x ll x kk
        grid.reTypeIds = np.ones(grid.shape, dtype=np.uint8)*grid.retNameToId["RX_DATA"]
        grid.noiseVar = self.noiseVar * bwp.nFFT                    # Convert the noise variance from time to frequency domain
        return grid
