# Copyright (c) 2024 InterDigital AI Lab
"""
This module implements the modulation and demodulation functionality based on **3GPP TR 38.211**. The :py:class:`Modem`
class is implemented to handle modulation and demodulation of bitstreams to and from complex symbols.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 11/03/2023    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np

# **********************************************************************************************************************
# The Modulator/Demodulator class
class Modem:
    r"""
    This class handles the process of modulating a bitstream to an array of complex symbols (Modulation) as 
    well as extracting Log-Likelihood ratios (LLRs) from an array of complex symbols (Demodulation). This 
    implementation is based on **3GPP TR 38.211 section 7.3.1.2**.
    """
    # TS 38.211 V17.0.0 (2021-12), Table 7.3.1.2-1
    mod2qm = {'BPSK':1, 'QPSK':2, '16QAM':4, '64QAM':6, '256QAM':8, '1024QAM':10}

    # ******************************************************************************************************************
    def __init__(self, modulation='QPSK'):
        r"""
        Parameters
        ----------
        modulation: str
            The modulation scheme based on table 7.3.1.2-1 in **3GPP TR 38.211**. Here is a list of supported 
            modulation schemes:
                        
            ===================  =========================
            Modulation Scheme    Modulation Order (qm)
            ===================  =========================
            BPSK                 1
            QPSK                 2
            16QAM                4
            64QAM                6
            256QAM               8
            1024QAM              10
            ===================  =========================


        **Other Properties:**

        In addition to the ``modulation`` parameter, here is a list of additional properties for this class.
        
            :qm: The modulation order. This is the number of bits per modulated symbol. See **3GPP TR 38.211, 
                Table 7.3.1.2-1** for more details.
            :constellation: The modulation constellation. This is a lookup table that converts each group of ``qm`` 
                bits from the input bitstream to a complex symbol.
        """
        self.modulation = modulation
        qm = self.mod2qm[modulation]
        self.qm = qm
        
        scale = 1/np.sqrt({1:2, 2:2, 4:10, 6:42, 8:170, 10:682}[qm])

        # The following function implements the equations in TS 38.211 V17.0.0 (2021-12),
        # sections 5.1.2, 5.1.3, 5.1.4, 5.1.5, 5.1.6, and 5.1.7
        def getConstellationValue(value):
            b = [int(x) for x in ("{0:0%db}"%(qm)).format(value)]
            real,img = 1,1
            for q in range(2,qm,2):
                real = (1<<(q//2)) - (1-2*b[qm-q])*real
                img  = (1<<(q//2)) - (1-2*b[qm+1-q])*img
            real *= 1-2*b[0]
            img *= 1-2*b[min(1,qm-1)]
            return scale*(real + 1j*img)

        self.constellation = np.array([ getConstellationValue(x) for x in range(1<<qm)])
        self.symbolOrder = np.argsort(1000*self.constellation.real - self.constellation.imag)   # Not used
        
        # Get a list of binary representation of all integers from 0 to 2^qm
        allBinaries = np.int8( [list(("{0:0%db}"%(self.qm)).format(i)) for i in range(1<<qm)] ) # Shape: (2^qm, qm)

        # c is a 2 x 2^(qm-1) x qm tensor.
        # c[0,:,i] is a list of indices of constellation points where the i'th bit is 0  i∈{0,...,qm-1}
        # c[1,:,i] is a list of indices of constellation points where the i'th bit is 1  i∈{0,...,qm-1}
        self.c = np.int16([np.stack([np.where(allBinaries[:,i]==bit)[0] for i in range(self.qm)],
                                    axis=1) for bit in [0,1]])                                  # Shape: 2, 2^(qm-1), qm

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`Modem` object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If ``None`` (default), the text 
            "Modem Properties:" is used for the title.

        getStr : Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        repStr = "\n" if indent==0 else ""
        if title is None:   title = "Modem Properties:"
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  Modulation Type ...........: %s\n"%(self.modulation)
        repStr += indent*' ' + "  Qm ........................: %d\n"%(self.qm)
        repStr += indent*' ' + "  Num constellation points ..: %d\n"%(len(self.constellation))
        if self.qm <=4:
            numPerLine = {1:2, 2:2, 4:4}[self.qm]
            repStr += indent*' ' + "  Constellation points ......:\n"
            for i in range(len(self.constellation)//numPerLine):
                repStr += indent*' ' + 20*" " + "%s\n"%("   ".join("%11s"%(str(p)[1:-1]) for p in np.round(self.constellation[i*numPerLine:(i+1)*numPerLine],2)))
            repStr += indent*' ' + "  Symbol Order ..............: %s\n"%(str(self.symbolOrder))

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def modulateOneBlock(self, bitstream):      # Not documented
        # This function modulates a single code block. This is called by the method "modulate" below.
        # See TS 38.211 V17.0.0 (2021-12), Section 7.3.1.2
        qm = self.qm
        if len(bitstream)%qm > 0:
            raise ValueError("The length of 'bitstream' (%d) must be a multiple of 'qm' (%d)!"%(len(bitstream), qm))
        
        # Get symbol Indexes
        symIndexes = ( np.uint16(bitstream).reshape((-1,qm)) * [[1<<(qm-i-1) for i in range(qm)]] ).sum(1)
        symbols = self.constellation[symIndexes]
        return symbols

    # ******************************************************************************************************************
    def modulate(self, bitstreams):
        r"""
        Modulates the given bitstream to an array(s) of complex symbols using current modulation scheme.

        Parameters
        ----------
        bitstreams : numpy array of bits
            A 1-D (one code block) or 2-D (several code blocks) array of bits.

        Returns
        -------
        Numpy array of complex values
            Returns a 1-D or 2-D (depending on shape of ``bitstreams``) numpy complex array of modulated symbols.
        """
        if bitstreams.ndim>1:
            return np.complex128( [ self.modulateOneBlock(bitstream) for bitstream in bitstreams] )
        return self.modulateOneBlock(bitstreams)

    # ******************************************************************************************************************
    def getLLRsFromSymbols(self, symbols, noiseVar, useMax=True):
        r"""
        This function calculates the Log Likelihood Ratios (LLRs) for each bit from the received noisy symbols. The
        LLR values can then be used by :py:class:`PolarDecoder` or :py:class:`LdpcDecoder` to extract the decoded
        bitstream.

        Parameters
        ----------
        symbols : 1-D or 2-D Complex numpy array
            An ``m``x``n`` complex numpy array where ``m`` is the number of coded blocks and ``n`` is the length of
            each code block. If it is a 1-D array, it means there is only one code block to demodulate.
            
        noiseVar : float
            The noise variance obtained using noise estimation or using the actual noise variance value used in
            simulation.

        useMax : Boolean
            If ``True`` (the default), this implementation uses the ``Max`` function in the calculation of the LLR 
            values. This is faster but uses an approximation and is slightly less accurate than the actual Log 
            Likelihood method which uses logarithm and exponential functions. If ``False``, the slower more accurate 
            method is used.

        Returns
        -------
        Numpy array of floating point
            A 1-D or 2-D numpy array of LLR values depending on the dimensionality of ``symbols``. In case of 2-D array,
            the return value is an ``m``x``l`` array of LLR values where ``l= n * qm``. In case of 1-D array, the
            output is a 1-D array of ``l`` LLR values.
        """

        # First calculate all distances:
        d = np.abs(symbols[...,None]-self.constellation)        # shape: symbols.shape + (2^qm,)
        exponents = -d**2/noiseVar
        
        exponents = exponents[...,self.c]                       # A tensor of shape: symbols.shape + (2, 2^(qm-1), qm)
        # exponents[...,0,:,:] for '0' bits
        # exponents[...,1,:,:] for '1' bits

        # Sum or Max over the second from last axis  => Shape of lls: symbols.shape + (2, qm)
        MAX_EXPONENT = 700      # Prevent overflow
        
        # All Log-Likelihood values
        lls = exponents.max(-2) if useMax else np.log(np.exp(np.clip(exponents,-MAX_EXPONENT,MAX_EXPONENT)).sum(-2))
        llrs = lls[...,0,:] - lls[...,1,:]          # The Log-Likelihood Ratio (LLR)
        # llrs shape: symbols.shape + (qm,)
        return llrs.reshape(llrs.shape[:-2]+(-1,))  # Merge last 2 dimensions

    # ******************************************************************************************************************
    def demodulate(self, symbols, noiseVar, useMax=True):
        r"""
        Demodulates the received noisy symbols to a bitstream using "hard decision" to convert Log Likelihood Ratios
        (LLRs) to bits. This function first calls the :py:meth:`Modem.getLLRsFromSymbols` method to get the
        LLR values, and then uses "hard decision" to convert LLRs to bits.

        Parameters
        ----------
        symbols : 2-D complex numpy array
            A ``m``x``n`` complex numpy array where ``m`` is the number of coded blocks and ``n`` is the length of
            each code block.
            
        noiseVar : float
            The noise variance obtained using noise estimation or using the actual noise variance value used in
            simulation.

        useMax : Boolean
            If ``True``, this implementation uses the ``Max`` function in the calculation of the LLR values. This is
            faster but uses an approximation and is slightly less accurate than the actual Log Likelihood method which
            uses logarithm and exponential functions. If ``False``, the slower more accurate method is used.

        Returns
        -------
        Numpy array of bit values
            Returns a 2-D array of demodulated bits.
        """
        llrs = self.getLLRsFromSymbols(symbols, noiseVar, useMax)
        return np.int8((llrs<=0)*1)

