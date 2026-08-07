# Copyright (c) 2024-2026, InterDigital AI Lab
"""
This module implements the modulation and demodulation functionality based on **3GPP TS 38.211**. The :py:class:`Modem`
class handles modulation and demodulation of bitstreams to and from complex symbols.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 11/03/2023    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np

from .utils import validateRange, deprecated

docFile = "Modulation"          # Used by the 'deprecated' decorators

# **********************************************************************************************************************
# The Modulator/Demodulator class
class Modem:
    r"""
    This class handles the process of modulating a bitstream to an array of complex symbols (Modulation) as 
    well as extracting log-likelihood ratios (LLRs) from an array of complex symbols during demodulation. This 
    implementation is based on **3GPP TS 38.211 section 5.1**.
    """
    # TS 38.211, Section 5.1
    mod2qm = {'BPSK':1, 'QPSK':2, '16QAM':4, '64QAM':6, '256QAM':8, '1024QAM':10}

    # ******************************************************************************************************************
    def __init__(self, modulation='QPSK'):
        r"""
        Parameters
        ----------
        modulation : str
            The modulation scheme based on Section 5.1 in **3GPP TS 38.211**. The supported modulation schemes
            are:
                        
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
        
            :qm: The modulation order. This is the number of bits per modulated symbol. See **3GPP TS 38.211, 
                Table 7.3.1.2-1** for more details.
            :constellation: The modulation constellation. This is a lookup table that converts each group of ``qm`` 
                bits from the input bitstream to a complex symbol.
        """
        self.modulation = modulation
        validateRange(self.modulation, list(self.mod2qm.keys()))
            
        qm = self.mod2qm[modulation]
        self.qm = qm
        
        scale = 1/np.sqrt({1:2, 2:2, 4:10, 6:42, 8:170, 10:682}[qm])

        # The following function implements the equations in TS 38.211,
        # sections 5.1.2, 5.1.3, 5.1.4, 5.1.5, 5.1.6, and 5.1.7
        def getConstellationValue(value):
            b = [int(x) for x in ("{0:0%db}"%(qm)).format(value)]
            real,imag = 1,1
            for q in range(2,qm,2):
                real = (1<<(q//2)) - (1-2*b[qm-q])*real
                imag = (1<<(q//2)) - (1-2*b[qm+1-q])*imag
            real *= 1-2*b[0]
            imag *= 1-2*b[min(1,qm-1)]
            return scale*(real + 1j*imag)

        self.constellation = np.array([ getConstellationValue(x) for x in range(1<<qm)])

        # Get a list of binary representations of all integers from 0 to 2^qm-1
        allBinaries = np.int8( [list(("{0:0%db}"%(self.qm)).format(i)) for i in range(1<<qm)] ) # Shape: (2^qm, qm)

        # c is a 2 x 2^(qm-1) x qm tensor.
        # c[0,:,i] is a list of indices of constellation points where the i-th bit is 0  i∈{0,...,qm-1}
        # c[1,:,i] is a list of indices of constellation points where the i-th bit is 1  i∈{0,...,qm-1}
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
            If specified, it is used as the title for the printed information. If `None` (the default), the text 
            "Modem Properties:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
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

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def modulateOneBlock(self, bitstream):      # Undocumented
        # This function modulates a single code block. This is called by the method "modulate" below.
        # See TS 38.211, Section 7.3.1.2
        qm = self.qm
        if len(bitstream)%qm > 0:
            raise ValueError("The length of 'bitstream' (%d) must be a multiple of 'qm' (%d)!"%(len(bitstream), qm))
        
        # Get symbol indices
        symIndexes = ( np.uint16(bitstream).reshape((-1,qm)) * [[1<<(qm-i-1) for i in range(qm)]] ).sum(1)
        symbols = self.constellation[symIndexes]
        return symbols

    # ******************************************************************************************************************
    def modulate(self, bitstreams):
        r"""
        Modulates the given bitstream into one or more arrays of complex symbols using the current modulation scheme.
        
        Parameters
        ----------
        bitstreams : NumPy array of bits
            A 1-D (one code block) or 2-D (several code blocks) array of bits.

        Returns
        -------
        NumPy array of complex values
            Returns a 1-D or 2-D (depending on shape of ``bitstreams``) NumPy complex array of modulated symbols.
        """
        if bitstreams.ndim>1:
            return np.complex128( [ self.modulateOneBlock(bitstream) for bitstream in bitstreams] )
        return self.modulateOneBlock(bitstreams)

    # ******************************************************************************************************************
    def getLLRs(self, symbols, noiseVar, useMax=True):
        r"""
        This function calculates the log-likelihood ratios (LLRs) for each bit from the received noisy symbols. The
        LLR values can then be used by :py:class:`PolarDecoder` or :py:class:`LdpcDecoder` to extract the decoded
        bitstream.

        Parameters
        ----------
        symbols : 1-D or 2-D Complex NumPy array
            An ``m``x``n`` complex NumPy array where ``m`` is the number of coded blocks and ``n`` is the length of
            each code block. If it is a 1-D array, it means there is only one code block to demodulate.
            
        noiseVar : float
            The noise variance obtained using noise estimation or using the actual noise variance value used in
            simulation.

        useMax : bool
            If `True` (the default), this implementation uses the ``Max`` function in the calculation of the LLR 
            values. This is faster but uses an approximation and is slightly less accurate than the actual Log 
            Likelihood method which uses logarithm and exponential functions. If `False`, the slower more accurate 
            method is used.

        Returns
        -------
        NumPy array of floating-point values
            A 1-D or 2-D NumPy array of LLR values depending on the dimensionality of ``symbols``. In the case of a 2-D
            array, the return value is an ``m``x``l`` array of LLR values where ``l= n * qm``. In case of 1-D array, 
            the output is a 1-D array of ``l`` LLR values.
        """
        if noiseVar <= 0:
            raise ValueError(f"'noiseVar' must be positive (got {noiseVar}).")

        # First calculate all distances:
        d = np.abs(symbols[...,None]-self.constellation)        # shape: symbols.shape + (2^qm,)
        exponents = -d**2/noiseVar

        exponents = exponents[...,self.c]                       # A tensor of shape: symbols.shape + (2, 2^(qm-1), qm)
        # exponents[...,0,:,:] for '0' bits
        # exponents[...,1,:,:] for '1' bits

        # Sum or Max over the second-to-last axis  => Shape of lls: symbols.shape + (2, qm)
        MAX_EXPONENT = 700      # Prevent overflow

        # All Log-Likelihood values
        lls = exponents.max(-2) if useMax else np.log(np.exp(np.clip(exponents,-MAX_EXPONENT,MAX_EXPONENT)).sum(-2))
        llrs = lls[...,0,:] - lls[...,1,:]          # The Log-Likelihood Ratio (LLR)
        # llrs shape: symbols.shape + (qm,)
        return llrs.reshape(llrs.shape[:-2]+(-1,))  # Merge last 2 dimensions

    # ******************************************************************************************************************
    @deprecated("getLLRs", docFile)
    def getLLRsFromSymbols(self, symbols, noiseVar, useMax=True):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`getLLRs` method instead.
        """
        return self.getLLRs(symbols, noiseVar, useMax)

    # ******************************************************************************************************************
    def demodulate(self, symbols, noiseVar, useMax=True):
        r"""
        Demodulates the received noisy symbols to a bitstream using hard decisions to convert log-likelihood ratios
        (LLRs) to bits. This function first calls the :py:meth:`Modem.getLLRs` method to get the
        LLR values, and then uses "hard decision" to convert LLRs to bits.

        Parameters
        ----------
        symbols : 1-D or 2-D Complex NumPy array
            An ``m x n`` complex NumPy array where ``m`` is the number of coded blocks and ``n`` is the length of
            each code block. If it is a 1-D array, it means there is only one code block to demodulate.
            
        noiseVar : float
            The noise variance obtained using noise estimation or using the actual noise variance value used in
            simulation.

        useMax : bool
            If `True`, this implementation uses the ``Max`` function in the calculation of the LLR values. This is
            faster but uses an approximation and is slightly less accurate than the actual Log Likelihood method which
            uses logarithm and exponential functions. If `False`, the slower more accurate method is used.

        Returns
        -------
        NumPy array of bit values
            Returns a 1-D or 2-D NumPy array of demodulated bits, depending on the dimensionality of symbols.
        """
        llrs = self.getLLRs(symbols, noiseVar, useMax)
        return np.int8((llrs<=0)*1)

