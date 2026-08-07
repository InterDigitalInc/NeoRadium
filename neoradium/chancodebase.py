# Copyright (c) 2024-2026, InterDigital AI Lab
"""
The module ``chancodebase.py`` implements the base class :py:class:`ChanCodeBase` for all **NeoRadium**'s channel
coding classes. It encapsulates some basic functionality such as creating, appending, and checking various types of
CRC processing based on **3GPP TS 38.212**.

Here is the hierarchy of current channel coding classes in **NeoRadium**:

- :py:class:`ChanCodeBase` (The base class for all channel coding)

  + :py:class:`~neoradium.ldpccodec.LdpcCodecCW`
  
  + :py:class:`~neoradium.polar.PolarBase` (The base class for Polar coding)

    * :py:class:`~neoradium.polar.PolarEncoder`
    * :py:class:`~neoradium.polar.PolarDecoder`
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 01/05/2024    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np

from .utils import validateRange

# This file is based on 3GPP TS 38.212

# **********************************************************************************************************************
strToPoly = {   # 3GPP TS 38.212, Section 5.1
             '6':   [1, 1, 0, 0, 0, 0, 1],
             '11':  [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             '16':  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             '24A': [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
             '24B': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
             '24C': [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]
            }

# **********************************************************************************************************************
# The base class for all channel coding classes
class ChanCodeBase:
    r"""
    Base class for all channel coding classes in **NeoRadium**. Provides shared CRC helpers and
    constants used by Polar and LDPC code paths per **3GPP TS 38.212, Section 5.1**.

    Inherited members
    -----------------
    - :py:attr:`LARGE_LLR` — a large LLR magnitude (``1e20``) representing absolute certainty,
      used by decoders to encode filler bits as "definitely zero".
    - :py:meth:`getCrc` — computes the CRC bits for a 1-D or 2-D bit array given a polynomial name.
    - :py:meth:`checkCrc` — validates that a bit array (with CRC appended at the end) is consistent.
    - :py:meth:`appendCrc` — convenience wrapper that returns the input bits with the CRC appended.
    - :py:meth:`getCrcLen` — returns the CRC length in bits for a given polynomial name.

    Known subclasses
    ----------------
    - :py:class:`~neoradium.polar.PolarBase` (Polar coding).
    - :py:class:`~neoradium.ldpccodec.LdpcCodecCW` (LDPC coding, current API).
    - :py:class:`~neoradium.ldpc.LdpcBase` (deprecated LDPC API).
    """
    LARGE_LLR = 1e20    # Large LLR value used for values with absolute certainty

    # ******************************************************************************************************************
    def __init__(self):
        pass

    # ******************************************************************************************************************
    @classmethod
    def getCrcLen(cls, poly):               # Undocumented
        # Return the number of CRC bits
        if poly[:2]=="24":  return 24
        return int(poly)

    # ******************************************************************************************************************
    @classmethod
    def getCrc(cls, bits, poly):
        r"""
        Calculates and returns the CRC based on the bitstream ``bits`` and the generator polynomial specified by 
        ``poly``.
        
        Parameters
        ----------
        bits : NumPy array
            A 1D or 2D NumPy array of bits. If it is a 1D NumPy array, the CRC bits are calculated for the given 
            bitstream and a 1D NumPy array containing the CRC bits is returned. If ``bits`` is an ``N x L`` NumPy 
            array, it is assumed that we have ``N`` bitstreams of length ``L``. In this case the CRC bits are
            calculated for each one of ``N`` bitstreams and an ``N x C`` NumPy array is returned where ``C`` is
            the CRC length.
            
        poly : str
            The string specifying the generator polynomial. The following generator polynomials are supported.

                ======================  ===========================
                The value of ``poly``   Generator polynomial
                ======================  ===========================
                '6'                     1100001
                '11'                    111000100001
                '16'                    10001000000100001
                '24A'                   1100001100100110011111011
                '24B'                   1100000000000000001100011
                '24C'                   1101100101011000100010111
                ======================  ===========================
                
            For more details please refer to **3GPP TS 38.212, Section 5.1**.

        Returns
        -------
        NumPy array
            If ``bits`` is a 1D NumPy array, the CRC bits are returned in a 1D NumPy array. If ``bits`` is an
            ``N x L`` NumPy array, the CRC bits of ``N`` bitstreams are returned in an ``N x C`` NumPy array where
            ``C`` is the CRC length.
        """
        if not isinstance(poly, str):   raise ValueError("'poly' must be a string!")
        validateRange(poly, ['6', '11', '16', '24A', '24B', '24C'])
        poly = np.uint8(strToPoly[poly])
        polyLen = len(poly)
        isFlat = bits.ndim==1
        if isFlat: bits = bits[None,:]
        m,n = bits.shape
        paddedBitArray = np.concatenate((bits, np.zeros((m, polyLen-1), dtype=bits.dtype)), axis=1)  # m x (n+polyLen-1)
        for d in range(n):
            paddedBitArray[:,d:d+polyLen] ^= np.outer(paddedBitArray[:,d],poly)
        return paddedBitArray[0,n:] if isFlat else paddedBitArray[:,n:]

    # ******************************************************************************************************************
    @classmethod
    def checkCrc(cls, bits, poly):
        r"""
        Checks the CRC bits at the end of the bitstream ``bits`` and returns `True` if the CRC is valid and `False` 
        if it is not.
        
        Parameters
        ----------
        bits : NumPy array
            A 1D or 2D NumPy array of bits. If it is a 1D NumPy array, the CRC bits are checked for the given
            bitstream and a boolean value is returned. If ``bits`` is an ``N x L`` NumPy array, it is assumed that 
            we have ``N`` bitstreams of length ``L``. In this case the CRC bits are checked for each one of ``N``
            bitstreams and a boolean NumPy array of length ``N`` is returned, specifying the CRC check result
            for each bitstream.
            
        poly : str
            The string specifying the generator polynomial. See the :py:meth:`getCrc` method above for a list of
            generator polynomials.

        Returns
        -------
        boolean or NumPy array
            If ``bits`` is a 1D NumPy array, the CRC check result is returned as a boolean value. If ``bits`` is an
            ``N x L`` NumPy array, the CRC check results of ``N`` bitstreams are returned in a boolean NumPy array
            of length ``N``.
        """
        return np.count_nonzero( cls.getCrc(bits, poly), -1)==0

    # ******************************************************************************************************************
    @classmethod
    def appendCrc(cls, bits, poly):
        r"""
        Calculates the CRC bits for the bitstream ``bits``, appends them to the end of the bitstream, and returns the
        new bitstream containing the original bits followed by the CRC bits.
        
        This function calls the :py:meth:`getCrc` method to get the CRC bits and then appends them to the end of
        ``bits``.
        
        Parameters
        ----------
        bits : NumPy array
            A 1D or 2D NumPy array of bits. If it is a 1D NumPy array, the CRC bits are calculated for the given
            bitstream and a 1D NumPy array containing the original bitstream and the CRC bits is returned. If 
            ``bits`` is an ``N x L`` NumPy array, it is assumed that we have ``N`` bitstreams of length ``L``. In
            this case the CRC bits are calculated and appended to the end of each one of ``N`` bitstreams and an
            ``N x M`` NumPy array is returned where ``M=L+C`` and ``C`` is the CRC length.

        poly : str
            The string specifying the generator polynomial. See the :py:meth:`getCrc` method above for a list of
            generator polynomials.

        Returns
        -------
        NumPy array
            If ``bits`` is a 1D NumPy array, the new bitstream with CRC appended to the end is returned in a 1D NumPy
            array. If ``bits`` is an ``N x L`` NumPy array, the CRC bits for each one of ``N`` bitstreams are appended
            to the end and an ``N x M`` NumPy array is returned where ``M=L+C`` and ``C`` is the CRC length.
        """
        return np.append(bits, cls.getCrc(bits, poly), axis=-1)

