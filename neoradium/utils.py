# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``utils.py`` contains utility classes and functions used by other modules in **NeoRadium**.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 01/10/2024    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np
from .random import random
from scipy.interpolate import RBFInterpolator, interp1d


# **********************************************************************************************************************
def toRadian(angle):    return (None if angle is None else np.float64(angle)*np.pi/180.0)
def toDegrees(angle):   return (None if angle is None else np.float64(angle)*180.0/np.pi)
def toLinear(x):        return 10.0**(x/10.0)
def toDb(x):            return 10.0*np.log10(x)

# **********************************************************************************************************************
def interpolate(x, y, xNew, method, numNeighbors=None, smoothing=10):   # Not documented - Not intended for direct use
    if method=='thin_plate_spline': f = RBFInterpolator(x[:,None], y, numNeighbors, smoothing, 'thin_plate_spline', 1)
    elif method == 'multiquadric':  f = RBFInterpolator(x[:,None], y, numNeighbors, smoothing, 'multiquadric', 1)
    elif method == 'linear':        f = interp1d(x, y, kind='linear', axis=0, fill_value='extrapolate')
    elif method == 'quadratic':     f = interp1d(x, y, kind='quadratic', axis=0, fill_value='extrapolate')
    elif method == 'nearest':       f = interp1d(x, y, kind='nearest', axis=0, fill_value='extrapolate')

    if method in ['thin_plate_spline', 'multiquadric']: yNew = f(xNew[:,None])
    else:                                               yNew = f(xNew)
    return yNew

# **********************************************************************************************************************
def polarInterpolate(x, y, xNew, method, numNeighbors=None, smoothing=10):# Not documented - Not intended for direct use
    theta, r = np.unwrap(np.angle(y),axis=0), np.abs(y)
    thetaNew = interpolate(x, theta, xNew, method, numNeighbors, smoothing)
    rNew = interpolate(x, r, xNew, method, numNeighbors, smoothing)
    return rNew * (np.cos(thetaNew) + 1j*np.sin(thetaNew))

# **********************************************************************************************************************
def intToBits(n, length=None):                                          # Not documented - Not intended for direct use
    if length is None:
        return np.uint8([1 if digit=='1' else 0 for digit in bin(n)[2:]])
        
    bits = [1 if digit=='1' else 0 for digit in bin(n)[2:]]
    return np.uint8((length-len(bits))*[0] + bits)

# **********************************************************************************************************************
def herm(x):                                                            # Not documented - Not intended for direct use
    return np.swapaxes(np.conj(x),-1,-2)

# **********************************************************************************************************************
def getMse(h, hEst):                                                    # Not documented - Not intended for direct use
    error = np.abs(hEst-h)
    mse = np.square(error).mean()
    return mse

# **********************************************************************************************************************
def getNmse(u, uEst):                                                   # Not documented - Not intended for direct use
    # Source: https://www.mathworks.com/help/ident/ref/goodnessoffit.html
    uMean = u.mean()
    nmse = np.square(np.abs(uEst-u)).sum()/np.square(np.abs(uMean-u)).sum()
    return nmse

# **********************************************************************************************************************
def goldSequence(cInit, numBits):                                       # Not documented - Not intended for direct use
    # This function creates a "numBits"-bit Gold sequence bitstream
    # using binary arithmetic with pre-calculated x1
    x1 = 0x42054D21     # Pre-calculated X1 (After 51 iterations)
    x2 = cInit          # X2 depends on "cInit".
    # Now pre-calculate x2:
    for _ in range(51):
        x2 ^= (x2>>3) ^ (x2>>2) ^ (x2>>1)
        x2 ^= ((x2<<28) ^ (x2<<29) ^ (x2<<30))&0x7FFFFFFF

    # First time, compute 12 bits
    c = (x1^x2)                             # 12 bits
    bits = [(c>>i)&1 for i in range(19,31)] # Pick the 12 MSBs

    remainingBits = numBits-12
    while remainingBits>0:
        x1 ^= (x1>>3)
        x1 ^= (x1<<28)&0x7FFFFFFF
        x2 ^= (x2>>3) ^ (x2>>2) ^ (x2>>1)
        x2 ^= ((x2<<28) ^ (x2<<29) ^ (x2<<30))&0x7FFFFFFF
        c = (x1^x2)                # 31 bits
        bits += [(c>>i)&1 for i in range(31)]
        remainingBits -=31
    
    return bits[:numBits]

# **********************************************************************************************************************
def getMultiLineStr(label, values, indent, formatStr, length, numPerLine):# Not documented - Not intended for direct use
    # This is used mostly in "print" methods of different classes where the value of a property occupies multiple
    # lines.
    indentStr = indent*' ' + '  '
    label = label.rstrip()+':'+' '*(len(label)-len(label.rstrip()))
    labelLen = len(label)
    r, retStr = 0, ""
    while r<len(values):
        if r == 0:
            retStr += indentStr + label + " %s\n"%(" ".join( (formatStr % p)[:length] for p in values[r:r+numPerLine] ))
        else:
            retStr += indentStr + labelLen*' ' + " %s\n"%(" ".join( (formatStr % p)[:length] for p in values[r:r+numPerLine] ))
        r += numPerLine
    return retStr

# **********************************************************************************************************************
def freqStr(f):
    if f>1000000000000000: return f"{f:.4g} Hz"
    if f>1000000000000: return f"{f/1000000000000:.4g} THz"
    if f>1000000000:    return f"{f/1000000000:.4g} GHz"
    if f>1000000:       return f"{f/1000000:.4g} MHz"
    if f>1000:          return f"{f/1000:.4g} kHz"
    return f"{f} Hz"

# **********************************************************************************************************************
def makeComplexNoiseLike(x, **kwargs):                                  # Not documented - Not intended for direct use
    # This function may be removed in the future. Use the Waveform and Grid objects' addNoise methods instead.
    # NOTE: To add noise to a waveform, you need to specify the nFFT value used for OFDM modulation. When adding
    #       noise to RX grid directly, you should not specify the nFFT.
    noiseStd = kwargs.get('noiseStd', None)
    if noiseStd is not None:
        return  (random.normal(0, noiseStd, x.shape+(2,))*[1,1j]).sum(-1)/np.sqrt(2)

    snrDb = kwargs.get('snrDb', None)
    if snrDb is not None:
        snr = toLinear(snrDb)
        nr = kwargs.get('nr', 1)
        nFFT = kwargs.get('nFFT', 1)
        noiseStd = 1/np.sqrt(snr*nr*nFFT)
        return makeComplexNoiseLike(x, noiseStd=noiseStd)

    noiseVar = kwargs.get('noiseVar', None)
    if noiseVar is not None:
        return makeComplexNoiseLike(x, noiseStd=np.sqrt(noiseVar))

    raise ValueError("You must specify one of 'snrDb', 'noiseVar', or 'noiseStd'!")
