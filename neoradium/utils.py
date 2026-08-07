# Copyright (c) 2024-2026, InterDigital AI Lab
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
from scipy.interpolate import RBFInterpolator, interp1d
from scipy.stats import norm
import warnings, functools

__all__ = ['toRadian', 'toDegrees', 'toLinear', 'toDb', 'herm', 'getMse', 'getNmse']
DOCS_LOC = "https://ail-wireless.pages.interdigital.com/neoradium/"   # For internal GitLab

# **********************************************************************************************************************
def toRadian(angle):
    r"""
    Converts an angle (or array of angles) from degrees to radians. Returns ``None`` unchanged so
    that optional-angle parameters can be passed through transparently.

    Parameters
    ----------
    angle : float, NumPy array, or None
        The angle(s) in degrees.

    Returns
    -------
    float, NumPy array, or None
        The same input converted to radians, or ``None`` if ``angle`` is ``None``.
    """
    return (None if angle is None else np.float64(angle)*np.pi/180.0)

# **********************************************************************************************************************
def toDegrees(angle):
    r"""
    Converts an angle (or array of angles) from radians to degrees. Returns ``None`` unchanged so
    that optional-angle parameters can be passed through transparently.

    Parameters
    ----------
    angle : float, NumPy array, or None
        The angle(s) in radians.

    Returns
    -------
    float, NumPy array, or None
        The same input converted to degrees, or ``None`` if ``angle`` is ``None``.
    """
    return (None if angle is None else np.float64(angle)*180.0/np.pi)

# **********************************************************************************************************************
def toLinear(x):
    r"""
    Converts a value (or array of values) from decibels (dB) to linear scale using :math:`10^{x/10}`.

    Parameters
    ----------
    x : float or NumPy array
        Value(s) in dB.

    Returns
    -------
    float or NumPy array
        The corresponding linear value(s).
    """
    return 10.0**(x/10.0)

# **********************************************************************************************************************
def toDb(x):
    r"""
    Converts a value (or array of values) from linear scale to decibels (dB) using
    :math:`10\log_{10}(x)`.

    Parameters
    ----------
    x : float or NumPy array
        Linear value(s). Must be positive; ``toDb(0)`` returns ``-inf``.

    Returns
    -------
    float or NumPy array
        The corresponding value(s) in dB.
    """
    return 10.0*np.log10(x)

# **********************************************************************************************************************
def interpolate(x, y, xNew, method, numNeighbors=None, smoothing=10):   # Undocumented - Not intended for direct use
    if method=='thin_plate_spline': f = RBFInterpolator(x[:,None], y, numNeighbors, smoothing, 'thin_plate_spline', 1)
    elif method == 'multiquadric':  f = RBFInterpolator(x[:,None], y, numNeighbors, smoothing, 'multiquadric', 1)
    elif method == 'linear':        f = interp1d(x, y, kind='linear', axis=0, fill_value='extrapolate')
    elif method == 'quadratic':     f = interp1d(x, y, kind='quadratic', axis=0, fill_value='extrapolate')
    elif method == 'nearest':       f = interp1d(x, y, kind='nearest', axis=0, fill_value='extrapolate')
    else:                           raise ValueError(f"Unsupported interpolation method: {method}")
    if method in ['thin_plate_spline', 'multiquadric']: yNew = f(xNew[:,None])
    else:                                               yNew = f(xNew)
    return yNew

# **********************************************************************************************************************
def polarInterpolate(x, y, xNew, method, numNeighbors=None, smoothing=10):# Undocumented - Not intended for direct use
    theta, r = np.unwrap(np.angle(y),axis=0), np.abs(y)
    thetaNew = interpolate(x, theta, xNew, method, numNeighbors, smoothing)
    rNew = interpolate(x, r, xNew, method, numNeighbors, smoothing)
    return rNew * (np.cos(thetaNew) + 1j*np.sin(thetaNew))

# **********************************************************************************************************************
def herm(x):
    r"""
    Returns the Hermitian (conjugate) transpose of ``x`` along its last two axes — i.e.,
    :math:`x^H` for batched matrix operations where leading dimensions broadcast and only the
    trailing two axes are transposed.

    Parameters
    ----------
    x : NumPy array
        Input array of shape ``(..., M, N)``.

    Returns
    -------
    NumPy array
        Array of shape ``(..., N, M)`` containing the conjugate transpose of ``x`` along the last
        two axes.
    """
    return np.swapaxes(np.conj(x),-1,-2)

# **********************************************************************************************************************
def getMse(h, hEst):
    r"""
    Returns the *Mean Squared Error* between an estimate and a reference:

    .. math::

        \text{MSE} = \frac{1}{N} \sum |\hat{h} - h|^2

    where the sum runs over all elements of the input arrays.

    Parameters
    ----------
    h : NumPy array
        The reference (true) values.

    hEst : NumPy array
        The estimated values. Must have the same shape as ``h``.

    Returns
    -------
    float
        The mean squared error.
    """
    error = np.abs(hEst-h)
    mse = np.square(error).mean()
    return mse

# **********************************************************************************************************************
def getNmse(u, uEst):
    r"""
    Returns the *Normalized Mean Squared Error* between an estimate ``uEst`` and a reference ``u``,
    following the definition used by MATLAB's
    `goodnessoffit <https://www.mathworks.com/help/ident/ref/goodnessoffit.html>`_:

    .. math::

        \text{NMSE} = \frac{\sum |\hat{u} - u|^2}{\sum |\bar{u} - u|^2}

    where :math:`\bar{u}` is the mean of the reference. NMSE is dimensionless and equals ``1.0`` for
    a trivial estimator that just returns the reference mean.

    Parameters
    ----------
    u : NumPy array
        The reference values.

    uEst : NumPy array
        The estimated values. Must have the same shape as ``u``.

    Returns
    -------
    float
        The normalized mean squared error.
    """
    uMean = u.mean()
    nmse = np.square(np.abs(uEst-u)).sum()/np.square(np.abs(uMean-u)).sum()
    return nmse

# **********************************************************************************************************************
def goldSequence(cInit, numBits):                                       # Undocumented - Not intended for direct use
    # This function creates a "numBits"-bit Gold-sequence bitstream using binary arithmetic with a pre-calculated x1.
    x1 = 0x42054D21     # Pre-calculated X1 (After 51 iterations)
    x2 = cInit          # X2 depends on "cInit".
    # Now pre-calculate x2:
    for _ in range(51):
        x2 ^= (x2>>3) ^ (x2>>2) ^ (x2>>1)
        x2 ^= ((x2<<28) ^ (x2<<29) ^ (x2<<30))&0x7FFFFFFF

    # First, compute 12 bits.
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
def getMultiLineStr(label, values, indent, formatStr, length, numPerLine):      # Undocumented
    # This is used mostly in "print" methods of different classes where the value of a property spans multiple lines.
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
    if f >= 1e15: return f"{f/1e15:.4g} PHz"
    if f >= 1e12: return f"{f/1e12:.4g} THz"
    if f >= 1e9:  return f"{f/1e9:.4g} GHz"
    if f >= 1e6:  return f"{f/1e6:.4g} MHz"
    if f >= 1e3:  return f"{f/1e3:.4g} kHz"
    return f"{f:.4g} Hz"

# **********************************************************************************************************************
warnings.simplefilter('module', DeprecationWarning)     # Print the Deprecation Warning only once
warnedMessages = set()
def deprecated(replacement=None, docFile=None):
    # A decorator to mark functions as deprecated. It emits a warning when the function is used.
    # Usage:
    #   def new_add(a, b):
    #       """The new, preferred function."""
    #       return a + b
    #
    #   @deprecated(replacement="new_add")
    #   def old_add(a, b):
    #       """The old, deprecated function."""
    #       return a + b
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = f"Call to deprecated function {func.__name__}."
            if replacement:     message += f" Use {replacement} instead."
            if message in warnedMessages: return func(*args, **kwargs)
            warnedMessages.add(message)
            if docFile:
                message += (f" For more information please visit: " + DOCS_LOC +
                            f"source/API/{docFile}.html#{func.__module__}.{func.__qualname__}")
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator
    
# **********************************************************************************************************************
def warnOnce(message):
    if message in warnedMessages: return
    warnedMessages.add(message)
    warnings.warn(message, category=DeprecationWarning, stacklevel=3)

# **********************************************************************************************************************
def getNumBlocks(errorMargin=0.01, confidence=0.95, blerEst=0.5):
    # Returns how many transmissions are needed to measure BLER with the specified confidence
    # and relative margin of error.
    # 'blerEst' is an initial estimate of BLER.
    # The default values give the number of blocks for a margin of error of 0.01 (± 1%) with
    # 95% confidence.
    # Based on the following formula:
    #    n = \frac{Z^2 \cdot (1 - \hat{p})}{\hat{p} \cdot r^2}
    # where Z is the Z-Score, \hat{p} is the initial estimate of BLER, and r
    # is the relative margin of error.
    cumProb = 1 - (1 - confidence) / 2   # Cumulative Probability
    z = norm.ppf(cumProb)                # Z-score (how many standard deviations away from the mean)
    return int(np.ceil(z*z*(1-blerEst)/(blerEst*(errorMargin**2))))

# **********************************************************************************************************************
def validateRange(var, valids, context="", varName=None):
    if varName is None:
        import inspect
        frame = inspect.getouterframes(inspect.currentframe())[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        varName = string[string.find('(') + 1:-1].split(',')[0].strip("self.")

    if isinstance(valids, list):
        if var in valids:                      return
        fStr = "'%s'" if type(valids[0])==str else "%s"
        raise ValueError("Invalid '%s'! ('%s' ∈ {%s}%s)"%(varName, varName,
                                                        ", ".join([fStr%str(x) for x in valids]), context))

    if isinstance(valids, tuple) and len(valids)==2:
        if var in range(valids[0],valids[1]+1): return
        fStr = "'%s'" if type(valids[0])==str else "%s"
        raise ValueError("Invalid '%s'! ('%s' ∈ {%s}%s)"%(varName, varName,
                                                        ",...,".join([fStr%str(x) for x in valids]), context))

    if var==valids:                              return
    fStr = "'%s'" if type(valids)==str else "%s"
    raise ValueError("Invalid '%s'! (It must be "%(varName) + fStr%str(valids) + context + ")")
