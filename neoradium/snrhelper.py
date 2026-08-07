# Copyright (c) 2025-2026, InterDigital AI Lab
"""
The module ``snrhelper.py`` implements the adaptive SNR iterator :py:class:`SnrScheduler`.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 10/21/2025    Shahab Hamidi-Rad       First version of the file.
# **********************************************************************************************************************
import numpy as np

# **********************************************************************************************************************
class SnrScheduler:
    r"""
    Use this class as an adaptive SNR iterator to step through SNR values while measuring a metric such as
    Bit Error Rate (BER), Block Error Rate (BLER), or throughput. Starting from an initial SNR guess, it adaptively 
    searches until it brackets a usable range and then traverses from the low to the high operating point.

    The iterator enforces that `setData(...)` is called once per loop iteration to report the metric value(s) computed 
    at the current SNR.
    """
    # ******************************************************************************************************************
    def __init__(self, snr0=0, step=1, loSnrVal=100, hiSnrVal=0, fastStep=None, maxSnrs=100):
        r"""
        Parameters
        ----------
        snr0 : float
            Initial SNR in dB. Pick a mid-range guess if possible (default: 0.0).
    
        step : float
            SNR increment in dB used when moving up/down (default: 1.0).
    
        loSnrVal : float
            Target metric value associated with the *low* SNR operating point.
            Example for BLER (%): 100. (default: 100)
    
        hiSnrVal : float
            Target metric value associated with the *high* SNR operating point.
            Example for BLER (%): 0. (default: 0)
    
        fastStep : float or None
            If specified, the ``fastStep`` is used during initial searching until we find a point inside the range
            specified by loSnrVal and hiSnrVal. Once a point is found in the range, the ``step`` parameter is used.
            If not specified, it is set to ``2 * step``. 
            
        maxSnrs : int
            Hard cap on the number of recorded SNR points; used as a safeguard against non-monotonic behavior or 
            failure to converge (default: 100).

        Notes
        -----
        - For BER/BLER, metrics generally *decrease* with SNR (loSnrVal > hiSnrVal).
        - For throughput, metrics often *increase* with SNR (loSnrVal < hiSnrVal).
        - The scheduler handles both cases.

        .. code-block:: python
            :caption: Examples

            # Start at -8 dB, use increments of 0.2 dB for BLER using default values (loSnrVal=100, hiSnrVal=0).
            # The defaults are oriented for decreasing-with-SNR metrics like BLER/BER: at low SNR the metric is
            # near 100%, at high SNR it approaches 0%. Hence loSnrVal > hiSnrVal.
            snrScheduler = SnrScheduler(snr0=-8, step=.2)

            # Start at 0 dB, use increments of 0.5 dB for BER in the range %10 to %45
            snrScheduler = SnrScheduler(snr0=0, step=.5, loSnrVal=45, hiSnrVal=10)

            # Start at -4 dB, use increments of 0.5 dB for throughput in the range %0 to %100. Note that
            # direction of changes in this case is the opposite of BER/BLER cases. (loSnrVal < hiSnrVal)
            snrScheduler = SnrScheduler(snr0=-4, step=1, loSnrVal=0, hiSnrVal=100)
        """
        self.snr0 = snr0            # The starting SNR value
        if not (isinstance(step, (int, float)) and step > 0):  raise ValueError("`step` must be a positive number.")
        self.step = step            # SNR step
        if not (isinstance(maxSnrs, int) and maxSnrs > 0):     raise ValueError("`maxSnrs` must be a positive integer.")
        self.maxSnrs = maxSnrs      # Max number of SNR values
        if loSnrVal == hiSnrVal:    raise ValueError("`loSnrVal` and `hiSnrVal` must be distinct.")
        self.loSnrVal = loSnrVal    # The value of reference param at lowest SNR
        self.hiSnrVal = hiSnrVal    # The value of reference param at highest SNR
        if fastStep is None:                                        self.fastStep = 2*self.step
        elif isinstance(fastStep, (int, float)) and fastStep > 0:   self.fastStep = fastStep
        else:                       raise ValueError("`fastStep` must be a positive number.")
        self.reset()

    # ******************************************************************************************************************
    def reset(self):
        r"""Reset to the initial state; typically called before the SNR loop inside an outer-loop."""
        self.curSnr = self.snr0
        self.buffers = None
        self.state = 'Start'
        self.curLo = -np.inf
        self.curHi = np.inf
        self.setDataCalled = True

    # ******************************************************************************************************************
    def __iter__(self): return self
    def __next__(self):
        if self.state=='Done':      raise StopIteration
        if not self.setDataCalled:  raise ValueError("The \"setData\" was not called in the last iteration!")
        self.setDataCalled = False
        return self.curSnr
            
    # ******************************************************************************************************************
    def whereAmI(self, value):
        # Determines where in the SNR list we are based on the loSnrVal/hiSnrVal values.
        if self.loSnrVal<self.hiSnrVal:
            if value <= self.loSnrVal:  return 'LoSNR'
            if value >= self.hiSnrVal:  return 'HiSNR'
            return 'MidSNR'
        
        if value >= self.loSnrVal:  return 'LoSNR'
        if value <= self.hiSnrVal:  return 'HiSNR'
        return 'MidSNR'

    # ******************************************************************************************************************
    def updateBuffers(self, value, *otherValues):
        numValues = 2 + len(otherValues)
        if self.buffers is None:
            self.buffers = [ [] for i in range(numValues) ]
        elif numValues != len(self.buffers):
            raise ValueError("Inconsistent number of values passed to the \"setData\" function!")
        elif len(self.buffers[0])>= self.maxSnrs:
            raise ValueError(f"Did not converge after {self.maxSnrs} tries.")
        self.buffers[0] += [ self.curSnr ]
        self.buffers[1] += [ value ]
        for i, otherValue in enumerate(otherValues):    self.buffers[i+2] += [ otherValue ]
    
    # ******************************************************************************************************************
    def setData(self, value, *otherValues):
        r"""
        Record the metric(s) for the current SNR and advance the internal state.
    
        You MUST call this at the end of each loop iteration; otherwise, a ValueError is raised on the next iteration.
    
        Parameters
        ----------
        value : float
            Primary metric used to drive scheduling (e.g., BLER, BER, throughput).
    
        otherValues : tuple of float, optional
            Additional values to store per SNR. The arity must be consistent across iterations. These are returned
            by :py:meth:`~SnrScheduler.getSnrsAndData()` after the loop.

    
        .. code-block:: python
            :caption: Example
       
            snrScheduler = SnrScheduler(snr0=7, step=.2)    # Instantiate the SnrScheduler for BLER

            for snrDb in snrScheduler:                      # Start your loop
                # ... 
                # Calculate "ber" and "bler" values in your loop for current snrDb
                # ...

                # Call setData with 'bler' as the main metric and 'ber' as 'otherValues'.
                snrScheduler.setData(bler, ber)

            # After the loop, call the 'getSnrsAndData' method. It returns 3 numpy arrays for SNR values,
            # the corresponding BLER values, and the corresponding BER values.
            snrs, blers, bers = snrScheduler.getSnrsAndData()
        """
        # MUST be called near the end of each iteration
        self.setDataCalled = True                               # Record this function being called
        self.updateBuffers(value, *otherValues)                 # Save information
        while self.curSnr in self.buffers[0]:
            curSnrIdx = self.buffers[0].index(self.curSnr)
            self.updateState(self.buffers[1][curSnrIdx])
            if self.curSnr is None:     break
            self.curSnr = np.round(self.curSnr,4).item()        # Fixing the rounding errors
            
    # ******************************************************************************************************************
    def updateState(self, value):                           # The state machine
        # Round midpoint to the nearest multiple of `step` (biased toward zero via int truncation),
        # so that every SNR the algorithm visits lies on the same step-aligned grid.
        def midPoint():     return self.step * int((self.curHi + self.curLo)/(2*self.step))
        if self.state == 'Start':                               # Starting State
            if self.whereAmI(value)=='LoSNR':                       # At low point
                self.curLo = max(self.curSnr, self.curLo)               # Update the highest low point
                self.state = 'SearchingUp'                              # Go up searching for an in-range point
                self.curSnr += self.fastStep                            # Go up fast
            elif self.whereAmI(value)=='HiSNR':                     # At high point
                self.curHi = min(self.curSnr, self.curHi)               # Update the lowest high point
                self.state = 'SearchingDown'                            # Go down searching for an in-range point
                self.curSnr -= self.fastStep                            # Go down fast
            else:                                                   # Found a point in the range
                self.upStart = self.curSnr + self.step                  # Set return point to start going up after reaching low
                self.curSnr -= self.step                                # Go down
                self.state = 'GoingDown'                                # Update State
        elif self.state == 'SearchingUp':                       # Going up searching for an in-range point
            if self.whereAmI(value)=='LoSNR':                       # Was at low point, still at low point
                self.curLo = max(self.curSnr, self.curLo)               # Update the highest low point
                if self.curHi<np.inf:   self.curSnr = midPoint()        # We do have a high point. -> Go to midpoint
                else:                   self.curSnr += self.fastStep    # We don't have a high point -> Keep going up fast
            elif self.whereAmI(value)=='HiSNR':                     # Was at low point, now at high point -> Go to midpoint
                self.curHi = min(self.curSnr, self.curHi)               # Update the lowest high point
                self.state = 'SearchingDown'                            # Going back down
                self.curSnr = midPoint()                                # Go to midpoint (A multiple of step)
            else:                                                   # Found a point in the range
                self.upStart = self.curSnr + self.step                  # Set return point to start going up after reaching low
                self.curSnr -= self.step                                # Go down until we hit low
                self.state = 'GoingDown'                                # Update State
        elif self.state == 'SearchingDown':                     # Going down searching for an in-range point
            if self.whereAmI(value)=='HiSNR':                       # Was at high point, still at high point
                self.curHi = min(self.curSnr, self.curHi)               # Update the lowest high point
                if self.curLo>-np.inf:  self.curSnr = midPoint()        # We do have a low point -> Go to midpoint
                else:                   self.curSnr -= self.fastStep    # We don't have a low point -> Keep going down fast
            elif self.whereAmI(value)=='LoSNR':                     # Was at high point, now at low point -> Go to midpoint between low and high
                self.curLo = max(self.curSnr, self.curLo)               # Update the highest low point
                self.curSnr = midPoint()                                # Go to the midpoint (A multiple of step)
                self.state = 'SearchingUp'                              # Going back up
            else:                                                   # Found a point in the range
                self.upStart = self.curSnr + self.step                  # Set return point to start going up after reaching low
                self.curSnr -= self.step                                # Go down until we hit low
                self.state = 'GoingDown'                                # Update State
        elif self.state == 'GoingDown':                         # Going down to reach low point
            if self.whereAmI(value)=='LoSNR':                       # Was going down, hit low point
                self.curLo = max(self.curSnr, self.curLo)               # Update the highest low point
                self.curSnr -= self.step                                # Go down one more step
                self.state = 'AtLow'                                    # Update State
            else:                                                   # Was going down, not reached low point yet
                self.curSnr -= self.step                                # Keep going
        elif self.state == 'AtLow':                             # Reached low point
            if self.whereAmI(value)=='LoSNR':                       # Reached low point, and again at low point
                self.curSnr = self.upStart                              # Go back to midpoint and start going up until hitting  high point
                self.state = 'GoingUp'                                  # Update State
            elif self.whereAmI(value)=='HiSNR':                     # Reached low point, and now at high point => Unexpected behavior
                raise RuntimeError(f"Unexpected state reached in algorithm. (LoSNR -> going down -> HiSNR) SNR:{self.curSnr} Value:{value}")
            else:                                                   # Reached low point, but not at low any more
                self.curSnr -= self.step                                # Go down until we hit a new low
                self.state = 'GoingDown'                                # Update State
        elif self.state == 'GoingUp':                           # Going up to reach high point
            if self.whereAmI(value)=='HiSNR':                       # Was going up, hit high point
                self.curHi = min(self.curSnr, self.curHi)               # Update the lowest high point
                self.curSnr += self.step                                # Go up one more step
                self.state = 'AtHigh'                                   # Update State
            else:                                                   # Was going up, not reached high point yet
                self.curSnr += self.step                                # Keep going
        elif self.state == 'AtHigh':                            # Reached high point
            if self.whereAmI(value)=='HiSNR':                       # Reached high point, and again at high point
                self.state = 'Done'                                     # Update State
                self.curSnr = None                                      # This is needed to break the loop in the last line of 'setData'
            elif self.whereAmI(value)=='LoSNR':                     # Reached high point, and now at low point => Unexpected behavior
                raise RuntimeError(f"Unexpected state reached in algorithm. (HiSNR -> going up - LoSNR) SNR:{self.curSnr} Value:{value}")
            else:                                                   # Reached high point, but not at high any more
                self.curSnr += self.step                                # Go up until we hit a new high
                self.state = 'GoingUp'                                  # Update State
        
    # ******************************************************************************************************************
    def getSnrsAndData(self):
        r"""
        Return recorded SNRs and their associated metric arrays. Only SNRs within the converged
        bracket :math:`[curLo, curHi]` are returned; the bracketing-phase endpoints outside that
        range are silently dropped.

        Returns
        -------
        list or None
            ``None`` if no iterations have been run yet (i.e., :py:meth:`setData` was never called).
            Otherwise a list ``l`` of length ``n+2``, where ``n`` is the number of parameters in
            the ``otherValues`` argument of :py:meth:`setData`:

            - ``l[0]``: 1-D ``float32`` NumPy array of SNR values, sorted ascending.
            - ``l[1]``: 1-D ``float32`` NumPy array of the primary metric (e.g. BLER) values
              corresponding to ``l[0]``.
            - ``l[i+2]`` (``0 <= i < n``): a Python list of values for the i'th ``otherValues``
              parameter, ordered to match ``l[0]``. Elements may be scalars or NumPy arrays of
              variable shape (NumPy can't always stack them, so a list is returned).
        """
        if not self.buffers:    return None                                     # The case where no iterations happened
        snrs = self.buffers[0]
        idx = np.argsort(snrs)                                                  # Sort based on SNR values
        idx = [i for i in idx if snrs[i]>=self.curLo and snrs[i]<=self.curHi ]  # Drop the search SNRs at both sides
        retList = [ np.float32(self.buffers[0])[idx], np.float32(self.buffers[1])[idx] ]
        # Note that the size of otherValues for different SNRs may not be the same. So, we cannot
        # always make them a numpy array. We return a list of "objects" for each otherValue. The "objects"
        # in these lists may be numpy arrays and the numpy arrays in these list may have different shapes.
        # The SNRs and Values are still returned as numpy arrays.
        retList += [ [buffer[i] for i in idx] for buffer in self.buffers[2:] ]
        return retList
