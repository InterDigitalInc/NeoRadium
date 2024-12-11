# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``carrier.py`` implements the classes :py:class:`Carrier` and :py:class:`BandwidthPart`.
Each :py:class:`Carrier` class can be associated with several :py:class:`BandwidthPart` objects.
This implementation is based on **3GPP TR 38.211**.
"""
# ****************************************************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------------------------------------
# 05/18/2023    Shahab Hamidi-Rad       First version of the file.
# 11/02/2023    Shahab Hamidi-Rad       Completed the documentation
# ****************************************************************************************************************************************************
import numpy as np
import scipy.io
import matplotlib.colors as colors
import matplotlib.patches as patches
import time

from .grid import Grid

MAX_CARRIER_BW = 400e6      # 400MHz (It was 20MHz for LTE)
MAX_RESOURCE_BLOCKS = 275   # Max Number of RBs in a carrier
MIN_RESOURCE_BLOCKS = 20    # Min Number of RBs in a carrier

# ****************************************************************************************************************************************************
# Numerology Constants
# ****************************************************************************************************************************************************
Tc = 1./(480000*4096)
𝜅 = 64
Tc𝜅 = Tc*𝜅
SAMPLE_RATE = 1/Tc𝜅        # = 30,720,000

# ****************************************************************************************************************************************************
class BandwidthPart:
    r"""
    This class implements the functionality of a bandwidth part. A bandwidth
    part is a subset of contiguous common resource blocks for a given numerology
    on a given carrier. For more information please refer to **3GPP TR 38.211,
    section 4.4.5**
    """
    sampleRate = SAMPLE_RATE    # = 30,720,000
    # ************************************************************************************************************************************************
    def __init__(self, carrier, **kwargs):
        r"""
        Parameters
        ----------
        carrier: Carrier object
        kwargs : dict
            A set of optional arguments.

                :startRb: The starting resource block (RB). This is the
                    number of RBs from CRB 0. The default is 0.
                :numRbs: The number of RBs included in the bandwidth
                    part. The default is 50.
                :spacing: The subcarrier spacing in KHz. This also specifies
                    the Numerology used. To specify the subcarrier spacing,
                    you can use 15, 30, 60, 120, 240, 480, or 960. To specify
                    the Numerology, you can use 0, 1, ..., 6. Please refer to
                    **3GPP TR 38.211, section 4.2** for more details.
                :cpType: Cyclic Prefix type. It can be either "Normal" or
                    "Extended". The "Extended" type is only available for
                    60 KHz subcarrier spacing.
                :nFFT: The FFT Size used for OFDM modulation of the resource
                    grids (see :py:class:`~neoradium.grid.Grid`) which are
                    created based on this bandwidth part. If not specified,
                    nFFT is calcualted based on the number of subcarriers
                    (:math:`K`) as follows:
                    
                        - nFFT must be a power of 2 which is larger than the
                          number of subcarriers
                          
                        - At least 1/8 of FFT size must be unused (Guard bands
                          with zero power subcarriers). This means:
                          :math:`\frac {nFFT-K} {nFFT} \ge \frac 1 8`.


        **Other Properties:**
        
        Here is a list of additional properties:
        
            :u: The Numerology value. It is one of (0, 1, ..., 6).
            :bandwidth: The bandwidth of this bandwidth part in Hz.
            :symbolsPerSlot: The number of OFDM symbols in each slot. This is
                equal to 14 and 12 for "Normal" and "Extended" Cyclic Prefix types
                correspondingly.
            :slotsPerSubFrame: The number of slots per subframe based on current
                Numerology.
            :symbolLens: A list of symbol length values in number of time samples. It
                contains the symbol lengths for every symbol in a subframe.
            :slotsPerFrame: The number of slots per frame based on current Numerology.
            :symbolsPerSubFrame: The number of OFDM Symbols per subframe based on
                current Numerology.
            :slotNoInFrame: The slot number in current frame.
            :slotNoInSubFrame: The slot number in current subframe.
            :cellId: The Cell identifier of the Carrier containing this bandwidth
                part.
            :slotNo: Current slot number. A counter that can be used in simulation.
            :frameNo: Current frame number. A counter that can be used in simulation.
                This is incremented every ``slotsPerFrame`` slots.
            :sampleRate: The sample rate. For 3GPP, this is set to 30,720,000 samples
                per second.
            :dataTimeRatio: The average ratio of the amount of time in an OFDM symbol
                spent transmitting user data to total OFDM symbol time. This is always
                less than one because some duration of time is spend transmitting the
                Cyclic Prefix which doesn't carry useful information.
        """
        self.carrier = carrier
        self.startRb = kwargs.get('startRb', 0)         # Number of RBs from CRB 0
        self.numRbs = kwargs.get('numRbs', 50)
        
        spacing = kwargs.get('spacing', 15)
        scsps = [15,30,60,120,240,480,960]              # All allowed subcarrier spacings in KHz. See 3GPP TR 38.211 section 4.2
        if spacing in scsps:        self.u, self.spacing = scsps.index(spacing), spacing
        elif spacing in range(7):   self.u, self.spacing = spacing, scsps[spacing]
        else:                       raise ValueError("Invalid \"spacing\" values (%s)!"%(str(spacing)))
        assert self.u in range(7)
        assert self.spacing in scsps, ("Spacing:" + str(self.spacing))
        
        self.cpType = kwargs.get('cpType', 'normal').lower()    # Cyclic Prefix Type: 'normal' or 'extended'
        if self.cpType not in ['normal','extended']:
            raise ValueError("Unsupported cpType \"%s\"! It must be one of 'Normal' or 'Extended'"%(self.cpType))

        numSubCar = self.numRbs * 12
        
        minFFT = 1<<int(np.ceil(np.log2(numSubCar)))    # A power of 2 which is larger than number of subcarriers
        self.nFFT = kwargs.get('nFFT', None)
        if self.nFFT is None:
            # Deciding the FFT Size:
            #    - It must be a power of 2 which is larger than number of subcarriers (self.numRbs*12)
            #    - We must have at least 1/8 of FFT size unused (Guard bands with zero power subcarriers)
            # Note: Matlab uses 85% occupancy as the threshold.
            
            # The above method matches the following tables (except for the cases where the FFT size in the
            # tables are not a power of 2):
            #    - 3GPP TS 38.101-1, Tables F.5.3-1, F.5.3-2, F.5.3-3, F.5.4-1
            #    - 3GPP TS 38.101-2, Tables F.5.3-1, F.5.3-2, F.5.4-1
            #    - 3GPP TS 38.104, Tables B.5.2-1, B.5.2-2, B.5.2-3, B.5.2-4, C.5.2-1, C.5.2-2, C.5.2-3
            self.nFFT = minFFT if numSubCar/minFFT < (7/8) else minFFT<<1
        
            # This was used before the above change. I don't remember the source of this!
            # self.nFFT = max(minFFT, 2048//(1<<self.u))
        elif self.nFFT < minFFT:
            raise ValueError("'nFFT' must be equal or larger than %d!"%(minFFT))
        elif (self.nFFT&(self.nFFT-1)):
            raise ValueError("'nFFT' must be a power of 2!")
            
        self.bandwidth = numSubCar * self.spacing * 1000
        self.symbolsPerSlot = 14 if self.cpType=='normal' else 12
        self.slotsPerSubFrame = 1<<(self.u)
        self.symbolLens = np.int32([(self.nFFT+self.getCpLen(l)) for l in range(self.symbolsPerSubFrame)])
        self.dataTimeRatio = self.nFFT/(self.symbolLens.mean())
        
    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="Bandwidth Part Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`BandwidthPart` object.

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
        repStr += indent*' ' + "  Resource Blocks: %d RBs starting at %d (%d subcarriers)\n"%(self.numRbs, self.startRb, self.numRbs*12)
        repStr += indent*' ' + "  Subcarrier Spacing: %d KHz\n"%(self.spacing)
        repStr += indent*' ' + "  CP Type: %s\n"%(self.cpType)
        repStr += indent*' ' + "  bandwidth: %d Hz\n"%(self.bandwidth)
        repStr += indent*' ' + "  symbolsPerSlot: %d\n"%(self.symbolsPerSlot)
        repStr += indent*' ' + "  slotsPerSubFrame: %d\n"%(self.slotsPerSubFrame)
        repStr += indent*' ' + "  nFFT: %d\n"%(self.nFFT)
        if getStr: return repStr
        print(repStr)

    # ************************************************************************************************************************************************
    @property
    def slotsPerFrame(self):        return 10*self.slotsPerSubFrame
    @property
    def symbolsPerSubFrame(self):   return (self.symbolsPerSlot * self.slotsPerSubFrame)
    @property
    def slotNoInFrame(self):        return (self.slotNo % self.slotsPerFrame)
    @property
    def slotNoInSubFrame(self):     return (self.slotNo % self.slotsPerSubFrame)

    # ************************************************************************************************************************************************
    def __getattr__(self, property):
        # Get these properties from the 'carrier' object
        if property not in ["cellId", "slotNo", "frameNo"]:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.carrier, property)
    
    # ************************************************************************************************************************************************
    def createGrid(self, numPlanes, useReDesc=False):
        r"""
        Creates a resource grid and returns an empty :py:class:`~neoradium.grid.Grid`
        object based on this bandwidth part.

        Parameters
        ----------
        numPlanes : int
            The number of "planes" in the resource grid. See the
            :py:class:`~neoradium.grid.Grid` class for more information.
            
        useReDesc : Boolean (default: False)
            If ``True``, the resource grid created will also include additional
            fields that describe the content of each resource element (RE). This
            can be used during the debugging to make sure the resources are
            allocated correctly. See the :py:class:`~neoradium.grid.Grid` class
            for more information.

        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            An empty :py:class:`~neoradium.grid.Grid` object based on this
            bandwidth part object.
        """
        return Grid(self, numPlanes, useReDesc=useReDesc)

    # ************************************************************************************************************************************************
    def getCpTime(self, symIdxInSubframe):
        r"""
        Returns the duration of Cyclic Prefix in seconds for the OFDM symbol specified
        by ``symIdxInSubframe``.

        Parameters
        ----------
        symIdxInSubframe : int
            The index of symbol from the beginning of subFrame.

        Returns
        -------
        floating point number
            The duration of Cyclic Prefix in seconds for the OFDM symbol specified by
            ``symIdxInSubframe``.
        """
        if symIdxInSubframe>=self.symbolsPerSubFrame:
            raise ValueError("The 'symIdxInSubframe' must be less than number of OFDM Symbols in a subframe (%d)."%(self.symbolsPerSubFrame))
        return Tc𝜅 * self.getCpLen(symIdxInSubframe)
        
    # ************************************************************************************************************************************************
    def getCpLen(self, symIdxInSubframe):
        r"""
        Returns the number of time samples in Cyclic Prefix for the OFDM symbol
        specified by ``symIdxInSubframe``. This is based on **TS 38.211, Section 5.3.1**.

        Parameters
        ----------
        symIdxInSubframe : int
            The index of symbol from the beginning of subFrame.

        Returns
        -------
        int
            The number of time samples in Cyclic Prefix for the OFDM symbol
            specified by ``symIdxInSubframe``. .
        """
        # NOTE: The returned value is Ncp//𝜅 (Ncp as defined in the above section)
        if symIdxInSubframe>=self.symbolsPerSubFrame:
            raise ValueError("The 'symIdxInSubframe' must be less than number of OFDM Symbols in a subframe (%d)."%(self.symbolsPerSubFrame))
        if self.cpType=='normal':
            cpLen = 144//(1<<self.u)
            if symIdxInSubframe in [0, 7*(1<<self.u)]:  cpLen += 16
        else:
            cpLen = 512//(1<<self.u)
        return cpLen

    # ************************************************************************************************************************************************
    def getSymTime(self, symIdxInSubframe):
        r"""
        Returns the duration of the OFDM symbol specified by ``symIdxInSubframe``
        in seconds.

        Parameters
        ----------
        symIdxInSubframe : int
            The index of symbol from the beginning of subFrame.

        Returns
        -------
        floating point number
            The duration of the OFDM symbol specified by ``symIdxInSubframe`` in seconds.
        """
        # See TS 38.211 V17.0.0 (2021-12), Section 5.3.1
        if symIdxInSubframe>=self.symbolsPerSubFrame:
            raise ValueError("The 'symIdxInSubframe' must be less than number of OFDM Symbols in a subframe (%d)."%(self.symbolsPerSubFrame))
        return self.symbolLens[symIdxInSubframe] * Tc𝜅

    # ************************************************************************************************************************************************
    def getSlotTime(self, slotIndex=None):
        r"""
        Returns the duration of the slot specified by ``slotIndex`` in seconds.

        Parameters
        ----------
        slotIndex : int (default: None)
            The index of slot from the beginning of subFrame. If this is None, current
            slot (``slotNoInSubFrame``) is used.

        Returns
        -------
        floating point number
            The duration of the slot specified by ``slotIndex`` in seconds.
        """
        return self.getSlotLen(slotIndex) * Tc𝜅
        
    # ************************************************************************************************************************************************
    def getSlotLen(self, slotIndex=None):
        r"""
        Returns the total number of time samples in the slot specified by ``slotIndex``.

        Parameters
        ----------
        symIdxInSubframe : int (default: 0)
            The index of symbol from the beginning of subFrame.

        Returns
        -------
        int
            The number of time samples in Cyclic Prefix for the OFDM symbol
            specified by ``symIdxInSubframe``. .
        """
        # s is the slot numer in subframe
        if slotIndex is None:   slotIndex = self.slotNoInSubFrame
        if slotIndex>=self.slotsPerSubFrame:
            raise ValueError("The 'slotIndex' must be less than number of slots in a subframe (%d)."%(self.slotsPerSubFrame))

        ls = range(slotIndex*self.symbolsPerSlot, (slotIndex+1)*self.symbolsPerSlot)
        return sum( self.symbolLens[ls] )

    # ************************************************************************************************************************************************
    def getSlotsDuration(self, numSlots):
        r"""
        Returns the total durations in seconds for the next ``numSlots`` slots,
        starting from current slot in current subframe (``slotNoInSubFrame``).

        Parameters
        ----------
        numSlots : int
            The number of next slots to include.

        Returns
        -------
        floating point number
            The total durations in seconds for the next ``numSlots`` slots.
        """
        return self.getSlotsSamples(numSlots) * Tc𝜅
        
    # ************************************************************************************************************************************************
    def getSlotsSamples(self, numSlots):
        r"""
        Returns total number of samples in the next ``numSlots`` slots,
        starting from current slot in current subframe (``slotNoInSubFrame``).

        Parameters
        ----------
        numSlots : int
            The number of next slots to include.

        Returns
        -------
        int
            The total number of samples in the next ``numSlots`` slots.
        """
        # The number of samples in the next 'numSlots' starting from current 'slotNoInSubFrame'
        duration = (numSlots//self.slotsPerSubFrame)*self.sampleRate//1000
        numRemainingSlots = numSlots % self.slotsPerSubFrame
        for slot in range(numRemainingSlots):
            slotIdxInSubFrame = (slot + self.slotNoInSubFrame) % self.slotsPerSubFrame
            duration += self.getSlotLen(slotIdxInSubFrame)
        return duration

    # ************************************************************************************************************************************************
    def getNumSlotsForSamples(self, ns):
        r"""
        Returns the number of slots that are completely covered by the next
        ``ns`` samples, starting from current slot in current subframe
        (``slotNoInSubFrame``).

        Parameters
        ----------
        ns : int
            The number time samples.

        Returns
        -------
        int
            The number of slots that are completely covered by the next
            ``ns`` samples.
        """
        numSlots = 0
        remainingSamples = ns
        s = self.slotNoInSubFrame
        slotLen = self.getSlotLen(s)
        while remainingSamples >= slotLen:
            numSlots += 1
            remainingSamples -= slotLen
            s = (s+1)%self.slotsPerSubFrame
            slotLen = self.getSlotLen(s)

        return numSlots

    # ************************************************************************************************************************************************
    def getNumSymbolsForSamples(self, ns):
        r"""
        Returns the number of OFDM symbols that are completely covered by the next
        ``ns`` samples, starting from the beginning of current slot in current
        subframe (``slotNoInSubFrame``).

        Parameters
        ----------
        ns : int
            The number time samples.

        Returns
        -------
        int
            The number of OFDM symbols that are completely covered by the next
            ``ns`` samples.
        """
        l = self.slotNoInSubFrame * elf.symbolsPerSlot
        numSymbols = 0
        remainingSamples = ns
        symLen = self.symbolLens[l]
        while remainingSamples>symLen:
            numSymbols += 1
            remainingSamples -= symLen
            l = (l+1)%self.symbolsPerSubFrame
            symLen = self.symbolLens[l]

        return numSymbols

    # ************************************************************************************************************************************************
    def getSymbolDurationsForNextSlots(self, numSlots):
        # Returns a list containing the OFDM symbol durations for the
        # OFDM symbols in the next "numSlots" starting from current
        # slot in the current subframe. (No Documentation)
        return self.getSymLensForNextSlots(numSlots) * Tc𝜅

    # ************************************************************************************************************************************************
    def getSymLensForNextSlots(self, numSlots):
        # Returns a list containing the number of samples in each
        # OFDM symbol in the next "numSlots" starting from current
        # slot in the current subframe. (No Documentation)
        s = 0
        symLens = []
        remainingSlots = numSlots
        if self.slotNoInSubFrame > 0:
            start = self.symbolsPerSlot * self.slotNoInSubFrame
            end = start + remainingSlots * self.symbolsPerSlot
            symLens += self.symbolLens[ start : end ].tolist()
            remainingSlots -= len(symLens)//self.symbolsPerSlot
        
        if remainingSlots > self.slotsPerSubFrame:
            symLens += (remainingSlots//self.slotsPerSubFrame) * self.symbolLens.tolist()
            remainingSlots %= self.slotsPerSubFrame
        
        if remainingSlots > 0:
            symLens += self.symbolLens[: remainingSlots*self.symbolsPerSlot].tolist()

        return np.int32(symLens)

# ****************************************************************************************************************************************************
class Carrier:
    r"""
    This class implements the functionality of a Carrier. A Carrier object
    is used to specify a group of resource blocks used for uplink or downlink
    communication. A Carrier object can be associated with several
    :py:class:`BandwidthPart` objects but only one can be active at any time.
    """
    # See TS 38.211 V17.0.0 (2021-12), Section 4.4.2
    sampleRate = SAMPLE_RATE    # = 30,720,000
    # ************************************************************************************************************************************************
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments.

                :startRb: The starting resource block (RB). This is the number
                    of RBs from CRB 0. The default is 0.
                :numRbs: The number of RBs included in the bandwidth part. The
                    default is 50.
                :bwps: A list of :py:class:`BandwidthPart` objects associated
                    with this Carrier. If this is not specified, a single bandwidth
                    part is automatically created which covers the whole carrier.
                :cellId: The Cell identifier of this Carrier. The default is 1.
                :curBwpIndex: The index of current bandwidth part. The default is 0.


        **Other Properties:**

        Here is a list of additional properties:

            :slotNo: Current slot number. A counter that can be used in simulation.
            :frameNo: Current frame number. A counter that can be used in simulation.
                This is incremented every ``slotsPerFrame`` slots.
            :curBwp: The currently active :py:class:`BandwidthPart` object.
            :frameNoRel: The remainder of current frame number divided by 1024.
            :slotNoInFrame: The slot number in current frame.
            :symbolsPerSlot: The number of OFDM symbols in each slot based on the
                Numerology of the currently active :py:class:`BandwidthPart`.
            :slotsPerSubFrame: The number of slots per subframe based on the Numerology
                of the currently active :py:class:`BandwidthPart`.
            :slotsPerFrame: The number of slots per frame based on the Numerology of
                the currently active :py:class:`BandwidthPart`.
            :symbolsPerSubFrame: The number of OFDM Symbols per subframe based on the
                Numerology of the currently active :py:class:`BandwidthPart`.
        """
        self.startRb = kwargs.get('startRb', 0)         # Number of RBs from CRB 0
        self.numRbs = kwargs.get('numRbs', 50)

        # If no BWP is given, we automatically create one which covers the whole carrier
        self.bwps = kwargs.get('bwps', [ BandwidthPart(self, **kwargs) ])
        self.cellId = kwargs.get('cellId', 1)
        self.curBwpIndex = kwargs.get('curBwpIndex', 0)
        self.dcLocation = kwargs.get('dcLocation', 0)   # 0-3299 (or 3300 to indicate that DC subcarrier is out side of the carrier)
        
        # Absolute counters (These values keep increasing when used in a loop)
        self.slotNo = 0
        self.frameNo = 0

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="Carrier Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`Carrier` object.

        Parameters
        ----------
        indent : int (default: 0)
            The number of indentation characters.
            
        title : str (default: None)
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
        repStr += indent*' ' + "  startRb: %d\n"%(self.startRb)
        repStr += indent*' ' + "  numRbs: %d\n"%(self.numRbs)
        repStr += indent*' ' + "  Cell Id: %d\n"%(self.cellId)
        repStr += indent*' ' + "  Active Bandwidth Part: %d\n"%(self.curBwpIndex)
        repStr += indent*' ' + "  Bandwidth Parts: %d\n"%(len(self.bwps))
        for i,bwp in enumerate(self.bwps):
            repStr += bwp.print(indent+2, "Bandwidth Part %d:"%(i), True)
        if getStr: return repStr
        print(repStr)

    # ************************************************************************************************************************************************
    @property
    def curBwp(self):           return self.bwps[self.curBwpIndex]
    @property
    def frameNoRel(self):       return (self.frameNo + self.slotNo//self.slotsPerFrame)%1024
    @property
    def slotNoInFrame(self):    return self.slotNo % self.slotsPerFrame

    # ************************************************************************************************************************************************
    def __getattr__(self, property):
        # Get these properties from the 'curBwp' object
        if property not in ["symbolsPerSlot", "slotsPerSubFrame", "slotsPerFrame", "symbolsPerSubFrame"]:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.curBwp, property)

    # ************************************************************************************************************************************************
    def goNext(self):
        r"""
        Increments current slot number in this carrier (``slotNo``). If
        the slot number passes the boundary of a frame, the a frame
        number (``frameNo``) is also incremented.
        """
        self.slotNo += 1
        if (self.slotNo % self.slotsPerFrame)==0:
            self.frameNo += 1
            
    # ************************************************************************************************************************************************
    def createGrid(self, numPorts, useReDesc=False):
        r"""
        Creates a resource grid and returns an empty
        :py:class:`~neoradium.grid.Grid` object based on the
        currently active :py:class:`BandwidthPart`. See
        :py:meth:`BandwidthPart.createGrid` for more details.
        """
        return self.curBwp.createGrid(numPorts, useReDesc=useReDesc)

