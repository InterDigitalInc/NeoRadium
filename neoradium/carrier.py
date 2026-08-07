# Copyright (c) 2024-2026, InterDigital AI Lab
"""
The module ``carrier.py`` implements the classes :py:class:`Carrier` and :py:class:`BandwidthPart`. Each 
:py:class:`Carrier` can be associated with several :py:class:`BandwidthPart` objects. This implementation is 
based on **3GPP TS 38.211**.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 05/18/2023    Shahab Hamidi-Rad       First version of the file.
# 11/02/2023    Shahab                  Completed the documentation
# 05/01/2025    Shahab                  Updated the nFFT calculations and removed some unused functions. Also
#                                       improved the BandwidthPart documentation.
# 03/12/2026    Shahab                  Changes in NeoRadium version 0.5.0:
#                                       * Interleaving process now happens at the BWP level. See the new
#                                         interleavingBundleSize parameter for the BandwidthPart class.
#                                       * You can now directly create a bandwidth part without explicitly creating
#                                         a Carrier. A Carrier object is automatically created in this case.
#                                       * Removed the 'useReDesc' from the 'createGrid' methods of 'BandwidthPart' and
#                                         'Carrier' classes.
# **********************************************************************************************************************
import numpy as np
import scipy.io
import time

from .grid import Grid
from .utils import freqStr, validateRange, warnOnce

MAX_CARRIER_BW = 400e6      # 400 MHz (It was 20 MHz for LTE)
MAX_RESOURCE_BLOCKS = 275   # Per-spec max RBs in a carrier (TS 38.101). The *effective* per-BWP max depends
                            # on the subcarrier spacing — see the 'numRbs < nFFT//12' check in BandwidthPart.
MIN_RESOURCE_BLOCKS = 20    # Min number of RBs in a carrier

# **********************************************************************************************************************
# Numerology Constants
Tc = 1./(480000*4096)
𝜅 = 64
Tc𝜅 = Tc*𝜅
SAMPLE_RATE = 1/Tc𝜅        # = 30,720,000

# **********************************************************************************************************************
class BandwidthPart:
    r"""
    This class encapsulates the functionality of a bandwidth part. A bandwidth part is a subset of contiguous common
    resource blocks for a specific numerology on a given carrier. For more detailed information, please refer to 
    3GPP TS 38.211, section 4.4.5. You typically create a :py:class:`Carrier` object and retrieve its current
    BandwidthPart using its ``curBwp`` property. Since we usually work with bandwidth parts in simulations, you can 
    also instantiate a :py:class:`BandwidthPart` object directly passing both numerology and carrier information. For
    example, the following line create both a :py:class:`BandwidthPart` and a :py:class:`Carrier` object. A reference
    to the carrier objects is available as ``bwp.carrier``.
    
    .. code-block:: python

        # Create a BandwidthPart. This internally creates a carrier with 24 resource blocks. 
        bwp = BandwidthPart(numRbs=24, spacing=30, interleavingBundleSize=2)

    """
    sampleRate = SAMPLE_RATE    # = 30,720,000
    # ******************************************************************************************************************
    def __init__(self, carrier=None, **kwargs):
        r"""
        Parameters
        ----------
        carrier : :py:class:`Carrier` or None
            The carrier associated with this bandwidth part. If `None`, a new carrier is created with only one
            bandwidth part (this object) and its properties can be set using the parameters passed in kwargs. For 
            example:
            
            .. code-block:: python

                # Create a BandwidthPart. This internally creates a carrier with 24 resource blocks. 
                bwp = BandwidthPart(numRbs=24, spacing=30, interleavingBundleSize=2)
            
        kwargs : dict
            A set of optional arguments.

                :startRb: The starting resource block (RB). This is the number of RBs from CRB 0. The default is 0.
                
                :numRbs: The number of RBs included in the bandwidth part. The default is 50.
                
                :spacing: The subcarrier spacing in kHz. This also specifies the numerology used. To specify the 
                    subcarrier spacing, you can use 15, 30, 60, 120, 240, 480, or 960. To specify the numerology, you
                    can use 0, 1, ..., 6. Please refer to **3GPP TS 38.211, section 4.2** for more details.
                    
                :cpType: Cyclic Prefix type. It can be either "Normal" or "Extended". The "Extended" type is only
                    available for 60 kHz subcarrier spacing.
                    
                :interleavingBundleSize: The bundle size of the interleaving process. It can be one of 0 (default),
                    2, or 4. The value 0 means interleaving is disabled (default). See **3GPP TS 38.211, 
                    Section 7.3.1.6** for more information.


        **Other Properties:**
        
        Here is a list of additional properties:
        
            :u: The numerology value, which falls within the range of 0 to 6 (:math:`\mu`). See 3GPP TS 38.211, 
                table 4.2-1.
            
            :bandwidth: The bandwidth of this bandwidth part in Hz.
            
            :nFFT: The FFT size used for OFDM modulation of the resource grids (See :py:class:`~neoradium.grid.Grid`),
                which are created based on this bandwidth part. It is calculated as follows:
                
                .. math::

                    N_{FFT} = \big [\frac {\frac {f_s} {1000} - \sum_{l=0}^{N_{symb}^{slot}-1} N_{CP,l}^{\mu}} {N_{symb}^{subframe,\mu}} \big ]
                    
                where :math:`f_s=\frac 1 {T_s}`, is the 5G sample rate (:math:`f_s=30,720,000` Hz), 
                :math:`N_{symb}^{slot}` is the number of symbols per slot, :math:`N_{CP,l}^{\mu}` is the number of 
                samples in the cyclic prefix of symbol :math:`l` based on numerology :math:`\mu`, and
                :math:`N_{symb}^{subframe,\mu}` is the number of symbols in each subframe for numerology :math:`\mu`.
                    
            :symbolsPerSlot: The number of OFDM symbols in each slot (:math:`N_{symb}^{slot}`). This is equal to 14 and 
                12 for "Normal" and "Extended" Cyclic Prefix types, respectively.
                
            :slotsPerSubFrame: The number of slots per subframe based on the current numerology 
                (:math:`N_{slot}^{subframe,\mu}`).
                
            :symbolLens: A list of symbol length values in number of time samples for every symbol in a subframe. The
                symbol length for symbol ``l``, ``symbolLens[l]``, is the sum of :math:`N_{FFT}` and 
                :math:`N_{CP,l}^{\mu}`.
                
            :slotsPerFrame: The number of slots per frame based on current numerology (:math:`N_{slot}^{frame,\mu}`).
                
            :symbolsPerSubFrame: The number of OFDM Symbols per subframe based on current numerology 
                (:math:`N_{symb}^{subframe,\mu}`).
                
            :slotNoInFrame: The slot number in the current frame (:math:`n_{s,f}^{\mu}`).
            
            :slotNoInSubFrame: The slot number in the current subframe (:math:`n_{s}^{\mu}`).
            
            :avgSlotDuration: The average slot duration in seconds. 
            
            :cellId: The cell identifier of the Carrier containing this bandwidth part.
            
            :slotNo: Current slot number. A counter that can be used in simulation.
            
            :frameNo: Current frame number. A counter that can be used in simulation. This is incremented every 
                ``slotsPerFrame`` slots.
                
            :sampleRate: The sample rate. For 3GPP, this is set to 30,720,000 samples per second 
                (:math:`f_s=\frac 1 {T_s}`).
            
            :dataTimeRatio: The average ratio of the amount of time in an OFDM symbol spent transmitting user data 
                to total OFDM symbol time. This is always less than one because some duration of time is spent 
                transmitting the Cyclic Prefix, which does not carry useful information.
        """
        self.carrier = carrier
        if self.carrier is None:    self.carrier = Carrier(bwps=[self], **kwargs)
        self.startRb = kwargs.get('startRb', 0)         # Number of RBs from CRB 0
        self.numRbs = kwargs.get('numRbs', 50)
        
        spacing = kwargs.get('spacing', 15)
        scsps = [15,30,60,120,240,480,960]              # All allowed subcarrier spacings in kHz. See 3GPP TS 38.211 section 4.2
        if spacing in scsps:        self.u, self.spacing = scsps.index(spacing), spacing
        elif spacing in range(7):   self.u, self.spacing = spacing, scsps[spacing]
        else:                       raise ValueError("Invalid \"spacing\" value (%s)!"%(str(spacing)))
        assert self.u in range(7)
        assert self.spacing in scsps, ("Spacing:" + str(self.spacing))
        
        self.cpType = kwargs.get('cpType', 'normal').lower()    # Cyclic Prefix Type: 'normal' or 'extended'
        validateRange(self.cpType, ['normal','extended'])
        if self.cpType == 'extended' and self.spacing != 60:
            raise ValueError(f"'Extended' cpType is only supported for 60 kHz subcarrier spacing "
                             f"(got {self.spacing} kHz). See TS 38.211 Section 5.3.1.")

        self.interleavingBundleSize = kwargs.get('interleavingBundleSize', 0)
        validateRange(self.interleavingBundleSize, [0,2,4])
        if self.interleavingBundleSize>0:
            if (self.numRbs % self.interleavingBundleSize) != 0:
                raise ValueError(f"'numRbs' must be a multiple of {self.interleavingBundleSize}")
            if (self.startRb % self.interleavingBundleSize) != 0:
                raise ValueError(f"'startRb' must be a multiple of {self.interleavingBundleSize}")
        self.setVrbToPrbMapping()
                
        numSubCar = self.numRbs * 12
        
        self.bandwidth = numSubCar * self.spacing * 1000
        self.symbolsPerSlot = 14 if self.cpType=='normal' else 12
        self.slotsPerSubFrame = 1<<(self.u)
        cpLens = np.int32([self.getCpLen(l) for l in range(self.symbolsPerSubFrame)]) # CP len for all subframe symbols

        # nFFT is calculated based on subframe length and CP lengths
        self.nFFT = int((self.sampleRate//1000-cpLens.sum())//self.symbolsPerSubFrame)
        if self.numRbs>=self.nFFT//12:
            raise ValueError(f"'numRbs' must be less than nFFT/12 (={self.nFFT//12})!")
        assert (self.nFFT&(self.nFFT-1))==0, f"ERROR: nFFT ({self.nFFT}) is not a power of 2!"

        self.symbolLens = cpLens + self.nFFT
        # Adding the first symbol length to the end to help with the fact that we always get symbolsPerSlot+1 symLen
        # values.
        self.symbolLens = np.append(self.symbolLens, self.symbolLens[0])
        self.dataTimeRatio = self.nFFT/(self.symbolLens.mean())
        
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`BandwidthPart` object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "Bandwidth Part Properties:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns
            the information in a string. Otherwise, nothing is returned.
        """
        if title is None:   title = "Bandwidth Part Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        rbStr = f"{self.numRbs} RBs starting at {self.startRb} ({self.numRbs*12} subcarriers)"
        repStr += indent*' ' + f"  Resource Blocks:    {rbStr}\n"
        repStr += indent*' ' + f"  Subcarrier Spacing: {self.spacing} kHz\n"
        repStr += indent*' ' + f"  CP Type:            {self.cpType}\n"
        interleavingStr = 'No' if self.interleavingBundleSize==0 else f'Yes (Bundle size:{self.interleavingBundleSize})'
        repStr += indent*' ' + f"  Interleaving:       {interleavingStr}\n"
        repStr += indent*' ' + f"  Bandwidth:          {freqStr(self.bandwidth)}\n"
        repStr += indent*' ' + f"  symbolsPerSlot:     {self.symbolsPerSlot}\n"
        repStr += indent*' ' + f"  slotsPerSubFrame:   {self.slotsPerSubFrame}\n"
        repStr += indent*' ' + f"  nFFT:               {self.nFFT}\n"
        repStr += indent*' ' + f"  frameNo:            {self.frameNo}\n"
        repStr += indent*' ' + f"  slotNo:             {self.slotNo}\n"

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    def slotsPerFrame(self):        return 10*self.slotsPerSubFrame
    @property
    def symbolsPerSubFrame(self):   return (self.symbolsPerSlot * self.slotsPerSubFrame)
    @property
    def slotNoInFrame(self):        return (self.slotNo % self.slotsPerFrame)
    @property
    def slotNoInSubFrame(self):     return (self.slotNo % self.slotsPerSubFrame)
    @property
    def avgSlotDuration(self):      return 1000./self.slotsPerSubFrame

    # ******************************************************************************************************************
    def __getattr__(self, attrName):
        # Get these properties from the 'carrier' object
        if attrName not in ["cellId", "slotNo", "frameNo", "goNext", "restart"]:
            raise AttributeError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, attrName))
        return getattr(self.carrier, attrName)
    
    # ******************************************************************************************************************
    def setVrbToPrbMapping(self):                               # Undocumented - Not intended for direct use
        # If interleaving is enabled for this bandwidth part, this function creates the mapping between the
        # virtual resource blocks and physical resource blocks based on TS 38.211, Section 7.3.1.6.
        # See also Fig. 9.12 in the "5G NR" book
        if self.interleavingBundleSize == 0:
            self.vrb2Prb = self.prb2Vrb = np.arange(self.numRbs)   # Interleaving is disabled => VRB ≡ PRB
            return
        
        # We are assuming 'startRb' and 'numRbs' are multiples of 'interleavingBundleSize':
        numBundles = self.numRbs//self.interleavingBundleSize
        
        # Creating the f function as defined in TS 38.211 - Section 7.3.1.6
        numBundles2 = numBundles-(numBundles%2)
        f = np.arange(numBundles)
        f[:numBundles2] = f[:numBundles2].reshape(2,-1).T.reshape(-1)
        
        self.vrb2Prb = (self.interleavingBundleSize*f[:,None] + np.arange(self.interleavingBundleSize)).flatten()
        self.prb2Vrb = np.argsort(self.vrb2Prb)

    # ******************************************************************************************************************
    def createGrid(self, numPorts=1, numPlanes=None):
        r"""
        Creates a resource grid and returns an empty :py:class:`~neoradium.grid.Grid`
        object based on this bandwidth part.

        Parameters
        ----------
        numPorts : int
            The number of "ports" in the resource grid. See the
            :py:class:`~neoradium.grid.Grid` class for more information.
            
        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            An empty :py:class:`~neoradium.grid.Grid` object based on this
            bandwidth part object.
        """
        if numPlanes is not None:
            warnOnce("createGrid: 'numPlanes' argument is deprecated. Use 'numPorts' instead.")
            numPorts = numPlanes
        return Grid(self, numPorts)

    # ******************************************************************************************************************
    def getCpLen(self, symIdxInSubFrame):
        r"""
        Returns the number of time samples in the Cyclic Prefix for the OFDM symbol
        specified by ``symIdxInSubFrame``. This is based on **TS 38.211, Section 5.3.1**.

        Parameters
        ----------
        symIdxInSubFrame : int
            The index of the symbol from the beginning of the subframe.

        Returns
        -------
        int
            The number of time samples in the cyclic prefix for the OFDM symbol
            specified by ``symIdxInSubFrame``.
        """
        # NOTE: The returned value is Ncp//𝜅 (Ncp as defined in the above section)
        if symIdxInSubFrame>=self.symbolsPerSubFrame:
            raise ValueError("'symIdxInSubFrame' must be less than the number of OFDM Symbols in a " +
                            f"subframe ({self.symbolsPerSubFrame}).")
        if self.cpType=='normal':
            cpLen = 144//(1<<self.u)
            # Normal-CP "long" symbols: each 0.5 ms half-subframe boundary gets a slightly longer CP.
            # Indices are 0 and 7*2^μ (the first symbol of the second 0.5 ms half), per TS 38.211 §5.3.1.
            if symIdxInSubFrame in [0, 7*(1<<self.u)]:  cpLen += 16
        else:
            cpLen = 512//(1<<self.u)
        return cpLen

    # ******************************************************************************************************************
    def getSlotLen(self, slotIndex=None):
        r"""
        Returns the total number of time samples in the slot specified by ``slotIndex``.

        Parameters
        ----------
        slotIndex : int
            The index of the slot from the beginning of subframe.

        Returns
        -------
        int
            The total number of time samples in the slot specified by ``slotIndex``.
        """
        # s is the slot number in subframe
        if slotIndex is None:   slotIndex = self.slotNoInSubFrame
        if slotIndex>=self.slotsPerSubFrame:
            raise ValueError(f"'slotIndex' must be less than number of slots in a subframe ({self.slotsPerSubFrame}).")

        ls = range(slotIndex*self.symbolsPerSlot, (slotIndex+1)*self.symbolsPerSlot)
        return sum( self.symbolLens[ls] )

    # ******************************************************************************************************************
    def getSymLens(self):
        r"""
        Returns an array containing the symbol lengths for the symbols in the current slot, plus the first symbol of 
        the next slot. The symbol length represents the total number of samples (at a sampling rate of 30,720,000 
        samples per second) for each symbol.

        Returns
        -------
        NumPy array
            An array containing the symbol lengths for all the symbols in the current slot, plus the first symbol of 
            the next slot. Therefore, the length of the returned array is ``symbolsPerSlot+1``.
        """
        # Returns symbol lengths for the next symbolsPerSlot+1 symbols
        start = self.symbolsPerSlot * self.slotNoInSubFrame
        return self.symbolLens[start: start+self.symbolsPerSlot+1 ]

# **********************************************************************************************************************
class Carrier:
    r"""
    This class encapsulates the functionality of a Carrier. A Carrier object serves as a container for a group of
    resource blocks dedicated to either uplink or downlink communication. It is possible to associate a Carrier
    object with multiple instances of the :py:class:`BandwidthPart` class, but only one instance can be active at any
    time.
    """
    # See TS 38.211, Section 4.4.2
    sampleRate = SAMPLE_RATE    # = 30,720,000
    # ******************************************************************************************************************
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments.

                :startRb: The starting resource block (RB). This is the number of RBs from CRB 0. The default is 0.
                
                :numRbs: The number of RBs included in the carrier. The default is 52.
                
                :bwps: A list of :py:class:`BandwidthPart` objects associated with this Carrier. If this is not 
                    specified, a single bandwidth part is automatically created covering the whole carrier. In this 
                    case, the following additional :py:class:`BandwidthPart` parameters can also be specified when 
                    creating the Carrier object:
                    
                        :spacing: The subcarrier spacing in kHz. This also specifies the numerology used. To specify
                            the subcarrier spacing, you can use 15, 30, 60, 120, 240, 480, or 960. To specify the 
                            numerology, you can use 0, 1, ..., 6. Please refer to **3GPP TS 38.211, section 4.2** 
                            for more details.
                    
                        :cpType: Cyclic Prefix type. It can be either "Normal" or "Extended". The "Extended" type is
                            only available for 60 kHz subcarrier spacing.
                            
                        :interleavingBundleSize: The bundle size of the interleaving process. It can be one of 0 
                            (default), 2, or 4. The value 0 means interleaving is disabled (default). See **3GPP 
                            TS 38.211 Section 7.3.1.6** for more information.
                            
                :cellId: The cell identifier of this carrier. The default is 1.
                
                :curBwpIndex: The index of the current bandwidth part. The default is 0.


        **Other Properties:**

        Here is a list of additional properties:

            :slotNo: Current slot number. A counter that can be used in simulation.
            
            :frameNo: Current frame number. A counter that can be used in simulation. This is incremented every
                ``slotsPerFrame`` slots.
                
            :curBwp: The currently active :py:class:`BandwidthPart` object.
            
            :frameNoRel: The remainder of the current frame number divided by 1024.
            
            :slotNoInFrame: The slot number in the current frame (:math:`n_{s,f}^{\mu}`).
            
            :symbolsPerSlot: The number of OFDM symbols in each slot (:math:`N_{symb}^{slot}`) based on the numerology
                of the currently active :py:class:`BandwidthPart`. This is equal to 14 and 12 for "Normal" and 
                "Extended" Cyclic Prefix types, respectively.

            :slotsPerSubFrame: The number of slots per subframe based on the numerology of the currently active 
                :py:class:`BandwidthPart` (:math:`N_{slot}^{subframe,\mu}`).

            :slotsPerFrame: The number of slots per frame based on the numerology of the currently active 
                :py:class:`BandwidthPart` (:math:`N_{slot}^{frame,\mu}`).

            :symbolsPerSubFrame: The number of OFDM symbols per subframe based on the numerology of the currently 
                active :py:class:`BandwidthPart` (:math:`N_{symb}^{subframe,\mu}`).

                
        **Example:**
        
        .. code-block:: python

            # Create a carrier with a single BandwidthPart:
            carrier = Carrier(startRb=0, numRbs=25, spacing=30, cpType="Normal")

        """
        self.startRb = kwargs.get('startRb', 0)         # Number of RBs from CRB 0
        self.numRbs = kwargs.get('numRbs', 52)

        # If no BWP is given, we automatically create one, which covers the whole carrier
        self.bwps = kwargs.get('bwps', [ BandwidthPart(self, **kwargs) ])
        self.cellId = kwargs.get('cellId', 1)
        self.curBwpIndex = kwargs.get('curBwpIndex', 0)
        self.dcLocation = kwargs.get('dcLocation', 0)   # 0-3299 (or 3300 to indicate that DC subcarrier is
                                                        # outside of the carrier)
        
        # Absolute counters (these values keep increasing when used in a loop)
        self.slotNo = 0
        self.frameNo = 0

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`Carrier` object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "Carrier Properties:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Carrier Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  Cell Id:              {self.cellId}\n"
        repStr += indent*' ' + f"  Bandwidth Parts:      {len(self.bwps)}\n"
        repStr += indent*' ' + f"  Active BWP:           {self.curBwpIndex}\n"
        for i, bwp in enumerate(self.bwps):
            repStr += bwp.print(indent+2, f"Bandwidth Part {i}:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    def curBwp(self):           return self.bwps[self.curBwpIndex]
    @property
    def frameNoRel(self):       return (self.frameNo + self.slotNo//self.slotsPerFrame)%1024
    @property
    def slotNoInFrame(self):    return self.slotNo % self.slotsPerFrame

    # ******************************************************************************************************************
    def __getattr__(self, attrName):
        # Get these properties from the 'curBwp' object
        if attrName not in ["symbolsPerSlot", "slotsPerSubFrame", "slotsPerFrame", "symbolsPerSubFrame"]:
            raise AttributeError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, attrName))
        return getattr(self.curBwp, attrName)

    # ******************************************************************************************************************
    def restart(self):
        r"""
        Resets this carrier's slot and frame counters to zero. Use this to rewind the carrier's timing
        state to the beginning of a simulation run (e.g., between Monte-Carlo trials). The counterpart
        :py:meth:`goNext` advances the counters by one slot.
        """
        self.slotNo = 0
        self.frameNo = 0

    # ******************************************************************************************************************
    def goNext(self):
        r"""
        Increments the current slot number in this carrier (``slotNo``). If the slot number crosses a frame boundary, 
        the frame number (``frameNo``) is also incremented.
        """
        self.slotNo += 1
        if (self.slotNo % self.slotsPerFrame)==0:   self.frameNo += 1
            
    # ******************************************************************************************************************
    def createGrid(self, numPorts):
        r"""
        Creates a resource grid and returns an empty :py:class:`~neoradium.grid.Grid` object based on the currently
        active :py:class:`BandwidthPart`. See :py:meth:`BandwidthPart.createGrid` for more details.
        """
        return self.curBwp.createGrid(numPorts)

# **********************************************************************************************************************
class ReservedPrbSet:  # See TS 38.214, Section 5.1.4.1
    """
    Represents a set of reserved resource blocks and OFDM symbols within a BWP, as defined in **3GPP TS 38.214
    Section 5.1.4.1**. A reserved resource set is used to mark time-frequency resources that should not be used for
    PDSCH transmission. The reservation is defined by:

        - a set of reserved PRB indices,
        - a set of reserved OFDM symbol indices within one or two slots, and
        - a periodic activity pattern.

    The RB and symbol reservations may be provided either as bitmaps or as lists of indices. The activity pattern
    is provided as a bitmap string, where a '1' indicates that the corresponding reservation unit is active.

    Notes
    -----
    - Bitmap strings use the least significant bit (LSB) on the right, i.e. the last character in the string
      corresponds to bit 0.
    - The symbol bitmap may describe one slot or two consecutive slots.
    - The pattern length must be one of the values allowed by **3GPP TS 38.214 Section 5.1.4.1**.
    """
    # ******************************************************************************************************************
    def __init__(self, bwp, prbs, symbols, pattern="1"):
        r"""
        Parameters
        ----------
        bwp : BandwidthPart
            The bandwidth part to which this reservation applies.

        prbs : str or list of int
            Reserved physical resource blocks, corresponding to ``resourceBlocks`` in **3GPP TS 38.214**.

            - If a string is provided, it is interpreted as a bitmap of '0's and '1's, with the LSB on the right. A 
              '1' means the corresponding RB is reserved.
            - If a list is provided, it is interpreted as the reserved symbol indices.

        symbols : str or list of int
            Reserved OFDM symbols, corresponding to ``symbolsInResourceBlock`` in **3GPP TS 38.214**.

            - If a string is provided, it is interpreted as a bitmap of '0's and '1's, with the LSB on the right. A 
              '1' means the corresponding OFDM symbol is reserved.
            - The bitmap length must be either ``symbolsPerSlot`` or ``2*symbolsPerSlot``.
            - If a list is provided, it is interpreted as the list of reserved symbol indices. If any index is greater 
              than or equal to the slot length, the reservation is assumed to span two slots.

        pattern : str, optional
            Periodicity and activity pattern, corresponding to ``periodicityAndPattern`` in **3GPP TS 38.214**. This 
            must be a bitmap string of '0's and '1's, with the LSB on the right. A '1' means the corresponding 
            reservation unit is active. The pattern length must be one of: ``1, 2, 4, 5, 8, 10, 20, 40``. A unit 
            corresponds to one slot or two slots, depending on the symbol configuration. The maximum pattern duration 
            is 40 ms.
        """
        # For the bitmap case, all values are strings of '0's and '1's, with the LSB on the right
        # (i.e., the last character in the string is the LSB).
        #   - prbs: A '1' means the corresponding PRB is reserved. The length can be up to the number
        #     of PRBs in the BWP.
        #   - symbols: A '1' means the corresponding symbol is reserved. The length can be either
        #     n or 2*n, where n is the number of symbols per slot.
        #
        # For the array/list case:
        #   - prbs is a list of reserved PRB indices.
        #   - symbols is a list of reserved symbol indices within one slot. If at least one value
        #     exceeds the slot length, it is assumed to represent a two-slot case.
        #
        # pattern: A '1' means the corresponding unit is active. Its length must be one of
        # [1, 2, 4, 5, 8, 10, 20, 40]. The maximum periodicity is 40 ms.
        #
        # A unit is one or two slots.
        self.bwp = bwp
        self.prbs = prbs            # Corresponding to 'resourceBlocks' in TS 38.214
        self.symbols = symbols      # Corresponding to 'symbolsInResourceBlock' in TS 38.214
        self.pattern = pattern      # Corresponding to 'periodicityAndPattern' in TS 38.214
        self.symLen = None
        self.patLen = None
        self.slotLen = self.bwp.symbolsPerSlot
        
        if type(self.prbs) == str:
            # bitmap to list
            self.prbs = [i for i,bit in enumerate(self.prbs[::-1]) if bit=='1']
            
        if type(self.symbols) == str:
            # bitmap to list
            self.symLen = len(self.symbols)
            self.symbols = [i for i,bit in enumerate(self.symbols[::-1]) if bit=='1']
            validateRange(self.symLen, [self.slotLen, 2*self.slotLen], varName="symbols length")
        else:
            self.symLen = self.slotLen
            if len(self.symbols)>0 and max(self.symbols) >= self.slotLen: self.symLen *= 2

        if type(self.pattern) != str:       raise ValueError("'pattern' must be a string of 1's and 0's.")
        
        self.patLen = len(self.pattern)
        self.pattern = [i for i,bit in enumerate(self.pattern[::-1]) if bit=='1']
        validateRange(self.patLen, [1, 2, 4, 5, 8, 10, 20, 40], varName="pattern length")

        patDuration = self.patLen/self.bwp.slotsPerSubFrame     # Pattern duration in ms.
        if patDuration > 40:
            raise ValueError("'pattern' duration must be less than 40 milliseconds.")

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="ReservedPrbSet Properties:", getStr=False):
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  PRBs:             {self.prbs}\n"
        repStr += indent*' ' + f"  symbols:          {self.symbols}\n"
        repStr += indent*' ' + f"  pattern:          {self.pattern} ({self.patLen} units)\n"
        repStr += indent*' ' + f"  pattern duration: {self.patLen/self.bwp.slotsPerSubFrame} milliseconds\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def populateGrid(self, pxxch):                              # Undocumented - Not intended for direct use
        if len(self.prbs)==0:                                   return
        if len(self.symbols)==0:                                return
        slotNo = self.bwp.slotNo

        if self.symLen==self.slotLen:
            if (slotNo % self.patLen) not in self.pattern:      return
            reservedSymbols = self.symbols
        else:
            if ((slotNo//2) % self.patLen) not in self.pattern: return
            if (slotNo%2)==0:       reservedSymbols = [x for x in self.symbols if x <self.slotLen]
            else:                   reservedSymbols = [x-self.slotLen for x in self.symbols if x>=self.slotLen]
        
        for l in reservedSymbols:
            for prb in self.prbs:
                if prb not in pxxch.prbSet: continue
                ks = np.arange(12) + 12*pxxch.prb2Grb[prb]
                pxxch.grid[:,l,ks] = "RESERVED"
