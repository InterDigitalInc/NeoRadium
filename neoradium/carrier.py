# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``carrier.py`` implements the classes :py:class:`Carrier` and :py:class:`BandwidthPart`. Each 
:py:class:`Carrier` can be associated with several :py:class:`BandwidthPart` objects. This implementation is 
based on **3GPP TR 38.211**.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 05/18/2023    Shahab Hamidi-Rad       First version of the file.
# 11/02/2023    Shahab Hamidi-Rad       Completed the documentation
# 05/01/2025    Shahab Hamidi-Rad       Updated the nFFT calculations and removed some unused functions. Also
#                                       improved the BandwidthPart documentation.
# **********************************************************************************************************************
import numpy as np
import scipy.io
import matplotlib.colors as colors
import matplotlib.patches as patches
import time

from .grid import Grid
from .utils import freqStr

MAX_CARRIER_BW = 400e6      # 400 MHz (It was 20 MHz for LTE)
MAX_RESOURCE_BLOCKS = 275   # Max number of RBs in a carrier
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
    3GPP TR 38.211, section 4.4.5. It is important to note that BandwidthPart objects are not directly created. Instead,
    you typically create a :py:class:`Carrier` object and retrieve its current BandwidthPart using its ``curBwp`` 
    property.
    """
    sampleRate = SAMPLE_RATE    # = 30,720,000
    # ******************************************************************************************************************
    def __init__(self, carrier, **kwargs):
        r"""
        Parameters
        ----------
        carrier: Carrier object
        kwargs : dict
            A set of optional arguments.

                :startRb: The starting resource block (RB). This is the number of RBs from CRB 0. The default is 0.
                
                :numRbs: The number of RBs included in the bandwidth part. The default is 50.
                
                :spacing: The subcarrier spacing in kHz. This also specifies the numerology used. To specify the 
                    subcarrier spacing, you can use 15, 30, 60, 120, 240, 480, or 960. To specify the numerology, you
                    can use 0, 1, ..., 6. Please refer to **3GPP TR 38.211, section 4.2** for more details.
                    
                :cpType: Cyclic Prefix type. It can be either "Normal" or "Extended". The "Extended" type is only
                    available for 60 kHz subcarrier spacing.
                    

        **Other Properties:**
        
        Here is a list of additional properties:
        
            :u: The numerology value, which falls within the range of 0 to 6 (:math:`\mu`). See 3GPP TR 38.211, 
                table 4.2-1.
            
            :bandwidth: The bandwidth of this bandwidth part in Hz.
            
            :nFFT: The FFT size used for OFDM modulation of the resource grids (See :py:class:`~neoradium.grid.Grid`)
                which are created based on this bandwidth part. It is calculated as follows:
                
                .. math::

                    N_{FFT} = \big [\frac {\frac {f_s} {1000} - \sum_{l=0}^{N_{symb}^{slot}-1} N_{CP,l}^{\mu}} {N_{symb}^{subframe,\mu}} \big ]
                    
                where :math:`f_s=\frac 1 {T_s}`, is the 5G sample rate (:math:`f_s=30,720,000` Hz), 
                :math:`N_{symb}^{slot}` is the number of symbols per slot, :math:`N_{CP,l}^{\mu}` is the number of 
                samples in cyclic prefix of symbol :math:`l` based on numerology :math:`\mu`, and
                :math:`N_{symb}^{subframe,\mu}` is the number of symbols in each subframe for numerology :math:`\mu`.
                    
            :symbolsPerSlot: The number of OFDM symbols in each slot (:math:`N_{symb}^{slot}`). This is equal to 14 and 
                12 for "Normal" and "Extended" Cyclic Prefix types, respectively.
                
            :slotsPerSubFrame: The number of slots per subframe based on current numerology 
                (:math:`N_{slot}^{subframe,\mu}`).
                
            :symbolLens: A list of symbol length values in number of time samples for every symbol in a subframe. The
                symbol length for symbol ``l``, ``symbolLens[l]``, is the sum of :math:`N_{FFT}` and 
                :math:`N_{CP,l}^{\mu}`.
                
            :slotsPerFrame: The number of slots per frame based on current numerology (:math:`N_{slot}^{frame,\mu}`).
                
            :symbolsPerSubFrame: The number of OFDM Symbols per subframe based on current numerology 
                (:math:`N_{symb}^{subframe,\mu}`).
                
            :slotNoInFrame: The slot number in current frame (:math:`n_{s,f}^{\mu}`).
            
            :slotNoInSubFrame: The slot number in current subframe (:math:`n_{s}^{\mu}`).
            
            :avgSlotDuration: The average slot duration in seconds. 
            
            :cellId: The Cell identifier of the Carrier containing this bandwidth part.
            
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
        self.startRb = kwargs.get('startRb', 0)         # Number of RBs from CRB 0
        self.numRbs = kwargs.get('numRbs', 50)
        
        spacing = kwargs.get('spacing', 15)
        scsps = [15,30,60,120,240,480,960]              # All allowed subcarrier spacings in kHz. See 3GPP TR 38.211 section 4.2
        if spacing in scsps:        self.u, self.spacing = scsps.index(spacing), spacing
        elif spacing in range(7):   self.u, self.spacing = spacing, scsps[spacing]
        else:                       raise ValueError("Invalid \"spacing\" values (%s)!"%(str(spacing)))
        assert self.u in range(7)
        assert self.spacing in scsps, ("Spacing:" + str(self.spacing))
        
        self.cpType = kwargs.get('cpType', 'normal').lower()    # Cyclic Prefix Type: 'normal' or 'extended'
        if self.cpType not in ['normal','extended']:
            raise ValueError("Unsupported cpType \"%s\"! It must be one of 'Normal' or 'Extended'"%(self.cpType))

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
        # Adding the first symbol len to the end to help with the fact that we always get symbolsPerSlot+1 symLen
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
            If specified, it is used as a title for the printed information. If ``None`` (default), the text
            "Bandwidth Part Properties:" is used for the title.

        getStr : Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns
            the information in a text string. Otherwise, nothing is returned.
        """
        if title is None:   title = "Bandwidth Part Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  Resource Blocks:    {self.numRbs} RBs starting at {self.startRb} ({self.numRbs*12} subcarriers)\n"
        repStr += indent*' ' + f"  Subcarrier Spacing: {self.spacing} kHz\n"
        repStr += indent*' ' + f"  CP Type:            {self.cpType}\n"
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
    def __getattr__(self, property):
        # Get these properties from the 'carrier' object
        if property not in ["cellId", "slotNo", "frameNo", "goNext", "restart"]:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.carrier, property)
    
    # ******************************************************************************************************************
    def createGrid(self, numPlanes, useReDesc=False):
        r"""
        Creates a resource grid and returns an empty :py:class:`~neoradium.grid.Grid`
        object based on this bandwidth part.

        Parameters
        ----------
        numPlanes : int
            The number of "planes" in the resource grid. See the
            :py:class:`~neoradium.grid.Grid` class for more information.
            
        useReDesc : Boolean
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

    # ******************************************************************************************************************
    def getCpLen(self, symIdxInSubframe):
        r"""
        Returns the number of time samples in the Cyclic Prefix for the OFDM symbol
        specified by ``symIdxInSubframe``. This is based on **TS 38.211, Section 5.3.1**.

        Parameters
        ----------
        symIdxInSubframe : int
            The index of symbol from the beginning of subframe.

        Returns
        -------
        int
            The number of time samples in Cyclic Prefix for the OFDM symbol
            specified by ``symIdxInSubframe``.
        """
        # NOTE: The returned value is Ncp//𝜅 (Ncp as defined in the above section)
        if symIdxInSubframe>=self.symbolsPerSubFrame:
            raise ValueError("'symIdxInSubframe' must be less than the number of OFDM Symbols in a " +
                            f"subframe ({self.symbolsPerSubFrame}).")
        if self.cpType=='normal':
            cpLen = 144//(1<<self.u)
            if symIdxInSubframe in [0, 7*(1<<self.u)]:  cpLen += 16
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
        numpy array
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
    # See TS 38.211 V17.0.0 (2021-12), Section 4.4.2
    sampleRate = SAMPLE_RATE    # = 30,720,000
    # ******************************************************************************************************************
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments.

                :startRb: The starting resource block (RB). This is the number of RBs from CRB 0. The default is 0.
                
                :numRbs: The number of RBs included in the carrier. The default is 50.
                
                :bwps: A list of :py:class:`BandwidthPart` objects associated with this Carrier. If this is not 
                    specified, a single bandwidth part is automatically created covering the whole carrier. In this 
                    case, the following additional :py:class:`BandwidthPart` parameters can also be specified when 
                    creating the Carrier object:
                    
                        :spacing: The subcarrier spacing in kHz. This also specifies the numerology used. To specify
                            the subcarrier spacing, you can use 15, 30, 60, 120, 240, 480, or 960. To specify the 
                            numerology, you can use 0, 1, ..., 6. Please refer to **3GPP TR 38.211, section 4.2** 
                            for more details.
                    
                        :cpType: Cyclic Prefix type. It can be either "Normal" or "Extended". The "Extended" type is
                            only available for 60 kHz subcarrier spacing.
                            
                        **Example:**
                        
                        .. code-block:: python
        
                            # Create a carrier with a single BandwidthPart:
                            carrier = Carrier(startRb=0, numRbs=25, spacing=30, cpType="Normal")
                            
                :cellId: The Cell identifier of this Carrier. The default is 1.
                
                :curBwpIndex: The index of current bandwidth part. The default is 0.


        **Other Properties:**

        Here is a list of additional properties:

            :slotNo: Current slot number. A counter that can be used in simulation.
            
            :frameNo: Current frame number. A counter that can be used in simulation. This is incremented every
                ``slotsPerFrame`` slots.
                
            :curBwp: The currently active :py:class:`BandwidthPart` object.
            
            :frameNoRel: The remainder of current frame number divided by 1024.
            
            :slotNoInFrame: The slot number in current frame (:math:`n_{s,f}^{\mu}`).
            
            :symbolsPerSlot: The number of OFDM symbols in each slot (:math:`N_{symb}^{slot}`) based on the numerology
                of the currently active :py:class:`BandwidthPart`. This is equal to 14 and 12 for "Normal" and 
                "Extended" Cyclic Prefix types, respectively.

            :slotsPerSubFrame: The number of slots per subframe based on the numerology of the currently active 
                :py:class:`BandwidthPart` (:math:`N_{slot}^{subframe,\mu}`).

            :slotsPerFrame: The number of slots per frame based on the numerology of the currently active 
                :py:class:`BandwidthPart` (:math:`N_{slot}^{frame,\mu}`).

            :symbolsPerSubFrame: The number of OFDM Symbols per subframe based on the numerology of the currently 
                active :py:class:`BandwidthPart` (:math:`N_{symb}^{subframe,\mu}`).
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
            If specified, it is used as a title for the printed information. If ``None`` (default), the text
            "Carrier Properties:" is used for the title.

        getStr : Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
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
    def __getattr__(self, property):
        # Get these properties from the 'curBwp' object
        if property not in ["symbolsPerSlot", "slotsPerSubFrame", "slotsPerFrame", "symbolsPerSubFrame"]:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.curBwp, property)

    # ******************************************************************************************************************
    def restart(self):
        self.slotNo = 0
        self.frameNo = 0

    # ******************************************************************************************************************
    def goNext(self):
        r"""
        Increments the current slot number in this carrier (``slotNo``). If the slot number passes the boundary of 
        a frame, the frame number (``frameNo``) is also incremented.
        """
        self.slotNo += 1
        if (self.slotNo % self.slotsPerFrame)==0:   self.frameNo += 1
            
    # ******************************************************************************************************************
    def createGrid(self, numPorts, useReDesc=False):
        r"""
        Creates a resource grid and returns an empty :py:class:`~neoradium.grid.Grid` object based on the currently
        active :py:class:`BandwidthPart`. See :py:meth:`BandwidthPart.createGrid` for more details.
        """
        return self.curBwp.createGrid(numPorts, useReDesc=useReDesc)

