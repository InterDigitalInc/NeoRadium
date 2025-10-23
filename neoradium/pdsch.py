# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``pdsch.py`` implements the :py:class:`PDSCH` class which encapsulates the Physical Downlink Shared Channel.
It is a downlink channel that delivers user data from gNB to UE. PDSCH occupies a grid of Resource Blocks (RBs) within
a slot. Usually, one or more OFDM symbols are used by the PDCCH, and the remaining resources are available for the
PDSCH.

The gNB schedules PDSCH resources for UEs based on their channel quality, data requirements, and fairness 
considerations. PDSCH uses LDPC (Low-Density Parity-Check) coding to provide forward error correction, enhancing the
robustness of data transmission over the wireless channel.

In Multiple Input, Multiple Output (MIMO) systems, a PDSCH is distributed among multiple layers. PDSCH includes
Demodulation Reference Signals (:py:class:`~neoradium.dmrs.DMRS`) to assist the UE in channel estimation and
demodulation, ensuring accurate data reception. It may also include Phase Tracking Reference Signals 
(:py:class:`~neoradium.dmrs.PTRS`) which enable suppression of phase noise and common phase error, particularly
important at high carrier frequencies such as millimeter-wave bands.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 12/30/2023    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np

from . import Modem, Grid
from .utils import goldSequence, getMultiLineStr
from .dmrs import DMRS, PTRS
# This implementation is based on:
#   - TS 38.211 V17.0.0 (2021-12)
#   - TS 38.212 V17.0.0 (2021-12)
#   - TS 38.214 V17.0.0 (2021-12)
# The following links can help with understanding some of the Standard ambiguities:
#   https://www.sharetechnote.com/html/5G/5G_PDSCH.html
#   https://www.sharetechnote.com/html/5G/5G_PDSCH_DMRS.html
#   https://www.sharetechnote.com/html/5G/5G_PTRS_DL.html

# **********************************************************************************************************************
class ReservedRbSet:  # See TS 38.214 V17.0.0 (2021-12), Section 5.1.4.1
    # ******************************************************************************************************************
    def __init__(self, carrier, rbs=[], symbols=[], pattern=[1]):
        # For bitmap case: All values are text strings of 0s and 1s with LSB on the right (the last character in text
        # is LSB)
        #   - rbs: a '1' character means the corresponding RB is reserved (length can be up to the number of RBs in
        #     the BWP)
        #   - symbols: a '1' character means the corresponding symbol is reserved (length can be 1*n or 2*n, where n
        #     is number of symbols per slot)
        #   - pattern: a '1' character means the corresponding 'unit' is active (length can be one of [1, 2, 4, 5, 8,
        #     10, 20, 40])
        # For array case:
        #   - rbs is an array of reserved RB indices
        #   - symbols is an array of symbols in one slot. If at least one value exceeds the slot length, then it is
        #     assumed to be a 2-slot case.
        #   - pattern is an array of 'unit' indices.
        # A 'unit' is one or two slots.
        self.carrier = carrier
        self.rbs = rbs              # Corresponding to 'resourceBlocks' in TS 38.214
        self.symbols = symbols      # Corresponding to 'symbolsInResourceBlock' in TS 38.214
        self.pattern = pattern      # Corresponding to 'periodicityAndPattern' in TS 38.214
        self.symLen = None
        self.patLen = None
        self.slotLen = self.carrier.symbolsPerSlot
        
        if type(self.rbs) == str:
            # bitmap to list
            self.rbs = [i for i,bit in enumerate(self.rbs[::-1]) if bit=='1']
            
        if type(self.symbols) == str:
            # bitmap to list
            self.symLen = len(self.symbols)
            self.symbols = [i for i,bit in enumerate(self.symbols[::-1]) if bit=='1']
            assert self.symLen in [slotLen, 2*slotLen], \
                    "Invalid bitmap length for \"ReservedRBs::symbols\". (Must be %d or %d)"%(slotLen, 2*slotLen)
        else:
            self.symLen = slotLen
            if max(self.symbols) >= slotLen: symLen *= 2

        if type(self.pattern) == str:
            self.patLen = len(self.pattern)
            self.pattern = [i for i,bit in enumerate(self.pattern[::-1]) if bit=='1']
        else:
            self.patLen = max(self.pattern)+1
            for p in [1, 2, 4, 5, 8, 10, 20, 40]:
                if self.patLen<=p:
                    self.patLen = p
                    break
        assert self.patLen in [1, 2, 4, 5, 8, 10, 20, 40], \
                    "Invalid bitmap length for \"ReservedRBs::pattern\". (Must be one of 1, 2, 4, 5, 8, 10, 20, or 40)"

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="ReservedRbSet Properties:", getStr=False):
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  rbs: %s\n"%(str(self.rbs))
        repStr += indent*' ' + "  symbols: %s (len:%d)\n"%(str(self.symbols), self.symLen)
        repStr += indent*' ' + "  pattern: %s (len:%d)\n"%(str(self.pattern), self.patLen)
        repStr += indent*' ' + "  slotLen: %d\n"%(self.slotLen)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def applyToMap(self, slotMap, slotNo):
        # slotMap: Each row corresponds to one symbol in slot and contains the RBs indices that are
        # available for that symbol with the order they should be assigned. Unscheduled symbols have an empty list.
        
        if len(self.rbs)==0:                                    return slotMap
        if len(self.symbols)==0:                                return slotMap
        
        if self.symLen==self.slotLen:
            if (slotNo % self.patLen) not in self.pattern:      return slotMap
            reservedSymbols = self.symbols
        else:
            if ((slotNo//2) % self.patLen) not in self.pattern: return slotMap
            if (slotNo%2)==0:       reservedSymbols = [x for x in self.symbols if x <self.slotLen]
            else:                   reservedSymbols = [x-self.slotLen for x in self.symbols if x>=self.slotLen]

        for sym, symRBs in enumerate(slotMap):
            if sym not in reservedSymbols:  continue
            slotMap[sym] = [rb for rb in symRBs if rb not in self.rbs]    # Keep only the RBs that are not reserved
        return slotMap

    # ******************************************************************************************************************
    def populateGrid(self, grid):
        if len(self.rbs)==0:                                    return
        if len(self.symbols)==0:                                return
        slotNo = self.carrier.slotNo

        if self.symLen==self.slotLen:
            if (slotNo % self.patLen) not in self.pattern:      return
            reservedSymbols = self.symbols
        else:
            if ((slotNo//2) % self.patLen) not in self.pattern: return
            if (slotNo%2)==0:       reservedSymbols = [x for x in self.symbols if x <self.slotLen]
            else:                   reservedSymbols = [x-self.slotLen for x in self.symbols if x>=self.slotLen]

        for p,portNo in enumerate(self.pdsch.portSet):
            for l in reservedSymbols:
                for rb in self.rbs:
                    for k in range(rb*12,rb*12+12):
                        grid[p,l,k] = "RESERVED"

# **********************************************************************************************************************
class PDSCH:
    r"""
    This class encapsulates the configuration and functionality of a Physical Downlink Shared Channel (PDSCH), that 
    delivers user data transmitted from gNB to UE.
    """
    # ******************************************************************************************************************
    def __init__(self, bwp, **kwargs):
        r"""
        Parameters
        ----------
        bwp: :py:class:`~neoradium.carrier.BandwidthPart`
            The :py:class:`~neoradium.carrier.BandwidthPart` object that represents the resources used by this 
            :py:class:`PDSCH` for transmission of user data from gNB to UE.
            
        kwargs : dict
            A set of optional arguments.

                :mappingType: The mapping type used by this PDSCH and its associated
                    :py:class:`~neoradium.dmrs.DMRS` object. It is a text string that can be either ``'A'`` or 
                    ``'B'``. The default is ``'A'``.
                    
                    In mapping type ``'A'``, the first DMRS OFDM symbol index is 2 or 3 and DMRS is mapped relative
                    to the start of slot boundary, regardless of where in the slot the actual data transmission
                    starts. The user data in this case usually occupies most of the slot.

                    In mapping type ``'B'``, the first DMRS OFDM symbol is the first OFDM symbol of the data
                    allocation, that is, the DMRS location is not given relative to the slot boundary but relative
                    to where the user data is located. The user data in this case usually occupies a small fraction
                    of the slot to support very low latency.

                :numLayers: The number of transmission layers for this :py:class:`PDSCH`. It must be an integer
                    from 1 to 8, with 1 as the default.

                :modulation: A text string or a tuple or list of 2 text strings specifying the modulation scheme 
                    used for data transmitted in this :py:class:`PDSCH` based on **3GPP TR 38.211, Table 7.3.1.2-1**.
                    The default is ``'16QAM'``. Here is a list of supported modulation schemes:
                        
                    ===================  =========================
                    Modulation Scheme    Modulation Order (``qm``)
                    ===================  =========================
                    BPSK                 1
                    QPSK                 2
                    16QAM                4
                    64QAM                6
                    256QAM               8
                    1024QAM              10
                    ===================  =========================

                    If ``modulation`` is a text string and there are two codewords in this :py:class:`PDSCH`, the
                    same modulation scheme is used for both codewords. If there are two codewords in this
                    :py:class:`PDSCH`, and you want to use different modulation schemes for the two codewords, you
                    can specify two different modulation schemes in a tuple or list of text strings. For example:
                    
                    .. code-block:: python

                        # Using "QPSK" for the first codeword and "16QAM" for the second codeword
                        modulation = ("QPSK", "16QAM")
                        
                    The specified modulation scheme(s) are used to create one or two 
                    :py:class:`~neoradium.modulation.Modem` objects.

                :reservedRbSets: A list of :py:class:`~neoradium.pdsch.ReservedRbSet` objects that are used to
                    reserve the specified resource blocks (RBs) at the specified OFDM symbols based on the patterns
                    defined in the :py:class:`~neoradium.pdsch.ReservedRbSet` objects. The default is an empty list
                    which means no reserved RBs.

                :reservedReMap: The map of additional reserved resource elements (REs) as a 3D list of the form
                    ``portNo x symNo x reNo``. The default is an empty list which means no reserved REs. The
                    following additional rules makes configuring reserved REs both easy and flexible.
                    
                    - If ``reservedReMap`` has data for only one port, it means that the map is the same for all ports.
                    
                    - If in a port, the reserved REs are given only for a single symbol, it is assumed the same REs
                      are reserved in all symbols.
                      
                    Here are some examples:
                    
                    .. code-block:: python

                        # REs 5, 17, and 29 are reserved on all ports and all symbols
                        reservedReMap = [[[5,17,29]]]
                   
                        # REs 5, 17, and 29 are reserved on all ports for OFDM symbol 2 only
                        reservedReMap = [[ [], [], [5,17,29], [], [], [], [], [], [], [], [], [], [], [] ]]
                        
                        # REs 5, 17, and 29 are reserved on all symbols of port index 1 (assuming we have 3 ports)
                        reservedReMap = [[], [[5,17,29]], []]
                    
                :sliv: *Start and Length Indicator Value*. If specified, it is used to determine the start and
                    length of consecutive OFDM symbols used by this :py:class:`PDSCH` based on **3GPP TS 38.214, 
                    Section 5.1.2.1**. The default is `None`. See :ref:`Specifying the OFDM symbols <SpecifyingSyms>`
                    below for more information.
                    
                :symStart: The index of the first OFDM symbol used for this :py:class:`PDSCH`. The default is `None`.
                    See :ref:`Specifying the OFDM symbols <SpecifyingSyms>` below for more information.
                    
                :symLen: The number of consecutive OFDM symbols used by this :py:class:`PDSCH` starting at 
                    ``symStart``. The default is `None`. See :ref:`Specifying the OFDM symbols <SpecifyingSyms>`
                    below for more information.
                
                :symSet: A list of OFDM symbol indices that are used by this :py:class:`PDSCH`. See 
                    :ref:`Specifying the OFDM symbols <SpecifyingSyms>` below for more information.
                
                :prbSet: The list of physical resource blocks (PRBs) used by this :py:class:`PDSCH`. The default is
                    all the RBs in the :py:class:`~neoradium.carrier.BandwidthPart` object ``bwp``.

                :interleavingBundleSize: The bundle size of interleaving process. It can be one of 0 (default), 2, 
                    or 4. The value 0 means interleaving is disabled (default). See **3GPP TS 38.211, Section 7.3.1.6**
                    for more information.

                :rnti: The *Radio Network Temporary Identifier*. The default is 1. It is used with ``nID`` below to
                    initialize a *golden sequence* used for the scrambling process. See **3GPP TS 38.211, 
                    Section 7.3.1.1** for more information.
                    
                :nID: The *scrambling identity*. The default is 1. It is used with ``rnti`` to initialize a *golden 
                    sequence* used for the scrambling process. See **3GPP TS 38.211, Section 7.3.1.1** for more
                    information.

                :prgSize: The size of Precoding RB Groups (PRGs). It can be one of 0 (default), 2, or 4. The value 0
                    means *Wideband Precoding* which means the same precoding is used for the whole bandwidth of
                    this :py:class:`PDSCH`. See **3GPP TS 38.214, Section 5.1.2.3** for more information.
                    

        .. _SpecifyingSyms:
        
        **Specifying the OFDM symbols:**
            
            You can specify the OFDM symbols used by this :py:class:`PDSCH` in different ways:
            
            - If ``sliv`` is specified, it is used to determine the start and length of consecutive OFDM symbols used
              by this :py:class:`PDSCH` based on **3GPP TS 38.214, Section 5.1.2.1**. In this case, the parameters
              ``symStart``, ``symLen``, and ``symSet`` are ignored.
              
            - If ``sliv`` is not specified and both ``symStart`` and ``symLen`` are specified, they are used to
              determine the OFDM symbols used by this :py:class:`PDSCH`. In this case the parameter ``symSet`` is
              ignored.
              
            - If ``sliv``, ``symStart``, and ``symLen`` are not specified but ``symSet`` is specified, it is used to
              determine the OFDM symbols used by this :py:class:`PDSCH`.
              
            - If neither of ``sliv``, ``symStart``, ``symLen``, and ``symSet`` are specified, the OFDM symbols are
              automatically assigned based on ``mappingType`` and ``cpType`` parameter of the
              :py:class:`~neoradium.carrier.BandwidthPart` object ``bwp``.
              

        **Other Properties:**
        
            :numCW: The number of codewords derived from the ``numLayers`` parameter. It is either 1 or 2.

            :modems: A list of one or two (depending on ``numCW``) :py:class:`~neoradium.modulation.Modem` object(s)
                used internally for modulation/demodulation of the codewords.
            
            :dmrs: The :py:class:`~neoradium.dmrs.DMRS` object associated with this :py:class:`PDSCH`. You can use
                :py:meth:`setDMRS` method to set the :py:class:`~neoradium.dmrs.DMRS` object associated with this
                :py:class:`PDSCH`.
                
            :portSet: The list of ports used by this :py:class:`PDSCH` and its associated 
                :py:class:`~neoradium.dmrs.DMRS` object. By default, this is set to the number of layers specified by
                ``numLayers``. This can be changed by the :py:class:`~neoradium.dmrs.DMRS` configuration.
                
            :slotNo: This returns the ``slotNo`` property of the :py:class:`~neoradium.carrier.Carrier` object 
                containing ``bwp``.
            
            :frameNo: This returns the ``frameNo`` property of the :py:class:`~neoradium.carrier.Carrier` object
                containing ``bwp``.
                
            :slotNoInFrame: This returns the ``slotNoInFrame`` property of the
                :py:class:`~neoradium.carrier.Carrier` object containing ``bwp``.
                
            :slotMap: This is the map of resource blocks used for each OFDM symbol in this :py:class:`PDSCH` in the
                form of a 2D list (A list of lists). Each element in the list corresponds to one OFDM symbol. If an
                OFDM symbol has no resource blocks allocation in this :py:class:`PDSCH`, its corresponding element is
                an empty list. Otherwise, it is a list of indices of all resource blocks used in that OFDM symbol in
                the order they should be allocated (based on the interleaving process). Note that the reserved
                resource blocks specified by ``reservedRbSets`` above are **not** included in the ``slotMap``.

        The notebook :doc:`../Playground/Notebooks/PDSCH/PDSCH-endToEnd` shows how to create an end-to-end pipeline of
        PDSCH communication.
        """
        self.bwp = bwp
        
        # Mapping Types:
        # A: First DM-RS is located in symbol 2/3 of the slot and the DM-RS is mapped relative to the start of the
        #    slot boundary, regardless of where in the slot the actual data transmission starts. Data usually occupy
        #    most of the slot.
        # B: First DM-RS is located in the first symbol of the data allocation, that is, the DM-RS location is not
        #    given relative to the slot boundary but relative to where the data are located. Data usually occupy a
        #    small fraction of the slot to support very low latency.
        self.mappingType = kwargs.get('mappingType', 'A')   # A:
        assert self.mappingType in "AB", "Unsupported mapping type \"%s\"!"%(self.mappingType)

        self.numLayers = kwargs.get('numLayers', 1)
        assert self.numLayers in range(1,9), "Number of Layers must be between 1 and 8!"
        self.numCW = 2 if self.numLayers>4 else 1

        self.reservedRbSets = kwargs.get('reservedRbSets', [])  # A list of ReservedRbSet classes
        self.reservedReMap = kwargs.get('reservedReMap', [])    # A 3-D list of the form portNo x symNo x reNo. See
        self.checkReservedREs()                                 # the "checkReservedREs" function for details

        modulation = kwargs.get('modulation', '16QAM')          # See TS 38.211 V17.0.0 (2021-12), Table 7.3.1.2-1
        if type(modulation)==str:                   modulation = self.numCW*[modulation]
        elif type(modulation) in [list, tuple]:     modulation = list(modulation)
        else:
            raise ValueError("'modulation' must be a string, a list strings, or a tuple strings. ('%s' is not supported)"%(type(modulation).__name__))
        if len(modulation)<self.numCW: modulation = 2*modulation
        modulation = modulation[:self.numCW]
        for modStr in modulation:
            if modStr not in ['QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']:
                raise ValueError("Unsupported modulation \"%s\"!"%(modStr))
        # Make a Modem object based on the modulation scheme for each codeword
        self.modems = [ Modem(modulation[0]) ]
        if self.numCW>1:  # Use the same Modem object if both modulations are the same
            self.modems += [ self.modems[0] if modulation[0]==modulation[1] else Modem(modulation[1]) ]

        sliv = kwargs.get('sliv', None)
        symStart, symLen = kwargs.get('symStart', None), kwargs.get('symLen', None)
        if sliv is not None:
            # SLIV specified. See TS 38.214 V17.0.0 (2021-12), Section 5.1.2.1
            s,l = sliv%14, sliv//14 + 1
            if s+l>14:  s,l = 13-s, 16-l
            check = (14*(l-1) + s) if l<=8 else (14*(14-l+1) + (14-1-s))
            assert sliv==check, "Failed to convert SLIV(%d) to start and length values!"%(sliv)
            self.symSet = np.uint32(range(s,s+l))
        elif (symStart is not None) and (symLen is not None):
            self.symSet = np.uint32(range(symStart, symStart+symLen))
        else:
            if self.mappingType=='A':           defaultSymSet = range(self.bwp.symbolsPerSlot)
            elif self.bwp.cpType=='normal':     defaultSymSet = range(13)
            else:                               defaultSymSet = range(6)
            self.symSet = np.uint32(kwargs.get('symSet', defaultSymSet))   # The set of symbols allocated
            self.symSet.sort()

        # Note that this is actually a "vrbSet". If “interleavingBundleSize” is zero, then vrb and prb are the same.
        # However, if it is nonzero, then the given set is used as the vrb set, which is mapped to a set of PRBs
        # that may be outside the given vrb set. Interleaving scatters the resource blocks of this PDSCH across
        # the entire BWP. If two PDSCH use different sets of RBs of a BWP, then their RBs are shuffled and mixed
        # after interleaving. See the getVrbToPrbMapping function.
        self.prbSet = np.uint32(kwargs.get('prbSet', range(0, self.bwp.numRbs)))    # The set of VRBs allocated
        self.prbSet.sort()
        
        if self.symSet[-1]>14 or self.symSet[0]<0:
            raise ValueError("Invalid 'symSet' values! (They must be in [0..13]")

        if self.prbSet[-1]>self.bwp.numRbs or self.prbSet[0]<0:
            raise ValueError("Invalid 'prbSet' values! (They must be in [0..%d]"%(self.bwp.numRbs))

        self.interleavingBundleSize = kwargs.get('interleavingBundleSize', 0)
        if self.interleavingBundleSize not in [0,2,4]:
            raise ValueError("'interleavingBundleSize' must be 0 (Interleaving disabled), 2, or 4")

        self._slotMap = None                            # Will be initialized later (See the slotMap property below)
        self.rnti = kwargs.get('rnti', 1)               # Radio Network Temporary Identifier
        self.nID = kwargs.get('nID', 1)                 # scrambling identity
        
        # The size of Precoding RB groups (PRGs). See 3GPP TS 38.214 V17.0.0 (2021-12), Section 5.1.2.3
        self.prgSize = kwargs.get('prgSize', 0) # 0 -> 'Wideband', which means a single precoding is used for all PRBs
        if self.prgSize not in [0,2,4]:     raise ValueError("'prgSize' must be 0 (Wideband), 2, or 4)")

        # Check Symbol allocation: (See TS 38.214 V17.0.0 (2021-12), Table 5.1.2.1-1: Valid S and L combinations)
        s,l,m = self.symSet[0], len(self.symSet), self.bwp.symbolsPerSlot
        if self.mappingType=='A':
            if l not in range(3,m+1):
                raise ValueError("Invalid symbol allocation: length = %d  ∉ [3..%d]"%(l,m))
            if (s+l) not in range(3,m+1):
                raise ValueError("Invalid symbol allocation: start+length = %d+%d = %d ∉ [3..%d]"%(s,l,s+l,m))
        elif self.bwp.cpType=='normal':
            if s not in range(13):
                raise ValueError("Invalid symbol allocation: start = %d ∉ [0..12]"%(s))
            if l not in range(2,14):
                raise ValueError("Invalid symbol allocation: length = %d ∉ [2..13]"%(l))
            if (s+l) not in range(2,15):
                raise ValueError("Invalid symbol allocation: start+length = %d+%d = %d ∉ [2..14]"%(s,l,s+l))
        else: # Extended cyclic prefix
            if s not in range(11):
                raise ValueError("Invalid symbol allocation: start = %d ∉ [0..10]"%(s))
            if l not in [2,4,6]:
                raise ValueError("Invalid symbol allocation: length = %d ∉ {2,4,6}"%(l))
            if (s+l) not in range(2,m+1):
                raise ValueError("Invalid symbol allocation: start+length = %d+%d = %d ∉ [2..12]"%(s,l,s+l))
            
        self.portSet = list(range(self.numLayers))  # This will be overwritten by the DMRS object.
        self.dmrs = None

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="PDSCH Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`PDSCH` object.

        Parameters
        ----------
        indent: int
            The number of indentation characters.
            
        title: str
            If specified, it is used as a title for the printed information.

        getStr: Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  mappingType: %s\n"%(self.mappingType)
        repStr += indent*' ' + "  nID: %d\n"%(self.nID)
        repStr += indent*' ' + "  rnti: %d\n"%(self.rnti)
        modStr = self.modems[0].modulation
        if (len(self.modems)>1) and (self.modems[0].modulation!=self.modems[1].modulation):
            modStr += ", " + self.modems[1].modulation
        repStr += indent*' ' + "  numLayers: %s\n"%(self.numLayers)
        repStr += indent*' ' + "  numCodewords: %s\n"%(self.numCW)
        repStr += indent*' ' + "  modulation: %s\n"%(modStr)
        repStr += indent*' ' + "  portSet: %s\n"%(str(self.portSet))
        repStr += getMultiLineStr("symSet", self.symSet, indent, "%3d", 3, numPerLine=20)
        repStr += getMultiLineStr("prbSet", self.prbSet, indent, "%3d", 3, numPerLine=20)
        repStr += indent*' ' + "  interleavingBundleSize: %d\n"%(self.interleavingBundleSize)
        repStr += indent*' ' + "  PRG Size: %s\n"%("Wideband" if self.prgSize==0 else str(self.prgSize))
        repStr += self.bwp.print(indent+2, "Bandwidth Part:", True)
        if self.dmrs is not None:
            repStr += self.dmrs.print(indent+2, "DMRS:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def setDMRS(self, **kwargs):
        r"""
        Creates and initializes a :py:class:`~neoradium.dmrs.DMRS` object associated with this :py:class:`PDSCH` object.

        Parameters
        ----------
        kwargs: dict
            A dictionary of parameters passed directly to the constructor of the :py:class:`~neoradium.dmrs.DMRS`
            class. Please refer to this class for a list of parameters that can be used to configure DMRS.
        """
        self.dmrs = DMRS(self, **kwargs)

    # ******************************************************************************************************************
    def setPTRS(self, **kwargs):
        r"""
        Creates and initializes a :py:class:`~neoradium.dmrs.PTRS` object associated with this :py:class:`PDSCH`
        object. Please note that you **must** first use the :py:meth:`setDMRS` function to initialize the
        :py:class:`~neoradium.dmrs.DMRS` object before calling this function.

        Parameters
        ----------
        kwargs: dict
            A dictionary of parameters passed directly to the constructor of the :py:class:`~neoradium.dmrs.PTRS`
            class. Please refer to this class for a list of parameters that can be used to configure PTRS.
        """
        if self.dmrs is None: raise ValueError("Cannot set PTRS without first defining a DMRS object for this PDSCH!")
        self.dmrs.setPTRS(**kwargs)

    # ******************************************************************************************************************
    # These are already documented in the __init__ function.
    @property
    def slotNo(self):               return self.bwp.carrier.slotNo
    @property
    def frameNo(self):              return self.bwp.carrier.frameNo
    @property
    def slotNoInFrame(self):        return self.bwp.carrier.slotNoInFrame

    # ******************************************************************************************************************
    @property
    def slotMap(self):                                          # Already documented in the __init__ function
        if self._slotMap is None:
            prbIndexes = self.getVrbToPrbMapping()
            
            # slotMap: Each row corresponds to one symbol in slot and contains the PRBs indices that are
            # available for that symbol with the order they should be assigned. Unscheduled symbols have an empty list.
            # See section 9.10 of the "5G NR" book
            # See TS 38.214 V17.0.0 (2021-12), Section 5.1.4.1
            self._slotMap = [[] if sym not in self.symSet else prbIndexes.tolist()
                                    for sym in range(self.bwp.symbolsPerSlot)]
            # Remove all reserved RBs
            for reservedRbSet in self.reservedRbSets:
                self._slotMap = reservedRbSet.applyToMap(self._slotMap, self.slotNo, self.bwp.symbolsPerSlot)
            
        return self._slotMap

    # ******************************************************************************************************************
    def checkReservedREs(self):                                 # Not documented
        # This function checks to make sure the "reservedReMap" parameter provided by the user contains valid
        # information. If reservedReMap has data for only one port, it is assumed that the map is the same for all
        # ports. If in a port, the reserved REs are given only for a single symbol, it is assumed the same REs are
        # reserved in all symbols.
        # Some Examples:
        #   [[[5,17,29]]]: REs 5, 17, and 29 are reserved on all ports and all symbols
        #   [[ [], [], [5,17,29], [], [], [], [], [], [], [], [], [], [], [] ]]: REs 5, 17, and 29 are reserved
        #   on all ports on symbol 2 only
        #   [[], [[5,17,29]], [] ]: REs 5, 17, and 29 are reserved on all symbols of port index 1 (Assuming we have
        #   3 ports)

        if len(self.reservedReMap) == 0:    return
        numPorts = len(self.portSet)
        numSym = self.bwp.symbolsPerSlot
        
        if len(self.reservedReMap) not in [1, numPorts]:
            raise ValueError("The reserved REs must be given for exactly 1 or %d ports."%(numPorts))
            
        for p, portMap in enumerate(reservedReMap):
            if len(portMap) not in [0, 1, numSym]:
                raise ValueError("The reserved REs must be given for exactly 1 or %d symbols."%(numSym))

    # ******************************************************************************************************************
    def getVrbToPrbMapping(self):                               # Not documented
        # If interleaving is enabled for this PDSCH, this function returns the mapping between the
        # virtual resource blocks and physical resource blocks based on TS 38.211, Section 7.3.1.6.
        # See also Fig. 9.12 in the "5G NR" book
        if self.interleavingBundleSize == 0:
            return self.prbSet          # Interleaving is disabled => VRB ≡ PRB
        
        # First Bundle has L-(startRb%L) resource blocks (L: interleavingBundleSize)
        # Last Bundle has (self.bwp.start+self.bwp.numRbs)%L or L resource block if end of BWP is on bundle boundary
        numBundles = int(np.ceil( (self.bwp.numRbs + (self.bwp.startRb % self.interleavingBundleSize))/
                                  self.interleavingBundleSize ))
        
        # Creating the f function as defined in TS 38.211 - Section 7.3.1.6
        rr = 2                  # R
        cc = numBundles//rr     # C
        f = np.int32(numBundles*[0])
        f[:rr*cc] = np.arange(rr*cc).reshape(rr,cc).T.reshape(-1)
        f[numBundles-1] = numBundles-1

        deltaBundle0 = (self.bwp.startRb % self.interleavingBundleSize)
        prbIndexes = np.int32( [j*self.interleavingBundleSize + b
                                    for j in f for b in range(self.interleavingBundleSize) ] )
        prbIndexes = prbIndexes[deltaBundle0 : deltaBundle0+self.bwp.numRbs] - deltaBundle0

        # Only use the prbs that are in this PDSCH's prbSet
        prbIndexes = np.int32(prbIndexes)[self.prbSet]
        return prbIndexes
        
    # ******************************************************************************************************************
    def populateReservedREs(self, grid):                        # Not documented
        # This function finds all the reserved resource elements specified in "reservedReMap" and marks
        # them as "RESERVED" in the resource grid provided.
        if len(self.reservedReMap)==0:  return
        
        for p,portNo in enumerate(self.portSet):
            if len(self.reservedReMap) == 1:    portMap = self.reservedReMap[0]     # Use the same map for all ports
            else:                               portMap = self.reservedReMap[p]
            if len(portMap) == 0:               continue
            for l in range(self.bwp.symbolsPerSlot):
                if len(portMap) == 1:           removedSymbolREs = portMap[0]       # Use the same map for all symbols
                else:                           removedSymbolREs = portMap[l]
                if len(removedSymbolREs) == 0:  continue

                # removedSymbolREs is a list of REs for a port 'p' and a Symbol 'sym', where the REs need to be
                # marked as reserved
                for k in removedSymbolREs:
                    grid[p,l,k] = "RESERVED"

    # ******************************************************************************************************************
    def scrambleBits(self, q, bits):                            # Not documented
        # See TS 38.211 V17.0.0 (2021-12), Section 7.3.1.1
        cInit = self.rnti * (1<<15) + q * (1<<14) + self.nID
        scramblingSeq = goldSequence(cInit, len(bits))
        scrambledBits = bits ^ scramblingSeq
        return scrambledBits

    # ******************************************************************************************************************
    def scrambleLLRs(self, q, llrs):                            # Not documented
        # See TS 38.211 V17.0.0 (2021-12), Section 7.3.1.1
        cInit = self.rnti * (1<<15) + q * (1<<14) + self.nID
        scramblingSeq = 1-2*np.float64(goldSequence(cInit, len(llrs)))
        scrambledLLRs = llrs * scramblingSeq
        return scrambledLLRs

    # ******************************************************************************************************************
    def getLayerMapIndexes(self, psdchIndexes, numREsInCw=None):   # Not documented
        # 'numREsInCw' is a list of number of REs in each codeword for one or two codewords. If None, get all the
        # REs indexed by the "psdchIndexes"
        if numREsInCw is None:      numREsInCw = self.getNumREsFromIndexes(psdchIndexes)
        
        # See TS 38.211 V17.0.0 (2021-12), Section 7.3.1.3
        layerStartIndexes = np.append([0], np.where(np.diff(psdchIndexes[0])==1)[0]+1)
        cw1Layers = self.numLayers if self.numCW==1 else self.numLayers//2
        layerStartIndexes1 = layerStartIndexes[:cw1Layers]
        n = (numREsInCw[0]+cw1Layers-1)//cw1Layers
        mapIndexes1 = (layerStartIndexes1[None,:] + np.arange(n)[:,None]).reshape(-1)[:numREsInCw[0]]
        if self.numCW==1:
            return [ (psdchIndexes[0][mapIndexes1], psdchIndexes[1][mapIndexes1], psdchIndexes[2][mapIndexes1]) ]

        layerStartIndexes2 = layerStartIndexes[cw1Layers:]
        cw2Layers = self.numLayers - cw1Layers
        n = (numREsInCw[1]+cw2Layers-1)//cw2Layers
        mapIndexes2 = (layerStartIndexes2[None,:] + np.arange(n)[:,None]).reshape(-1)[:numREsInCw[1]]
            
        return [ (psdchIndexes[0][mapIndexes1], psdchIndexes[1][mapIndexes1], psdchIndexes[2][mapIndexes1]),
                 (psdchIndexes[0][mapIndexes2], psdchIndexes[1][mapIndexes2], psdchIndexes[2][mapIndexes2]) ]

    # ******************************************************************************************************************
    def mapToLayers(self, symbols):                             # Not documented
        # NOTE: This function is not used. It will be removed later. Use the "getLayerMapIndexes" function instead.
        # symbols is a numCW x m array of complex values

        # See TS 38.211 V17.0.0 (2021-12), Section 7.3.1.3

        # Get the layers for the 1st codeword
        period1 = self.numLayers if self.numCW==1 else self.numLayers//2
        n = len(symbols[0])
        nZeros = (period1-n%period1)%period1
        layersSymbols1 = np.append(symbols[0], nZeros*[0]).reshape(-1,period1).T        # Shape: period1 x M1

        if self.numCW==1:    return layersSymbols1

        period2 = (self.numLayers+1)//2
        n = len(symbols[1])
        nZeros = (period2-n%period2)%period2
        layersSymbols2 = np.append(symbols[1], nZeros*[0]).reshape(-1,period2).T        # Shape: period1 x M1

        # Now combine the layers for CW1 and CW2
        # First make sure they have the same width
        diffLen = layersSymbols1.shape[1] - layersSymbols2.shape[1]
        if diffLen>0:       layersSymbols2 = np.concatenate([layersSymbols2, np.zeros((period2,diffLen))], 1)
        elif diffLen<0:     layersSymbols1 = np.concatenate([layersSymbols1, np.zeros((period1,-diffLen))], 1)

        return np.concatenate([layersSymbols1, layersSymbols2])

    # ******************************************************************************************************************
    def getGrid(self, useReDesc=False):
        r"""
        Creates a :py:class:`~neoradium.grid.Grid` object for this :py:class:`PDSCH` and a populates it with the
        configured :py:class:`~neoradium.dmrs.DMRS` and :py:class:`~neoradium.dmrs.PTRS` reference signals. It also
        marks all the resources corresponding to the ``reservedRbSets`` and ``reservedReMap`` parameters as reserved
        in the newly created resource grid.
        
        The returned resource grid contains all reference signals and is ready to be populated with the user data (See
        :py:meth:`populateGrid` method).

        Parameters
        ----------
        useReDesc : Boolean
            If `True`, the resource grid will also include additional fields that describe the content of each 
            resource element (RE). This can be used during the debugging to make sure the resources are allocated
            correctly.

        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A :py:class:`~neoradium.grid.Grid` object representing the resource grid for this :py:class:`PDSCH`
            pre-populated with reference signals.
        """
        grid = self.bwp.createGrid(self.numLayers, useReDesc)
        self.allocateResources(grid)
        return grid

    # ******************************************************************************************************************
    def getReIndexes(self, grid, reTypeStr):
        r"""
        Returns the indices of all resource elements of the specified resource grid that are allocated for this
        :py:class:`PDSCH` with the content type specified by the ``reTypeStr``.
        
        This is similar to the :py:meth:`~neoradium.grid.Grid.getReIndexes` of the :py:class:`~neoradium.grid.Grid`
        class with two main differences:
        
            - This function considers only the resource elements in ``grid`` that are assigned to this
              :py:class:`PDSCH`.
              
            - The indices are ordered based on the interleaving configuration given by ``interleavingBundleSize``
              according to **3GPP TS 38.214, Section 5.1.4.1**.
              
        For example, the following code first gets the indices of all DMRS values in ``myPdsch`` and uses the
        returned indices to retrieve the DMRS values.

        .. code-block:: python
            
            myGrid = myPdsch.getGrid()
            dmrsIdx = myPdsch.getReIndexes(myGrid, "DMRS")  # Get the indices of all DMRS values
            dmrsValues = myGrid[indexes]                    # Get all DMRS values as a 1-D array.


        Parameters
        ----------
        grid: :py:class:`~neoradium.grid.Grid`
            The resource grid associated with this :py:class:`PDSCH`. This can be obtained using the
            :py:class:`getGrid` function for example.
            
        reTypeStr: str
            The content type of the desired resource elements in ``grid`` that are used by this :py:class:`PDSCH`.
            Here is a list of values that can be used:
            
                :"UNASSIGNED": The *un-assigned* resource elements.
                :"RESERVED": The reserved resource elements. This includes the REs reserved by ``reservedRbSets``
                    or ``reservedReMap`` parameters of this :py:class:`PDSCH`.
                :"NO_DATA": The resource elements that should not contain any data. For example when the corresponding
                    REs in a different layer is used for transmission of data for a different UE. (See 
                    ``otherCdmGroups`` parameter of :py:class:`~neoradium.dmrs.DMRS` class)
                :"DMRS": The resource elements used for :py:class:`~neoradium.dmrs.DMRS`.
                :"PTRS": The resource elements used for :py:class:`~neoradium.dmrs.PTRS`.
                :"CSIRS_NZP": The resource elements used for Zero-Power (ZP) CSI-RS (See :py:mod:`~neoradium.csirs`).
                :"CSIRS_ZP": The resource elements used for Non-Zero-Power (NZP) CSI-RS (See :py:mod:`~neoradium.csirs`).
                :"PDSCH": The resource elements used for user data in a Physical Downlink Shared Channel 
                    (:py:class:`PDSCH`)
                :"PDCCH": The resource elements used for user data in a Physical Downlink Control Channel
                    (:py:class:`~neoradium.pdcch.PDCCH`)
                :"PUSCH": The resource elements used for user data in a Physical Uplink Shared Channel
                    (:py:class:`~neoradium.pusch.PUSCH`)
                :"PUCCH": The resource elements used for user data in a Physical Uplink Control Channel
                    (:py:class:`~neoradium.pucch.PUCCH`)


        Returns
        -------
        3-tuple
            A tuple of three 1-D NumPy arrays specifying a list of locations in ``grid``. This value can be used
            directly to access the REs at the specified locations. (see example above)
        """
        # This is similar to "Grid::getReIndexes" but only considers the REs for this PDSCH.
        indexes = [[], [], []]
        for p in self.portSet:
            for sym, symRBs in enumerate(self.slotMap):
                for rb in symRBs:
                    idx = np.where(grid.reTypeIds[p,sym,rb*12:rb*12+12] == grid.retNameToId[reTypeStr])[0]
                    count = len(idx)
                    if count>0:
                        indexes[0] += count*[p]
                        indexes[1] += count*[sym]
                        indexes[2] += (rb*12 + idx).tolist()
        return (np.int32(indexes[0]), np.int32(indexes[1]), np.int32(indexes[2]))

    # ******************************************************************************************************************
    def getNumREsFromIndexes(self, indexes):
        r"""
        Returns the number of resource elements included in ``indexes`` for each codeword. The returned value is a
        list of one or two integers depending on number of codewords (``numCW``).

        Parameters
        ----------
        indexes: 3-tuple
            A tuple of 3 lists specifying locations of a set of resource elements in the resource grid. For example 
            this can be obtained using the :py:meth:`getReIndexes` function.

        Returns
        -------
        list
            A list of one or two integers depending on the number of codewords (``numCW``), indicating the number of
            resource elements (REs) included in ``indexes`` for each codeword.
        """
        numAllREs = len(indexes[0])
        if self.numCW == 1:  return [ numAllREs ]                       # Number of REs for 1st (the only) codeword
       
        # We have 2 codewords:
        layerStartIndexes = np.append([0], np.where(np.diff(indexes[0])==1)[0]+1)
        numREsInCw  = [ layerStartIndexes[ self.numLayers//2 ] ]        # Number of REs for 1st codeword
        numREsInCw += [ numAllREs - numREsInCw[0] ]                     # Number of REs for 2nd codeword
        return numREsInCw

    # ******************************************************************************************************************
    def getBitSizes(self, grid, reTypeStr="PDSCH"):
        r"""
        Returns total number of bits corresponding to the resource elements in ``grid`` assigned to this
        :py:class:`PDSCH` with content type specified by ``reTypeStr`` for each codeword. The returned value is a
        list of one or two integers depending on the number of codewords (``numCW``).
        
        The default value of ``reTypeStr="PDSCH"`` is for a common use case where we want to get total number
        of bits available in this :py:class:`PDSCH` for user data after setting aside the REs for DMRS, PTRS, and
        the reserved resources.

        Parameters
        ----------
        grid: :py:meth:`~neoradium.grid.Grid`
            The resource grid associated with this :py:class:`PDSCH`. This can be obtained using the 
            :py:meth:`getGrid` function for example.

        reTypeStr: str
            The content type of the desired resource elements used to count the returned number of bits. The default
            values of ``"PDSCH"`` causes this function to return total number of (unassigned) bits that are
            available for user data. Please refer to the :py:meth:`getReIndexes` function for a list of values that
            can used for ``reTypeStr``.

        Returns
        -------
        list
            A list of one or two integers depending on the number of codewords (``numCW``), indicating the number of
            bits allocated for ``reTypeStr`` for each codeword.
        """
        # Return the number of data bits that can be carried by this PDSCH in current slot for each codeword
        symbolsIndexes = self.getReIndexes(grid, reTypeStr)
        numREsInCw = self.getNumREsFromIndexes(symbolsIndexes)
        return [ numREsInCw[i] * self.modems[i].qm for i in range(self.numCW) ]

    # ******************************************************************************************************************
    def allocateResources(self, grid):
        # Allocate resources for this PSDCH in the given resource grid.
        for reservedRbSet in self.reservedRbSets:   reservedRbSet.populateGrid(grid) # Set all reserved RBs to RESERVED
        self.populateReservedREs(grid)                                               # Set all reserved REs to RESERVED
        if self.dmrs is not None:                   self.dmrs.populateGrid(grid)     # Set the DMRS/PTRS values

        pdschIdx = []
        for port in self.portSet:
            for sym in self.symSet:
                for prb in self.slotMap[sym]:
                    for r in range(12):
                        re = prb*12 + r
                        curReType = grid.reTypeAt(port,sym,re)
                        if curReType in ["DMRS", "CSIRS_ZP", "CSIRS_NZP", "RESERVED", "PTRS", "NO_DATA"]: continue
                        if curReType not in ["UNASSIGNED", "PDSCH"]:
                            raise ValueError(f"Trying to allocate the RE at ({port},{sym},{re}) for PDSCH," +
                                             f"while it is currently allocated for \"{curReType}\"!")
                        grid[port,sym,re] = (0, "PDSCH")
                        pdschIdx += [ [port,sym,re] ]
        self.dataIndices = tuple( np.int32(pdschIdx).T )

    # ******************************************************************************************************************
    def populateGrid(self, grid, bits=None):
        r"""
        *Populates* the resource grid specified by ``grid`` with the user data provided in ``bits``.
        
        This function performs the following operations:
        
            :Scrambling: Scrambling of the specified ``bits`` using the ``rnti`` and ``nID`` properties of this
                :py:class:`PDSCH`. These properties are used to initialize a *golden sequence* which is then used
                for the scrambling process according to **3GPP TS 38.211, Section 7.3.1.1**. The data bits for each
                codeword are scrambled separately.
                
            :Modulation: Converting the scrambled binary data stream into complex symbols for each resource elements
                assigned for user data. The modulation process is performed by the 
                :py:class:`~neoradium.modulation.Modem` objects in the ``modems`` list of this :py:class:`PDSCH`. The
                modulation for each codeword is performed separately by its own dedicated 
                :py:class:`~neoradium.modulation.Modem` object.
                
            :Layer Mapping: Distributing the modulated complex symbols across one or more transmission layers of this
                :py:class:`PDSCH` according to **3GPP TS 38.211, Section 7.3.1.3**.

            :Interleaving:  Converting Virtual Resource Blocks (VRBs) to Physical Resource Blocks (PRBs). If enabled, 
                the resources are re-ordered based on the interleaving configuration given by ``interleavingBundleSize``
                according to **3GPP TS 38.214, Section 5.1.4.1**.

        Parameters
        ----------
        grid: :py:class:`~neoradium.grid.Grid`
            The :py:class:`~neoradium.grid.Grid` object that gets populated with the user data bits.
            
        bits: list, tuple, NumPy array, or None
            Specifies the user data bits that are used to populate the specified resource grid. It can be one of the
            following:
            
            :tuple of NumPy arrays: Depending on the number of codewords (``numCW``), the tuple can have one or two
                1D NumPy arrays of bits each specifying the user data bits for each codeword.
                
            :NumPy array: A one or two dimensional NumPy array. It is a 1D array, only if we have one codeword and the
                given NumPy array is used for the single codeword. The 2D NumPy array can be used for one or two
                codeword cases. The first dimension of the NumPy array in this case should match the number
                of codewords (``numCW``).
                
            :list of NumPy arrays: Depending on the number of codewords (``numCW``), the list can have one or two 1D
                NumPy arrays of bits each specifying the user data bits for each codeword.
                
            :None: If this is None, ``grid`` data is not updated. This is used for the (rare) case where we only want
                to update the resource element descriptions in the ``grid`` object. See the ``useReDesc`` parameter 
                of the :py:class:`~neoradium.grid.Grid` class for more information.
        """
        if bits is not None:
            if type(bits)==tuple:           bits = list(bits)
            elif type(bits)==np.ndarray:
                if bits.ndim==1:            bits = [bits]
                else:                       bits = [ bits[i] for i in range(bits.shape[0]) ]
            elif type(bits)!=list:
                raise ValueError("'bits' must be a NumPy array, a tuple of NumPy arrays, or a list of NumPy arrays.")
            if self.numCW!=len(bits):
                raise ValueError(f"Number of codewords is {self.numCW} but {len(bits)} set(s) of bits are provided!")

        if bits is not None:
            symbols = []    # One item in the list for each codeword
            for cw in range(self.numCW):
                scrambledBits = self.scrambleBits(cw, bits[cw])
                symbols += [ self.modems[cw].modulate(scrambledBits) ]
            
            numREsInCw = [len(s) for s in symbols]
            layerMappedIndexes = self.getLayerMapIndexes(self.dataIndices, numREsInCw)
            for cw, layerMappedIndex in enumerate(layerMappedIndexes):
                grid[ layerMappedIndex ] = (symbols[cw], "PDSCH")
            
        if grid.reDesc is not None:
            if bits is None:
                layerStarts = np.append([0], np.where(np.diff(self.dataIndices[0])==1)[0]+1)
                numREsInCw = np.append(np.diff(layerStarts), [len(self.dataIndices[0])-layerStarts[-1]])
            else:
                numREsInCw = [(len(bits[cw]) + self.modems[cw].qm - 1)//self.modems[cw].qm for cw in range(self.numCW)]
            layerMappedIndexes = self.getLayerMapIndexes(self.dataIndices, numREsInCw)
            for cw, layerMappedIndex in enumerate(layerMappedIndexes):
                grid.reDesc[ layerMappedIndex ] = ["CW%d-%d"%(cw,i) for i in range(numREsInCw[cw])]

    # ******************************************************************************************************************
    def getLLRsFromGrid(self, rxGrid, pdschIndexes, llrScales=None, noiseVar=None, useMax=True):
        r"""
        This method is used at the receiving side where the Log-likelihood-Ratios (LLRs) are extracted from the
        received resource grid ``rxGrid``. This is in some sense the opposite of the :py:meth:`populateGrid` method
        since it does the following:
        
            :Deinterleaving: Converting Physical Resource Blocks (PRBs) to Virtual Resource Blocks (VRBs). If enabled,
                the resources are re-ordered based on the interleaving configuration given by ``interleavingBundleSize``
                according to **3GPP TS 38.214, Section 5.1.4.1** so get the data in its original order.

            :Layer Demapping: Extracting the modulated complex symbols for each codeword from different layers of this
                :py:class:`PDSCH` according to **3GPP TS 38.211, Section 7.3.1.3**.

            :Demodulation: Converting complex symbols to Log-likelihood-Ratios (LLRs) using the 
                :py:class:`~neoradium.modulation.Modem` objects in the ``modems`` list of this :py:class:`PDSCH`. The
                demodulation for each codeword is performed separately by its own dedicated 
                :py:class:`~neoradium.modulation.Modem` object. This produces one or two sets of LLRs for each codeword.
                
            :Descrambling: The descrambling of the demodulated LLRs using the ``rnti`` and ``nID`` properties of this
                :py:class:`PDSCH`. These properties are used to initialize a *golden sequence* which is then used for
                the descrambling process according to **3GPP TS 38.211, Section 7.3.1.1**. The LLRs for each codeword
                are descrambled separately.
        
        This function returns a list of one or two NumPy arrays representing the LLRs for each codeword.

        Parameters
        ----------
        rxGrid: :py:class:`~neoradium.grid.Grid`
            The equalized received resource grid associated with this :py:class:`PDSCH`. Usually this is the
            :py:class:`~neoradium.grid.Grid` object obtained after equalization in the receiver pipeline (See the
            :py:meth:`~neoradium.grid.Grid.equalize` function).

        pdschIndexes: 3-tuple
            A tuple of 3 lists specifying locations of the set of resource elements in ``rxGrid`` that are assigned 
            to the user data. The function :py:meth:`getReIndexes` is typically used to obtain this.
            
        llrScales: 3-D NumPy array
            The Log-Likelihood Ratios (LLR) scaling factors which are used by demodulating process when extracting 
            Log-Likelihood Ratios (LLRs) from the equalized resource grid. The shape of this array **must** be the
            same shape as ``rxGrid``.
            
        noiseVar: float or None
            The variance of the Additive White Gaussian Noise (AWGN) present in the received resource grid. If this 
            is not provided (``noiseVar=None``), This function uses the ``noiseVar`` property of the ``rxGrid`` object.
            
        useMax : Boolean
            If `True`, this implementation uses the ``Max`` function in the calculation of the LLR values. This is
            faster but uses an approximation and is slightly less accurate than the actual Log Likelihood method 
            which uses logarithm and exponential functions. If `False`, the slower more accurate method is used.

        Returns
        -------
        list
            A list of one or two NumPy arrays each representing the LLRs for each codeword.
        """
        # First get the layer-mapped indices from the pdschIndexes
        layerMappedIndexes = self.getLayerMapIndexes(pdschIndexes)
        
        if noiseVar is None: noiseVar = rxGrid.noiseVar
        noiseVar = max(noiseVar, 1e-10)
        
        llrs = []
        for cw in range(self.numCW):
            demappedSymbols = rxGrid[ layerMappedIndexes[cw] ]              # The demapped symbols for this codeword
            cwLLRs = self.modems[cw].getLLRsFromSymbols(demappedSymbols, noiseVar, useMax)  # Demodulate symbols to LLRs
            cwLLRs = self.scrambleLLRs(cw, cwLLRs)                                          # Descramble the LLRs
            if llrScales is not None:
                demappedScales = llrScales[ layerMappedIndexes[cw] ]
                cwLLRs *= np.repeat(demappedScales, self.modems[cw].qm)
            llrs += [ cwLLRs ]                                                              # Add to the list
        return llrs
    
    # ******************************************************************************************************************
    def getHardBitsFromGrid(self, rxGrid, pdschIndexes, llrScales=None, noiseVar=None, useMax=True):
        r"""
        This method first calls the :py:meth:`getLLRsFromGrid` function above and then uses hard-decisions on the
        returned LLRs to get the output user bits.
        
        This can be used when there is no channel coding in the communication pipeline. It returns a list of one or
        two NumPy arrays of bits for each codeword.

        Parameters
        ----------
        rxGrid: :py:class:`~neoradium.grid.Grid`
            The equalized received resource grid associated with this :py:class:`PDSCH`. Usually this is the
            :py:class:`~neoradium.grid.Grid` object obtained after equalization in the receiver pipeline (See the
            :py:meth:`~neoradium.grid.Grid.equalize` function).

        pdschIndexes: 3-tuple
            A tuple of 3 lists specifying locations of the set of resource elements in ``rxGrid`` that are assigned
            to the user data. The function :py:meth:`getReIndexes` is typically used to obtain this.
            
        llrScales: 3-D NumPy array
            The Log-Likelihood Ratios (LLR) scaling factors which are used by demodulating process when extracting 
            Log-Likelihood Ratios (LLRs) from the equalized resource grid. The shape of this array **must** be the
            same shape as ``rxGrid``.
            
        noiseVar: float or None
            The variance of the Additive White Gaussian Noise (AWGN) present in the received resource grid. If this
            is not provided (``noiseVar=None``), This function uses the ``noiseVar`` property of the ``rxGrid`` object.
            
        useMax : Boolean
            If `True`, this implementation uses the ``Max`` function in the calculation of the LLR values. This
            is faster but uses an approximation and is slightly less accurate than the actual Log Likelihood method 
            which uses logarithm and exponential functions. If `False`, the slower more accurate method is used.

        Returns
        -------
        list
            A list of one or two NumPy arrays of bits for each codeword.
        """
        llrs = self.getLLRsFromGrid(rxGrid, pdschIndexes, llrScales, noiseVar, useMax)
        return [ np.int8( llrs[cw]<0 ) for cw in range(self.numCW) ]
    
    # ******************************************************************************************************************
    def getDataSymbols(self, grid):
        r"""
        This is a helper function that returns the modulated complex symbols for all user data in ``grid`` for this
        :py:class:`PDSCH` object. The following code shows two different ways to do this:
        
        .. code-block:: python

            # Getting the indices of user data in "grid" and then using them to get "dataSymbols1":
            dataReIndexes = myPdsch.getReIndexes(grid, "PDSCH")
            dataSymbols1 = grid[ dataReIndexes ]
            
            # Using the "getDataSymbols" function:
            dataSymbols2 = myPdsch.getDataSymbols(grid)
            
            assert np.all(dataSymbols1==dataSymbols2)             # The results are the same


        Parameters
        ----------
        grid: :py:class:`~neoradium.grid.Grid`
            The resource grid associated with this :py:class:`PDSCH` containing the user data.

        Returns
        -------
        NumPy array
            A 1D NumPy array of modulated complex symbols corresponding to the user data in ``grid``.
        """
        return grid[ self.dataIndices ]
            
    # ******************************************************************************************************************
    def getPrecodingMatrix(self, channelMatrix):
        r"""
        This function calculates the precoding matrix that can be applied to a resource grid. This function supports
        *Precoding RB groups (PRGs)* which means different precoding matrices could be applied to different groups
        of subcarriers in the resource grid. See **3GPP TS 38.214, Section 5.1.2.3** for more details. The ``prgSize``
        property of :py:class:`PDSCH` determines what type of precoding matrix is returned by this function:
        
            :Wideband: If ``prgSize`` is set to zero, a single ``Nt x Nl``, matrix is returned where ``Nt`` is the
                number of transmitter antenna and ``Nl`` is the number of layers in this :py:class:`PDSCH`. In this
                case the same precoding is applied to all subcarriers of the resource grid.
            
            :Using PRGs: If ``prgSize`` is set to 2 or 4, a list of tuples of the form (``groupRBs``, ``groupF``)
                is returned. For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all
                subcarriers of the resource blocks listed in ``groupRBs``.
        
        .. Note:: It is assumed that the ``channelMatrix`` is obtained based on the same 
            :py:class:`~neoradium.carrier.BandwidthPart` object as the one used by this :py:class:`PDSCH`.

        Parameters
        ----------
        channelMatrix: NumPy array
            An ``L x K x Nr x Nt`` complex NumPy array representing the channel matrix. It can be the actual channel 
            matrix obtained directly from a channel model using the 
            :py:meth:`~neoradium.channelmodel.ChannelModel.getChannelMatrix` method (perfect estimation), or an 
            estimated channel matrix obtained using the :py:meth:`~neoradium.grid.Grid.estimateChannelLS` method.

        Returns
        -------
        NumPy array or list of tuples
            Depending on the ``prgSize`` property of this :py:class:`PDSCH`, the returned value can be:
            
            :NumPy Array: If ``prgSize`` is set to zero, a single *Wideband* ``Nt x Nl``, matrix is returned where
                ``Nt`` is the number of transmitter antenna and ``Nl`` is the number of layers in this 
                :py:class:`PDSCH`. In this case the same precoding is applied to all subcarriers of the resource grid.
            
            :list of tuples: If ``prgSize`` is set to 2 or 4, a list of tuples of the form (``groupRBs``, ``groupF``)
                is returned. For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all
                subcarriers of the resource blocks listed in ``groupRBs``.
        """
        # Channel Matrix Shape: numSym x numREs x Nr x Nt
        numRBs = channelMatrix.shape[1]//12
        if numRBs < len(self.prbSet):
            raise ValueError("The number of RBs in the 'channelMatrix' (%d) cannot be less than RBs in the PDSCH (%d)!"
                             %(numRBs, len(self.prbSet)))

        def getGroupPrecoder(channelMatrix, groupREs):      # Get a precoder matrix (Nt x Nl) for the specified group
            groupChannel = channelMatrix[:,groupREs,:,:]        # Channel matrix for the group specified groupREs
            groupChannel = groupChannel.mean(axis=(0,1))        # Average over time and frequency => Shape (Nr x Nt)
            _, _, vH = np.linalg.svd(groupChannel)              # vH Shape: Nt x Nt
            groupPrecoder = (np.conj(vH).T)[:,:self.numLayers]  # Nt x Nl
            return groupPrecoder/np.sqrt(self.numLayers)        # Normalize the group precoder

        curGroup = -1
        curGroupRBs = []    # The RBs in the current group that are used by this PDSCH
        # The precoding matrix can take the following forms:
        # a) A single Nt x Nl matrix which is applied to the whole resource grid.
        # b) A list of tuples of the form (rbList, precodingMatrix). Each entry in the list
        #    means: apply "precodingMatrix" to the RBs specified in the "rbList". The "rbList"
        #    entries only contain the RBs in this PDSCH. So, if for example there are multiple
        #    PDSCHs on the same BWP, the precoder lists can be calculated separately, aggregated
        #    together, and applied to one grid containing the data for all PDSCHs.
        f = []
        for prb in self.prbSet:
            # For the "Wideband" case (prgSize=0), everything is in the same group
            group = 0 if self.prgSize==0 else (prb + self.bwp.startRb)//self.prgSize
            curGroupRBs += [prb]
            
            if group != curGroup:
                # A new group is found; process the current group, then start the new group.
                reIndexes = np.int32([rb*12+re for rb in curGroupRBs for re in range(12)])
                f += [ (curGroupRBs, getGroupPrecoder(channelMatrix, reIndexes)) ]
                curGroup = group
                curGroupRBs = []

        # Process the last group
        if group != curGroup:
            # New group, process current group first, then start the new group
            reIndexes = np.int32([rb*12+re for rb in curGroupRBs for re in range(12)])
            f += [ (curGroupRBs, getGroupPrecoder(channelMatrix, reIndexes)) ]

        if (len(self.prbSet) == numRBs) and (self.prgSize==0):
            # If all PRBs are used by this PDSCH and prgSize=0 (Wideband) => A single Nt x Nl precoding
            # matrix could be applied to the whole grid.
            return f[0][1]  # Shape: Nt x Nl
        
        return f            # A list of tuples of the form (rbList, precodingMatrix)

    # ******************************************************************************************************************
    def getTxBlockSize(self, codeRates, xOverhead=0, scaleFactor=1.0):
        r"""
        This function calculates the transport block size based on the desired code rate (``codeRates``), the number
        of additional overhead resource elements (``xOverhead``), and the scaling factor (``scaleFactor``). It returns
        a list of one or two integer values specifying the size of transport blocks for each codeword. This
        implementation is based on **3GPP TS 38.214, Section 5.1.3.2**.
        
        Parameters
        ----------
        codeRates: float, list, NumPy array, or tuple
            If ``codeRates`` is a float value, it specifies the same code rate for all codewords. If it is a list, 
            NumPy array, or tuple, it should contain one or two code rate values for each codeword. This is the
            value :math:`R` in **3GPP TS 38.214, Section 5.1.3.2**.
            
        xOverhead: int
            The number of additional *overhead* resource elements (REs) that should be considered when calculating 
            the transport block size. This is the value :math:`N^{PRB}_{oh}` in **3GPP TS 38.214, Section 5.1.3.2**.
            
        scaleFactor: float
            The scaling factor, which **must** be one of: 0.25, 0.5, or 1.0. This is the value :math:`S` in 
            **3GPP TS 38.214, Table 5.1.3.2-2**.

        Returns
        -------
        list
            A list of one or two integers depending on the number of codewords (``numCW``), indicating the transport
            block size for each codeword.
        """
        if type(codeRates) in [float, np.float32, np.float64]:  codeRates = [codeRates]
        elif type(codeRates) in [list, np.ndarray, tuple]:      codeRates = list(codeRates)
        else:
            raise ValueError("'codeRates' must be a float value, or a list, tuple, or NumPy array of 1 or 2 float values. ('%s' is not supported)"%(type(codeRates).__name__))
        if len(codeRates)<self.numCW:           codeRates = self.numCW * codeRates
        codeRates = codeRates[:self.numCW]

        if scaleFactor not in [1/4, 1/2, 1]:    raise ValueError("'scaleFactor' must be one of: 0.25, 0.5, or 1")

        # 3GPP TS 38.214 V17.0.0 (2021-12), Section 5.1.3.2
        # Step 1:
        numPRBs = len(self.prbSet)
        npRE = 12*len(self.symSet)
        if self.dmrs is not None:   npRE -= len(self.dmrs.symSet)*(12 - len(self.dmrs.dataREs))
        assert npRE>0
        if npRE<=xOverhead:                     raise ValueError("'xOverhead' must be less than %d."%(npRE))
        npRE-=xOverhead
        numREs = min(156, npRE)*numPRBs
        
        cwLayers = [self.numLayers] if self.numCW==1 else [self.numLayers//2, self.numLayers-self.numLayers//2]
        
        # Step 2:
        txBlockSize = []
        for c in range(self.numCW):
            nInfo = scaleFactor * numREs * codeRates[c] * self.modems[c].qm * cwLayers[c]
            if nInfo <= 3824:
                # Step 3:
                n = max(3, int(np.log2(nInfo))-6)
                npInfo = max(24, (1<<n)*(nInfo//(1<<n)))
                # TS 38.214 V17.0.0 (2021-12), Table 5.1.3.2-1
                txBlockSizes = np.int32([24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
                                         168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384,
                                         408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
                                         888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352,
                                         1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216,
                                         2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368,
                                         3496, 3624, 3752, 3824])
                txBlockSize += [ txBlockSizes[txBlockSizes>=npInfo][0] ]
            else:
                # Step 4:
                n = int(np.log2(nInfo-24))-5
                npInfo = max(3840, (1<<n)*np.round((nInfo-24)/(1<<n)))

                if codeRates[c] <= 0.25:    eightC = 8*np.ceil((npInfo + 24)/3816)
                elif npInfo > 8424:         eightC = 8*np.ceil((npInfo + 24)/8424)
                else:                       eightC = 8
                
                txBlockSize += [ int(eightC*np.ceil((npInfo + 24)/eightC)) - 24 ]

        return txBlockSize

