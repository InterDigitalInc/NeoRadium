# Copyright (c) 2024-2026, InterDigital AI Lab
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
# 12/18/2025    Shahab Hamidi-Rad       Some bug fixes related to the "getPrecodingMatrix" method.
# 04/09/2026    Shahab Hamidi-Rad       Changes in NeoRadium version 0.5.0:
#                                       * Allow PDSCH to use a subset of OFDM symbols and/or resource blocks. Added
#                                         'grb2Prb' and 'prb2Grb' lookup tables to convert from the Grid RBs to physical
#                                         RBs and vice versa.
#                                       * Moved the interleaving processing to the BandwidthPart object. The parameter
#                                         'interleavingBundleSize' will be removed from PDSCH in the future.
#                                       * PDSCH now receives a CsiRsConfig object that is used to reserve REs for
#                                         CSI-RS when populating the resource grid for the PDSCH.
#                                       * Removed the support for 'reservedReMap'. The functionality can be implemented
#                                         easily by the user by manually assigning specific REs to "RESERVED".
#                                       * Replaced some of the assertions to real validations that raise ValueError.
#                                       * New functions: 'initGrid', 'getBitCapacity', 'setPdschData', 'precodeTo',
#                                         'equalize', and 'getLdpcCodec'.
#                                       * Deprecated functions: 'getGrid', 'getReIndexes', 'getBitSizes',
#                                         'populateGrid'.
# 08/01/2026    Shahab Hamidi-Rad       Changes in NeoRadium version 0.5.1:
#                                       * Added "Automatic overhead selection" to the transport block size calculations
#                                         in the 'getTxBlockSize' function.
#                                       * Added a new optional input parameter 'w' to the 'precodeTo' function.
# **********************************************************************************************************************
import numpy as np
import scipy
from scipy.interpolate import interp1d

from .grid import Grid
from .modulation import Modem
from .utils import goldSequence, getMultiLineStr, herm, warnOnce, deprecated, validateRange
from .dmrs import DMRS
from .ldpccodec import LdpcCodec
from .harq import HarqEntity

docFile = "PhyChannels"         # Used by the 'deprecated' decorators

# This implementation is based on:
#   - TS 38.211
#   - TS 38.212
#   - TS 38.214
# The following links can help clarify some ambiguities in the standard:
#   https://www.sharetechnote.com/html/5G/5G_PDSCH.html
#   https://www.sharetechnote.com/html/5G/5G_PDSCH_DMRS.html
#   https://www.sharetechnote.com/html/5G/5G_PTRS_DL.html

# **********************************************************************************************************************
class PDSCH:
    r"""
    This class encapsulates the configuration and functionality of a Physical Downlink Shared Channel (PDSCH) that 
    delivers user data transmitted from gNB to UE.
    """
    # ******************************************************************************************************************
    def __init__(self, bwp, **kwargs):
        r"""
        Parameters
        ----------
        bwp : :py:class:`~neoradium.carrier.BandwidthPart`
            The :py:class:`~neoradium.carrier.BandwidthPart` object that represents the resources used by this 
            :py:class:`PDSCH` for transmission of user data from gNB to UE.
            
        kwargs : dict
            A set of optional arguments.

                :mappingType: The mapping type used by this PDSCH and its associated
                    :py:class:`~neoradium.dmrs.DMRS` object. It is a string that can be either ``'A'`` or 
                    ``'B'``. The default is ``'A'``.
                    
                    In mapping type ``'A'``, the first DM-RS OFDM symbol index is 2 or 3 and DM-RS is mapped relative
                    to the start of slot boundary, regardless of where in the slot the actual data transmission
                    starts. The user data in this case usually occupies most of the slot.

                    In mapping type ``'B'``, the first DM-RS OFDM symbol is the first OFDM symbol of the data
                    allocation, that is, the DM-RS location is not given relative to the slot boundary but relative
                    to where the user data is located. The user data in this case usually occupies a small fraction
                    of the slot to support very low latency.

                :numLayers: The number of transmission layers for this :py:class:`PDSCH`. It must be an integer
                    from 1 to 8, with 1 as the default.

                :modulation: A string, or a tuple or list of two strings specifying the modulation scheme 
                    used for data transmitted in this :py:class:`PDSCH` based on **3GPP TS 38.211, Table 7.3.1.2-1**.
                    The default is ``'16QAM'``. Here is a list of supported modulation schemes:
                        
                    ===================  =========================
                    Modulation Scheme    Modulation Order (``qm``)
                    ===================  =========================
                    QPSK                 2
                    16QAM                4
                    64QAM                6
                    256QAM               8
                    1024QAM              10
                    ===================  =========================

                    If ``modulation`` is a string and there are two codewords in this :py:class:`PDSCH`, the
                    same modulation scheme is used for both codewords. If there are two codewords in this
                    :py:class:`PDSCH`, and you want to use different modulation schemes for the two codewords, you
                    can specify two different modulation schemes in a tuple or list of strings. For example:
                    
                    .. code-block:: python

                        # Using "QPSK" for the first codeword and "16QAM" for the second codeword
                        modulation = ("QPSK", "16QAM")
                        
                    The specified modulation scheme(s) are used to create one or two 
                    :py:class:`~neoradium.modulation.Modem` objects.

                :csiRsConfig: A :py:class:`~neoradium.csirs.CsiRsConfig` object that contains CSI-RS configuration
                    information. If specified, it is used to reserve CSI-RS resources in the grid so that they are 
                    not assigned to PDSCH, DM-RS, or PT-RS.             

                :reservedPrbSets: A list of :py:class:`~neoradium.carrier.ReservedPrbSet` objects that are used to
                    reserve the specified resource blocks (RBs) at the specified OFDM symbols based on the patterns
                    defined in the :py:class:`~neoradium.carrier.ReservedPrbSet` objects. The default is an empty list
                    which means no reserved PRBs.

                :portSet: A list of ports used by this :py:class:`PDSCH` and its associated  
                    :py:class:`~neoradium.dmrs.DMRS` object. If not specified, by default, this is set based on the 
                    number of layers specified by ``numLayers``. For example, for a 2-layer PDSCH, the ``portSet`` is
                    set to ``{0, 1}``, which corresponds to DM-RS port numbers ``{1000, 1001}``.
                
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

                :interleavingBundleSize: This is for backward compatibility and will be removed in future versions. Set
                    ``interleavingBundleSize`` in the :py:class:`~neoradium.carrier.BandwidthPart` object.

                :rnti: The *Radio Network Temporary Identifier*. The default is 1. It is used with ``nID`` below to
                    initialize a *Gold sequence* used for the scrambling process. See **3GPP TS 38.211, 
                    Section 7.3.1.1** for more information.
                    
                :nID: The *scrambling identity*. It is used with ``rnti`` to initialize a *Gold sequence* used
                    for the scrambling process. See **3GPP TS 38.211, Section 7.3.1.1** for more
                    information. If not specified, it is set to ``bwp.cellId``.

                :prgSize: The size of Precoding RB Groups (PRGs). It can be one of 0 (default), 2, or 4. The value 0
                    means *Wideband Precoding* which means the same precoding is used for the whole bandwidth of
                    this :py:class:`PDSCH`. Subband values (``2`` or ``4``) enable a separate precoder per PRG;
                    use a smaller ``prgSize`` when the channel is more frequency-selective (e.g., long delay
                    spread, low antenna correlation across frequency), at the cost of higher PMI feedback
                    overhead. ``prgSize=4`` is the natural midpoint when subband precoding is needed without
                    the full overhead of ``prgSize=2``. See **3GPP TS 38.214, Section 5.1.2.3** for more
                    information.
                    

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
                
            :slotNo: This returns the ``slotNo`` property of the :py:class:`~neoradium.carrier.Carrier` object 
                containing ``bwp``.
            
            :frameNo: This returns the ``frameNo`` property of the :py:class:`~neoradium.carrier.Carrier` object
                containing ``bwp``.
                
            :slotNoInFrame: This returns the ``slotNoInFrame`` property of the
                :py:class:`~neoradium.carrier.Carrier` object containing ``bwp``.
                
        The notebook :doc:`../Playground/Notebooks/PDSCH/PDSCH-endToEnd` shows how to create an end-to-end PDSCH 
        communication pipeline.
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
        validateRange(self.mappingType, ["A", "B"])

        self.numLayers = kwargs.get('numLayers', 1)
        validateRange(self.numLayers, (1, 8))

        self.numCW = 2 if self.numLayers>4 else 1

        self.csiRsConfig = kwargs.get('csiRsConfig', None)      # CSI-RS configuration
        self.reservedPrbSets = kwargs.get('reservedPrbSets', [])  # A list of ReservedPrbSet objects
        
        modulation = kwargs.get('modulation', '16QAM')          # See TS 38.211, Table 7.3.1.2-1
        if type(modulation)==str:                   modulation = self.numCW*[modulation]
        elif type(modulation) in [list, tuple]:     modulation = list(modulation)
        else:
            raise ValueError(f"'modulation' must be a string, a list of strings, or a tuple of strings. " +
                             f"('{type(modulation).__name__}' is not supported)")
        if len(modulation)<self.numCW: modulation = 2*modulation
        modulation = modulation[:self.numCW]
        for modStr in modulation:
            if modStr not in ['QPSK', '16QAM', '64QAM', '256QAM', '1024QAM']:
                raise ValueError(f"Unsupported modulation \"{modStr}\"!")
        # Make a Modem object based on the modulation scheme for each codeword
        self.modems = [ Modem(modulation[0]) ]
        if self.numCW>1:  # Use the same Modem object if both modulations are the same
            self.modems += [ self.modems[0] if modulation[0]==modulation[1] else Modem(modulation[1]) ]

        # Note that PDSCH is always a contiguous set of OFDM symbols.
        sliv = kwargs.get('sliv', None)
        symStart, symLen = kwargs.get('symStart', None), kwargs.get('symLen', None)
        if sliv is not None:
            # SLIV specified. See TS 38.214, Section 5.1.2.1
            s,l = sliv%14, sliv//14 + 1
            if s+l>14:  s,l = 13-s, 16-l
            check = (14*(l-1) + s) if l<=8 else (14*(14-l+1) + (14-1-s))
            if sliv != check:   raise ValueError(f"Failed to convert SLIV({sliv}) to start and length values!")
            self.symSet = np.uint32(range(s,s+l))
        elif (symStart is not None) and (symLen is not None):
            self.symSet = np.uint32(range(symStart, symStart+symLen))
        else:
            if self.mappingType=='A':           defaultSymSet = range(self.bwp.symbolsPerSlot)
            elif self.bwp.cpType=='normal':     defaultSymSet = range(13)
            else:                               defaultSymSet = range(6)
            self.symSet = np.sort(np.uint32(kwargs.get('symSet', defaultSymSet)))   # The set of symbols allocated

        if len(self.symSet)==0:             raise ValueError(f"'symSet' must not be empty!")
        if self.symSet[-1]>=self.bwp.symbolsPerSlot or self.symSet[0]<0:
            raise ValueError(f"Invalid 'symSet' values! (They must be in [0..{self.bwp.symbolsPerSlot-1}])")
        if len(self.symSet)>1 and np.diff(self.symSet).max()>1:
            raise ValueError(f"Invalid 'symSet' values! The OFDM symbol allocation must be contiguous!")

        # The size of Precoding RB groups (PRGs). See 3GPP TS 38.214, Section 5.1.2.3
        self.prgSize = kwargs.get('prgSize', 0) # 0 -> 'Wideband', which means a single precoding is used for all PRBs
        if self.prgSize not in [0,2,4]:     raise ValueError("'prgSize' must be 0 (Wideband), 2, or 4)")

        # If we are using subband precoding (prgSize>0) or using interleaving, then we have these rules:
        #   - bundle/group size (bgs) is maximum of prgSize and interleavingBundleSize
        #   - Number of PRBs must be a multiple of bgs
        #   - Each contiguous PRB group must start at a multiple of bgs
        #   - The number of PRBs in each contiguous group must be a multiple of bgs
        self.prbSet = np.sort(np.unique(np.uint32(kwargs.get('prbSet', range(0, self.bwp.numRbs)))))
        if len(self.prbSet)==0:             raise ValueError(f"'prbSet' must not be empty!")
        if self.prbSet[-1]>=self.bwp.numRbs or self.prbSet[0]<0:
            raise ValueError(f"Invalid 'prbSet' values! (They must be in [0..{self.bwp.numRbs-1}])")

        # This is only for backward compatibility. Will be removed in future.
        # Interleaving is implemented in the BandwidthPart class starting in NeoRadium 0.5.
        interleavingBundleSize = kwargs.get('interleavingBundleSize', None)
        if interleavingBundleSize is not None:
            warnOnce("'interleavingBundleSize' is a property of the BandwidthPart class. "+
                     "It will be removed from the PDSCH class in future releases.")
            # Warning: You should set interleavingBundleSize for the bandwidth part.
            if interleavingBundleSize not in [0,2,4]:
                raise ValueError("'interleavingBundleSize' must be 0 (Interleaving disabled), 2, or 4")
            if len(self.prbSet)==self.bwp.numRbs:       # If only one PDSCH in the whole BWP, and
                if self.bwp.interleavingBundleSize==0:  # interleaving not set for BWP, set it in BWP, otherwise, ignore
                    self.bwp.interleavingBundleSize = interleavingBundleSize
                    self.bwp.setVrbToPrbMapping()

        bgs = max(self.prgSize, self.bwp.interleavingBundleSize)    # Bundle/Group size
        if bgs>0:
            if len(self.prbSet)%bgs: raise ValueError(f"Length of 'prbSet' must be a multiple {bgs}!")
            if self.bwp.numRbs%bgs:  raise ValueError(f"Number of RBs in BandwidthPart must be a multiple of {bgs}!")
            if self.bwp.startRb%bgs: raise ValueError(f"BandwidthPart's 'startRb' must be a multiple of {bgs}!")

            if self.prbSet[0]%bgs:   raise ValueError(f"'prbSet' must start at a multiples of {bgs}!")
            if sum((self.prbSet[1:]-self.prbSet[:-1]-1) % bgs)>0:
                raise ValueError(f"'prbSet' must contain groups whose length and start are both multiples of {bgs}!")

        # NOTE:
        # We have 3 types of RBs: PRB, VRB, GRB
        # VRB <-> PRB: This conversion handles the interleaving. It is done in BandwidthPart object. There is
        #              always a one-to-one relationship (e.g., shuffling)
        # GRP <-> PRB: Direct conversion from GRB to PRB. This is not a one-to-one relationship. If the "prbSet" has
        #              less RBs than the BWP, then GRBs are mapped to a subset of PRBs that are included in "prbSet".
        # GRBs are always contiguous (all GRBs are used for PDSCH). Depending on prbSet, some PRBs and VRBs may not
        # be used by the PDSCH.
        # In the simple case when all BWP PRBs are used for PDSCH and interleaving is disabled: GRB=VRB=PRB.
        # See the page "Interleaving, GRBs, VRBs, and PRBs" in the "ImplementationNotes.key".
        self.grb2Prb = self.bwp.vrb2Prb[ np.sort(self.bwp.prb2Vrb[self.prbSet]) ]       # GRB to PRB
        self.prb2Grb = np.int32([-1]*self.bwp.numRbs)                                   # PRB to GRB
        self.prb2Grb[self.grb2Prb] = np.arange(len(self.prbSet))                        # PRB to GRB

        self.rnti = kwargs.get('rnti', 1)               # Radio Network Temporary Identifier
        self.nID = kwargs.get('nID', bwp.cellId)        # scrambling identity
        
        # Check Symbol allocation: (See TS 38.214, Table 5.1.2.1-1: Valid S and L combinations)
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
            
        self.portSet = np.int32(kwargs.get('portSet', list(range(self.numLayers))))%1000
        if len(self.portSet) < self.numLayers:
            raise ValueError(f"The number of ports must be at least equal to number of layers ({self.numLayers})!")
        self.dmrs = None
        self.grid = None
        self.dataIndices = None

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="PDSCH Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`PDSCH` object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str
            If specified, it is used as the title for the printed information.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
            Otherwise, nothing is returned.
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"

        repStr += indent*' ' + f"  mappingType:               {self.mappingType}\n"
        repStr += indent*' ' + f"  nID:                       {self.nID}\n"
        repStr += indent*' ' + f"  rnti:                      {self.rnti}\n"
        repStr += indent*' ' + f"  numLayers:                 {self.numLayers}\n"
        repStr += indent*' ' + f"  numCodewords:              {self.numCW}\n"
        modStr = self.modems[0].modulation
        if (len(self.modems)>1) and (self.modems[0].modulation!=self.modems[1].modulation):
            modStr += ", " + self.modems[1].modulation
        repStr += indent*' ' + f"  modulation:                {modStr}\n"
        repStr += indent*' ' + f"  PRG Size:                  {'Wideband' if self.prgSize==0 else self.prgSize}\n"
        repStr += getMultiLineStr("portSet                  ", self.portSet, indent, "%-3d", 3, numPerLine=20)
        repStr += getMultiLineStr("symSet                   ", self.symSet, indent, "%-3d", 3, numPerLine=20)
        repStr += getMultiLineStr("prbSet                   ", self.prbSet, indent, "%-3d", 3, numPerLine=20)
        if self.dmrs is not None:
            repStr += self.dmrs.print(indent+2, "DMRS:", True)
            if self.dmrs.ptrs is not None:
                repStr += self.dmrs.ptrs.print(indent+2, "PTRS:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def setDMRS(self, **kwargs):
        r"""
        Creates and initializes a :py:class:`~neoradium.dmrs.DMRS` object associated with this :py:class:`PDSCH` object.

        Parameters
        ----------
        kwargs : dict
            A dictionary of parameters passed directly to the constructor of the :py:class:`~neoradium.dmrs.DMRS`
            class. Please refer to this class for a list of parameters that can be used to configure DM-RS.
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
        kwargs : dict
            A dictionary of parameters passed directly to the constructor of the :py:class:`~neoradium.dmrs.PTRS`
            class. Please refer to this class for a list of parameters that can be used to configure PT-RS.
        """
        if self.dmrs is None: raise ValueError("Cannot set PTRS without first defining a DMRS object for this PDSCH!")
        self.dmrs.setPTRS(**kwargs)

    # ******************************************************************************************************************
    def __getattr__(self, attrName):        # already documented in the __init__ function.
        # Get these attributes from the 'bwp' object
        if attrName not in ["slotNo", "frameNo", "slotNoInFrame"]:
            raise AttributeError(f"Class '{self.__class__.__name__}' does not have any property named '{attrName}'!")
        return getattr(self.bwp, attrName)

    # ******************************************************************************************************************
    def scrambleBits(self, q, bits):                            # Undocumented
        # See TS 38.211, Section 7.3.1.1
        cInit = self.rnti * (1<<15) + q * (1<<14) + self.nID
        scramblingSeq = goldSequence(cInit, len(bits))
        scrambledBits = bits ^ scramblingSeq
        return scrambledBits

    # ******************************************************************************************************************
    def scrambleLLRs(self, q, llrs):                            # Undocumented
        # See TS 38.211, Section 7.3.1.1
        cInit = self.rnti * (1<<15) + q * (1<<14) + self.nID
        scramblingSeq = 1-2*np.float64(goldSequence(cInit, len(llrs)))
        scrambledLLRs = llrs * scramblingSeq
        return scrambledLLRs

    # ******************************************************************************************************************
    def getLayerMapIndexes(self, psdchIndexes, numREsInCw=None):   # Undocumented
        # 'numREsInCw' is a list of number of REs in each codeword for one or two codewords. If None, get all the
        # REs indexed by the "psdchIndexes"
        if numREsInCw is None:      numREsInCw = self.getNumREsFromIndexes(psdchIndexes)
        
        # See TS 38.211, Section 7.3.1.3
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
    @deprecated("initGrid", docFile)
    def getGrid(self):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`initGrid` method instead.
        """
        return self.initGrid()
    
    # ******************************************************************************************************************
    def initGrid(self):
        r"""
        Creates a :py:class:`~neoradium.grid.Grid` object for this :py:class:`PDSCH` and populates it with the
        configured :py:class:`~neoradium.dmrs.DMRS` and :py:class:`~neoradium.dmrs.PTRS` reference signals. 
        
        If a :py:class:`~neoradium.csirs.CsiRsConfig` object was provided when this :py:class:`PDSCH` was created,
        it will be used to reserve the CSI-RS resources so that they are not assigned to PDSCH, DM-RS, or PT-RS.
        
        This function also marks all resources corresponding to the ``reservedPrbSets`` parameter as "RESERVED" in
        the newly created resource grid.

        The returned resource grid contains all reference signals and is ready to be populated with the user data (See
        :py:meth:`setPdschData` method).

        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A :py:class:`~neoradium.grid.Grid` object representing the resource grid for this :py:class:`PDSCH`
            pre-populated with DM-RS and PT-RS.
        """
        # Creates PDSCH's internal grid object. This grid may have less RBs than the bandwidth part, and the RBs in
        # this grid may map to non-contiguous RBs in the bandwidth part (depending on `prbSet`)
        self.grid = Grid(self.bwp, self.numLayers, contents="PDSCH", numRbs=len(self.prbSet))
        self.allocateResources()
        return self.grid
        
    # ******************************************************************************************************************
    @deprecated("Grid.getReIndexes", docFile)
    def getReIndexes(self, grid, reTypeStr):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`~neoradium.grid.Grid.getReIndexes` method instead.
        """
        return grid.getReIndexes(reTypeStr)

    # ******************************************************************************************************************
    def getNumREsFromIndexes(self, indexes):
        r"""
        Returns the number of resource elements included in ``indexes`` for each codeword. The returned value is a
        list of one or two integers depending on the number of codewords (``numCW``).

        Parameters
        ----------
        indexes : 3-tuple
            A tuple of 3 lists specifying locations of a set of resource elements in the resource grid. For example, 
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
    def getBitCapacity(self):
        r"""
        Returns the total number of bits corresponding to PDSCH resource elements. The returned value is a
        list of one or two integers depending on the number of codewords (``numCW``).
        
        Returns
        -------
        list
            A list of one or two integers depending on the number of codewords (``numCW``), indicating the number of
            PDSCH bits for each codeword.

            
        .. Note:: This function replaces the deprecated function :py:meth:`getBitSizes`. The following example 
            shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                grid = pdsch.getGrid()
                numBits = pdsch.getBitSizes(grid)

                # New:
                pdsch.initGrid()
                numBits = pdsch.getBitCapacity()                       
        """
        if self.dataIndices is None:
            raise ValueError("The 'initGrid' function must be called before calling 'getBitCapacity'.")

        # Return the number of data bits that can be carried by this PDSCH in current slot for each codeword
        numREsInCw = self.getNumREsFromIndexes(self.dataIndices)
        return [ numREsInCw[i] * self.modems[i].qm for i in range(self.numCW) ]

    # ******************************************************************************************************************
    @deprecated("getBitCapacity", docFile)
    def getBitSizes(self, grid=None, reTypeStr="PDSCH"):
        r"""
        :red:`DEPRECATED`: This function is deprecated and will be removed in future releases. Please use the 
        :py:meth:`getBitCapacity` method instead.
        """
        return self.getBitCapacity()

    # ******************************************************************************************************************
    def allocateResources(self):                        # Undocumented
        # Allocate resources for this PDSCH
        for reservedPrbSet in self.reservedPrbSets: reservedPrbSet.populateGrid(self)       # Reserve RBs
        if self.csiRsConfig is not None: self.csiRsConfig.reserveGridResources(self.grid)   # Reserve CSI-RS resources
        if self.dmrs is not None:                   self.dmrs.populateGrid(self.grid)       # Allocate DM-RS/PT-RS

        # Now mark every remaining RE in the grid as "PDSCH"
        pdschIdx = []
        for layer in range(self.numLayers):
            for l in self.symSet:
                for vrb in range(len(self.prbSet)):
                    for re in range(12):
                        k = vrb*12 + re
                        curReType = self.grid.reTypeAt(layer,l,k)
                        if curReType in ["DMRS", "CSIRS_ZP", "CSIRS_NZP", "RESERVED", "PTRS", "NO_DATA"]: continue
                        if curReType not in ["UNASSIGNED", "PDSCH"]:
                            raise ValueError(f"Trying to allocate the RE at ({layer},{l},{re}) for PDSCH," +
                                             f"while it is currently allocated for \"{curReType}\"!")
                        self.grid[layer,l,k] = (0, "PDSCH")
                        pdschIdx += [ [layer,l,k] ]
        self.dataIndices = tuple( np.int32(pdschIdx).T )

    # ******************************************************************************************************************
    def setPdschData(self, dataBits):
        r"""
        *Populates* this PDSCH's resource grid with the user data provided in ``dataBits``.
        
        This function performs the following operations:
        
            :Scrambling: Scrambling of the specified ``dataBits`` using the ``rnti`` and ``nID`` properties of this
                :py:class:`PDSCH`. These properties are used to initialize a *Gold sequence* which is then used
                for the scrambling process according to **3GPP TS 38.211, Section 7.3.1.1**. The data bits for each
                codeword are scrambled separately.
                
            :Modulation: Converting the scrambled binary data stream into complex symbols for each resource element
                assigned for user data. The modulation process is performed by the 
                :py:class:`~neoradium.modulation.Modem` objects in the ``modems`` list of this :py:class:`PDSCH`. The
                modulation for each codeword is performed separately by its own dedicated 
                :py:class:`~neoradium.modulation.Modem` object.
                
            :Layer Mapping: Distributing the modulated complex symbols across one or more transmission layers of this
                :py:class:`PDSCH` according to **3GPP TS 38.211, Section 7.3.1.3**.

        Parameters
        ----------
        dataBits : list, tuple, or NumPy array
            Specifies the user data bits that are used to populate the specified resource grid. It can be one of the
            following:
            
            :tuple of NumPy arrays: Depending on the number of codewords (``numCW``), the tuple can have one or two
                1D NumPy arrays of bits each specifying the user data bits for each codeword.
                
            :NumPy array: A one or two dimensional NumPy array. It is a 1D array, only if we have one codeword and the
                given NumPy array is used for the single codeword. The 2D NumPy array can be used for cases with one 
                or two codewords. The first dimension of the NumPy array in this case should match the number
                of codewords (``numCW``).
                
            :list of NumPy arrays: Depending on the number of codewords (``numCW``), the list can have one or two 1D
                NumPy arrays of bits each specifying the user data bits for each codeword.
                
                
        .. Note:: This function replaces the deprecated function :py:meth:`populateGrid`. The following example 
            shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                grid = pdsch.getGrid()                       # Create a resource grid already populated with DMRS
                txBlockSize = pdsch.getTxBlockSize(codeRate) # Calculate the Transport Block Size
                txBlock = random.bits(txBlockSize[0])        # Create random binary data
                numBits = pdsch.getBitSizes(grid)            # Actual number of bits available in the resource grid
                rateMatchedCodeBlocks = ldpcEncoder.getRateMatchedCodeBlocks(txBlock, numBits[0])
                pdsch.populateGrid(grid, rateMatchedCodeBlocks)

                # New:
                pdsch.initGrid()                             # Create and initialize PDSCH's internal grid
                txBlock = random.bits(ldpc.txBlockSizes[0])  # Create random binary data
                numBits = pdsch.getBitCapacity()             # Actual number of bits available in the resource grid   
                rateMatchedCodeBlocks = ldpc.encode(txBlock, numBits[0])
                pdsch.setPdschData(rateMatchedCodeBlocks)
        """
        if self.grid is None:
            raise ValueError("The 'initGrid' function must be called before calling 'setPdschData'.")
        
        if type(dataBits)==tuple:           dataBits = list(dataBits)
        elif type(dataBits)==np.ndarray:
            if dataBits.ndim==1:            dataBits = [dataBits]
            else:                           dataBits = [ dataBits[i] for i in range(dataBits.shape[0]) ]
        elif type(dataBits)!=list:
            raise ValueError("'dataBits' must be a NumPy array, a tuple of NumPy arrays, or a list of NumPy arrays.")
        if self.numCW!=len(dataBits):
            raise ValueError(f"Number of codewords is {self.numCW} but {len(dataBits)} set(s) of data bits are provided!")

        maxBits = self.getBitCapacity()
        for cw in range(self.numCW):
            if len(dataBits[cw])>maxBits[cw]:
                raise ValueError(f"The number of bits provided ({len(dataBits[cw]):,}) exceeds "
                                 f"the maximum ({maxBits[cw]:,}) for codeword {cw}.")

        symbols = []    # One item in the list for each codeword
        for cw in range(self.numCW):
            scrambledBits = self.scrambleBits(cw, dataBits[cw])
            symbols += [ self.modems[cw].modulate(scrambledBits) ]
        
        numREsInCw = [len(s) for s in symbols]
        layerMappedIndexes = self.getLayerMapIndexes(self.dataIndices, numREsInCw)
        for cw, layerMappedIndex in enumerate(layerMappedIndexes):
            self.grid[ layerMappedIndex ] = (symbols[cw], "PDSCH")
            
    # ******************************************************************************************************************
    @deprecated("setPdschData", docFile)
    def populateGrid(self, grid, bits=None):
        r"""
        :red:`DEPRECATED`: This function is deprecated and will be removed in future releases. Please use the
        :py:meth:`setPdschData` method instead.
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
            

    # ******************************************************************************************************************
    def getLLRs(self, eqGrid, llrScales=None, noiseVar=None, useMax=True):
        r"""
        This method is used at the receiving side where the log-likelihood ratios (LLRs) are extracted from the
        equalized resource grid ``eqGrid``. This is in some sense the opposite of the :py:meth:`setPdschData` method
        since it does the following:
        
            :Deinterleaving: Converting Physical Resource Blocks (PRBs) to Virtual Resource Blocks (VRBs). If enabled,
                the resources are re-ordered based on the interleaving configuration given by ``interleavingBundleSize``
                according to **3GPP TS 38.214, Section 5.1.4.1** to get the data in its original order.

            :Layer Demapping: Extracting the modulated complex symbols for each codeword from different layers of this
                :py:class:`PDSCH` according to **3GPP TS 38.211, Section 7.3.1.3**.

            :Demodulation: Converting complex symbols to log-likelihood ratios (LLRs) using the 
                :py:class:`~neoradium.modulation.Modem` objects in the ``modems`` list of this :py:class:`PDSCH`. The
                demodulation for each codeword is performed separately by its own dedicated 
                :py:class:`~neoradium.modulation.Modem` object. This produces one or two sets of LLRs for each codeword.
                
            :Descrambling: The descrambling of the demodulated LLRs using the ``rnti`` and ``nID`` properties of this
                :py:class:`PDSCH`. These properties are used to initialize a *Gold sequence* which is then used for
                the descrambling process according to **3GPP TS 38.211, Section 7.3.1.1**. The LLRs for each codeword
                are descrambled separately.
        
        This function returns a list of one or two NumPy arrays representing the LLRs for each codeword.

        Parameters
        ----------
        eqGrid : :py:class:`~neoradium.grid.Grid`
            The equalized received resource grid associated with this :py:class:`PDSCH`. Usually this is the
            :py:class:`~neoradium.grid.Grid` object obtained after equalization in the receiver pipeline (See the
            :py:meth:`equalize` function).

        llrScales : 3-D NumPy array or None
            The log-likelihood ratio (LLR) scaling factors which are used by demodulation process when extracting
            log-likelihood ratios (LLRs) from the equalized resource grid. The shape of this array **must** be the
            same shape as ``eqGrid``. Typically obtained as the second return value of :py:meth:`equalize`:

            .. code-block:: python

                eqGrid, llrScales = pdsch.equalize(rxGrid, channelMatrix)
                llrs = pdsch.getLLRs(eqGrid, llrScales)

            If ``None``, no per-RE scaling is applied.
            
        noiseVar : float or None
            The variance of the Additive White Gaussian Noise (AWGN) present in the received resource grid. If this 
            is not provided (``noiseVar=None``), This function uses the ``noiseVar`` property of the ``eqGrid`` object.
            
        useMax : bool
            If `True`, this implementation uses the ``Max`` function in the calculation of the LLR values. This is
            faster but uses an approximation and is slightly less accurate than the actual log-likelihood method 
            which uses logarithm and exponential functions. If `False`, the slower more accurate method is used.

        Returns
        -------
        list
            A list of one or two NumPy arrays each representing the LLRs for each codeword.
            
            
        .. Note:: This function replaces the deprecated function :py:meth:`getLLRsFromGrid`. The 
            following example shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                llrs = pdsch.getLLRsFromGrid(eqGrid, pdschIndexes, llrScales)

                # New:
                llrs = pdsch.getLLRs(eqGrid, llrScales)                       
        """
        # First get the layer-mapped indices
        layerMappedIndexes = self.getLayerMapIndexes(self.dataIndices)
        
        if noiseVar is None: noiseVar = eqGrid.noiseVar
        noiseVar = max(noiseVar, 1e-10)
        
        llrs = []
        for cw in range(self.numCW):
            demappedSymbols = eqGrid[ layerMappedIndexes[cw] ]                  # The demapped symbols for this codeword
            cwLLRs = self.modems[cw].getLLRs(demappedSymbols, noiseVar, useMax) # Demodulate symbols to LLRs
            cwLLRs = self.scrambleLLRs(cw, cwLLRs)                              # Descramble the LLRs
            if llrScales is not None:
                demappedScales = llrScales[ layerMappedIndexes[cw] ]
                cwLLRs *= np.repeat(demappedScales, self.modems[cw].qm)
            llrs += [ cwLLRs ]                                                  # Add to the list
        return llrs

    # ******************************************************************************************************************
    @deprecated("getLLRs", docFile)
    def getLLRsFromGrid(self, rxGrid, pdschIndexes=None, llrScales=None, noiseVar=None, useMax=True):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`getLLRs` method instead.

        This method is used at the receiving side where the log-likelihood ratios (LLRs) are extracted from the
        received resource grid ``rxGrid``. This is in some sense the opposite of the :py:meth:`populateGrid` method
        since it does the following:
        
            :Deinterleaving: Converting Physical Resource Blocks (PRBs) to Virtual Resource Blocks (VRBs). If enabled,
                the resources are re-ordered based on the interleaving configuration given by ``interleavingBundleSize``
                according to **3GPP TS 38.214, Section 5.1.4.1** to get the data in its original order.

            :Layer Demapping: Extracting the modulated complex symbols for each codeword from different layers of this
                :py:class:`PDSCH` according to **3GPP TS 38.211, Section 7.3.1.3**.

            :Demodulation: Converting complex symbols to log-likelihood ratios (LLRs) using the 
                :py:class:`~neoradium.modulation.Modem` objects in the ``modems`` list of this :py:class:`PDSCH`. The
                demodulation for each codeword is performed separately by its own dedicated 
                :py:class:`~neoradium.modulation.Modem` object. This produces one or two sets of LLRs for each codeword.
                
            :Descrambling: The descrambling of the demodulated LLRs using the ``rnti`` and ``nID`` properties of this
                :py:class:`PDSCH`. These properties are used to initialize a *Gold sequence* which is then used for
                the descrambling process according to **3GPP TS 38.211, Section 7.3.1.1**. The LLRs for each codeword
                are descrambled separately.
        
        This function returns a list of one or two NumPy arrays representing the LLRs for each codeword.

        Parameters
        ----------
        rxGrid : :py:class:`~neoradium.grid.Grid`
            The equalized received resource grid associated with this :py:class:`PDSCH`. Usually this is the
            :py:class:`~neoradium.grid.Grid` object obtained after equalization in the receiver pipeline (See the
            :py:meth:`~neoradium.grid.Grid.equalize` function).

        pdschIndexes : 3-tuple
            A tuple of 3 lists specifying locations of the set of resource elements in ``rxGrid`` that are assigned 
            to the user data. The function :py:meth:`getReIndexes` is typically used to obtain this. 
            If not specified, the internal parameter ``dataIndices`` is used. 
            
        llrScales : 3-D NumPy array
            The log-likelihood ratio (LLR) scaling factors which are used by demodulation process when extracting 
            log-likelihood ratios (LLRs) from the equalized resource grid. The shape of this array **must** be the
            same shape as ``rxGrid``.
            
        noiseVar : float or None
            The variance of the Additive White Gaussian Noise (AWGN) present in the received resource grid. If this 
            is not provided (``noiseVar=None``), This function uses the ``noiseVar`` property of the ``rxGrid`` object.
            
        useMax : bool
            If `True`, this implementation uses the ``Max`` function in the calculation of the LLR values. This is
            faster but uses an approximation and is slightly less accurate than the actual log-likelihood method 
            which uses logarithm and exponential functions. If `False`, the slower more accurate method is used.

        Returns
        -------
        list
            A list of one or two NumPy arrays each representing the LLRs for each codeword.
        """
        # First get the layer-mapped indices from the pdschIndexes
        if pdschIndexes is None:    pdschIndexes = self.dataIndices
        layerMappedIndexes = self.getLayerMapIndexes(pdschIndexes)
        
        if noiseVar is None: noiseVar = rxGrid.noiseVar
        noiseVar = max(noiseVar, 1e-10)
        
        llrs = []
        for cw in range(self.numCW):
            demappedSymbols = rxGrid[ layerMappedIndexes[cw] ]                  # The demapped symbols for this codeword
            cwLLRs = self.modems[cw].getLLRs(demappedSymbols, noiseVar, useMax) # Demodulate symbols to LLRs
            cwLLRs = self.scrambleLLRs(cw, cwLLRs)                              # Descramble the LLRs
            if llrScales is not None:
                demappedScales = llrScales[ layerMappedIndexes[cw] ]
                cwLLRs *= np.repeat(demappedScales, self.modems[cw].qm)
            llrs += [ cwLLRs ]                                                  # Add to the list
        return llrs
    
    # ******************************************************************************************************************
    def getHardBits(self, eqGrid, llrScales=None, noiseVar=None, useMax=True):
        r"""
        This method first calls the :py:meth:`getLLRs` function above and then uses hard-decisions on the
        returned LLRs to get the output user bits.
        
        This can be used when there is no channel coding in the communication pipeline. It returns a list of one or
        two NumPy arrays of bits for each codeword.

        Parameters
        ----------
        eqGrid : :py:class:`~neoradium.grid.Grid`
            The equalized received resource grid associated with this :py:class:`PDSCH`. Usually this is the
            :py:class:`~neoradium.grid.Grid` object obtained after equalization in the receiver pipeline (See the
            :py:meth:`equalize` function).

        llrScales : 3-D NumPy array
            The log-likelihood ratio (LLR) scaling factors which are used by demodulation process when extracting 
            log-likelihood ratios (LLRs) from the equalized resource grid. The shape of this array **must** be the
            same shape as ``eqGrid``.
            
        noiseVar : float or None
            The variance of the Additive White Gaussian Noise (AWGN) present in the received resource grid. If this
            is not provided (``noiseVar=None``), This function uses the ``noiseVar`` property of the ``eqGrid`` object.
            
        useMax : bool
            If `True`, this implementation uses the ``Max`` function in the calculation of the LLR values. This
            is faster but uses an approximation and is slightly less accurate than the actual log-likelihood method 
            which uses logarithm and exponential functions. If `False`, the slower more accurate method is used.

        Returns
        -------
        list
            A list of one or two NumPy arrays of bits for each codeword.


        Returns
        -------
        list
            A list of one or two NumPy arrays each representing the LLRs for each codeword.
            
            
        .. Note:: This function replaces the deprecated function :py:meth:`getHardBitsFromGrid`. The 
            following example shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                llrs = pdsch.getHardBitsFromGrid(eqGrid, pdschIndexes, llrScales)

                # New:
                llrs = pdsch.getHardBits(eqGrid, llrScales)                       
        """
        llrs = self.getLLRs(eqGrid, llrScales, noiseVar, useMax)
        return [ np.int8( llrs[cw]<0 ) for cw in range(self.numCW) ]

    # ******************************************************************************************************************
    @deprecated("getHardBits", docFile)
    def getHardBitsFromGrid(self, rxGrid, pdschIndexes, llrScales=None, noiseVar=None, useMax=True):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`getHardBits` method instead.
        """
        return self.getHardBits(rxGrid, llrScales, noiseVar, useMax)
    
    # ******************************************************************************************************************
    def getDataSymbols(self, grid=None):
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
        grid : :py:class:`~neoradium.grid.Grid`
            The resource grid associated with this :py:class:`PDSCH` containing the user data. 

        Returns
        -------
        NumPy array
            A 1D NumPy array of modulated complex symbols corresponding to the user data in ``grid``.
        """
        if grid is None: return self.grid[ self.dataIndices ]
        return grid[ self.dataIndices ]

    # ******************************************************************************************************************
    def getPrecodingMatrix(self, channelMatrix):
        r"""
        This function calculates the precoding matrix that can be applied to a resource grid. This function supports
        *Precoding RB groups (PRGs)* which means different precoding matrices could be applied to different groups
        of subcarriers in the resource grid. See **3GPP TS 38.214, Section 5.1.2.3** for more details. The ``prgSize``
        property of :py:class:`PDSCH` determines what type of precoding matrix is returned by this function:
        
            :Wideband: If ``prgSize`` is set to zero, a single ``Nt x Nl`` matrix is returned where ``Nt`` is the
                number of transmitter antennas and ``Nl`` is the number of layers in this :py:class:`PDSCH`. In this
                case the same precoding is applied to all subcarriers of the resource grid.
            
            :Using PRGs: If ``prgSize`` is set to 2 or 4, a list of tuples of the form (``groupRBs``, ``groupF``)
                is returned. For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all
                subcarriers of the resource blocks listed in ``groupRBs``.
        
        .. Note:: It is assumed that the ``channelMatrix`` is obtained based on the same 
            :py:class:`~neoradium.carrier.BandwidthPart` object as the one used by this :py:class:`PDSCH`.

        Parameters
        ----------
        channelMatrix : NumPy array
            An ``L x K x Nr x Nt`` complex NumPy array representing the channel matrix. It can be obtained directly 
            from a channel model using the :py:meth:`~neoradium.channelmodel.ChannelModel.getChannelMatrix` method.

        Returns
        -------
        NumPy array or list of tuples
            Depending on the ``prgSize`` property of this :py:class:`PDSCH`, the returned value can be:
            
            :NumPy Array: If ``prgSize`` is set to zero, a single *Wideband* ``Nt x Nl``, matrix is returned where
                ``Nt`` is the number of transmitter antennas and ``Nl`` is the number of layers in this 
                :py:class:`PDSCH`. In this case the same precoding is applied to all subcarriers of the resource grid.
            
            :list of tuples: If ``prgSize`` is set to 2 or 4, a list of tuples of the form (``groupRBs``, ``groupF``)
                is returned. For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all
                subcarriers of the resource blocks listed in ``groupRBs``.

            .. Note:: The returned precoding matrix (or each ``groupF`` in the subband case) is normalized by 
                :math:`1/\sqrt{N_l}` per **3GPP TS 38.211, Section 6.3.1.5** so that the total transmit power is
                preserved across layers. If you compose this output with your own additional precoder factors,
                do not re-apply this normalization.
        """
        # NOTE: For wideband, precoder is Nt x Nl. No GRB/PRB conversion is needed and the same precoder is applied to
        # all RBs. For subband, the groups are based on PRBs and the returned list includes group precoders only for
        # the PRBs of this PDSCH.
        # Channel Matrix Shape: L x K x Nr x Nt
        numPRBs = channelMatrix.shape[1]//12     # This is the number of PRBs in the BWP
        if numPRBs < len(self.prbSet):
            raise ValueError("The number of RBs in the 'channelMatrix' (%d) cannot be less than RBs in the PDSCH (%d)!"
                             %(numPRBs, len(self.prbSet)))

        def getGroupPrecoder(groupREs):             # Get a precoder matrix (Nt x Nl) for the specified group of REs
            groupChannel = channelMatrix[:,groupREs,:,:]        # Channel matrix for the group specified groupREs
            groupChannel = groupChannel.mean(axis=(0,1))        # Average over time and frequency => Shape (Nr x Nt)
            _, _, vH = np.linalg.svd(groupChannel)              # vH Shape: Nt x Nt
            groupPrecoder = (np.conj(vH).T)[:,:self.numLayers]  # Nt x Nl
            return groupPrecoder/np.sqrt(self.numLayers)        # Normalize the group precoder

        if self.prgSize==0:
            # Wideband case: A single Nt x Nl matrix is returned
            allREs = (12*self.prbSet[:,None]+np.arange(12)[None,:]).flatten()   # These REs are PRB-based
            return getGroupPrecoder(allREs)
        
        # Note that precoding happens in PRB grid (after interleaving)
        precoder = []
        groupsPrbs = self.prbSet.reshape(-1,self.prgSize)   # Note that the length of prbSet is a multiple of prgSize
        for groupPrbs in groupsPrbs:
            groupREs = (groupPrbs[:,None]*12 + np.arange(12)[None,:]).flatten()
            precoder += [ (groupPrbs, getGroupPrecoder(groupREs)) ]
        return precoder

    # ******************************************************************************************************************
    def precodeTo(self, txGrid, precoder, w=None):
        r"""
        Applies the specified precoding matrix to this grid object and returns a new *precoded* grid. Optionally, a
        steering vector can be applied before the precoder. This function supports *Precoding RB Groups (PRGs)*, 
        which means different precoding matrices can be applied to different groups of subcarriers in the resource
        grid. See **3GPP TS 38.214, Section 5.1.2.3** for more details.

        Parameters
        ----------
        txGrid : :py:class:`~neoradium.grid.Grid`
            The transmitted resource grid of shape ``Nt x L x K`` where ``Nt`` is the number of transmitter antennas, 
            ``L`` is the number of OFDM symbols, and ``K`` is the number of subcarriers. The precoded information is 
            placed in this resource grid.
            
        precoder : NumPy array or list of tuples
            This function supports two types of precoding:
        
            :Wideband: ``precoder`` is an ``Nt x Nl`` matrix where ``Nt`` is the number of transmitter antennas and 
                ``Nl`` is the number of layers which **must** match the number of layers in this PDSCH. In this case
                the same precoding is applied to all subcarriers of this PDSCH.
            
            :Using PRGs: ``precoder`` is a list of tuples of the form (``groupRBs``, ``groupF``).
                For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all subcarriers
                of the resource blocks listed in ``groupRBs``.
                
        w : NumPy array, optional
            An optional steering vector of shape ``Nt x 1`` applied before the precoder, where ``Nt`` is the number of
            transmitter antennas. When provided, each row of the precoding matrix is multiplied by the corresponding
            element of ``w``. This allows a directional steering vector to be combined with a PMI-based precoder. If
            omitted, only the specified precoder is applied.
            
                            
        .. Note:: This function replaces the deprecated function :py:meth:`~neoradium.grid.Grid.precode`. The 
            following example shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                precodedGrid = grid.precode(precoder)

                # New:
                txGrid = bwp.createGrid(channel.txAntenna.numEl)
                pdsch.precodeTo(txGrid, precoder)                       
        """
        if self.grid is None:
            raise ValueError("The 'initGrid' function must be called before calling 'precodeTo'.")

        if w is not None:
            nt = txGrid.shape[0]
            w = np.asarray(w, dtype=np.complex128)
            if w.ndim == 1:     w = w[:, None]
            if w.shape != (nt, 1):  raise ValueError(f"'w' must have shape ({nt},1); received {w.shape}.")

        # Current grid is a PDSCH grid. It may have non-contiguous RBs.
        numPrbs = len(self.prbSet)
        if type(precoder)==list:
            # The precoder matrix is a list of tuples of the form (groupRBs, groupF)
            nt, nl = precoder[0][1].shape
            # The precoder matrix "f" is an Nt x Nl matrix or a list of tuples of the form (groupPRBs, groupF)
            f = np.zeros((nt, nl, numPrbs*12), dtype=np.complex128)         # Shape: Nt, Nl, K   (GRB-based)
            grbPrecoded = [0]*len(self.prbSet)                              # Flag for each GRB in the self.grid
            for groupPrbs, groupF in precoder:
                if groupF.shape[0] != txGrid.shape[0]:
                    raise ValueError(f"Mismatch in number of TX antenna for PRB group {groupPrbs}!")
                if groupF.shape[1] != self.numLayers:
                    raise ValueError(f"Mismatch in number of Layers for PRB group {groupPrbs}!")
                
                groupREs = []
                for prb in groupPrbs:
                    if prb not in self.prbSet:  continue
                    grb = self.prb2Grb[prb]
                    grbPrecoded[grb] = 1
                    groupREs += [ grb*12 + re for re in range(12) ]

                f[:,:,groupREs] = groupF[:,:,None]
            
            # Make sure all VRBs in the self.grid are covered (e.g., set to 1)
            if 0 in grbPrecoded:
                missingPrb = self.grb2Prb[ grbPrecoded.index(0) ]
                raise ValueError(f"Missing PRB in the precoder! (PRB: {missingPrb})")
            #     f       . self.grid   ->      precodedGrid        <--- Tensors
            # (Nt, Nl, K) . (Nl, L, K)  ->      (Nt, L, K)          <--- Shapes
            #  0   1         0   1               0   1              <--- Axes
            axes = [(0,1), (0,1), (0,1)]
            if w is not None:   f = w[:,:,None] * f
        elif type(precoder) != np.ndarray:
            raise ValueError("'precoder' must be a 2D NumPy array or a list of tuples.")
        else:
            # precoder is a 2D matrix of shape Nt x Nl (PRB-based)
            if precoder.shape[0] != txGrid.shape[0]:
                raise ValueError(f"Mismatch in number of TX antenna!")
            if precoder.shape[1] != self.numLayers:
                raise ValueError(f"Mismatch in number of Layers!")
            #      f   . self.grid      ->      precodedGrid        <--- Tensors
            # (Nt, Nl) . (Nl, L, K)     ->      (Nt, L, K)          <--- Shapes
            #  0   1      0   1                  0   1              <--- Axes
            axes = [(0,1), (0,1), (0,1)]
            f = precoder
            if w is not None:   f = w * f

        precodedGrid = np.matmul(f, self.grid.grid, axes=axes)       # Nt x L x K   (GRB-based)

        # First get the PRB-based subcarrier indices (txKs).
        # The RE types in the precoded grid for each L/K is the maximum of the RE types in the original
        # grid (i.e., self.grid) at L/K.
        txKs = (self.grb2Prb[:,None]*12+np.arange(12)).flatten()
        txGrid.reTypeObjIds[:, self.symSet[:,None], txKs] = self.grid.reTypeObjIds.max(0)[self.symSet, :]

        # Copy only the RE values related to this PDSCH from precodedGrid to the txGrid:
        txGrid.grid[:, self.symSet[:,None], txKs] = precodedGrid[:, self.symSet, :]

    # ******************************************************************************************************************
    def estimateChannel(self, rxGrid, **kwargs):
        """
        Estimate the effective PDSCH channel from DMRS and return an associated residual-error variance.

        This function estimates the effective channel between the PDSCH layers and the receive antennas using 
        the DMRS REs. A local channel estimate is first computed for each CDM group by dividing the received DMRS
        values by the transmitted DMRS reference values and averaging over the REs in the CDM group.

        The per-CDM-group channel estimates are then interpolated:

        * across subcarriers, using the center subcarrier of each CDM group, and
        * across OFDM symbols, using the center OFDM symbol of each DMRS symbol position or DMRS symbol pair.

        The returned channel estimate is always in physical PRB order. Please refer to the notebook 
        :doc:`../Playground/Notebooks/PDSCH/ChannelNoiseEst` for examples of using this function. 

        Parameters
        ----------
        rxGrid : Grid or numpy.ndarray
            The received resource grid used for channel estimation.

        **kwargs : dict, optional
            Optional keyword arguments.

            extrapolate : bool or None
                If ``True``, linear extrapolation is used outside the DMRS-supported
                frequency and time ranges. If ``False``, the channel estimate is
                clipped to the boundary values at both ends during interpolation. If 
                not specified, linear extrapolation is used when the DMRS's ``additionalPos``
                is non-zero.

            estimateNoiseVar : bool
                If ``True``, the function also returns the estimated noise variance,
                in addition to the estimated channel and residual error variance. The
                noise variance is computed using a calibration table. Note that the
                accuracy of the noise variance estimate degrades at high
                SNR values.

        Returns
        -------
        chanEst : numpy.ndarray
            The estimated effective channel as a NumPy complex array of shape
            ``(L, K, nr, numLayers)``, where:

            * ``L`` is the number of OFDM symbols in the received grid,
            * ``K`` is the number of subcarriers in the **full bandwidth part** (i.e.,
              ``bwp.numRbs * 12``), not just the PDSCH allocation — the channel estimate
              is returned over the entire BWP so it can be fed directly to
              :py:meth:`equalize` / :py:meth:`getPrecodingMatrix`,
            * ``nr`` is the number of receive antennas, and
            * ``numLayers`` is the number of PDSCH layers.

        errVar : float
            The variance of the effective residual uncertainty associated with the
            estimated channel. This quantity is computed from the residual between
            the received CDM-group DMRS values and their reconstruction from the
            local channel estimates.

            This is **not** the AWGN noise variance applied to the received signal.
            It is intended for use as the ``noiseVar`` input of
            :py:meth:`equalize` when the channel is estimated using this function.

        estNoiseVar : float
            The estimated noise variance. This is returned only if ``estimateNoiseVar`` is
            set to ``True`` in ``kwargs``. The estimate may be less accurate at high SNR.


        .. Note:: 
            * The local CDM-group channel estimate is computed as the average of
              ``cdmY / cdmX`` over the REs of each CDM group.

            * For double-symbol DMRS configurations, each adjacent DMRS symbol pair is
              represented by a single time anchor located at the center of the pair.

            * If only one DMRS symbol position is present, the frequency-interpolated
              channel estimate is copied to all OFDM symbols.

            * The residual-based ``errVar`` reflects the uncertainty associated with the
              estimated channel and may include contributions from AWGN, channel
              estimation error, interpolation error, and model mismatch.
            
            * This function replaces the deprecated function :py:meth:`~neoradium.grid.Grid.estimateChannelLS`. The 
              following example shows how to migrate existing code to use this method:

                .. code-block:: python

                    # Old:
                    estChannelMatrix, noiseVar = rxGrid.estimateChannelLS(pdsch.dmrs)

                    # New:
                    estChannelMatrix, errVar = pdsch.estimateChannel(rxGrid)
        """
        nr, ll, kk = rxGrid.shape                                   # nr x L x K
        cdmLKs = self.dmrs.getCdmLKs()                              # numSym x numSymCdms x cdmSize x 2
        cdmXs = self.dmrs.getCdmValues(cdmLKs)[:,:,None]            # numSym x numSymCdms x 1 x nl x cdmSize
        cdmYs = self.dmrs.getCdmValues(cdmLKs, rxGrid)[:,:,:,None]  # numSym x numSymCdms x nr x 1 x cdmSize

        # Set cdmHs to zero where cdmXs is zero. (avoids division by zero)
        zeroOut = np.zeros(cdmYs.shape[:3] + cdmXs.shape[3:], np.complex128)
        cdmHs = np.divide(cdmYs, cdmXs, out=zeroOut, where=(cdmXs!=0)).mean(-1) # numSym x numSymCdms x nr x nl

        # CDM group locations are different per port (layer). We need to do this interpolation per port
        numCdm=len(np.unique(list(self.dmrs.cdmGroups.values())))
        extrapolate = kwargs.get('extrapolate', (self.dmrs.additionalPos>0))
        symHs = []
        for p in self.portSet:
            portCdmGroup = self.dmrs.cdmGroups[p]               # CDM group number for this port
            portCdmLKs = cdmLKs[:,portCdmGroup::numCdm,:,:]     # CDM L,K values for this port's CDM group
            portCdmHs = cdmHs[:,portCdmGroup::numCdm,:,p:p+1]   # CDM H values for this port's CDM group
        
            gKs = portCdmLKs[0,:,:,1].mean(1)                   # Center subcarrier for each CDM group in GRB order
            pKs = self.grb2Prb[np.int32(gKs//12)]*12 + gKs%12   # Center subcarrier for each CDM group in PRB order
        
            fillValue = 'extrapolate' if extrapolate else (portCdmHs[:,0], portCdmHs[:,-1])
            f = interp1d(pKs, portCdmHs, axis=1, kind='linear', bounds_error=False, fill_value=fillValue)
            symHs += [ f(np.arange(12*self.bwp.numRbs)) ]       # numSym x K x nr x 1
        symHs = np.concatenate(symHs,-1)                        # numSym x K x nr x nl

        # Interpolate OFDM symbols
        ls = cdmLKs[:,0,:,0].mean(1)                            # Center OFDM symbols for each CDM group
        if len(ls)==1:          # If there is only one symbol, just copy it to all OFDM symbols
            chanEst = np.tile(symHs,[ll,1,1,1])                 # L x K x nr x nl
        else:
            fillValue = 'extrapolate' if extrapolate else (symHs[0], symHs[-1])
            f = interp1d(ls, symHs, axis=0, kind='linear', bounds_error=False, fill_value=fillValue)
            chanEst = f(np.arange(ll))                          # L x K x nr x nl

        cdmXs = cdmXs[:,:,0]    # numSym x numSymCdms x nl x cdmSize
        cdmYs = cdmYs[:,:,:,0]  # numSym x numSymCdms x nr x cdmSize
        
        # Get channel at CDM REs from the interpolated channels along subcarriers
        cdmHsInt = symHs[:, cdmLKs[0,:,:,1]].mean(2)    # numSym x numSymCdms x nr x nl
        errVar = (cdmYs - (cdmHsInt @ cdmXs)).var()     # Residual Error Variance
        
        estimateNoiseVar = kwargs.get('estimateNoiseVar', False)
        if estimateNoiseVar: return chanEst, errVar, self.dmrs.resVarToNoiseVar(errVar, nr)
        return chanEst, errVar

    # ******************************************************************************************************************
    def equalize(self, rxGrid, h, noiseVar=None):
        r"""
        Equalizes the received resource grid ``rxGrid`` using the effective channel ``h``. The effective channel is 
        assumed to include the effect of the precoding matrix, therefore, its shape is ``L x K x Nr x Nl`` where ``L`` 
        is the number of OFDM symbols, ``K`` is the number of subcarriers in the whole bandwidth part, ``Nr`` is the  
        number of receiver antennas, and ``Nl`` is the number of layers. The output of the equalization process is a new
        :py:class:`~neoradium.grid.Grid` object of shape ``Nl x L x Kp``, where ``Kp`` is the number of subcarriers 
        used by this PDSCH (``Kp <= K``).
        
        The process of de-interleaving resource blocks and mapping from PRBs to VRBs to GRBs (the RBs in the returned 
        resource grid) is performed in this function. Both ``rxGrid`` and ``h`` are in PRBs. The returned equalized 
        resource grid is in GRBs.
        
        This function also outputs log-likelihood ratio (LLR) scaling factors which are used by the demodulation 
        process when extracting log-likelihood ratios (LLRs) from the equalized resource grid.
        
        This method uses the Minimum Mean Squared Error (MMSE) algorithm for the equalization.

        Parameters
        ----------
        rxGrid : :py:class:`~neoradium.grid.Grid`
            The received resource grid. It is an ``Nr x L x K`` resource grid where ``Nr`` is the number of receiver
            antennas, ``L`` is the number of OFDM symbols, and ``K`` is the number of subcarriers in the whole 
            bandwidth part.
            
        h : 4-D complex NumPy array
            This is an ``L x K x Nr x Nl`` NumPy array representing the estimated channel matrix, where ``L`` is
            the number of OFDM symbols, ``K`` is the number of subcarriers in the whole bandwidth part, ``Nr`` is the 
            number of receiver antennas, and ``Nl`` is the number of layers.
            
        noiseVar : float or None
            The variance of noise applied to the received resource grid. If this is not provided, this method
            tries to use the noise variance of the resource grid obtained by the OFDM demodulation process for
            the time-domain case or the variance of the noise applied to the received resource grid by the
            :py:meth:`addNoise` method for the frequency domain case (See the ``noiseVar`` property of 
            :py:class:`~neoradium.grid.Grid` class).
            
            .. Note:: When the function :py:meth:`estimateChannel` is used to estimate the channel, the ``errVar``
                returned by that function must be passed to this function through the ``noiseVar`` argument. Although
                ``errVar`` is not exactly the AWGN noise variance applied to the received signal, it should be used for
                equalization whenever an estimated channel is used. The following example shows equalization with 
                perfect channel knowledge versus equalization with an estimated channel:

                .. code-block:: python

                    # Using the perfect channel from a channel model 'channel'
                    channelMatrix = channel.getChannelMatrix()
                    precoder = pdsch.getPrecodingMatrix(channelMatrix)
                    effChannelMatrix = CdlChannel.getEffChannel(channelMatrix, precoder)
                    eqGrid, llrScales = pdsch.equalize(rxGrid, effChannelMatrix)    # 'noiseVar' is stored in 'rxGrid'

                    # Using an estimated channel
                    estChannelMatrix, errVar = pdsch.estimateChannel(rxGrid)
                    eqGrid, llrScales = pdsch.equalize(rxGrid, estChannelMatrix, errVar)
                    

        Returns
        -------
        eqGrid : :py:class:`~neoradium.grid.Grid`
            The equalized grid object of shape ``Nl x L x Kp`` where ``Nl`` is the number of layers, ``L`` is the
            number of OFDM symbols, and ``Kp`` is the number of subcarriers used by this PDSCH.
            
        llrScales : 3-D NumPy array
            The log-likelihood ratio (LLR) scaling factors which are used by the demodulation process when extracting
            log-likelihood ratios (LLRs) from the equalized resource grid. The shape of this array is ``Nl x L x Kp``
            which is similar to ``eqGrid`` above.
            
            
        .. Note:: This function replaces the deprecated function :py:meth:`~neoradium.grid.Grid.equalize`. The 
            following example shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                eqGrid, llrScales = rxGrid.equalize(effectiveChannelMatrix)

                # New:
                eqGrid, llrScales = pdsch.equalize(rxGrid, effectiveChannelMatrix)                       
        """
        # See the page "Equalization and LLR Scaling" in the Implementation Notes.
        nr,l,k = rxGrid.shape
        lh,kh,nrh,nl = h.shape
        if nr != nrh:       raise ValueError(f"Mismatch in number of RX antennas in 'rxGrid'({nr}) and 'h'({nrh})")
        if l != lh:         raise ValueError(f"Mismatch in number of symbols in 'rxGrid'({l}) and 'h'({lh})")
        if k != kh:         raise ValueError(f"Mismatch in number of REs in 'rxGrid'({k}) and 'h'({kh})")
        if self.numLayers != nl:
            raise ValueError(f"Mismatch in number of layers in 'PDSCH'({self.numLayers}) and 'h'({nl})")
        
        if noiseVar is None:    noiseVar = rxGrid.noiseVar
        noiseVar = max(noiseVar, 1e-12)

        pres = (self.grb2Prb[:,None]*12 + np.arange(12)).flatten() # PRB to VRB to GRB in one step!
        kp = len(pres)                                              # The PDSCH K (number of REs in the PDSCH grid)
        # h and rxGrid both cover the whole BWP. We now keep only the PRBs of this PDSCH
        rxGridPdsch = rxGrid.grid[:,:,pres]                         # nr x L x Kp
        h = h[:,pres,:,:]                                           # L x Kp x nr x nl
        
        if nr>nl:   # SU-MIMO cases
            a = herm(h) @ h + np.eye(nl) * noiseVar                 # L x Kp x nl x nl
            la = np.linalg.cholesky(a)  # a is Hermitian Positive Definite => a=la*herm(la) <= Cholesky decomposition
            g = scipy.linalg.cho_solve((la, True), herm(h))         # Use Cholesky to get the equalizer
            
            aInv = scipy.linalg.cho_solve((la, True), np.eye(nl))   # L x Kp x nl x nl
        else:   # nr <= nl -> MU-MIMO cases
            b = h @ herm(h) + np.eye(nr) * noiseVar                 # L x Kp x nr x nr
            lb = np.linalg.cholesky(b)  # b is Hermitian Positive Definite => a=lb*herm(lb) <= Cholesky decomposition
            bInv = scipy.linalg.cho_solve( (lb, True), np.eye(nr))  # Use Cholesky to get inverse of b
            g = herm(h) @ bInv                                      # L x Kp x nl x nr
            aInv = (np.eye(nl)-herm(h) @ bInv @ h)/noiseVar         # L x Kp x nl x nl

        # Equalized grid
        eqGrid = Grid(self.bwp, numPlanes=nl, numRbs=len(self.prbSet))
        eqGrid.grid = np.matmul(g, rxGridPdsch[:,None,:,:], axes = [(2,3), (0,1), (0,1)])[:,0,:,:]   # Nl x L x Kp
        # Copy RE types and object IDs from the first rxGrid port for all layers
        eqGrid.reTypeObjIds = np.stack(nl*[rxGrid.reTypeObjIds[0,:,pres]])
        eqGrid.noiseVar = noiseVar

        llrScales = np.transpose( 1/(np.diagonal(aInv,axis1=2,axis2=3).real + 1e-12), (2,0,1))

        return eqGrid, llrScales   # Both have the same Shape: nl x ll x kk

    # ******************************************************************************************************************
    def getTxBlockSize(self, coderates, xOverhead=None, scaleFactor=1.0):
        r"""
        This function calculates the transport block size based on the desired code rate (``coderates``), the number
        of additional overhead resource elements (``xOverhead``), and the scaling factor (``scaleFactor``). It returns
        a list of one or two integer values specifying the size of transport blocks for each codeword. This
        implementation is based on **3GPP TS 38.214, Section 5.1.3.2**.
        
        Parameters
        ----------
        coderates : float, list, NumPy array, or tuple
            If ``coderates`` is a float value, it specifies the same code rate for all codewords. If it is a list, 
            NumPy array, or tuple, it should contain one or two code rate values for each codeword. This is the
            value :math:`R` in **3GPP TS 38.214, Section 5.1.3.2**.
            
        xOverhead : int or None, optional
            The number of additional overhead resource elements per PRB to consider when calculating the transport
            block size. This corresponds to :math:`N^{PRB}_{oh}` in **3GPP TS 38.214, Section 5.1.3.2** and,
            when explicitly specified, must be one of ``0``, ``6``, ``12``, or ``18`` according to **3GPP TS 38.331**.

            If set to ``None`` (the default), **NeoRadium** automatically selects a conservative value based on
            the configured DM-RS overhead, the maximum CSI-RS overhead that may occur in a slot, and the requested
            code rate. This helps prevent the effective code rate from becoming excessively large when reference-signal
            overhead reduces the number of REs available for PDSCH. The automatically selected value remains
            fixed for the TBS calculation and is chosen from ``0``, ``6``, ``12``, and ``18``.

            The automatic selection is a **NeoRadium** implementation convenience based on conservative empirical
            thresholds. It is not a procedure specified or recommended by 3GPP. To disable the automatic behavior,
            explicitly provide the desired ``xOverhead`` value.
                        
        scaleFactor : float
            The scaling factor, which **must** be one of: 0.25, 0.5, or 1.0. This is the value :math:`S` in
            **3GPP TS 38.214, Table 5.1.3.2-2** and reduces the effective TBS by the same proportion. The
            value is signaled by higher layers; ``1.0`` is the normal case (full capacity), while ``0.5``
            and ``0.25`` are used for more robust/fallback transmissions (e.g., certain DCI Format 1_0 cases).

        Returns
        -------
        list
            A list of one or two integers depending on the number of codewords (``numCW``), indicating the transport
            block size for each codeword.
        """
        if type(coderates) in [float, np.float32, np.float64]:  coderates = [coderates]
        elif type(coderates) in [list, np.ndarray, tuple]:      coderates = list(coderates)
        else:
            raise ValueError(f"'coderates' must be a float value, list, tuple, or NumPy array of 1 or 2 float values. "+
                             f"('{type(coderates).__name__}' is not supported)")
        if len(coderates)<self.numCW:           coderates = self.numCW * coderates  # Repeat the coderates
        coderates = coderates[:self.numCW]

        if xOverhead is None:
            # Automatic overhead selection is a NeoRadium convenience intended to prevent
            # excessively large effective code rates when substantial DM-RS and CSI-RS
            # overhead reduces the number of REs available for PDSCH. The CSI-RS term uses
            # the maximum overhead that can occur in any slot over the CSI-RS configuration
            # period so that the TBS can remain unchanged from slot to slot.
            #
            # The thresholds below were determined empirically using conservative worst-case
            # combinations of modulation order, transmission rank, and code rate. The selected
            # value is rounded up to one of the standardized values: 0, 6, 12, or 18.
            # This automatic selection is an implementation safeguard and is not specified or
            # recommended by 3GPP. Users who require explicit standards-controlled behavior can
            # provide xOverhead directly.
            xOverhead = 0
            dmrsREs = 0 if self.dmrs is None else self.dmrs.numDmrsOhREs
            csiRsREs = 0 if self.csiRsConfig is None else self.csiRsConfig.getMaxOverheadREs()
            for coderate in coderates:
                if coderate>0.9:    overheadTolerance = 17
                elif coderate>0.8:  overheadTolerance = 27
                elif coderate>0.7:  overheadTolerance = 41
                else:               continue
                remainingREs = overheadTolerance - dmrsREs - csiRsREs
                if remainingREs >= 0:       oh = 0
                elif remainingREs >= -6:    oh = 6
                elif remainingREs >= -12:   oh = 12
                else:                       oh = 18
                if oh>xOverhead:            xOverhead = oh

        if scaleFactor not in [1/4, 1/2, 1]:    raise ValueError("'scaleFactor' must be one of: 0.25, 0.5, or 1")

        # 3GPP TS 38.214, Section 5.1.3.2
        # Step 1:
        numPRBs = len(self.prbSet)
        npRE = 12*len(self.symSet)  # Number of REs allocated for PDSCH within a PRB (N'_{RE})
        if self.dmrs is not None:   npRE -= self.dmrs.numDmrsOhREs

        if xOverhead not in [0,6,12,18]:        raise ValueError("'xOverhead' must be one of 0, 6, 12, or 18")
        if npRE<=xOverhead:                     raise ValueError("'xOverhead' must be less than %d."%(npRE))
        npRE-=xOverhead
        numREs = min(156, npRE)*numPRBs
        
        cwLayers = [self.numLayers] if self.numCW==1 else [self.numLayers//2, self.numLayers-self.numLayers//2]
        
        # Step 2:
        txBlockSize = []
        for c in range(self.numCW):
            nInfo = scaleFactor * numREs * coderates[c] * self.modems[c].qm * cwLayers[c]   # A floating point value
            if nInfo <= 3824:
                # Step 3:
                n = max(3, int(np.log2(nInfo))-6)
                npInfo = max(24, (1<<n)*(nInfo//(1<<n)))
                # 3GPP TS 38.214, V18.1.0 (2023-12), Table 5.1.3.2-1
                txBlockSizes = np.int32([24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160,
                                         168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384,
                                         408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848,
                                         888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352,
                                         1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216,
                                         2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368,
                                         3496, 3624, 3752, 3824])
                txBlockSize += [ int(txBlockSizes[txBlockSizes>=npInfo][0]) ]
            else:
                # Step 4:
                n = int(np.log2(nInfo-24))-5
                npInfo = max(3840, (1<<n)*np.round((nInfo-24)/(1<<n)))

                if coderates[c] <= 0.25:    eightC = 8*np.ceil((npInfo + 24)/3816)
                elif npInfo > 8424:         eightC = 8*np.ceil((npInfo + 24)/8424)
                else:                       eightC = 8
                
                txBlockSize += [ int(eightC*np.ceil((npInfo + 24)/eightC)) - 24 ]

        return txBlockSize

    # ******************************************************************************************************************
    def getLdpcCodec(self, coderates, numIter=5, nRef=0):
        r"""
        Creates and returns an :py:class:`~neoradium.ldpccodec.LdpcCodec` object configured based on the current
        :py:class:`PDSCH` settings.

        This method derives the required LDPC coding parameters (modulation schemes and transport block sizes)
        from the :py:class:`PDSCH` instance and combines them with the specified ``coderates`` to initialize an
        :py:class:`LdpcCodec` object. The resulting codec can be used for encoding and decoding the transport
        block(s) associated with this :py:class:`PDSCH`.

        Parameters
        ----------
        coderates : float, list, tuple, or NumPy array
            One or two code rate values corresponding to the codeword(s) of this :py:class:`PDSCH`. If a single
            value is provided and two codewords are present (i.e., ``numLayers > 4``), the same code rate is used
            for both codewords.

        numIter : int
            The number of iterations used in the LDPC decoder (Layered Belief Propagation). Higher values may
            improve decoding performance at the cost of increased complexity. The default is 5.

        nRef : int
            The reference buffer size used for Low-Buffer Rate Matching (LBRM). This corresponds to
            :math:`N_{ref}` in **3GPP TS 38.212, Section 5.4.2.1**. The default is 0 (LBRM disabled).

        Returns
        -------
        :py:class:`~neoradium.ldpccodec.LdpcCodec`
            An LDPC codec object configured for the current :py:class:`PDSCH`, supporting one or two codewords
            depending on ``numLayers``.

        Notes
        -----
        - The modulation schemes are automatically extracted from the internal ``modems`` of this
          :py:class:`PDSCH`.
        - The transport block size(s) are computed using :py:meth:`getTxBlockSize` based on the provided
          ``coderates``.
        - The returned :py:class:`LdpcCodec` object is fully configured and ready for encoding and decoding
          operations.
        """
        return LdpcCodec([m.modulation for m in self.modems], coderates, self.getTxBlockSize(coderates),
                         self.numLayers, numIter, nRef)


    # ******************************************************************************************************************
    def getHarq(self, coderates, numIter=5, nRef=0,
                harqType="CC", numProc=8, rvSequence=[0,2,3,1], maxTries=4, eventCallback=None):
        r"""
        Creates and returns a :py:class:`~neoradium.harq.HarqEntity` object configured based on the current
        :py:class:`PDSCH` settings.

        This method first creates an :py:class:`~neoradium.ldpccodec.LdpcCodec` object using the 
        :py:meth:`getLdpcCodec` method and then uses it to create a :py:class:`~neoradium.harq.HarqEntity` object. 
        
        * For more details about ``coderates``, ``numIter``, and ``nRef`` refer to :py:meth:`getLdpcCodec`.
        * For more details about ``harqType``, ``numProc``, ``rvSequence``, ``maxTries``, ``eventCallback`` refer to
          :py:class:`~neoradium.harq.HarqEntity`.

        Returns
        -------
        :py:class:`~neoradium.harq.HarqEntity`
            A HARQ entity object created based on the given parameters.
        """
        ldpc = self.getLdpcCodec(coderates, numIter, nRef)
        return HarqEntity(ldpc, harqType, numProc, rvSequence, maxTries, eventCallback)

