# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``dmrs.py`` implements the :py:class:`DMRS` and :py:class:`PTRS` classes which encapsulate the Demodulation 
Reference Signals (DMRS) and the Phase-Tracking Reference Signals (PTRS) respectively.

Demodulation reference signals are intended for channel estimation on the receiver side and enable coherent 
demodulation. They are used with all types of communication channels for both data and control, and downlink and
uplink. This means a :py:class:`DMRS` object can be associated with :py:class:`~neoradium.pdsch.PDSCH`,
:py:class:`~neoradium.pdcch.PDCCH`, :py:class:`~neoradium.pusch.PUSCH`, or :py:class:`~neoradium.pucch.PUCCH` classes.

:py:class:`PTRS` is used for tracking the phase of the local oscillators at the receiver and transmitter. If 
transmitted, a PTRS is always associated with one DMRS port.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 12/26/2023    Shahab Hamidi-Rad       Completed the documentation.
# 06/27/2025    Shahab Hamidi-Rad       Updated the documentation.
# **********************************************************************************************************************
import numpy as np

from . import Modem
from .utils import goldSequence, toLinear

# This implementation is based on:
#   - TS 38.211 V17.0.0 (2021-12)
#   - TS 38.212 V17.0.0 (2021-12)
#   - TS 38.214 V17.0.0 (2021-12)

# The following links can help understand the standard more easily:
#   https://www.sharetechnote.com/html/5G/5G_PDSCH_DMRS.html
#   https://www.sharetechnote.com/html/5G/5G_PTRS_DL.html

# **********************************************************************************************************************
dmrsPositions = { 1:     # DMRS.symbols: Single
                  {     # Table 7.4.1.1.2-3: PDSCH DM-RS positions for single-symbol DM-RS.
                    'A':    # 0 and 1 represent l0 and l1 in the table
                         # Pos0,  Pos1,   Pos2,     Pos3          # ld
                        [[ [],    [],     [],       []         ], # 0
                         [ [],    [],     [],       []         ], # 1
                         [ [],    [],     [],       []         ], # 2
                         [ [0],   [0],    [0],      [0]        ], # 3
                         [ [0],   [0],    [0],      [0]        ], # 4
                         [ [0],   [0],    [0],      [0]        ], # 5
                         [ [0],   [0],    [0],      [0]        ], # 6
                         [ [0],   [0],    [0],      [0]        ], # 7
                         [ [0],   [0,7],  [0,7],    [0,7]      ], # 8
                         [ [0],   [0,7],  [0,7],    [0,7]      ], # 9
                         [ [0],   [0,9],  [0,6,9],  [0,6,9]    ], # 10
                         [ [0],   [0,9],  [0,6,9],  [0,6,9]    ], # 11
                         [ [0],   [0,9],  [0,6,9],  [0,5,8,11] ], # 12
                         [ [0],   [0,11], [0,7,11], [0,5,8,11] ], # 13 - Assuming l1=11 (Ignoring case l1=12). See
                                                                  #      section 7.4.1.1.2 for more details
                         [ [0],   [0,11], [0,7,11], [0,5,8,11] ]],# 14 - Assuming l1=11 (Ignoring case l1=12). See
                                                                  #      section 7.4.1.1.2 for more details

                    'B':    # 0 represents l0 in the table
                         # Pos0,  Pos1,  Pos2,     Pos3          # ld
                        [[ [],    [],    [],       []         ], # 0
                         [ [],    [],    [],       []         ], # 1
                         [ [0],   [0],   [0],      [0]        ], # 2
                         [ [0],   [0],   [0],      [0]        ], # 3
                         [ [0],   [0],   [0],      [0]        ], # 4
                         [ [0],   [0,4], [0,4],    [0,4]      ], # 5
                         [ [0],   [0,4], [0,4],    [0,4]      ], # 6
                         [ [0],   [0,4], [0,4],    [0,4]      ], # 7
                         [ [0],   [0,6], [0,3,6],  [0,3,6]    ], # 8
                         [ [0],   [0,7], [0,4,7],  [0,4,7]    ], # 9
                         [ [0],   [0,7], [0,4,7],  [0,4,7]    ], # 10
                         [ [0],   [0,8], [0,4,8],  [0,3,6,9]  ], # 11
                         [ [0],   [0,9], [0,5,9],  [0,3,6,9]  ], # 12
                         [ [0],   [0,9], [0,5,9],  [0,3,6,9]  ], # 13
                         [ [],    [],    [],       []         ]] # 14
                  },
                  2:     # DMRS.symbols: Double
                  {     # Table 7.4.1.1.2-4: PDSCH DM-RS positions for double-symbol DM-RS.
                    'A':    # 0 represents l0 in the table
                         # Pos0,  Pos1,   Pos2,     Pos3          # ld
                        [[ [],    [],     [],       []         ], # 0
                         [ [],    [],     [],       []         ], # 1
                         [ [],    [],     [],       []         ], # 2
                         [ [],    [],     [],       []         ], # 3
                         [ [0],   [0],    [],       []         ], # 4
                         [ [0],   [0],    [],       []         ], # 5
                         [ [0],   [0],    [],       []         ], # 6
                         [ [0],   [0],    [],       []         ], # 7
                         [ [0],   [0],    [],       []         ], # 8
                         [ [0],   [0],    [],       []         ], # 9
                         [ [0],   [0,8],  [],       []         ], # 10
                         [ [0],   [0,8],  [],       []         ], # 11
                         [ [0],   [0,8],  [],       []         ], # 12
                         [ [0],   [0,10], [],       []         ], # 13
                         [ [0],   [0,10], [],       []         ]],# 14
                    'B':    # 0 represents l0 in the table
                         # Pos0,  Pos1,   Pos2,     Pos3          # ld
                        [[ [],    [],     [],       []         ], # 0
                         [ [],    [],     [],       []         ], # 1
                         [ [],    [],     [],       []         ], # 2
                         [ [],    [],     [],       []         ], # 3
                         [ [],    [],     [],       []         ], # 4
                         [ [0],   [0],    [],       []         ], # 5
                         [ [0],   [0],    [],       []         ], # 6
                         [ [0],   [0],    [],       []         ], # 7
                         [ [0],   [0,5],  [],       []         ], # 8
                         [ [0],   [0,5],  [],       []         ], # 9
                         [ [0],   [0,7],  [],       []         ], # 10
                         [ [0],   [0,7],  [],       []         ], # 11
                         [ [0],   [0,8],  [],       []         ], # 12
                         [ [0],   [0,8],  [],       []         ], # 13
                         [ [],    [],     [],       []         ]] # 14
                  },
                }
       
# **********************************************************************************************************************
ptrsRefREs = [[],   # See 3GPP TS 38.211. Table 7.4.1.2.2-1
              [     # DMRS configType = 1
               #  Offset 00  01  10  11
                        [0,  2,  6,  8],        # Port 1000
                        [2,  4,  8,  10],       # Port 1001
                        [1,  3,  7,  9],        # Port 1002
                        [3,  5,  9,  11]        # Port 1003
              ],
              [     # DMRS configType = 2
               #  Offset 00  01  10  11
                        [0,  1,  6,  7],        # Port 1000
                        [1,  6,  7,  0],        # Port 1001
                        [2,  3,  8,  9],        # Port 1002
                        [3,  8,  9,  2],        # Port 1003
                        [4,  5,  10, 11],       # Port 1004
                        [5,  10, 11, 4]         # Port 1005
              ]
             ]
    
# **********************************************************************************************************************
class DMRS:
    r"""
    This class encapsulates the configuration and functionality of Demodulation Reference Signals. A :py:class:`DMRS`
    object can be associated with a :py:class:`~neoradium.pdsch.PDSCH`, a :py:class:`~neoradium.pdcch.PDCCH`, a 
    :py:class:`~neoradium.pusch.PUSCH`, or a :py:class:`~neoradium.pucch.PUCCH`. (Currently only 
    :py:class:`~neoradium.pdsch.PDSCH` is implemented in **NeoRadium**. Support for other channels is coming soon.)
    
    For every PDSCH, at least one DMRS OFDM symbol is mandatory. It is also possible to have one, two, or three 
    additional OFDM symbols assigned to DMRS.
    """
    # ******************************************************************************************************************
    def __init__(self, pxxch=None, **kwargs):
        r"""
        Parameters
        ----------
        pxxch: :py:class:`~neoradium.pdcch.PDSCH`
            The :py:class:`~neoradium.pdcch.PDSCH` object associated with this :py:class:`DMRS` object. Technically
            this can be any of :py:class:`~neoradium.pdsch.PDSCH`, :py:class:`~neoradium.pdcch.PDCCH`, 
            :py:class:`~neoradium.pusch.PUSCH`, or :py:class:`~neoradium.pucch.PUCCH` classes, but currently only
            :py:class:`~neoradium.pdsch.PDSCH` is implemented in **NeoRadium**.
            
        kwargs : dict
            A set of optional arguments.

                :configType: The DMRS configuration type. It can be either 1 (default) or 2. In Configuration type 1,
                    the minimum resource element group in frequency domain is one RE. In Configuration type 2, the 
                    minimum resource element group in frequency domain is two consecutive REs.
                    
                :symbols: The number of OFDM symbols used with each group of DMRS REs. It can be 1 (*Single*) or 
                    2 (*Double*). The default is *Single*.

                :typeA1stPos: This is the OFDM symbol index for the first DMRS symbol when Mapping type A is being
                    used. It can be either 2 (default) or 3.

                :additionalPos: Position(s) for additional DMRS symbols. For ``symbols==1``,
                    it can be 0, 1, 2, or 3 and for ``symbols==2`` it can be 0 or 1. This allows up to 4 OFDM symbols 
                    to be used for DMRS.
                    
                :portSet: A list of port numbers used for DMRS allocation. See **3GPP TS 38.211, Table 7.4.1.1.2-5**
                    for more information. If not specified, it is set based on the number of layers in ``pxxch``. If
                    specified, the number of items in the list should match the number of layers in ``pxxch``.
                    
                :otherCdmGroups: A list of CDM groups used by other PXXCH objects (For example, in PDSCH, some CDM 
                    groups may be used for other UEs). When allocating resources, this class makes sure that no data
                    is allocated in the CDM groups used for other UEs. See 
                    :doc:`../Playground/Notebooks/DMRS/CDMsWithNoData` for examples of how to use this parameter.
                    
                :scID: The number specifying which one of the ``nIDs`` (see below) should be used for scrambling. It
                    can be 0 (default) or 1.

                :nIDs: A list of one or 2 integer values (``nIDs[scID] ∈ {0,1,...,65535}, scID ∈ {0,1}``) The nIDs[0] 
                    and nIDs[1] are explained in **3GPP TS 38.211, Section 7.4.1.1.1** (*scramblingID0*, 
                    *scramblingID1*).
                    
                :sameSeq: A boolean value set to ``True`` by default. If ``True``, the same binary sequence is created
                    for all CDM Groups. Otherwise the sequences for different CDM Groups are initialized differently. 
                    This corresponds to the parameter setting ``dmrs-Downlink`` in **3GPP TS 38.211, 
                    Section 7.4.1.1.1**.

                :epreRatioDb: The ratio of PXXCH energy per resource element (EPRE) to DMRS EPRE in dB. If not 
                    specified, **3GPP TS 38.214, Table 4.1-1** is used to set this parameter.


        **Other Properties:**
        
            :cdmGroups: A list of CDM groups used by this DMRS. This property is set based on the ``portSet`` and 
                ``configType`` parameters.

            :symSet: A numpy array containing the indices of the OFDM symbols used by this DMRS.
            
            :ptrs: The :py:class:`PTRS` object associated with this DMRS object or ``None`` if PTRS is not configured.
                
            :ptrsEnabled: A boolean read-only property. If ``True`` it means PTRS is enabled, and therefore the ``ptrs``
                property above should not be ``None``. Otherwise PTRS is disabled and the ``ptrs`` property above
                should be set to ``None``.
                
        The notebook :doc:`../Playground/Notebooks/DMRS/DMRS` shows some examples of configuring DMRS.
        """
        self.pxxch = pxxch
        
        # A CDM Group:
        # The DMRS signals that share the same subcarriers but are separated in the code domain by using different
        # orthogonal sequences.
        
        self.configType = kwargs.get('configType', 1)               # DMRS Configuration Type (1 or 2)
        if self.configType not in [1,2]:    raise ValueError("Invalid DMRS 'configType' value! (It must be 1 or 2)")

        self.symbols = kwargs.get('symbols', 1)                     # DMRS symbols 1->Single, 2->Double
        if self.symbols not in [1,2]:       raise ValueError("Invalid DMRS 'symbols' value! (It must be 1 or 2)")

        self.typeA1stPos = kwargs.get('typeA1stPos', 2)             # dmrs-TypeA-Position
        if self.typeA1stPos not in [2,3]:   raise ValueError("Invalid 'typeA1stPos' value! (It must be 2 or 3)")
        if (pxxch.symSet[0] not in [0,1,2]) and (pxxch.symSet[0]!=3 or self.typeA1stPos!=3):
            raise ValueError("Invalid symbol allocation: start = %d"%(pxxch.symSet[0]))

        self.additionalPos = kwargs.get('additionalPos', 0)         # dmrs-AdditionalPosition
        if self.symbols == 1:
            if self.additionalPos not in range(4):
                raise ValueError("Invalid 'additionalPos' value! (It must be in [0..3])")
        elif self.additionalPos not in [0,1]:
            raise ValueError("Invalid 'additionalPos' value! (It must be 0 or 1 for 2-symbol DMRS)")

        ports = kwargs.get('portSet', list(range(pxxch.numLayers)))   # If not specified, use numLayers
        if len(ports) != pxxch.numLayers:
            raise ValueError("The number of ports in 'portSet' must match the number of layers (%d)"%(pxxch.numLayers))
        # See TS 38.211 V17.0.0 (2021-12), Table 7.4.1.1.2-5 for the valid range of port numbers
        if self.configType == 1:  validRange = list(range(4)) if self.symbols == 1 else list(range(8))
        else:                     validRange = list(range(6)) if self.symbols == 1 else list(range(12))
        for p in ports:
            if p not in validRange:
                raise ValueError("Invalid DMRS 'port number' %d! (Valid Range: %d..%d)"%(p, validRange[0],
                                                                                         validRange[-1]))
        self.pxxch.portSet = ports

        if self.pxxch.numLayers > len(validRange):
            raise ValueError("Invalid DMRS 'symbols' specified (%d) for a %d-layer PDSCH!"%(self.symbols,
                                                                                            self.pxxch.numLayers))

        # See TS 38.211 V17.0.0 (2021-12), Tables 7.4.1.1.2-1 and 7.4.1.1.2-2 for "cdmGroups" (𝜆) Values and
        # "deltaShifts" (∆)
        # See also Fig. 9.18 and 9.19 in the "5G NR" book.
        self.cdmGroups   = [(p//2)%2 for p in self.pxxch.portSet] if self.configType==1 else [(p//2)%3
                                    for p in self.pxxch.portSet]
        self.deltaShifts = self.cdmGroups if self.configType==1 else [2*g for g in self.cdmGroups]
        
        # otherCdmGroups is the list of CDM groups used by the PXXCH for other devices
        otherCdmGroups = kwargs.get('otherCdmGroups', []) # List of CDM groups used by other PXXCH objects (other UEs)
        for cdm in otherCdmGroups:
            if cdm in self.cdmGroups:
                raise ValueError("Invalid 'otherCdmGroups' value (%d)! It is already used by this PDSCH."%(cdm))
            if self.configType==1:
                if cdm not in [0, 1]:
                    raise ValueError("Invalid 'otherCdmGroups' value (%d)! Valid CDM groups are 0 and 1."%(cdm))
            elif cdm not in [0, 1, 2]:
                raise ValueError("Invalid 'otherCdmGroups' value (%d)! Valid CDM groups are 0, 1, and 2."%(cdm))
        self.allCdmGroups = sorted(list(set(self.cdmGroups)) + otherCdmGroups) # List of all CDM groups(this and others)
        
        self.dataREs = []   # The RE indices in the DMRS symbol RBs that can be used for data

        # The nID0 and nID1 as specified in 3GPP TS 38.211 V17.0.0 (2021-12), Section 7.4.1.1.1 (scramblingID0,
        # scramblingID1)
        self.nIDs = kwargs.get('nIDs', [])
        self.scID = kwargs.get('scID', 0)
        if self.scID not in [0,1]:   raise ValueError("Invalid 'scID' value! (It must be 0 or 1)")
        
        self.sameSeq = kwargs.get('sameSeq', True)  # If True, the same binary sequence is created for all CDM
                                                    # Groups. Otherwise the sequences for different CDM Groups are
                                                    # initialized differently. This is the parameter "dmrs-Downlink"
                                                    # in 3GPP TS 38.211 V17.0.0 (2021-12), Section 7.4.1.1.1
        lBar, self.symSet = self.getSymSet()
        
        # The ratio of PXXCH EPRE to DM-RS EPRE (EPRE: Energy Per RE)
        # For the default, we use the TS 38.214 V17.0.0 (2021-12), Table 4.1-1.
        # This means: [0] -> 0 dB      [0,1] -> -3 dB      [0,1,2] -> -4.77 dB
        self.epreRatioDb = kwargs.get('epreRatioDb', [0, -3, -4.77][max(self.allCdmGroups)])

        self.ptrs = None

    # ******************************************************************************************************************
    @property
    def ptrsEnabled(self):  return False if self.ptrs is None else (self.ptrs.timeDensity!=0)
    
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="DMRS Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`DMRS` object.

        Parameters
        ----------
        indent: int
            The number of indentation characters.
            
        title: str
            If specified, it is used as a title for the printed information.

        getStr: Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  configType: %d\n"%(self.configType)
        repStr += indent*' ' + "  nIDs: %s\n"%(str(self.nIDs))
        repStr += indent*' ' + "  scID: %d\n"%(self.scID)
        repStr += indent*' ' + "  sameSeq: %d\n"%(self.sameSeq)
        repStr += indent*' ' + "  symbols: %s\n"%("Single" if self.symbols==1 else "Double")
        repStr += indent*' ' + "  typeA1stPos: %d\n"%(self.typeA1stPos)
        repStr += indent*' ' + "  additionalPos: %d\n"%(self.additionalPos)
        repStr += indent*' ' + "  cdmGroups: %s\n"%(str(self.cdmGroups))
        repStr += indent*' ' + "  deltaShifts: %s\n"%(str(self.deltaShifts))
        repStr += indent*' ' + "  allCdmGroups: %s\n"%(str(self.allCdmGroups))
        repStr += indent*' ' + "  symSet: %s\n"%(str(self.symSet))
        repStr += indent*' ' + "  REs (before shift): %s\n"%(str(list(range(0,11,2)) if self.configType==1 else [0,1,6,7]))
        repStr += indent*' ' + "  epreRatioDb: %s (dB)\n"%(str(self.epreRatioDb))
        if self.ptrs is not None:
            repStr += self.ptrs.print(indent+2, "PTRS:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def setPTRS(self, **kwargs):
        r"""
        Creates a new :py:class:`PTRS` object based on the parameters given in ``kwargs`` and associates it with 
        this :py:class:`DMRS` object. For more information please refer to the :py:class:`PTRS` documentation.
        """
        self.ptrs = PTRS(self, **kwargs)

    # ******************************************************************************************************************
    def getSymSet(self):                        # Not documented (Used only internally)
        if len(self.pxxch.symSet)==0:  return [],[]
        # Note: The following code assumes pxxch.symSet is sorted.
        # See 3GPP TS 38.211 V17.0.0 (2021-12), Section 7.4.1.1.2
        if self.pxxch.mappingType == 'A':
            l0 = self.typeA1stPos               # 2 or 3
            ld = self.pxxch.symSet[-1]+1        # This is PXXCH duration ∈ [1..14]
            if self.additionalPos == 3:
                assert self.typeA1stPos==2, "Unsupported combination of 'additionalPos' and 'typeA1stPos'!"
            if ld in [2,3]:
                assert self.typeA1stPos==2, "Unsupported combination of 'ld' and 'typeA1stPos'!"
            
            lBar = np.int32(dmrsPositions[self.symbols]['A'][ld][self.additionalPos])
            dmrsSymSet = np.int32([l0] + lBar[1:].tolist())
        else:   # Mapping Type B
            # Note that in this case position values in lBar are relative to first symbol scheduled for
            # PXXCH in pxxch.symSet
            l0 = 0
            ld = self.pxxch.symSet[-1]-self.pxxch.symSet[0]+1  # This is PXXCH duration
            if ld==7:
                assert self.pxxch.bwp.cpType=='normal', "Unsupported configuration: ld=7 with extended cyclic prefix!"
            if ld==6:
                assert self.pxxch.bwp.cpType=='extended', "Unsupported configuration: ld=6 with normal cyclic prefix!"
            lBar = np.int32(dmrsPositions[self.symbols]['B'][ld][self.additionalPos])
            dmrsSymSet = lBar + self.pxxch.symSet[0]        # Start at the first symbol scheduled in pxxch.symSet

        if self.symbols == 2:
            # Add the second DMRS symbol. For example [1,5] -> [1,2,5,6]
            lBar =       np.int32([l+ll for l in lBar for ll in [0,1]])
            dmrsSymSet = np.int32([l+ll for l in dmrsSymSet for ll in [0,1]])

        # Remove the ones that are not scheduled in PXXCH symSet:
        keepIndexes = [i for i,l in enumerate(dmrsSymSet) if l in self.pxxch.symSet]
        dmrsSymSet = dmrsSymSet[keepIndexes]
        lBar = lBar[keepIndexes]
        return lBar, dmrsSymSet

    # ******************************************************************************************************************
    def getUnusedREs(self):                     # Not documented
        # First get all the REs that are used by DMRS in the PRBs containing DMRS
        if self.configType==1:  dmrsREs, noDataShifts = np.int32(range(0,11,2)), np.int32(self.allCdmGroups)
        else:                   dmrsREs, noDataShifts = np.int32([0, 1, 6, 7]),  2*np.int32(self.allCdmGroups)
        
        usedREs = set(dmrsREs.tolist())     # This set contains the REs that are not available for data in a DMRS RB
        for shift in self.deltaShifts:      usedREs.update( (dmrsREs + shift).tolist() )
        
        # noDataShifts: No data should be allocated in these REs. We include them in the "usedREs".
        for shift in noDataShifts:          usedREs.update( (dmrsREs + shift).tolist() )
        
        # Return the REs in a DMRS RB that can be used for data
        return [x for x in range(12) if x not in usedREs]

    # ******************************************************************************************************************
    def populateGrid(self, grid):
        r"""
        Uses the information in this :py:class:`DMRS` to calculate Demodulation
        Reference Signal values and update the :py:class:`~neoradium.grid.Grid`
        object specified by ``grid``.
        
        If PTRS is enabled, it calls the :py:meth:`~PTRS.populateGrid` method
        of the :py:class:`PTRS` class to update the specified ``grid`` with Phase
        Tracking Reference Signals.

        Parameters
        ----------
        grid : :py:class:`~neoradium.grid.Grid`
            The :py:class:`~neoradium.grid.Grid` object that gets populated with
            the Demodulation Reference Signals.
        """
        slotMap = self.pxxch.slotMap
        # slotMap: Each row corresponds to one symbol in slot and contains the PRBs indices that are
        # available for that symbol with the order they should be assigned. Unscheduled symbols have an empty list.

        # See 3GPP TS 38.211 V17.0.0 (2021-12), Section 7.4.1.1.1
        # See 3GPP TS 38.214 V17.0.0 (2021-12), Section 4.1
        # See Fig. 9.18 and 9.19 in the "5G NR" book.

        dmrsREs = np.int32(range(0,11,2)) if self.configType==1 else np.int32([0, 1, 6, 7])
        noDataShifts = self.configType * np.int32(self.allCdmGroups)
        nREs = len(dmrsREs)         # 6/4 for Config Type 1/2 respectively
        numBitsPerRB = 2 * nREs     # QPSK Modulation=>2 bits per RE, times nREs
        
        # The sequence of bits is always generated starting from CRB 0. The bits before start of BWP are not used.
        offsetBits = self.pxxch.bwp.startRb * numBitsPerRB              # The number of bits before the start of BWP.
        totalBits = offsetBits + (self.pxxch.bwp.numRbs * numBitsPerRB) # Generate sequences with this many bits
        
        # DMRS Beta: See TS 38.214 V17.0.0 (2021-12), Section 4.1
        dmrsBeta = toLinear(-self.epreRatioDb/2)
        
        dmrsAndNoDataREs = []
        for p,portNo in enumerate(self.pxxch.portSet):
            portDmrsREs = dmrsREs + self.deltaShifts[p]
            cdmGroup = self.cdmGroups[p]
            # See TS 38.211 V17.0.0 (2021-12), Tables 7.4.1.1.2-1 and 7.4.1.1.2-2
            wf = [1,-1] if portNo%2 else [1,1]
            wt = [1,-1] if portNo//([4,6][self.configType-1]) else [1,1]
            
            for li,l in enumerate(self.symSet):
                if self.sameSeq:
                    nCSIDlambda = self.scID
                    lambdaBar = 0
                else:
                    nCSIDlambda = self.scID if cdmGroup in [0,2] else 1-self.scID
                    lambdaBar = cdmGroup
                    
                if len(self.nIDs) > nCSIDlambda:    nId = self.nIDs[nCSIDlambda]
                else:                               nId = self.pxxch.bwp.cellId
                
                # Generate sequence of bits (pseudo-random)
                cInit = ((1<<17)*(self.pxxch.bwp.symbolsPerSlot * self.pxxch.bwp.slotNoInFrame + l + 1)*(2*nId + 1) +
                         (1<<17)*(lambdaBar//2) + 2*nId + nCSIDlambda) & 0x7FFFFFFF
                symbolBits = goldSequence(cInit, totalBits)[offsetBits:]    # c(n) in TS 38.211, Section 7.4.1.1.1
                
                # Convert every 2 bits to one complex value per RE:
                rawSymbols = (1-2*np.float64(symbolBits).reshape(-1,2))/np.sqrt(2)
                rawSymbols = rawSymbols[:,0] + 1j*rawSymbols[:,1]           # r(n) in TS 38.211, Section 7.4.1.1.1
                
                lPrime = 0 if self.symbols==1 else li%2                     # See 3GPP TS 38.211 Table 7.4.1.1.2-5
                for ri,dmrsRB in enumerate(slotMap[l]):
                    for reIdx,re in enumerate(portDmrsREs):
                        kPrime = reIdx%2
                        k = 12*dmrsRB + re
                        symIdx = dmrsRB * nREs + reIdx
                        curReType = grid.reTypeAt(p,l,k)
                        if curReType=="RESERVED":  continue
                        assert curReType=="UNASSIGNED", \
                          "Assigning DMRS to the RE(%d,%d,%d) which is already allocated for \"%s\"!"%(p,l,k, curReType)
                        
                        grid[p,l,k] = (dmrsBeta * wf[kPrime] * wt[lPrime] * rawSymbols[symIdx], "DMRS")
                        if grid.reDesc is not None:
                            grid.reDesc[p,l,k] = "DMRS,%s"%('+' if wf[kPrime]*wt[lPrime]>0 else '-')

                        if (li==0) and (self.ptrs is not None):
                            # Save the first symbol's value to be used by PTRS
                            self.ptrs.saveDmrsL0Value(portNo,k,rawSymbols[symIdx])
                        if li==0 and ri==0:
                            dmrsAndNoDataREs += [re]
                                
                    for noDataShift in noDataShifts:
                        for re in dmrsREs:
                            k = 12*dmrsRB + re + noDataShift    # Subcarrier index containing "NO_DATA"
                            if grid.reTypeAt(p,l,k)=="UNASSIGNED":
                                grid[p,l,k] = "NO_DATA"
                                if li==0 and ri==0:        dmrsAndNoDataREs += [re + noDataShift]

        self.dataREs = [x for x in range(12) if x not in dmrsAndNoDataREs]  # RE indices that can be used for data
        if self.ptrsEnabled:    self.ptrs.populateGrid(grid)
     
# **********************************************************************************************************************
class PTRS:
    """
    This class encapsulates the functionality of Phase Tracking Reference Signals (PTRS). A :py:class:`PTRS` object 
    can be associated with a :py:class:`~neoradium.pdsch.PDSCH` or a :py:class:`~neoradium.pusch.PUSCH`. (Currently 
    only :py:class:`~neoradium.pdsch.PDSCH` is implemented in **NeoRadium**. Support for other channels is coming soon.)
    
    The PTRS is used to track the phase of the local oscillators at the receiver and transmitter. This enables 
    suppression of phase noise and common phase error, particularly important at high carrier frequencies, such as
    millimeter-wave bands. Because of the properties of phase noise, PTRS may have low density in the frequency domain 
    but high density in the time domain.. If transmitted, PTRS is always associated with one DMRS port.
    
    This implementation is mostly based on **3GPP TS 38.211, Section 7.4.1.2** and **3GPP TS 38.214, Section 5.1.6.3**.
    """
    # See TS 38.211 V17.0.0 (2021-12), Section 7.4.1.2 Phase-tracking reference signals for PDSCH
    # See TS 38.214 V17.0.0 (2021-12), Section 5.1.6.3 PT-RS reception procedure
    # ******************************************************************************************************************
    def __init__(self, dmrs, **kwargs):
        r"""
        Parameters
        ----------
        dmrs: :py:class:`DMRS`
            The :py:class:`DMRS` object associated with this :py:class:`PTRS`.
            
        kwargs : dict
            A set of optional arguments.

                :mcsi: A list of 3 values for ``ptrs-MCS1``, ``ptrs-MCS2``, and ``ptrs-MCS3`` in **3GPP TS 38.214, 
                    table 5.1.6.3-1** or ``None`` (default). This is used with ``iMCS`` and ``nRBi`` to determine time
                    and frequency density of the PTRS signals. See :ref:`Specifying Time and Frequency
                    density <TimeFreqDensity>` below for more information.
                    
                :iMCS: The value from **3GPP TS 38.214 tables 5.1.3.1-1 to 5.1.3.1-4** or ``None`` (default). This is
                    used with ``mcsi`` and ``nRBi`` to determine time and frequency density of the PTRS signals. See
                    :ref:`Specifying Time and Frequency density <TimeFreqDensity>` below for more information.

                :nRBi: A list of 2 values for ``nRB0`` and ``nRB1`` in **3GPP TS 38.214, table 5.1.6.3-2** or 
                    ``None`` (default). This is used with ``mcsi`` and ``iMCS`` to determine time and frequency
                    density of the PTRS signals. See :ref:`Specifying Time and Frequency density <TimeFreqDensity>`
                    below for more information.

                :timeDensity: The time density of the PTRS signals. It can be 1 (default), 2, or 4. This is ignored if
                    parameters ``mcsi``, ``iMCS``, and ``nRBi`` are all specified. See :ref:`Specifying Time and 
                    Frequency density <TimeFreqDensity>` below for more information.
                    
                :freqDensity: The frequency density of the PTRS signals. It can be 2 (default) or 4. This is ignored if
                    parameters ``mcsi``, ``iMCS``, and ``nRBi`` are all specified. See :ref:`Specifying Time and
                    Frequency density <TimeFreqDensity>` below for more information.
                    
                :reOffset: The resource element (RE) offset. It can be one of 0 (default), 1, 2, or 3. This is the
                    ``resourceElementOffset`` value in **3GPP TS 38.211, Table 6.4.1.2.2.1-1**.

                :portSet: The set of antenna ports associated with this PTRS. If not specified, the first port of the
                    associated :py:class:`DMRS` is used.

                :epreRatio: The ``epre-Ratio`` value in **3GPP TS 38.214, Table 4.1-2**. It is used to determine the 
                    ratio of PTRS energy per resource element (EPRE) to PDSCH EPRE in dB. It can be 0 (default) or 1. 
                    See **3GPP TS 38.214, Table 4.1-2** for more information.


        .. _TimeFreqDensity:
        
        **Specifying Time and Frequency density:**
            
            There are two ways to specify the time and frequency density of the PTRS signals.
            
                :Using MCS Info: In this method, all of the values ``mcsi``, ``iMCS``, and ``nRBi`` **must** be 
                    specified. The values ``timeDensity`` and ``freqDensity`` are then derived from the provided MCS 
                    information based on **3GPP TS 38.214, Tables 5.1.6.3-1 and 5.1.6.3-2**.
                    
                :Direct Setting: In this method, the values ``timeDensity`` and ``freqDensity`` are provided directly.
                    In this case, all of the values ``mcsi``, ``iMCS``, and ``nRBi`` must be set to ``None`` (default).

        **Other Properties:**
        
            :symSet: A numpy array containing the indices of the OFDM symbols used by this :py:class:`PTRS`.

        The notebook :doc:`../Playground/Notebooks/DMRS/PTRS` shows some examples of configuring PTRS.
        """
        self.pxxch = dmrs.pxxch
        self.dmrs = dmrs
        
        self.mcsi = kwargs.get('mcsi', None)     # A list of 3 values for MCS1 to MCS3 (MCS4 is not configured)
        self.iMCS = kwargs.get('iMCS', None)     # The value from one of the tables 5.1.3.1-1 to 5.1.3.1-4 in TS 38.214
        self.nRBi = kwargs.get('nRBi', None)     # A list of 2 values for nRB0 and nRB1
        if (self.mcsi is not None) or (self.iMCS is not None) or (self.nRBi is not None):
            if (self.mcsi is None) or (self.iMCS is None) or (self.nRBi is None):
                raise ValueError("The parameters 'mcsi', 'iMCS', and 'nRBi' must all be None or all have valid values.")
            
            # See TS 38.214 V17.0.0 (2021-12), Table 5.1.6.3-1
            if type(self.mcsi)==list:       raise ValueError("The parameters 'mcsi' must be a list with 3 values!")
            if len(self.mcsi)!=3:           raise ValueError("The parameters 'mcsi' must be a list with 3 values!")
            if self.iMCS < self.mcsi[0]:    self.timeDensity = self.freqDensity = 0     # Disable PTRS
            elif self.iMCS < self.mcsi[1]:  self.timeDensity = 4
            elif self.iMCS < self.mcsi[2]:  self.timeDensity = 2
            else:                           self.timeDensity = 1
            
            # See TS 38.214 V17.0.0 (2021-12), Table 5.1.6.3-2
            numRBs = len(self.pxxch.prbSet)
            if type(self.nRBi)==list:       raise ValueError("The parameters 'nRBi' must be a list with 2 values!")
            if len(self.nRBi)!=2:           raise ValueError("The parameters 'nRBi' must be a list with 2 values!")
            if numRBs < self.nRBi[0]:       self.timeDensity = self.freqDensity = 0     # Disable PTRS
            elif numRBs < self.nRBi[1]:     self.freqDensity = 2
            else:                           self.freqDensity = 4

        else:
            # If 'mcsi', 'iMCS', and 'nRBi' are all None (not provided), then 'timeDensity' and 'freqDensity' can be
            # provided or the default values are used as specified in TS 38.214 V17.0.0 (2021-12), Section 5.1.6.3
            self.timeDensity = kwargs.get('timeDensity', 1)
            if self.timeDensity not in [1,2,4]:
                raise ValueError("Invalid 'timeDensity' value! (It must be 1, 2, or 4)")
            if self.timeDensity >= len(self.pxxch.symSet):
                self.timeDensity = 0    # Disable PTRS (See TS 38.214 V17.0.0 (2021-12), Section 5.1.6.3)

            self.freqDensity = kwargs.get('freqDensity', 2)
            if self.freqDensity not in [2,4]:
                raise ValueError("Invalid 'freqDensity' value! (It must be 2 or 4)")
        
        self.reOffset = kwargs.get('reOffset', 0)
        if self.reOffset in ['00', '01', '10', '11']: self.reOffset = {'00':0, '01':1, '10':2, '11':3}[self.reOffset]
        if self.reOffset not in [0,1,2,3]:
            raise ValueError("Invalid 'reOffset' value! (It must be 0, 1, 2, or 3)")

        # A PTRS can be associated with one or two ports.
        self.portSet = kwargs.get('portSet', self.pxxch.portSet[0:1])   # If not specified, use the first port of PXXCH
        self.dmrsL0Values = { portNo:{} for portNo in self.portSet}     # Save the DMRS values of the first symbol here
                                                                        # and use them when populating the grid with
                                                                        # PTRS values

        self.epreRatio = kwargs.get('epreRatio', 0)                     # EPRE (Energy Per RE) Ratio.
        if self.epreRatio not in [0,1]:
            raise ValueError("Invalid 'epreRatio' value! (It must be 0 or 1)")

        self.symSet = []
        skip = 0
        for s in range(self.pxxch.symSet[0], self.pxxch.symSet[-1]+1):
            if s in self.dmrs.symSet:   skip = self.timeDensity
            if skip==0:
                if s in self.pxxch.symSet:  self.symSet += [s]
                skip = self.timeDensity
            skip-=1

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="PTRS Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`PTRS` object.

        Parameters
        ----------
        indent: int
            The number of indentation characters.
            
        title: str
            If specified, it is used as a title for the printed information.

        getStr: Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        if (self.mcsi is not None) or (self.iMCS is not None) or (self.nRBi is not None):
            repStr += indent*' ' + "  MCS1,MCS2,MCS3: %d %d %d\n"%(self.mcsi[0], self.mcsi[1], self.mcsi[2])
            repStr += indent*' ' + "  Imcs: %d\n"%(self.iMCS)
            repStr += indent*' ' + "  Nrb1, Nrb2: %d %d\n"%(self.nRBi[0], self.nRBi[1])
        repStr += indent*' ' + "  timeDensity: %d\n"%(self.timeDensity)
        repStr += indent*' ' + "  freqDensity: %d\n"%(self.freqDensity)
        repStr += indent*' ' + "  reOffset: %d\n"%(self.reOffset)
        repStr += indent*' ' + "  portSet: %s\n"%(str(self.portSet))
        repStr += indent*' ' + "  epreRatio: %d\n"%(self.epreRatio)
        repStr += indent*' ' + "  symSet: %s\n"%(str(self.symSet))
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def saveDmrsL0Value(self, portNo, k, value):    # Not documented
        # This is called when populating the grid with DMRS values. We save the value of the first
        # DMRS OFDM symbol so that it can be used later when populating the grid with PTRS values. All
        # PTRS REs use the same value of the first DMRS OFDM symbol.
        if portNo not in self.portSet:  return
        self.dmrsL0Values[portNo][k] = value
        
    # ******************************************************************************************************************
    def populateGrid(self, grid):
        r"""
        Uses the information in this :py:class:`PTRS` object to calculate the Phase Tracking Reference Signal values
        and update the :py:class:`~neoradium.grid.Grid` object specified by ``grid``.
        
        Normally you don't need to call this function directly. Since every :py:class:`PTRS` object is associated 
        with a :py:class:`DMRS` object, this function is called automatically when the :py:meth:`~DMRS.populateGrid`
        method of the :py:class:`DMRS` class is called.

        Parameters
        ----------
        grid : :py:class:`~neoradium.grid.Grid`
            The :py:class:`~neoradium.grid.Grid` object that gets populated with the Phase Tracking Reference Signals.
        """
        # See Figure 9.22 in the "5G NR" book
        # See TS 38.211 V17.0.0 (2021-12), Section 7.4.1.2.2
        # Note: I think the Matlab implementation of PTRS allocation might be wrong (or outdated).
        #       According to the standard (See 38.211 Table 7.4.1.2.2-1), allocations are different for different
        #       ports. However, Matlab calculates the allocation for one port and then applies the same for all
        #       ports. Here we create different allocations for different port numbers as specified in the standard.
        if len(self.pxxch.symSet)==0:       return
        if len(self.dmrs.symSet)==0:        return
        
        slotMap = self.pxxch.slotMap
        
        # PTRS EPRE: See table 3GPP TS 38.214 V17.0.0 (2021-12), Table 4.1-2
        beta = 1.0
        if self.epreRatio==0:
            beta = toLinear([0,3,4.77,6,7,7.78][len(self.portSet)]/2)

        # For each port in 'portSet', symbol in 'symSet', and k'th subcarrier in the allocated REs, we copy the first
        # DMRS at that subcarrier to all symbols in 'symSet'
        for p,portNo in enumerate(self.pxxch.portSet):
            if portNo not in self.portSet:  continue
            refRE = ptrsRefREs[self.dmrs.configType][portNo][self.reOffset]
            for l in self.symSet:
                rbs = sorted(slotMap[l]) # The slotMap may have RBs in shuffled order because of interleaving.
                numRBs = len(rbs)
                numREs = 12*numRBs
                if (numRBs % self.freqDensity) == 0:    refRB = self.pxxch.rnti % self.freqDensity
                else:                                   refRB = self.pxxch.rnti % (numRBs % self.freqDensity)

                # Note that kc below is continuous from 0 to (numREs-1)
                # The actual RE index (k below) is from the RBs available for this symbol in the 'slotMap' which in
                # general may not be continuous (i.e. as a result of Reserved RBs being removed)
                k0 = refRE + 12*refRB
                for kc in range(k0, numREs, 12*self.freqDensity):
                    k = rbs[kc//12]*12+kc%12
                    curReType = grid.reTypeAt(p,l,k)
                    if curReType=="RESERVED":  continue
                    assert curReType=="UNASSIGNED", \
                        "Assigning PTRS to the RE(%d,%d,%d) which is already allocated for \"%s\"!"%(p,l,k, curReType)
                    grid[p,l,k] = (beta * self.dmrsL0Values[portNo][k], "PTRS")
                    if grid.reDesc is not None:     grid.reDesc[p,l,k] = "PTRS"

