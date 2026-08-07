# Copyright (c) 2024-2026, InterDigital AI Lab
"""
The module ``dmrs.py`` implements the :py:class:`DMRS` and :py:class:`PTRS` classes, which encapsulate the Demodulation 
Reference Signals (DM-RS) and the Phase-Tracking Reference Signals (PT-RS) respectively.

Demodulation reference signals are intended for channel estimation on the receiver side and enable coherent 
demodulation. They are used with all types of communication channels for both data and control, and both downlink and 
uplink. This means a :py:class:`DMRS` object can be associated with :py:class:`~neoradium.pdsch.PDSCH`,
:py:class:`~neoradium.pdcch.PDCCH`, :py:class:`~neoradium.pusch.PUSCH`, or :py:class:`~neoradium.pucch.PUCCH` classes.

:py:class:`PTRS` is used for tracking the phase of the local oscillators at the receiver and transmitter. If 
transmitted, a PT-RS is always associated with one or two DM-RS ports.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 12/26/2023    Shahab Hamidi-Rad       Completed the documentation.
# 06/27/2025    Shahab Hamidi-Rad       Updated the documentation.
# 10/21/2025    Shahab Hamidi-Rad       Added support for enhanced DM-RS (3GPP TS 38.211, section 7.4.1.1.2).
# 04/09/2026    Shahab Hamidi-Rad       Changes in NeoRadium version 0.5.0:
#                                       * Modified the code to support general cases of PDSCH configuration where PDSCH
#                                         does not cover the whole grid in time and/or frequency.
#                                       * Added more parameter validations for DMRS and PTRS classes.
#                                       * Updated the 'ptrsRefREs' table for enhanced DM-RS in release 18.
#                                       * Replaced 'otherCdmGroups' with 'numCdmGroupsWithoutData' for better
#                                         compatibility with 3GPP.
#                                       * Used dictionaries instead of lists for 'cdmGroups' and 'deltaShifts'. This
#                                         allows the code to work with any DM-RS port numbers used in the 'portSet'.
#                                       * Extended the valid range of port numbers in 'portSet' to support enhanced
#                                         DM-RS.
# **********************************************************************************************************************
import numpy as np

from .modulation import Modem
from .utils import goldSequence, toDb, toLinear, validateRange, getMultiLineStr

# This implementation is based on:
#   - TS 38.211
#   - TS 38.212
#   - TS 38.214

# The following links can help understand the standard more easily:
#   https://www.sharetechnote.com/html/5G/5G_PDSCH_DMRS.html
#   https://www.sharetechnote.com/html/5G/5G_PTRS_DL.html

# **********************************************************************************************************************
dmrsPositions = { 1:     # DMRS.symbols: Single
                  {     # Table 7.4.1.1.2-3: PDSCH DM-RS positions for single-symbol DM-RS.
                    'A': # This table does not contain l0. It contains only additional symbols
                         # Pos0,  Pos1,   Pos2,     Pos3          # ld
                        [[ [],    [],     [],       []         ], # 0
                         [ [],    [],     [],       []         ], # 1
                         [ [],    [],     [],       []         ], # 2
                         [ [],    [],     [],       []         ], # 3
                         [ [],    [],     [],       []         ], # 4
                         [ [],    [],     [],       []         ], # 5
                         [ [],    [],     [],       []         ], # 6
                         [ [],    [],     [],       []         ], # 7
                         [ [],    [7],    [7],      [7]        ], # 8
                         [ [],    [7],    [7],      [7]        ], # 9
                         [ [],    [9],    [6,9],    [6,9]      ], # 10
                         [ [],    [9],    [6,9],    [6,9]      ], # 11
                         [ [],    [9],    [6,9],    [5,8,11]   ], # 12
                         [ [],    [11],   [7,11],   [5,8,11]   ], # 13 - Assuming l1 = 11 (Ignoring case l1 = 12). See
                                                                  #      Section 7.4.1.1.2 for more details
                         [ [],    [11],   [7,11],   [5,8,11]   ]],# 14 - Assuming l1 = 11 (Ignoring case l1 = 12). See
                                                                  #      Section 7.4.1.1.2 for more details

                    'B': # 0 represents l0 in the table
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
                         [ [],    [],     [],       []         ], # 4
                         [ [],    [],     [],       []         ], # 5
                         [ [],    [],     [],       []         ], # 6
                         [ [],    [],     [],       []         ], # 7
                         [ [],    [],     [],       []         ], # 8
                         [ [],    [],     [],       []         ], # 9
                         [ [],    [8],    [],       []         ], # 10
                         [ [],    [8],    [],       []         ], # 11
                         [ [],    [8],    [],       []         ], # 12
                         [ [],    [10],   [],       []         ], # 13
                         [ [],    [10],   [],       []         ]],# 14
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
              [     # DM-RS configType = 1
               #  Offset 00  01  10  11
                        [0,  2,  6,  8],        # Port 1000
                        [2,  4,  8,  10],       # Port 1001
                        [1,  3,  7,  9],        # Port 1002
                        [3,  5,  9,  11],       # Port 1003
                        None,                   # Port 1004
                        None,                   # Port 1005
                        None,                   # Port 1006
                        None,                   # Port 1007
                        [4,  6, 10,  0],        # Port 1008
                        [6,  8,  0,  2],        # Port 1009
                        [5,  7, 11,  1],        # Port 1010
                        [7,  9,  1,  3],        # Port 1011
              ],
              [     # DM-RS configType = 2
               #  Offset 00  01  10  11
                        [0,  1,  6,  7],        # Port 1000
                        [1,  6,  7,  0],        # Port 1001
                        [2,  3,  8,  9],        # Port 1002
                        [3,  8,  9,  2],        # Port 1003
                        [4,  5,  10, 11],       # Port 1004
                        [5,  10, 11, 4],        # Port 1005
                        None,                   # Port 1006
                        None,                   # Port 1007
                        None,                   # Port 1008
                        None,                   # Port 1009
                        None,                   # Port 1010
                        None,                   # Port 1011
                        [6,  7,  0,  1],        # Port 1012
                        [7,  0,  1,  6],        # Port 1013
                        [8,  9,  2,  3],        # Port 1014
                        [9,  2,  3,  8],        # Port 1015
                        [10, 11, 4,  5],        # Port 1016
                        [11, 4,  5,  10],       # Port 1017
              ]
             ]

dmrsWs = [# Config Type 1 (3GPP TS 38.211, Table 7.4.1.1.2-1)
          #        Wf           Wt        p
          [([1,  1,  1,  1], [1,  1]),  # 1000
           ([1, -1,  1, -1], [1,  1]),  # 1001
           ([1,  1,  1,  1], [1,  1]),  # 1002
           ([1, -1,  1, -1], [1,  1]),  # 1003
           ([1,  1,  1,  1], [1, -1]),  # 1004
           ([1, -1,  1, -1], [1, -1]),  # 1005
           ([1,  1,  1,  1], [1, -1]),  # 1006
           ([1, -1,  1, -1], [1, -1]),  # 1007
           ([1,  1, -1, -1], [1,  1]),  # 1008
           ([1, -1, -1,  1], [1,  1]),  # 1009
           ([1,  1, -1, -1], [1,  1]),  # 1010
           ([1, -1, -1,  1], [1,  1]),  # 1011
           ([1,  1, -1, -1], [1, -1]),  # 1012
           ([1, -1, -1,  1], [1, -1]),  # 1013
           ([1,  1, -1, -1], [1, -1]),  # 1014
           ([1, -1, -1,  1], [1, -1])], # 1015
          # Config Type 2 (3GPP TS 38.211, Table 7.4.1.1.2-2)
          #        Wf           Wt        p
          [([1,  1,  1,  1], [1,  1]),  # 0
           ([1, -1,  1, -1], [1,  1]),  # 1
           ([1,  1,  1,  1], [1,  1]),  # 2
           ([1, -1,  1, -1], [1,  1]),  # 3
           ([1,  1,  1,  1], [1,  1]),  # 4
           ([1, -1,  1, -1], [1,  1]),  # 5
           ([1,  1,  1,  1], [1, -1]),  # 6
           ([1, -1,  1, -1], [1, -1]),  # 7
           ([1,  1,  1,  1], [1, -1]),  # 8
           ([1, -1,  1, -1], [1, -1]),  # 9
           ([1,  1,  1,  1], [1, -1]),  # 10
           ([1, -1,  1, -1], [1, -1]),  # 11
           ([1,  1, -1, -1], [1,  1]),  # 12
           ([1, -1, -1,  1], [1,  1]),  # 13
           ([1,  1, -1, -1], [1,  1]),  # 14
           ([1, -1, -1,  1], [1,  1]),  # 15
           ([1,  1, -1, -1], [1,  1]),  # 16
           ([1, -1, -1,  1], [1,  1]),  # 17
           ([1,  1, -1, -1], [1, -1]),  # 18
           ([1, -1, -1,  1], [1, -1]),  # 19
           ([1,  1, -1, -1], [1, -1]),  # 20
           ([1, -1, -1,  1], [1, -1]),  # 21
           ([1,  1, -1, -1], [1, -1]),  # 22
           ([1, -1, -1,  1], [1, -1])]  # 23
         ]

## Noise estimation from the residual error variance:
## 3-level dictionary: ConfigType(1 or 2) x symbols(1 or 2) x numLayers(1 to 8)
## Values in the lists are: maxX, kX, m, y0, a, b, c, d
##                          maxX: Maximum value of residual variance.
##                          kX: Knee position (the residual variance where the line and the tan curve meet)
##                          m: Slope of the line
##                          y0: Intercept of the line
##                          a, b, c, d: The parameters of the tan model: y = a + b*tan((x-c)/d)
#rvToSnrTable = {1:{1:   # ConfigType = 1, symbols = 1
#                    { 1: [21.948, 12.187, 1.008, -2.503, 7.728, 10.138, 10.093, 10.469],        # k=8
#                      2: [32.842, 24.752, 1.009, -10.285, 9.622, 10.146, 18.928, 12.565],       # k=9
#                      3: [28.603, 19.522, 1.007, -4.882, 10.895, 11.820, 15.398, 13.003],       # k=9
#                      4: [34.378, 24.948, 1.006, -10.304, 11.360, 12.350, 21.359, 13.226] },    # k=9
#                   2:   # ConfigType = 1, symbols = 2
#                    { 1: [9.319, 0.508, 1.017, -0.756, -2.025, 7.951, -1.306, 8.212],           # k=6
#                      2: [10.443, 1.963, 1.019, -2.261, -1.781, 7.495, 0.431, 7.661],           # k=6
#                      3: [12.073, 5.688, 1.026, -1.329, 3.400, 5.453, 4.581, 5.535],            # k=7
#                      4: [13.336, 6.607, 1.024, -2.213, 3.374, 5.802, 5.427, 5.902],            # k=7
#                      5: [18.195, 12.366, 1.024, -3.282, 7.429, 5.364, 10.293, 5.932],          # k=8
#                      6: [29.268, 19.598, 1.005, -4.895, 12.014, 12.446, 16.725, 13.003],       # k=9
#                      7: [32.241, 21.611, 1.004, -6.829, 11.785, 15.205, 18.457, 15.768],       # k=9
#                      8: [36.138, 25.122, 1.003, -10.319, 11.493, 16.756, 21.645, 17.385] } },  # k=9
#                2:{1:   # ConfigType = 2, symbols = 1
#                    { 1: [26.830, 17.563, 1.006, -2.884, 11.057, 12.159, 13.623, 13.222],       # k=9
#                      2: [36.603, 29.108, 1.015, -19.937, 8.204, 6.918, 27.689, 7.096],         # k=8
#                      3: [32.211, 20.713, 1.003, -5.863, 11.048, 19.041, 16.760, 19.769],       # k=9
#                      4: [38.492, 29.392, 1.010, -19.930, 7.905, 9.102, 27.515, 9.385],         # k=8
#                      5: [35.229, 27.274, 1.004, -7.543, 1.213, 19.800, -0.787, 37.174],        # k=10
#                      6: [39.907, 29.533, 1.007, -19.924, 7.547, 11.196, 27.218, 11.574]},      # k=8
#                   2:   # ConfigType = 2, symbols = 2
#                    { 1: [9.485, 0.630, 1.017, -0.880, -2.037, 8.003, -1.197, 8.267],           # k=6
#                      2: [10.458, 2.256, 1.020, -2.578, -1.736, 7.198, 0.787, 7.345],           # k=6
#                      3: [12.128, 5.863, 1.027, -1.531, 2.732, 5.584, 4.042, 5.977],            # k=7
#                      4: [13.354, 6.873, 1.025, -2.531, 3.391, 5.550, 5.746, 5.636],            # k=7
#                      5: [14.287, 6.530, 1.018, -1.990, 2.735, 7.142, 4.552, 7.524],            # k=7
#                      6: [15.075, 7.207, 1.017, -2.666, 2.710, 7.273, 5.192, 7.666],            # k=7
#                      7: [19.633, 12.781, 1.018, -3.479, 7.494, 6.494, 10.646, 7.009],          # k=8
#                      8: [30.221, 19.441, 1.004, -4.639, 11.676, 15.800, 16.164, 16.385]} } }   # k=9

# Noise estimation from the residual error variance:
# 3-level dictionary: ConfigType(1 or 2) x symbols(1 or 2) x numLayers(1 to 8)
# The difference errVar - SNR in dB is constant before a knee point after which it
# decreases on a 4th order polinomial.
# Values in the lists are: a, kX, d
#                          kX: Knee position (the residual variance where the difference starts decreasing)
#                          d: The constant difference before the knee point
#                          a: The factor of the 4th order polynomial: y = (a*(x-kX))**4 + d
rvToSnrTable = {1:{1:   # ConfigType = 1, symbols = 1
                #     numLayers: [a, kX, d]
                    { 1: [0.0550, -0.5000, 2.6271],
                      2: [0.0350, 2.0000, 10.3391],
                      3: [0.0550, 6.5000, 4.9559],
                      4: [0.0400, 7.0000, 10.3387] },
                   2:   # ConfigType = 1, symbols = 2
                    { 1: [0.0600, -14.0000, 1.1137],
                      2: [0.0640, -11.0000, 2.6190],
                      3: [0.0700, -8.0000, 1.7970],
                      4: [0.0680, -7.0000, 2.6178],
                      5: [0.0700, -1.0000, 3.6296],
                      6: [0.0670, 10.0000, 4.9505],
                      7: [0.0650, 13.0000, 6.8602],
                      8: [0.0400, 9.0000, 10.3395] } },
                2:{1:   # ConfigType = 2, symbols = 1
                    { 1: [0.0550, 5.5000, 2.9730],
                      2: [0.0400, 10.0000, 19.9054],
                      3: [0.0500, 9.5000, 5.8929],
                      4: [0.0400, 10.5000, 19.9097],
                      5: [0.0450, 12.5000, 7.5650],
                      6: [0.0400, 11.5000, 19.9120]},
                   2:   # ConfigType = 2, symbols = 2
                    { 1: [0.0600, -14.0000, 1.2324],
                      2: [0.0660, -10.0000, 2.9612],
                      3: [0.0700, -8.0000, 2.0004],
                      4: [0.0680, -7.0000, 2.9544],
                      5: [0.0800, -2.5000, 2.2917],
                      6: [0.0750, -3.0000, 2.9609],
                      7: [0.0600, -3.2000, 3.7404],
                      8: [0.0670, 11.5000, 4.6811]} } }

# **********************************************************************************************************************
class DMRS:
    r"""
    This class encapsulates the configuration and functionality of Demodulation Reference Signals. A :py:class:`DMRS`
    object can be associated with a :py:class:`~neoradium.pdsch.PDSCH`, a :py:class:`~neoradium.pdcch.PDCCH`, a 
    :py:class:`~neoradium.pusch.PUSCH`, or a :py:class:`~neoradium.pucch.PUCCH`. (Currently only 
    :py:class:`~neoradium.pdsch.PDSCH` is implemented in **NeoRadium**. Support for other channels is coming soon.)
    
    For every PDSCH, at least one OFDM symbol carrying DM-RS is required. It is also possible to have one, two, or 
    three additional OFDM symbols assigned to DM-RS.
    """
    # ******************************************************************************************************************
    def __init__(self, pxsch, **kwargs):
        r"""
        Parameters
        ----------
        pxsch : :py:class:`~neoradium.pdsch.PDSCH`
            The :py:class:`~neoradium.pdsch.PDSCH` object associated with this :py:class:`DMRS` object. Technically
            this can be any of the :py:class:`~neoradium.pdsch.PDSCH` or :py:class:`~neoradium.pusch.PUSCH` classes, 
            but currently only :py:class:`~neoradium.pdsch.PDSCH` has been implemented in **NeoRadium**.
            
        kwargs : dict
            A set of optional arguments.

                :configType: The DM-RS configuration type. It can be either 1 (default) or 2. In Configuration type 1,
                    the minimum resource element group in frequency domain is one RE. In Configuration type 2, the 
                    minimum resource element group in frequency domain is two consecutive REs.
                    
                :enhanced: This boolean parameter indicates whether the enhanced DM-RS, as introduced in 3GPP release
                    18, should be used. This parameter is equivalent to the ``enhanced-dmrs-Type`` as explained in 
                    **3GPP TS 38.211, section 7.4.1.1.2**. The default value is `False`.
                    
                :symbols: The number of OFDM symbols used with each group of DM-RS REs. It can be 1 (*Single*) or 
                    2 (*Double*). The default is *Single*.

                :typeA1stPos: This is the OFDM symbol index for the first DM-RS symbol when Mapping type A is being
                    used. It can be either 2 (default) or 3.

                :additionalPos: Position(s) for additional DM-RS symbols. For ``symbols==1``,
                    it can be 0, 1, 2, or 3 and for ``symbols==2`` it can be 0 or 1. This allows up to 4 OFDM symbols 
                    to be used for DM-RS.
                   
                :numCdmGroupsWithoutData: Specifies how many CDM groups in each RB are treated as reserved for DM-RS 
                    and therefore unavailable for payload data. It determines the extent of the DM-RS-associated 
                    "NO_DATA" region, in addition to the REs that carry the DM-RS symbols themselves. This value is 
                    used to match the DM-RS overhead and EPRE assumptions defined for PDSCH in **3GPP TS 38.214, 
                    Section 4.1**, where the DM-RS-to-PDSCH power ratio depends on the number of CDM groups without 
                    data. If not explicitly provided, the implementation derives a default value from the number of 
                    active DM-RS ports and the configured DM-RS symbol length. See 
                    :doc:`../Playground/Notebooks/DMRS/CDMsWithNoData` for examples of how to use this parameter.
                    
                :scID: The number specifying which one of the ``nIDs`` (see below) should be used for scrambling. It
                    can be 0 (default) or 1.

                :nIDs: A list of one or 2 integer values (``nIDs[scID] ∈ {0,1,...,65535}, scID ∈ {0,1}``) The nIDs[0] 
                    and nIDs[1] are explained in **3GPP TS 38.211, Section 7.4.1.1.1** (*scramblingID0*, 
                    *scramblingID1*).
                    
                :sameSeq: A boolean value set to `True` by default. If `True`, the same binary sequence is created
                    for all CDM groups. Otherwise, the sequences for different CDM groups are initialized differently. 
                    This is related to the parameter setting ``dmrs-Downlink`` in **3GPP TS 38.211, 
                    Section 7.4.1.1.1**. ``sameSeq=True`` means ``dmrs-Downlink`` is not provided.

                :epreRatioDb: The ratio of PXSCH energy per resource element (EPRE) to DM-RS EPRE in dB. If not 
                    specified, **3GPP TS 38.214, Table 4.1-1** is used to set this parameter.


        **Other Properties:**
        
            :cdmGroups: A dictionary mapping each DM-RS port to its CDM group. This property is set based on the 
                ``portSet`` and ``configType`` parameters.

            :deltaShifts: A dictionary mapping each DM-RS port to its frequency-domain delta shift.

            :symSet: A NumPy array containing the indices of the OFDM symbols used by this DM-RS.
            
            :ptrs: The :py:class:`PTRS` object associated with this DMRS object or `None` if PT-RS is not configured.
                
            :ptrsEnabled: A boolean read-only property. If `True` it means PT-RS is enabled, and therefore the ``ptrs``
                property above should not be `None`. Otherwise, PT-RS is disabled and the ``ptrs`` property above
                should be set to `None`.
                
        The notebook :doc:`../Playground/Notebooks/DMRS/DMRS` shows some examples of configuring DM-RS.
        """
        self.pxsch = pxsch
                
        self.configType = kwargs.get('configType', 1)               # DM-RS Configuration Type (1 or 2)
        validateRange(self.configType, [1,2])

        self.enhanced = kwargs.get('enhanced', False)               # enhanced-dmrs-Type

        self.symbols = kwargs.get('symbols', 1)                     # DM-RS symbols 1->Single, 2->Double
        validateRange(self.symbols, [1,2])

        self.typeA1stPos = kwargs.get('typeA1stPos', 2)             # dmrs-TypeA-Position
        validateRange(self.typeA1stPos, [2,3])
        if (pxsch.symSet[0] not in [0,1,2]) and (pxsch.symSet[0]!=3 or self.typeA1stPos!=3):
            raise ValueError("Invalid symbol allocation: start = %d"%(pxsch.symSet[0]))

        self.additionalPos = kwargs.get('additionalPos', 0)         # dmrs-AdditionalPosition
        if self.symbols == 1:
            if self.additionalPos not in range(4):
                raise ValueError("Invalid 'additionalPos' value! (It must be in [0..3])")
        elif self.additionalPos not in [0,1]:
            raise ValueError("Invalid 'additionalPos' value! (It must be 0 or 1 for 2-symbol DM-RS)")

        # Note: We internally use 0-based port numbers (e.g., 0, 1, ... instead of the standard 1000, 1001, ...)
        # See TS 38.211, Table 7.4.1.1.2-5 for the valid range of port numbers
        if self.configType == 1:        validRange = list(range(4)) if self.symbols == 1 else list(range(8))
        else:                           validRange = list(range(6)) if self.symbols == 1 else list(range(12))
        if self.enhanced:
            if self.configType == 1:    validRange += list(range(8,12)) if self.symbols == 1 else list(range(8,16))
            else:                       validRange += list(range(12,18)) if self.symbols == 1 else list(range(12,24))
        for p in self.pxsch.portSet:
            if p not in validRange:
                raise ValueError(f"Invalid DM-RS 'port number' {p}! (Valid Range: {validRange})")

        # CDM Group: The DM-RS signals that share the same subcarriers but are separated in the code domain by
        # using different orthogonal sequences.
        # See TS 38.211, Tables 7.4.1.1.2-1 and 7.4.1.1.2-2 for "cdmGroups" (𝜆) Values and
        # "deltaShifts" (∆)
        # See also Fig. 9.18 and 9.19 in the "5G NR" book.
        # cdmGroups and deltaShifts are actually lookup tables from ports to cdmGroup/deltaShift values. They have
        # one entry per active port.
        if self.configType==1:  self.cdmGroups = {int(p): int((p//2)%2) for p in self.pxsch.portSet}
        else:                   self.cdmGroups = {int(p): int((p//2)%3) for p in self.pxsch.portSet}
        self.deltaShifts = self.cdmGroups if self.configType==1 else {k:2*v for k,v in self.cdmGroups.items()}
        self.numCdmGroups = len(set(self.cdmGroups.values()))
        
        self.numCdmGroupsWithoutData = int(kwargs.get('numCdmGroupsWithoutData', self.numCdmGroups))
        validateRange(self.numCdmGroupsWithoutData, [1,2] if self.configType==1 else [1,2,3])

        # The nID0 and nID1 as specified in 3GPP TS 38.211, Section 7.4.1.1.1 (scramblingID0,
        # scramblingID1)
        self.nIDs = kwargs.get('nIDs', [])
        self.scID = kwargs.get('scID', 0)
        if self.scID not in [0,1]:   raise ValueError("Invalid 'scID' value! (It must be 0 or 1)")
        
        self.sameSeq = kwargs.get('sameSeq', True)  # If True, the same binary sequence is created for all CDM
                                                    # Groups. Otherwise, the sequences for different CDM Groups are
                                                    # initialized differently. This is the opposite of the parameter
                                                    # "dmrs-Downlink" in 3GPP TS 38.211, Section 7.4.1.1.1
        self.symSet = self.getSymSet()
        
        # The ratio of PXSCH EPRE to DM-RS EPRE (EPRE: Energy Per RE)
        # For the default, we use the TS 38.214, Table 4.1-1.
        # This means: 1 -> 0 dB, 2 -> -3 dB, 3 -> -4.77 dB
        self.epreRatioDb = kwargs.get('epreRatioDb',
                                      [0, -3, -4.77][self.numCdmGroupsWithoutData-1])

        # Total number of DM-RS overhead REs including DM-RS and the NO_DATA REs in one resource block
        numDmrsREs = 6 if self.configType==1 else 4
        self.numDmrsOhREs = len(self.symSet)*(numDmrsREs * self.numCdmGroupsWithoutData)

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
        repStr += indent*' ' + f"  configType:              {self.configType}\n"
        repStr += indent*' ' + f"  nIDs:                    {self.nIDs}\n"
        repStr += indent*' ' + f"  scID:                    {self.scID}\n"
        repStr += indent*' ' + f"  sameSeq:                 {self.sameSeq}\n"
        repStr += indent*' ' + f"  symbols:                 {'Single' if self.symbols==1 else 'Double'}\n"
        repStr += indent*' ' + f"  typeA1stPos:             {self.typeA1stPos}\n"
        repStr += indent*' ' + f"  additionalPos:           {self.additionalPos}\n"
        repStr += getMultiLineStr("cdmGroups (port:cdm)   ", [f"{k}:{v}" for k,v in self.cdmGroups.items()], indent, "%-4s", 5, numPerLine=20)
        repStr += getMultiLineStr("deltaShifts (port:cdm) ", [f"{k}:{v}" for k,v in self.deltaShifts.items()], indent, "%-4s", 5, numPerLine=20)
        repStr += indent*' ' + f"  numCdmGroupsWithoutData: {self.numCdmGroupsWithoutData}\n"
        repStr += getMultiLineStr("symSet               ", self.symSet, indent, "%3d", 3, numPerLine=20)
        repStr += indent*' ' + f"  REs (before shift):      {'0 2 4 6 8 10' if self.configType==1 else '0 1 6 7'}\n"
        repStr += indent*' ' + f"  epreRatioDb:             {self.epreRatioDb} (dB)\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def resVarToNoiseVar(self, resVar, nr):
        # Called from the channel estimation method "PDSCH.estimateChannel" to convert the calculated
        # residual variance values to the estimated noise variance using the calibration table and the
        # constant/polynomial difference curve combination.
        rvDb = toDb(1/(resVar*nr))      # Calibration is based on dB values. Convert residual variance to dB
        
        a, kX, d = rvToSnrTable[self.configType][self.symbols][self.pxsch.numLayers]
        diff = d if rvDb <= kX else (-(a*(rvDb-kX))**4 + d) # This is the difference: diff = rvDb - snrDb
        snrDb = min(50, rvDb - diff)                        # Clip the SNR at 50 dB
        noiseVar = 1/(toLinear(snrDb) * nr)                 # Convert the SNR to noise variance
        return noiseVar
        
    # ******************************************************************************************************************
    def setPTRS(self, **kwargs):
        r"""
        Creates a new :py:class:`PTRS` object based on the parameters given in ``kwargs`` and associates it with 
        this :py:class:`DMRS` object. For more information, please refer to the :py:class:`PTRS` documentation.
        """
        self.ptrs = PTRS(self, **kwargs)

    # ******************************************************************************************************************
    def getSymSet(self):                                                            # Undocumented
        if len(self.pxsch.symSet)==0:  return np.int32([])
        # Note: The following code assumes pxsch.symSet is sorted.
        # See 3GPP TS 38.211, Section 7.4.1.1.2
        if self.pxsch.mappingType == 'A':
            l0 = self.typeA1stPos               # 2 or 3
            ld = len(self.pxsch.symSet)         # This is PXSCH duration ∈ [1..14]
            if self.additionalPos == 3:
                if self.typeA1stPos!=2:
                    raise ValueError("Unsupported combination of 'additionalPos' and 'typeA1stPos'!")
            if ld in [2,3]:
                if self.typeA1stPos!=2:
                    raise ValueError("Unsupported combination of 'ld' and 'typeA1stPos'!")
            
            lBar = dmrsPositions[self.symbols]['A'][ld][self.additionalPos] # List of additional symbols
            dmrsSymSet = np.int32([l0] + lBar)
        else:   # Mapping Type B
            # Note that in this case position values in lBar are relative to first symbol scheduled for
            # PXSCH in pxsch.symSet
            l0 = 0
            ld = len(self.pxsch.symSet)         # This is PXSCH duration
            if ld==7 and self.pxsch.bwp.cpType=='extended':
                raise ValueError("Unsupported configuration: ld=7 with extended cyclic prefix!")
            if ld==6 and self.pxsch.bwp.cpType=='normal':
                raise ValueError("Unsupported configuration: ld=6 with normal cyclic prefix!")
            lBar = dmrsPositions[self.symbols]['B'][ld][self.additionalPos]
            dmrsSymSet = np.int32(lBar) + self.pxsch.symSet[0]  # Start at the first symbol scheduled in pxsch.symSet

        if self.symbols == 2:
            # Add the second DM-RS symbol. For example: [1,5] -> [1,2,5,6]
            dmrsSymSet = (dmrsSymSet[:,None] + [[0,1]]).flatten()

        # Note that PXSCH always uses a contiguous time block, so this DMRS symSet is always a subset of PXSCH symSet.
        return dmrsSymSet

    # ******************************************************************************************************************
    def populateGrid(self, grid):
        r"""
        Uses the information in this :py:class:`DMRS` to calculate demodulation reference signal values and update the
        :py:class:`~neoradium.grid.Grid` object specified by ``grid``.
        
        If PT-RS is enabled, it calls the :py:meth:`~PTRS.populateGrid` method of the :py:class:`PTRS` class to update
        the specified ``grid`` with phase-tracking reference signals.

        Parameters
        ----------
        grid : :py:class:`~neoradium.grid.Grid`
            The :py:class:`~neoradium.grid.Grid` object that is populated with the demodulation reference signals.
        """
        # See 3GPP TS 38.211, Section 7.4.1.1.1
        # See 3GPP TS 38.214, Section 4.1
        # See Fig. 9.18 and 9.19 in the "5G NR" book.

        dmrsREs = np.int32([0,2,4,6,8,10] if self.configType==1 else [0,1,6,7])
        nREs = len(dmrsREs)         # 6/4 for Config Type 1/2 respectively
        numBitsPerRB = 2 * nREs     # QPSK modulation -> 2 bits per RE, times nREs
        
        # noDataShifts is a list of 2 or 3 lists, one for each CDM group. For each CDM group, the list contains the
        # shift values applied to the REs in 'dmrsREs' to get the REs that should be marked as No-Data.
        # numCdmGroupsWithoutData: 0  1                 2                        3                        # ConfigType↓
        noDataShifts =       [[ None, None,             None,                    None],                             # 0
                              [ None, [[],[0]],         [[1],[0]],               None],                             # 1
#                             [ None, [[],[0,1],[0,1]], [[2,3],[0,1],[0,1,2,3]], [[2,3,4,5],[0,1,4,5],[0,1,2,3]] ]  # 2
                              [ None, [[],[0],[0]],     [[2],[0],[0,2]],          [[2,4],[0,4],[0,2]] ]  # 2
                             ][self.configType][self.numCdmGroupsWithoutData]
              
        # The sequence of bits is always generated starting from CRB 0. The bits before start of BWP are not used.
        offsetBits = self.pxsch.bwp.startRb * numBitsPerRB              # The number of bits before the start of BWP.
        totalBits = offsetBits + (self.pxsch.bwp.numRbs * numBitsPerRB) # Generate sequences with this many bits
        
        # DM-RS Beta: See TS 38.214, Section 4.1
        dmrsBeta = toLinear(-self.epreRatioDb/2)
        maxKprime = 4 if self.enhanced else 2
        for p,portNo in enumerate(self.pxsch.portSet):
            portDmrsREs = dmrsREs + self.deltaShifts[portNo]    # deltaShifts is a lookup table from portNo to delta
            cdmGroup = self.cdmGroups[portNo]                   # 𝝀
            wf, wt = dmrsWs[self.configType-1][portNo]          # Note that internally we use 0-based port numbers
            
            # self.symSet contains all DM-RS symbols
            for li,l in enumerate(self.symSet):
                if self.sameSeq:
                    # Same sequence is used for all CDM groups. This means the higher-layer parameter dmrs-Downlink
                    # in the DMRS-DownlinkConfig IE is NOT provided.
                    nCSIDlambda = self.scID
                    lambdaBar = 0
                else:
                    nCSIDlambda = self.scID if cdmGroup in [0,2] else 1-self.scID
                    lambdaBar = cdmGroup
                    
                if len(self.nIDs) > nCSIDlambda:    nId = self.nIDs[nCSIDlambda]
                else:                               nId = self.pxsch.bwp.cellId
                
                # Generate sequence of bits (pseudo-random)
                cInit = ((1<<17)*(self.pxsch.bwp.symbolsPerSlot * self.pxsch.bwp.slotNoInFrame + l + 1)*(2*nId + 1) +
                         (1<<17)*(lambdaBar//2) + 2*nId + nCSIDlambda) & 0x7FFFFFFF
                symbolBits = goldSequence(cInit, totalBits)[offsetBits:]    # c(n) in TS 38.211, Section 7.4.1.1.1
                
                # Convert every 2 bits to one complex value per RE:
                rawSymbols = (1-2*np.float64(symbolBits).reshape(-1,2))/np.sqrt(2)
                rawSymbols = rawSymbols[:,0] + 1j*rawSymbols[:,1]           # r(n) in TS 38.211, Section 7.4.1.1.1
                
                lPrime = 0 if self.symbols==1 else li%2                     # See 3GPP TS 38.211 Table 7.4.1.1.2-5
                for dmrsGrb, dmrsPrb in enumerate(self.pxsch.grb2Prb):
                    for reIdx,re in enumerate(portDmrsREs):
                        kPrime = reIdx % maxKprime
                        k = 12*dmrsGrb + re                                 # Using GRB for k
                        symIdx = dmrsPrb * nREs + reIdx                     # Using PRB for raw symbol index
                        curReType = grid.reTypeAt(p,l,k)
                        if curReType=="RESERVED": continue
                        if curReType not in ["UNASSIGNED", "DMRS"]:
                            raise ValueError(f"Trying to allocate the RE at ({p},{l},{k}) for DM-RS," +
                                             f"while it is currently allocated for \"{curReType}\"!")
                        grid[p,l,k] = (dmrsBeta * wf[kPrime] * wt[lPrime] * rawSymbols[symIdx], "DMRS")

                        if (li==0) and (self.ptrs is not None):
                            # Save the first symbol's value to be used by PT-RS
                            self.ptrs.saveDmrsL0Value(portNo,k,rawSymbols[symIdx])

                    for shift in noDataShifts[cdmGroup]:
                        for re in dmrsREs:
                            k = 12*dmrsGrb + re + shift
                            if grid.reTypeAt(p,l,k)=="UNASSIGNED":
                                grid[p,l,k] = "NO_DATA"

        if self.ptrsEnabled:    self.ptrs.populateGrid(grid)
     
    # ******************************************************************************************************************
    def getCdmLKs(self):                                                        # Undocumented
        # Return the DMRS resource-element indices grouped by CDM group.
        # This function returns the ``[l, k]`` indices of all resource elements (REs)
        # belonging to each DMRS CDM group for the current PDSCH DMRS configuration.
        #
        # For double-symbol DMRS configurations, each adjacent DMRS symbol pair is
        # counted once. In that case, the returned symbol dimension corresponds to
        # the number of DMRS symbol pairs, not the total number of DMRS OFDM symbols.

        # The returned subcarrier indices are in continuous virtual resource-block
        # order (GRB/VRB order), not physical PRB order.

        # Returns a NumPy integer array of shape ``(numSym, numSymCdms, cdmSize, 2)``,
        #   * ``numSym`` is the number of DMRS symbol positions. For double-symbol
        #     DMRS, each adjacent DMRS symbol pair is counted as one. (unlike self.symSet)
        #   * ``numSymCdms`` is the number of CDM groups per DMRS symbol position.
        #     The total number of CDM groups is therefore ``numSym * numSymCdms``.
        #   * ``cdmSize`` is the number of REs in each CDM group.
        #   * The last dimension contains the ``[l, k]`` indices of each RE.
        # Therefore, ``cdmLKs[s, i, j]`` is a length-2 array ``[l, k]`` giving
        # the OFDM-symbol index ``l`` and subcarrier index ``k`` of the
        # ``j``-th RE in the ``i``-th CDM group of the ``s``-th DMRS symbol
        # position.
        # NOTE: k's are in GRB order
        ls = self.symSet if self.symbols==1 else self.symSet[::2]   # List of DMRS ofdm symbols. (doubles counted once)
        cdms = sorted(set(self.cdmGroups.values()))                 # List of CDM groups
        cdmSize = 2*self.symbols
        # First, get LKs for one RB
        if self.configType==1:
            cdmLKs1Rb = np.int32([[[int(l)+lPrime,k+kPrime+c]
                                   for lPrime in range(self.symbols)
                                       for kPrime in [0,2]]
                                           for l in ls
                                               for k in [0,4,8]
                                                   for c in cdms ])
        else:
            cdmLKs1Rb = np.int32([[[int(l)+lPrime,k+kPrime+2*c]
                                   for lPrime in range(self.symbols)
                                       for kPrime in [0,1]]
                                              for l in ls
                                                  for k in [0,6]
                                                      for c in cdms])
        cdmLKs1Rb = cdmLKs1Rb.reshape(len(ls), -1, cdmSize, 2)   # numSym x numSymCdmsPerRB x cdmSize x 2
        numPRBs = len(self.pxsch.prbSet)
        cdmLKs = np.concatenate([cdmLKs1Rb+[[[0, 12*i]]] for i in range(numPRBs)], axis=1)
        return cdmLKs # numSym x numSymCdms x cdmSize x 2

    # ******************************************************************************************************************
    def getCdmValues(self, cdmLKs, rxGrid=None):                                # Undocumented
        # Return values at the DMRS CDM-group resource elements.
        # This function extracts the values corresponding to the REs of each CDM
        # group, either from the transmitted PDSCH grid DMRS symbols or from a
        # received resource grid.
        #
        # When ``rxGrid`` is not provided, the function returns the DMRS reference
        # values from the PDSCH grid at the CDM-group RE locations.
        #
        # When ``rxGrid`` is provided, the function returns the received values from
        # the receive resource grid at the same CDM-group RE locations. In this case,
        # the RE indices are converted from continuous virtual resource-block order
        # (GRB/VRB order) to physical PRB order before indexing the received grid.
        #
        # Returns a NumPy complex array of shape ``(numSym, numSymCdms, numPorts, cdmSize)``, where:
        #   * ``numSym`` is the number of DMRS symbol positions. For double-symbol
        #     DMRS, each adjacent DMRS symbol pair is counted as one.
        #   * ``numSymCdms`` is the number of CDM groups per DMRS symbol position.
        #   * ``numPorts`` is:
        #       - the number of PDSCH layers when ``rxGrid`` is ``None``, or
        #       - the number of receive antennas when ``rxGrid`` is provided.
        #   * ``cdmSize`` is the number of REs in each CDM group.
        # Thus, ``cdmValues[s, i, p]`` contains the ``cdmSize`` complex values
        # for port or antenna ``p`` in the ``i``-th CDM group of the ``s``-th
        # DMRS symbol position.
        numSyms, numSymCdms, cdmSize, _ = cdmLKs.shape
        flatCdmLKs = cdmLKs.reshape(-1,2).copy()            # Do not corrupt the original 'cdmLKs'
        
        if rxGrid is None:  # Getting the reference values at CDM group REs
            dmrsRetId = self.pxsch.grid.makeTypeObjId(self.pxsch.grid.retNameToId["DMRS"])
            lkTypes = self.pxsch.grid.reTypeObjIds[:,*zip(*flatCdmLKs)]
            lkVals = self.pxsch.grid[:,*zip(*flatCdmLKs)]   # All DMRS values at CDM L,K locations
            lkVals[lkTypes != dmrsRetId] = 0                # Make sure the values is set to zero for non-DMRS REs.
            numPorts = self.pxsch.numLayers                 # Number of PDSCH layers
        else:               # Getting the RX values at CDM group REs
            gre2Pre = np.int32([12*x+np.arange(12) if x>=0 else -1*np.ones(12) for x in self.pxsch.grb2Prb]).flatten()
            flatCdmLKs[:,1] = gre2Pre[flatCdmLKs[:,1]]      # convert k's from GRB to PRB
            lkVals = rxGrid[:,*zip(*flatCdmLKs)]            # All RX values at CDM L,K locations
            numPorts = rxGrid.shape[0]                      # Number of receive antenna
    
        cdmValues = np.transpose( lkVals.reshape(numPorts, numSyms, numSymCdms, cdmSize), [1,2,0,3])
        return cdmValues    # numSym x numSymCdms x numPorts x cdmSize  (numPorts is numLayers or num RX antenna)

# **********************************************************************************************************************
class PTRS:
    """
    This class encapsulates the functionality of Phase Tracking Reference Signals (PT-RS). A :py:class:`PTRS` object 
    can be associated with a :py:class:`~neoradium.pdsch.PDSCH` or a :py:class:`~neoradium.pusch.PUSCH`. (Currently 
    only :py:class:`~neoradium.pdsch.PDSCH` is implemented in **NeoRadium**. Support for other channels is coming soon.)
    
    The PT-RS is used to track the phase of the local oscillators at the receiver and transmitter. This enables 
    suppression of phase noise and common phase error, particularly important at high carrier frequencies, such as
    millimeter-wave bands. Because of the properties of phase noise, PT-RS may have low density in the frequency domain 
    but high density in the time domain. If transmitted, PT-RS is always associated with one or two DM-RS ports.
    
    This implementation is mostly based on **3GPP TS 38.211, Section 7.4.1.2** and **3GPP TS 38.214, Section 5.1.6.3**.
    """
    # See TS 38.211, Section 7.4.1.2 Phase-tracking reference signals for PDSCH
    # See TS 38.214, Section 5.1.6.3 PT-RS reception procedure
    # ******************************************************************************************************************
    def __init__(self, dmrs, **kwargs):
        r"""
        Parameters
        ----------
        dmrs : :py:class:`DMRS`
            The :py:class:`DMRS` object associated with this :py:class:`PTRS`.
            
        kwargs : dict
            A set of optional arguments.

                :mcsi: A list of 3 values for ``ptrs-MCS1``, ``ptrs-MCS2``, and ``ptrs-MCS3`` in **3GPP TS 38.214, 
                    Table 5.1.6.3-1** or `None` (default). This is used with ``iMCS`` and ``nRBi`` to determine time
                    and frequency density of the PT-RS signals. See :ref:`Specifying Time and Frequency
                    density <TimeFreqDensity>` below for more information.
                    
                :iMCS: The value from **3GPP TS 38.214 tables 5.1.3.1-1 to 5.1.3.1-4** or `None` (default). This is
                    used with ``mcsi`` and ``nRBi`` to determine time and frequency density of the PT-RS signals. See
                    :ref:`Specifying Time and Frequency density <TimeFreqDensity>` below for more information.

                :nRBi: A list of 2 values for ``nRB0`` and ``nRB1`` in **3GPP TS 38.214, Table 5.1.6.3-2** or 
                    `None` (default). This is used with ``mcsi`` and ``iMCS`` to determine time and frequency
                    density of the PT-RS signals. See :ref:`Specifying Time and Frequency density <TimeFreqDensity>`
                    below for more information.

                :timeDensity: The time density of the PT-RS signals. It can be 1 (default), 2, or 4. This is ignored if
                    parameters ``mcsi``, ``iMCS``, and ``nRBi`` are all specified. See :ref:`Specifying Time and 
                    Frequency density <TimeFreqDensity>` below for more information.
                    
                :freqDensity: The frequency density of the PT-RS signals. It can be 2 (default) or 4. This is ignored if
                    parameters ``mcsi``, ``iMCS``, and ``nRBi`` are all specified. See :ref:`Specifying Time and
                    Frequency density <TimeFreqDensity>` below for more information.
                    
                :reOffset: The resource element (RE) offset. It can be one of 0 (default), 1, 2, or 3. This is the
                    ``resourceElementOffset`` value in **3GPP TS 38.211, Table 7.4.1.2.2-1**.

                :portSet: The set of antenna ports associated with this PT-RS. If not specified, the first port of the
                    associated :py:class:`DMRS` is used.

                :epreRatio: The ``epre-Ratio`` value in **3GPP TS 38.214, Table 4.1-2**. It is used to determine the 
                    ratio of PT-RS energy per resource element (EPRE) to PDSCH EPRE in dB. It can be 0 (default) or 1. 
                    See **3GPP TS 38.214, Table 4.1-2** for more information.


        .. _TimeFreqDensity:
        
        **Specifying Time and Frequency density:**
            
            There are two ways to specify the time and frequency density of the PT-RS signals.
            
                :Using MCS Info: In this method, all of the values ``mcsi``, ``iMCS``, and ``nRBi`` **must** be 
                    specified. The values ``timeDensity`` and ``freqDensity`` are then derived from the provided MCS 
                    information based on **3GPP TS 38.214, Tables 5.1.6.3-1 and 5.1.6.3-2**.
                    
                :Direct Setting: In this method, the values ``timeDensity`` and ``freqDensity`` are provided directly.
                    In this case, ``mcsi``, ``iMCS``, and ``nRBi`` must all be set to `None` (default).

        **Other Properties:**
        
            :symSet: A NumPy array containing the indices of the OFDM symbols used by this :py:class:`PTRS`.

        The notebook :doc:`../Playground/Notebooks/DMRS/PTRS` shows some examples of configuring PTRS.
        """
        self.pxsch = dmrs.pxsch
        self.dmrs = dmrs
        
        self.mcsi = kwargs.get('mcsi', None)     # A list of 3 values for MCS1 to MCS3 (MCS4 is not configured)
        self.iMCS = kwargs.get('iMCS', None)     # The value from one of the tables 5.1.3.1-1 to 5.1.3.1-4 in TS 38.214
        self.nRBi = kwargs.get('nRBi', None)     # A list of 2 values for nRB0 and nRB1
        if (self.mcsi is not None) or (self.iMCS is not None) or (self.nRBi is not None):
            if (self.mcsi is None) or (self.iMCS is None) or (self.nRBi is None):
                raise ValueError("The parameters 'mcsi', 'iMCS', and 'nRBi' must all be None or all have valid values.")
            
            # See TS 38.214, Table 5.1.6.3-1
            if type(self.mcsi)!=list:       raise ValueError("The parameters 'mcsi' must be a list with 3 values!")
            if len(self.mcsi)!=3:           raise ValueError("The parameters 'mcsi' must be a list with 3 values!")
            if self.iMCS < self.mcsi[0]:    self.timeDensity = self.freqDensity = 0     # Disable PT-RS
            elif self.iMCS < self.mcsi[1]:  self.timeDensity = 4
            elif self.iMCS < self.mcsi[2]:  self.timeDensity = 2
            else:                           self.timeDensity = 1
            
            # See TS 38.214, Table 5.1.6.3-2
            numRBs = len(self.pxsch.prbSet)
            if type(self.nRBi)!=list:       raise ValueError("The parameters 'nRBi' must be a list with 2 values!")
            if len(self.nRBi)!=2:           raise ValueError("The parameters 'nRBi' must be a list with 2 values!")
            if numRBs < self.nRBi[0]:       self.timeDensity = self.freqDensity = 0     # Disable PT-RS
            elif numRBs < self.nRBi[1]:     self.freqDensity = 2
            else:                           self.freqDensity = 4

        else:
            # If 'mcsi', 'iMCS', and 'nRBi' are all None (not provided), then 'timeDensity' and 'freqDensity' can be
            # provided or the default values are used as specified in TS 38.214, Section 5.1.6.3
            self.timeDensity = kwargs.get('timeDensity', 1)
            validateRange(self.timeDensity, [1,2,4])

            if self.timeDensity >= len(self.pxsch.symSet):
                self.timeDensity = 0    # Disable PT-RS (See TS 38.214, Section 5.1.6.3)

            self.freqDensity = kwargs.get('freqDensity', 2)
            validateRange(self.freqDensity, [2,4])
        
        self.reOffset = kwargs.get('reOffset', 0)
        if self.reOffset in ['00', '01', '10', '11']: self.reOffset = {'00':0, '01':1, '10':2, '11':3}[self.reOffset]
        validateRange(self.reOffset, [0,1,2,3])

        # A PT-RS can be associated with one or two ports.
        self.portSet = kwargs.get('portSet', self.pxsch.portSet[0:1])   # If not specified, use the first port of PXSCH
        if len(self.portSet)>2:     raise ValueError("PTRS portSet can have at most 2 ports!")
        for portNo in self.portSet:
            if ptrsRefREs[self.dmrs.configType][portNo] is None:
                raise ValueError(f"Invalid PT-RS portNo {portNo}!")
        self.dmrsL0Values = { portNo:{} for portNo in self.portSet} # Save the DM-RS values of the first symbol here
                                                                    # and use them when populating the grid with
                                                                    # PT-RS values

        self.epreRatio = kwargs.get('epreRatio', 0)                 # EPRE (Energy Per RE) Ratio.
        validateRange(self.epreRatio, [0,1])

        self.symSet = []
        if self.timeDensity>0 and self.freqDensity>0:
            skip = 0
            for s in self.pxsch.symSet:
                if s in self.dmrs.symSet:   skip = self.timeDensity
                if skip==0:
                    self.symSet += [s]
                    skip = self.timeDensity
                skip-=1
        self.symSet = np.int32(self.symSet)

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="PTRS Properties:", getStr=False):
        r"""
        Prints the properties of this :py:class:`PTRS` object.

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
        if (self.mcsi is not None) or (self.iMCS is not None) or (self.nRBi is not None):
            repStr += indent*' ' + f"  MCS1,MCS2,MCS3:          {self.mcsi}\n"
            repStr += indent*' ' + f"  iMCS:                    {self.iMCS}\n"
            repStr += indent*' ' + f"  Nrb1, Nrb2:              {self.nRBi[0]}, {self.nRBi[1]}\n"
        repStr += indent*' ' + f"  timeDensity:             {self.timeDensity}\n"
        repStr += indent*' ' + f"  freqDensity:             {self.freqDensity}\n"
        repStr += indent*' ' + f"  reOffset:                {self.reOffset}\n"
        repStr += indent*' ' + f"  portSet:                 {self.portSet}\n"
        repStr += indent*' ' + f"  epreRatio:               {self.epreRatio}\n"
        repStr += getMultiLineStr("symSet               ", self.symSet, indent, "%3d", 3, numPerLine=20)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def saveDmrsL0Value(self, portNo, k, value):    # Undocumented
        # This is called when populating the grid with DM-RS values. We save the value of the raw symbols used for
        # the first DM-RS OFDM symbol (without orthogonal weights wt and wf and DM-RS beta) so that it can be used later
        # when populating the grid with PT-RS values. All PT-RS REs use the same value.
        # Notes:
        #  - k is VRB-based.
        #  - value is the raw symbol (QPSK modulation of every 2 bits in the gold sequence).
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
            The :py:class:`~neoradium.grid.Grid` object that is populated with the phase-tracking reference signals.
        """
        # See Figure 9.22 in the "5G NR" book
        # See TS 38.211, Section 7.4.1.2.2
        # Note that the REs allocated for PT-RS cannot be used for PDSCH in other ports/layers. They must be set to
        # NO_DATA.
        if len(self.pxsch.symSet)==0:       return
        if len(self.dmrs.symSet)==0:        return
        
        # PT-RS EPRE: See 3GPP TS 38.214, Table 4.1-2A.
        beta = 1.0
        if self.epreRatio==0:
            beta = toLinear([0,3,4.77,6,7,7.78,8.45,9][self.pxsch.numLayers-1]/2)

        if (grid.numRbs % self.freqDensity) == 0:   refRB = self.pxsch.rnti % self.freqDensity
        else:                                       refRB = self.pxsch.rnti % (grid.numRbs % self.freqDensity)

        # For each port in 'portSet', symbol in 'symSet', and k-th subcarrier in the allocated REs, we copy the first
        # DMRS at that subcarrier to all symbols in 'symSet'
        for p,portNo in enumerate(self.pxsch.portSet):
            if portNo not in self.portSet:  continue
            refRE = ptrsRefREs[self.dmrs.configType][portNo][self.reOffset]
            for l in self.symSet:
                # Note that kc below is continuous in the whole BWP. (PRB-based)
                # The actual RE index (k below) is in the GRB indexing of 'grid'
                k0 = refRE + 12*refRB
                for kc in range(k0, 12*self.pxsch.bwp.numRbs, 12*self.freqDensity):
                    if kc//12 not in self.pxsch.prbSet: continue
                    k = 12*self.pxsch.prb2Grb[ kc//12 ] + kc%12
                    curReType = grid.reTypeAt(p, l, k)
                    if curReType in ["DMRS", "CSIRS_ZP", "CSIRS_NZP", "RESERVED"]: continue
                    if curReType not in ["UNASSIGNED", "PTRS"]:
                        raise ValueError(f"Trying to allocate the RE at ({p},{l},{k}) for PT-RS," +
                                         f"while it is currently allocated for \"{curReType}\"!")
                    grid[p,l,k] = (beta * self.dmrsL0Values[portNo][k], "PTRS")

        # Set the PT-RS REs in other ports to NO_DATA:
        ptrsIdx = grid.getReIndexes("PTRS")
        for p,portNo in enumerate(self.pxsch.portSet):
            if portNo in self.portSet:  continue            # These are set to PT-RS already.
            for l,k in zip(ptrsIdx[1],ptrsIdx[2]):
                curReType = grid.reTypeAt(p, l, k)
                if curReType in ["DMRS", "CSIRS_ZP", "CSIRS_NZP", "RESERVED"]: continue
                if curReType not in ["UNASSIGNED"]:
                    raise ValueError(f"Trying to set the RE at ({p},{l},{k}) to 'NO_DATA'" +
                                     f"while it is currently allocated for \"{curReType}\"!")
                grid[p, l, k] = "NO_DATA"
