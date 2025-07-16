# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``csirs.py`` implements the Channel State Information Reference Signals (CSI-RS) based on **3GPP TS 38.211**
and **3GPP TS 38.214**. CSI-RS is implemented in **NeoRadium** in the following 3 classes:

    * :py:class:`CsiRs`: Implements one CSI-RS resource which can be used for both Zero-Power (ZP) and Non-Zero-Power
      (NZP) resources.
        
    * :py:class:`CsiRsSet`: Implements a CSI-RS Resource Set which contains one or more CSI-RS resources.
        
    * :py:class:`CsiRsConfig`: Implements the overall CSI-RS configuration information. It contains one or more CSI-RS 
      resource sets.
  
**NeoRadium**'s API implementation provides both flexibility and simplicity. Here are a couple of different ways to 
configure CSI-RS.

**Flexible Configuration:**
This method provides the most flexibility in configuration. Different numbers and types of CSI-RS resources can be 
configured in different numbers and types of CSI-RS resource sets. For example, the following code configures two 
CSI-RS resource sets, one ZP and one NZP, with one and two CSI-RS resources, respectively. Each CSI-RS resource is 
configured individually with different settings.

.. code-block:: python
    :caption: An example of flexibility in CSI-RS configuration

    from neoradium import Carrier, CsiRsConfig, CsiRsSet, CsiRs

    # Create the carrier and bandwidth part objects
    carrier = Carrier(startRb=0, numRbs=24, spacing=15)
    bwp = carrier.bwps[0]

    # Create a ZP CSI-RS resource and add it to the ZP CSI-RS resource set
    zpCsiRs = CsiRs(offset=1, symbols=[1], numPorts=1, freqMap="1000", density=0.5)
    zpCsiRsSet = CsiRsSet("ZP", bwp, csiRsList=[zpCsiRs], resourceType='semiPersistent', period=10)

    # Create two NZP CSI-RS resources and add them to the NZP CSI-RS resource set
    nzpCsiRs1 = CsiRs(offset=0, symbols=[1], numPorts=1, freqMap="000000001000", density=3)
    nzpCsiRs2 = CsiRs(offset=3, symbols=[3], numPorts=1, freqMap="1000", density=1)
    nzpCsiRsSet = CsiRsSet("NZP", bwp, csiRsList=[nzpCsiRs1, nzpCsiRs2], resourceType='periodic', period=5)

    # Now create the CSI-RS configuration using the ZP and NZP resource sets
    csiRsConfig = CsiRsConfig([zpCsiRsSet, nzpCsiRsSet])
    
**Easy and simple configuration:**
You can quickly create a typical CSI-RS configuration without dealing with the complexity of various CSI-RS 
configuration parameters. For example, the following code configures CSI-RS with one NZP and one ZP resource.

.. code-block:: python
    :caption: An example of simplicity in CSI-RS configuration

    from neoradium import Carrier, CsiRsConfig, CsiRsSet, CsiRs

    # Create the carrier and bandwidth part objects
    carrier = Carrier(startRb=0, numRbs=24, spacing=15)
    bwp = carrier.bwps[0]

    # Create the CSI-RS configuration with a single call. The CSI-RS resource and
    # resource set are automatically configured using the parameters provided.
    csiRsConfig = CsiRsConfig(csiType="NZP", bwp=bwp, symbols=[1], numPorts=1, freqMap="000000001000", density=3)

For more examples, see :doc:`../Playground/Notebooks/CSI-RS/CSI-RS`
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 12/01/2023    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np

from .utils import goldSequence, toLinear

# **********************************************************************************************************************
# This implementation is based on:
#   TS 38.211 V17.0.0 (2021-12)
#   TS 38.214 V17.0.0 (2021-12)
# Also see:
#   https://www.sharetechnote.com/html/5G/5G_CSI_RS.html
#   The book: 5G NR The next generation wireless access technology, Section 8.1

# **********************************************************************************************************************
# TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-1
# cdmSize: gives info about cdmType.
#   Examples: 1->noCDM   2->fd-CDM2   4->cdm4-FD2-TD2   8->cdm8-FD2-TD4
# kBar,lBar: 4 digit strs.
#   Examples: "1101"->(k1,l0+1) "0201"->(k0,l0+1),(k1,l0+1),(k2,l0+1) "4000"->(k0+4,l0) "0210"->(k0,l1),(k1,l1),(k2,l1)
csiRsLocations = [
    None,   # Row 0 not used
    # ports  Density  cdmSize   kBar,lBar                   cdmGroup Index      kPrime      lPrime          Row
    [ 1,     [3],     1,        "0000 4000 8000",           [0,0,0],            [0],        [0]       ],    # 1 (TRS)
    [ 1,     [1,.5],  1,        "0000",                     [0],                [0],        [0]       ],    # 2
    [ 2,     [1,.5],  2,        "0000",                     [0],                [0,1],      [0]       ],    # 3
    [ 4,     [1],     2,        "0000 2000",                [0,1],              [0,1],      [0]       ],    # 4 (FR2)
    [ 4,     [1],     2,        "0000 0001",                [0,1],              [0,1],      [0]       ],    # 5
    [ 8,     [1],     2,        "0300",                     [0,1,2,3],          [0,1],      [0]       ],    # 6
    [ 8,     [1],     2,        "0100 0101",                [0,1,2,3],          [0,1],      [0]       ],    # 7
    [ 8,     [1],     4,        "0100",                     [0,1],              [0,1],      [0,1]     ],    # 8
    [ 12,    [1],     2,        "0500",                     list(range(6)),     [0,1],      [0]       ],    # 9
    [ 12,    [1],     4,        "0200",                     [0,1,2],            [0,1],      [0,1]     ],    # 10
    [ 16,    [1,.5],  2,        "0300 0301",                list(range(8)),     [0,1],      [0]       ],    # 11
    [ 16,    [1,.5],  4,        "0300",                     [0,1,2,3],          [0,1],      [0,1]     ],    # 12
    [ 24,    [1,.5],  2,        "0200 0201 0210 0211",      list(range(12)),    [0,1],      [0]       ],    # 13
    [ 24,    [1,.5],  4,        "0200 0210",                list(range(6)),     [0,1],      [0,1]     ],    # 14
    [ 24,    [1,.5],  8,        "0200",                     [0,1,2],            [0,1],      [0,1,2,3] ],    # 15
    [ 32,    [1,.5],  2,        "0300 0301 0310 0311",      list(range(16)),    [0,1],      [0]       ],    # 16
    [ 32,    [1,.5],  4,        "0300 0310",                list(range(8)),     [0,1],      [0,1]     ],    # 17
    [ 32,    [1,.5],  8,        "0300",                     [0,1,2,3],          [0,1],      [0,1,2,3] ]]    # 18

# **********************************************************************************************************************
wFwTSequences = {
                    1:  # TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-2 ('noCDM', cdmSize=1)
                        #    Wf(0)       Wt(0)           Index (s)
                        [ [  [ 1 ],      [ 1 ]  ] ], #   0
                        
                    2:  # TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-3 ('fd-CDM2', cdmSize=2)
                        #      Wf(0)  Wf(1)    Wt(0)            Index (s)
                        [ [  [   1,     1   ], [ 1 ] ],     #   0
                          [  [   1,    -1   ], [ 1 ] ] ],   #   1
                          
                    4:  # TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-4 ('cdm4-FD2-TD2', cdmSize=4)
                        #      Wf(0)  Wf(1)      Wt(0)  Wt(1)             Index (s)
                        [ [  [   1,     1   ], [   1,     1   ]],     #   0
                          [  [   1,    -1   ], [   1,     1   ]],     #   1
                          [  [   1,     1   ], [   1,    -1   ]],     #   2
                          [  [   1,    -1   ], [   1,    -1   ]] ],   #   3
                          
                    8:  # TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-5 ('cdm8-FD2-TD4', cdmSize=8)
                        #      Wf(0)  Wf(1)      Wt(0)  Wt(1)  Wt(2)  Wt(3)             Index (s)
                        [ [  [   1,     1   ], [   1,     1,     1,     1   ]],     #   0
                          [  [   1,    -1   ], [   1,     1,     1,     1   ]],     #   1
                          [  [   1,     1   ], [   1,    -1,     1,    -1   ]],     #   2
                          [  [   1,    -1   ], [   1,    -1,     1,    -1   ]],     #   3
                          [  [   1,     1   ], [   1,     1,    -1,    -1   ]],     #   4
                          [  [   1,    -1   ], [   1,     1,    -1,    -1   ]],     #   5
                          [  [   1,     1   ], [   1,    -1,    -1,     1   ]],     #   6
                          [  [   1,    -1   ], [   1,    -1,    -1,     1   ]] ]    #   7
                }
                
# **********************************************************************************************************************
class CsiRs:
    r"""
    This class implements a CSI-RS resource which can be used to represent a Zero-Power (ZP - see **3GPP TS 38.214 
    Section 5.1.4.2**) or Non-Zero-Power (NZP - see **3GPP TS 38.214 Section 5.2.2.3.1**) resource."
    """
    # ******************************************************************************************************************
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments.
            
                :resourceId: The resource identifier of this CSI-RS resource. This is set to zero by default. This 
                    represents the values ``zp-CSI-RS-ResourceId`` or ``nzp-CSI-RS-ResourceId`` in **3GPP TS 38.214** 
                    for ZP and NZP resource types, respectively.
                    
                :offset: The slot offset for this CSI-RS resource. It is set to zero by default. This is the value 
                    :math:`T_{offset}` in **3GPP TS 38.211 section 7.4.1.5.3**. It is only used if the ``resourceType``
                    parameter of :py:class:`CsiRsSet` class containing this CSI-RS resource is set to ``'periodic'`` or
                    ``'semiPersistent'``. See :ref:`Time-Domain Configuration <TimeDomainConfig>` below for more 
                    information.
                    
                :numPorts: The number of antenna ports used by this CSI-RS resource. It can be one of 1, 2, 4, 8, 12, 
                    16, 24, or 32. This is the value :math:`X` in **3GPP TS 38.211 Table 7.4.1.5.3-1**.
                   
                :cdmSize: The CDM size of this CSI-RS resource. It can be one of 1, 2, 4, or 8 corresponding to 
                    ``noCDM``, ``fd-CDM2``, ``cdm4-FD2-TD2``, and ``cdm8-FD2-TD4`` in **3GPP TS 38.211 
                    Table 7.4.1.5.3-1**.
                    
                :density: The frequency density of the CSI-RS resource. It can be one of 0.5, 1, or 3. The value 3 is 
                    only available if ``numPorts`` is set to 1. This is the value :math:`\rho` in **3GPP TS 38.211 
                    Table 7.4.1.5.3-1**.
                    
                :freqMap: A bitmap string that specifies the frequency-domain locations of this CSI-RS resource. See
                    **3GPP TS 38.211 Section 7.4.1.5.3** for more information. This value determines which resource
                    elements in each resource block are used by this CSI-RS resource.
                    
                :symbols: The list of time symbol indices in each slot that are used by this CSI-RS resource.
                    
                :powerDb: The power (in dB) used by this CSI-RS resource for the NZP resources. This is ignored
                    for Zero-Power (ZP) resources. The default value is 0. This contains the value of 
                    :math:`\beta_{CSIRS}` as specified in **3GPP TS 38.211 Section 7.4.1.5.3**.
                    
                :scramblingID: The scrambling identity which is used for generating pseudo-random sequences. The
                    default is 0. This is ignored for ZP resources.


        **Other Properties:**
        
        All of the parameters mentioned above except ``freqMap`` and ``symbols`` are directly available. Here is a 
        list of additional properties:
        
            :ls: The time-domain symbol indices used by this CSI-RS resource. This is a list of integers that is
                extracted from the ``symbols`` parameter explained above. This list contains the :math:`l_0` and
                :math:`l_1` values in the 5\ :sup:`th` column of **3GPP TS 38.211 Table 7.4.1.5.3-1** (The column 
                titled :math:`(\bar k, \bar l)`).
                
            :ks: The frequency-domain RE indices used by this CSI-RS resource. This is a list of integers that is
                extracted from the ``freqMap`` parameter explained above. This list contains the :math:`k_0`,
                :math:`k_1`, :math:`k_2`, and :math:`k_3` values in the 5\ :sup:`th` column of **3GPP TS 38.211
                Table 7.4.1.5.3-1** (The column titled :math:`(\bar k, \bar l)`).
                
            :row: The row index of **3GPP TS 38.211 Table 7.4.1.5.3-1** corresponding to the configuration of this
                CSI-RS resource (The column titled "Row"). This class determines the row index using the parameters
                ``numPorts``, ``cdmSize``, ``density``, and ``freqMap``.
                   
            :mySet: The :py:class:`CsiRsSet` object containing this CSI-RS resource.


        Additionally you can access the :py:class:`CsiRsSet` class parameters ``period``, ``bwp``, ``csiType``,
        ``resourceType``, ``active``, ``startRb``, and ``numRbs`` directly. The parameter ``mySet`` is internally
        used to return these values from the :py:class:`CsiRsSet` class containing this CSI-RS resource.
        """
        self.resourceId = kwargs.get('resourceId', 0)               # zp-CSI-RS-ResourceId  or nzp-CSI-RS-ResourceId

        # The slot offset can be different for different CsiRs objects. But the period is the same for all CsiRs
        # objects in a CsiRsSet. This only applies when the CsiRsSet::resourceType is one of "periodic" or
        # "semiPersistent" For more info see TS 38.214 V17.0.0 (2021-12), Section 5.2.2.3.1
        self.offset = kwargs.get('offset', 0)

        self.numPorts = kwargs.get('numPorts', 1)                   # Number of CSI-RS Ports
        if self.numPorts not in [1,2,4,8,12,16,24,32]:
            raise ValueError("Invalid CSI-RS 'numPorts' value! numPorts ∈ {1,2,4,8,12,16,24,32}")

        self.cdmSize = kwargs.get('cdmSize', min(self.numPorts,2))
        if self.cdmSize not in [1,2,4,8]:
            raise ValueError("Invalid CSI-RS 'cdmSize' value! cdmSize ∈ {1,2,4,8}")

        self.density = kwargs.get('density', 1)                     # The frequency density 𝜌.
        validDensities = [1] if self.numPorts in [4,8,12] else ([0.5,1,3] if self.numPorts==1 else [0.5,1])
        if self.density not in validDensities:
            raise ValueError("Invalid CSI-RS 'density' value! density ∈ {%s}"%(",".join(str(x) for x in validDensities)))

        kMap = kwargs.get('freqMap', self.getDefaultKmap())         # The RE bitmap (See TS 38.211 V17.0.0 (2021-12), Section 7.4.1.5.3)
        self.row, self.ks = self.getRow(kMap)                       # Validate the values and infer the row number in table 7.4.1.5.3-1

        if self.row in [13,14,16,17]:
            # Both l0 and l1 needed
            self.ls = kwargs.get('symbols', [3,9])      # The symbol indices in the slot where CSI-RS are located (l0 and l1)
            if len(self.ls)!=2:
                raise ValueError("Second CSI-RS symbol index is missing!")
            if self.ls[0] not in range(0,14):
                raise ValueError("Invalid CSI-RS first symbol index value! l0 ∈ {0,1,...,13}")
            if self.ls[1] not in range(2,13):
                raise ValueError("Invalid CSI-RS second symbol index value! l1 ∈ {2,3,...,12}")
        else:
            # Only l0 is needed
            self.ls = kwargs.get('symbols', [5])        # The symbol indices in the slot where CSI-RS are located (l0)
            if len(self.ls)!=1:     print("Warning: Only the first specified CSI-RS symbol index will be used!")
            elif self.ls[0] not in range(0,14): raise ValueError("Invalid CSI-RS symbol index value! l0 ∈ {0,1,...,13}")
        
        # Note about "ks" and "ls":
        # self.ks contains the k0, k1, k2, k3 values in the 5th column of Table 7.4.1.5.3-1
        # self.ls contains the l0, l1 values in the 5th column of Table 7.4.1.5.3-1

        self.powerDb = kwargs.get('powerDb', 0) # This is the power for the NZP CSI-RS symbols (𝛃CSIRS) (Only for NZP)
        self.scramblingID = kwargs.get('scramblingID', 0) # Used for generating pseudo-random sequences (Only for NZP)
        
        self.mySet = None       # This will be set by the CsiRsSet object

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CSI-RS resource.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If ``None`` (default), the text
            "CSI-RS Properties:" is used for the title.

        getStr : Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "CSI-RS Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  resourceId:         {self.resourceId}\n"
        repStr += indent*' ' + f"  numPorts:           {self.numPorts}\n"
        cdmSizeNames = {1:'noCDM', 2:'fd-CDM2', 4:'cdm4-FD2-TD2', 8:'cdm8-FD2-TD4'}
        repStr += indent*' ' + f"  cdmSize:            {self.cdmSize} ({cdmSizeNames[self.cdmSize]})\n"
        repStr += indent*' ' + f"  density:            {self.density}\n"
        repStr += indent*' ' + f"  RE Indexes:         {'  '.join(str(ki) for ki in self.ks)}\n"
        repStr += indent*' ' + f"  Symbol Indexes:     {'  '.join(str(li) for li in self.ls)}\n"
        repStr += indent*' ' + f"  Table Row:          {self.row}\n"
        if self.resourceType in ['semiPersistent', 'periodic']:
            repStr += indent*' ' + f"  Slot Offset:        {self.offset}\n"
        if self.csiType == "NZP":
            repStr += indent*' ' + f"  Power:              {self.powerDb} dB\n"
            repStr += indent*' ' + f"  scramblingID:       {self.scramblingID}\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def __getattr__(self, property):            # Not documented (Already documented above)
        # Get these properties from the 'mySet' object
        if property not in ["period", "bwp", "csiType", "resourceType", "active", "startRb", "numRbs"]:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.mySet, property)

    # ******************************************************************************************************************
    def getDefaultKmap(self):                                       # Not documented
        # Based on the number of ports returns a valid frequency map that can be used as default.
        return {
                1: '000000001000' if self.density==3 else '1000',   # k0=3                  (Row 1 or 2)
                2: '001000',                                        # k0=3                  (Row 3)
                4: '010',                                           # k0=4                  (Row 4)
                8: '010100',                                        # k0,k1=4,8             (Row 7 or 8)
                12:'111111' if self.cdmSize==2 else '101010',       # k0,k1,k2,k3,k4,k5=0,2,4,6,8,10 for cdm2 or
                                                                    # k0,k1,k2=2,6,10 for cdm4 (Row 9 or 10)
                16:'110011',                                        # k0,k1,k2,k3=0,2,8,10  (Row 11 or 12)
                24:'101010',                                        # k0,k1,k2=2,6,10       (Row 13, 14, or 15)
                32:'110011'                                         # k0,k1,k2,k3=0,2,8,10  (Row 16, 17, or 18)
               }[self.numPorts]

    # ******************************************************************************************************************
    def getRow(self, kMap):                                         # Not documented
        # Given the number of ports, the frequency map, density, and CDM Size, this function finds the correct row
        # in 3GPP TS 38.211 Table 7.4.1.5.3-1. See TS 38.211 V17.0.0 (2021-12), Section 7.4.1.5.3 for more info"
        validNumKs, validLens = {
                                    1: ([1],[12]) if self.density==3 else ([1],[4]),
                                    2: ([1],[6]),
                                    4: ([1],[3,6]),
                                    8: ([2,4],[6]),
                                    12:([3,6],[6]),
                                    16:([4],[6]),
                                    24:([3],[6]),
                                    32:([4],[6])
                                }[self.numPorts]
                                
        numKs = sum(int(bit) for bit in kMap)
        mapLen = len(kMap)
        if numKs not in validNumKs:
            raise ValueError("Invalid combination of CSI-RS parameters. See TS 38.211 V17, Table 7.4.1.5.3-1")
        if mapLen not in validLens:
            raise ValueError("Invalid combination of CSI-RS parameters. See TS 38.211 V17, Table 7.4.1.5.3-1")

        row = {
                1: 1 if self.density==3 else 2,
                2: 3,
                4: 4 if len(kMap)==3 else 5,
                8: 6 if numKs==4 else {1:-1, 2:7, 4:8, 8:-1}[self.cdmSize],
                12: {1:-1, 2:9,  4:10, 8:-1}[self.cdmSize],
                16: {1:-1, 2:11, 4:12, 8:-1}[self.cdmSize],
                24: {1:-1, 2:13, 4:14, 8:15}[self.cdmSize],
                32: {1:-1, 2:16, 4:17, 8:18}[self.cdmSize]
              }[self.numPorts]

        if row in [1,2]:    ks=[i for i in range(mapLen) if kMap[mapLen-i-1]=='1']
        elif row==4:        ks=[4*i for i in range(mapLen) if kMap[mapLen-i-1]=='1']
        else:               ks=[2*i for i in range(mapLen) if kMap[mapLen-i-1]=='1']
                    
        return row, ks

    # ******************************************************************************************************************
    def anythingForCurSlot(self):                                   # Not documented
        # Return True if this CSI-RS resource has any allocations in the current slot in the bandwidth part
        if self.resourceType == 'aperiodic':        return self.active
        if self.resourceType == 'semiPersistent':
            if self.active == False:
                return False
        return ((self.bwp.slotNo - self.offset)%self.period)==0

    # ******************************************************************************************************************
    def populateGrid(self, grid):                                   # Not documented
        # This function calculates the reference signal based on the configuration of this CSI-RS resource and updates
        # the corresponding resource elements in the given Grid object.
        if self.anythingForCurSlot() == False:  return
        
        _, _, _, klBarsStr, cdmGroupIndexes, kPrimes, lPrimes = csiRsLocations[ self.row ]
        klBarStrs = klBarsStr.split(' ')
        klBarPairs = []     # The pairs in the 5th column of "TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-1"
        for klBarStr in klBarStrs:
            k1st, kLast, lIndex, ll = [int(c) for c in klBarStr]
            if k1st>kLast:  klBarPairs += [ (self.ks[0]+k1st, self.ls[lIndex]+ll) ]     # Special case in rows 1 and 4
            else:           klBarPairs += [ (self.ks[kk], self.ls[lIndex]+ll) for kk in range(k1st, kLast+1)]

        jkBarsForLBars = {}                         # Keys: lBar, Values: list of (j,kBar) pairs for the lBar
        for j,(kBar,lBar) in enumerate(klBarPairs): # j, kBar, lBar as in TS 38.211 V17.0.0 (2021-12), Table 7.4.1.5.3-1
            jkBarsForLBars[lBar] = jkBarsForLBars.get(lBar,[]) + [(j*(self.row!=1), kBar)]  # For first row, j = 0
        
        if self.row == 1:   symsPerRB = 3            # For row 1, we need 3 symbols per RB and all RBs are used
        else:               symsPerRB = len(kPrimes) # For row 2, we need 1 symbol, otherwise 2 symbols per RB
        bitsPerRB = symsPerRB*2                      # 2 bits per symbols
        
        totalRBs = self.startRb+self.numRbs                                     # Total RBs from CRB 0 to the end
        totalRBsUsed = totalRBs if self.density in [1,3] else (totalRBs+1)//2   # For density 0.5, half the RBs are used
        totalBits = totalRBsUsed * bitsPerRB                                    # Total bits from CRB 0
                
        # CSI-RS Beta
        csirsBeta = toLinear(self.powerDb/2)
        alpha = int(np.round(2*self.density) if self.numPorts>1 else self.density)  # Alpha is 1,2, or 3
        wFwTtable = wFwTSequences[self.cdmSize]

        for lBar, jkBars in jkBarsForLBars.items():
            for lPrime in lPrimes:
                l = lBar + lPrime

                if self.csiType == "NZP":
                    # Generate sequence of bits (pseudo-random) (See TS 38.211 V17.0.0 (2021-12), Section 7.4.1.5.2)
                    cInit = ((1<<10)*(self.bwp.symbolsPerSlot * self.bwp.slotNoInFrame+l+1)*(2*self.scramblingID+1) +
                             self.scramblingID) & 0x7FFFFFFF
                    symbolBits = goldSequence(cInit, totalBits)

                    # Convert every 2 bits to one complex value per RE:
                    rawSymbols = (1-2*np.float64(symbolBits).reshape(-1,2))/np.sqrt(2)
                    # 'rawSymbols' is r(m) in TS 38.211 V17.0.0 (2021-12), Section 7.4.1.5.2
                    rawSymbols = rawSymbols[:,0] + 1j*rawSymbols[:,1]

                for n in range(self.startRb, self.startRb+self.numRbs):
                    if (self.density<1) and (n%2==1):   continue    # Every other RB is populated when density is 0.5
                    for j, kBar in jkBars:
                        for kPrime in kPrimes:
                            mPrime = int( np.floor(n*alpha) + kPrime + np.floor(kBar*self.density/12) )
                            kCbr0 = 12*n + kBar + kPrime            # kCbr0 is the subcarrier index from CRB 0
                            k = kCbr0 - 12*grid.startRb             # The subcarrier index in the grid
                            for s, (wfs,wts) in enumerate(wFwTtable):   # s goes from 0 to (cdmSize-1)
                                wf = wfs[kPrime]
                                wt = wts[lPrime]
                                p = s + j*self.cdmSize
                                curReType = grid.reTypeAt(p,l,k)
                                assert curReType in ["UNASSIGNED", "RESERVED"], \
                                        "Assigning \"%s CSI-RS\" to the RE(%d,%d,%d) which is already allocated for \"%s\"!"%(self.mySet.csiType, p,l,k, curReType)
                                if self.mySet.csiType == "ZP":
                                    grid[p,l,k] = (0, "CSIRS_ZP")
                                else:
                                    grid[p,l,k] = (csirsBeta * wf * wt * rawSymbols[mPrime], "CSIRS_NZP")
                                if grid.reDesc is not None:
                                    if self.csiType == "ZP":
                                        grid.reDesc[p,l,k] = "CSI-RS,ZP"
                                    else:
                                        grid.reDesc[p,l,k] = "CSI-RS,NZP,%s"%(self.csiType, '+' if wf*wt>0 else '-')
        
# **********************************************************************************************************************
class CsiRsSet:  # A CSI-RS Resource Set (contains several CSI-RS Resources)
    r"""
    This class implements a CSI-RS resource set which contains one or more CSI-RS resources (:py:class:`CsiRs` 
    objects). A CSI-RS resource set can be either Zero-Power (ZP) or Non-Zero-Power (NZP). All CSI-RS resources
    in a set are of the same type (ZP or NZP). By default, a CSI-RS resource set is configured with one CSI-RS resource.
    More CSI-RS resources can be added using the :py:meth:`addCsiRs` method.
    """
    # ******************************************************************************************************************
    def __init__(self, csiType, bwp, **kwargs):
        r"""
        Parameters
        ----------
        csiType : str
            The type of this CSI-RS resource set. It **must** be either ``'NZP'`` or ``'ZP'``.
            
        bwp : :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part used by this CSI-RS resource set.
            
        kwargs : dict
            A set of optional arguments.
            
                :rsId: The resource set identifier of this CSI-RS resource set. This is set to zero by default. This
                    represents the values ``nzp-CSI-ResourceSetId`` or ``zp-CSI-RS-ResourceSetId`` in **3GPP TS
                    38.214** for NZP and ZP resource sets correspondingly.
                    
                :startRb: The index of starting resource block (RB) used by this CSI-RS resource set. By default this
                    is set to ``bwp.startRb``. The resources specified by ``startRb`` and ``numPorts`` **must** be
                    inside the specified bandwidth part ``bwp``.
                    
                :numRbs: The number of resource blocks used by this CSI-RS resource set starting at ``startRb``. By
                    default this is set to ``bwp.numRbs``. The resources specified by ``startRb`` and ``numPorts``
                    **must** be inside the specified bandwidth part ``bwp``.
                    
                :resourceType: The time-domain resource type of this CSI-RS resource set. It can be one of
                    ``'aperiodic'``, ``'semiPersistent'``, or ``'periodic'`` (default). See :ref:`Time-Domain
                    Configuration <TimeDomainConfig>` below for more information.

                :period: The period (in number of slots) of this CSI-RS resource set in time domain. This is the value :math:`T_{CSI-RS}` in **3GPP TS 38.211 section 7.4.1.5.3**. By default, it is set to 4 which means
                    CSI-RS is transmitted every 4\ :sup:`th` slot (for example slots numbers 0, 4, 8, ...). The
                    ``period`` can be 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, or 640 slots. This parameter is
                    ignored if ``resourceType`` is set to ``'aperiodic'``. See  
                    :ref:`Time-Domain Configuration <TimeDomainConfig>` below for more information.
                    
                :active: This boolean flag is used to *trigger* an ``'aperiodic'`` or *activate* a ``'semiPersistent'``
                    CSI-RS resource set. It is ignored if ``resourceType`` is set to ``'periodic'``. This is set to
                    ``True`` by default. See :ref:`Time-Domain Configuration <TimeDomainConfig>` below for more
                    information.
                    
                :csiRsList: A list of :py:class:`CsiRs` objects contained in this CSI-RS resource set. If not 
                    specified, this class creates a single CSI-RS resource and puts it in the list. More 
                    :py:class:`CsiRs` objects can be added using the :py:meth:`addCsiRs` method.


        **Other Properties:**
        
        All of the parameters mentioned above are directly available. Here is a list of additional properties:
        
            :numPorts: This read-only parameter returns the maximum number of ports of all CSI-RS resources in this
                CSI-RS resource set.


        .. _TimeDomainConfig:
        
        **Time-Domain Configuration:**
        
        In time-Domain, CSI-RS can be configured for *periodic*, *semi-persistent*, or *aperiodic* transmission.
        
        :periodic: In this case, the CSI-RS transmission occurs every N\ :sup:`th` slot, where N ranges from 4 to
            640. In this case, each CSI-RS resource is also configured with an ``offset`` value. Please note that
            while the ``period`` N is the same for all CSI-RS resources in a set, the ``offset`` can be different for
            different CSI-RS resources in the set. That is why the ``period`` is a property if the CSI-RS resource
            set while ``offset`` is a property of CSI-RS resource.
            
            .. figure:: ../Images/CSI-RS-Timing.png
                :align: center
                :figwidth: 600px

                The time-domain ``period`` and ``offset`` examples [1]_

        :semi-persistent: In this case, the periodicity and offset of the CSI-RS transmissions is configured similar
            to the *periodic* case but the actual transmission can be disabled/enabled using the ``active`` flag
            defined above. Once *activated*, the CSI-RS transmission behavior is just like the *periodic* case until
            it is *deactivated*. All CSI-RS resources in the set are activated/deactivated together. (The ``active``
            flag is a property of CSI-RS resource set)
            
        :aperiodic: In this case, there is no periodicity. The transmission of the CSI-RS can be *triggered* by
            setting the ``active`` flag to ``True``. In practice this is signaled in the DCI message. All CSI-RS
            resources in the set are *triggered* together. (The ``active`` flag is a property of CSI-RS resource set)
            
        Please refer to **3GPP TS 38.211 section 7.4.1.5.3** for more details. See also 
        :doc:`../Playground/Notebooks/CSI-RS/CSI-RS-Time`.
        """
        self.rsId = kwargs.get('rsId', 0)   # Resource Set Id  (nzp-CSI-ResourceSetId or zp-CSI-RS-ResourceSetId)

        self.bwp = bwp      # Same BWP is used for all CsiRs objects in this CsiRsSet
        
        # Resource Mapping parameters (CSI-RS-ResourceMapping)
        self.startRb = kwargs.get('startRb', self.bwp.startRb)  # Number of RBs from CRB 0.
        self.numRbs = kwargs.get('numRbs', self.bwp.numRbs)     # Number of RBs for this CSI-RS.
        if (self.startRb < self.bwp.startRb) or (self.startRb+self.numRbs > self.bwp.startRb+self.bwp.numRbs):
            raise ValueError("Invalid CSI-RS config! The whole CSI-RS resources must be inside the Bandwidth Part.")

        # Since everything still works without this condition, we are not enforcing it here. So it is still ok if
        # 'startRb' and 'numRbs' are not multiples of 4.
        # if (self.startRb%4)>0 or (self.numRbs%4)>0:
        #     raise ValueError("Invalid CSI-RS config! 'startRb' and 'numRbs' must be multiples of 4.")

        self.csiType = csiType                                  # Can be "ZP" or "NZP"
        if self.csiType not in ["ZP","NZP"]:        raise ValueError("Invalid CSI-RS type! csiType ∈ {\"ZP\",\"NZP\"}")

        self.resourceType = kwargs.get('resourceType', 'periodic')
        if self.resourceType not in ['aperiodic', 'semiPersistent', 'periodic']:
            raise ValueError("Invalid CSI-RS 'resourceType' value! resourceType ∈ {'aperiodic', 'semiPersistent', 'periodic'}")

        # The slot offset can be different for different CsiRs objects. But the period is the same for all CsiRs
        # objects in a CsiRsSet. This only applies when the CsiRsSet::resourceType is one of "periodic" or
        # "semiPersistent" For more info see TS 38.214 V17.0.0 (2021-12), Section 5.2.2.3.1
        self.period = kwargs.get('period', 4)   #  By default, CSI-RS present every 4 slots (slots numbers 0, 4, 8, ...)
        validPeriods = [4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640]
        if self.period not in validPeriods:
            raise ValueError(f"Invalid CSI-RS Resource Set 'period'! period ∈ {{{', '.join(str(i) for i in validPeriods)}}}")
        self.active = kwargs.get('active', True)    # Used for 'aperiodic' (Trigger) and 'semiPersistent' (Activate)

        if self.csiType=="NZP":
            # These are the parameters specific to the "NZP" type (NOTE: These are not currently used and therefore
            # not documented)
            self.repetition = kwargs.get('repetition', True)    # True means All CsiRs in this CsiRsSet are transmitted
                                                                # with same downlink spatial domain transmission filter
            self.trigOffset = kwargs.get('trigOffset', 0)
            if self.trigOffset not in range(5):
                raise ValueError("Invalid CSI-RS 'aperiodic triggering offset'! (It must be between 0 and 4)")

            # 'trs'=True => NZP CSI-RS used for tracking (See TS 38.214 V17.0.0 (2021-12), Section 5.1.6.1.1)
            self.trs = kwargs.get('trs', False)
            if (self.trs==True) and (self.csiType=="ZP"):
                raise ValueError("Invalid CSI-RS trs setting! It can only be used with \"NZP\" CSI-RS.")

        self.csiRsList = []
        self.addCsiRs( kwargs.get('csiRsList', [ CsiRs(**kwargs) ]) )

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CSI-RS resource set.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If ``None`` (default), the text
            "CSI-RS Resource Set Properties:" is used for the title.

        getStr : Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "CSI-RS Resource Set Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + f"({len(self.csiRsList)} {self.csiType} Resources)\n"
        repStr += indent*' ' + f"  Resource Set ID:      {self.rsId}\n"
        repStr += indent*' ' + f"  Resource Type:        {self.resourceType}\n"
        repStr += indent*' ' + f"  Resource Blocks:      {self.numRbs} RBs starting at {self.startRb}\n"
        if self.resourceType in ['semiPersistent', 'periodic']:
            repStr += indent*' ' + f"  Slot Period:          {self.period}\n"
        if self.resourceType in ['aperiodic', 'semiPersistent']:
            repStr += indent*' ' + f"  active:               {self.active}\n"
        repStr += self.bwp.print(indent+2, "Bandwidth Part:", True)
        for csiRs in self.csiRsList:
            repStr += csiRs.print(indent+2, "CSI-RS:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def addCsiRs(self, csiRsList):
        r"""
        Adds one or more CSI-RS resources to this CSI-RS resource set.

        Parameters
        ----------
        csiRsList : list
            A list of :py:class:`CsiRs` objects to be added to this CSI-RS resource set.
        """
        for csiRs in csiRsList:
            if csiRs.offset not in range(self.period):
                raise ValueError("Invalid CSI-RS 'offset'! offset ∈ {0,1,...,%d}"%(self.period-1))
            csiRs.mySet = self
            self.csiRsList += [csiRs]

    # ******************************************************************************************************************
    @property                           # Not documented (Already documented above)
    def numPorts(self): return max(csiRs.numPorts for csiRs in self.csiRsList)
    
    # ******************************************************************************************************************
    def populateGrid(self, grid):       # Not documented
        # The method calls the "populateGrid" function of all CSI-RS resources.
        if (self.resourceType in ['aperiodic','semiPersistent']) and (not self.active): return
        for csiRs in self.csiRsList: csiRs.populateGrid(grid)
        
# **********************************************************************************************************************
class CsiRsConfig:  # CSI-RS Configuration (contains several CSI-RS Resource Sets)
    r"""
    This class implements the overall CSI-RS configuration. It keeps a list of CSI-RS resource sets 
    (:py:class:`CsiRsSet` objects) each of which contains one or more CSI-RS resources (:py:class:`CsiRs` objects). By
    default, this class creates a single CSI-RS resource set. More CSI-RS resource sets can be added using the 
    :py:meth:`addCsiResourceSets` method
    """
    # ******************************************************************************************************************
    def __init__(self, csiRsSetList=[], **kwargs):
        r"""
        Parameters
        ----------
        csiRsSetList : list
            A list of :py:class:`CsiRsSet` objects contained in this CSI-RS configuration. If this list is not
            specified, but a :py:class:`~neoradium.carrier.BandwidthPart` object is specified in ``kwargs`` this class
            creates a single CSI-RS resource set containing a single CSI-RS resource. The parameters passed in 
            ``kwargs`` are used to initialize the CSI-RS resource set and its only CSI-RS resource.
            
            If this list is not specified, and a :py:class:`~neoradium.carrier.BandwidthPart` object is not given,
            then the list of CSI-RS resource sets remains empty and you **must** use the :py:meth:`addCsiResourceSets`
            to add CSI-RS resource sets.
                        
        kwargs : dict
            A set of optional arguments. These parameters are passed to the :py:class:`CsiRsSet` and :py:class:`CsiRs`
            objects when they are created the first time.
            
        
        **Other Properties:**
        
        All of the parameters mentioned above are directly available. Here is a list of additional properties:
        
            :numPorts: This read-only parameter returns the maximum number of ports in all CSI-RS resources in all
                CSI-RS resource sets.
        """
        self.csiRsSetList = []      # A list of CsiRsSet objects
        
        if len(csiRsSetList)==0:    # There is no list.
            # First check to see if a bandwidth part is given. If so, create a single CsiRsSet which
            # includes a CsiRs object in it configured by the information in kwargs.
            # Otherwise, no CsiRsSet objects are added. The resource sets must be added later
            # using the addCsiResourceSets method.
            bwp = kwargs.get('bwp', None)
            if bwp is not None:
                del kwargs['bwp']
                csiType = kwargs.get('csiType', None)
                if csiType is None:     csiType = "NZP"
                else:                   del kwargs['csiType']
                csiRsSetList = [ CsiRsSet(csiType, bwp, **kwargs) ]

        self.addCsiResourceSets(csiRsSetList)
    
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CSI-RS Configuration.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If ``None`` (default), the text
            "CSI-RS Configuration:" is used for the title.

        getStr : Boolean
            If ``True``, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "CSI-RS Configuration:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + " (%d Resource Sets)\n"%(len(self.csiRsSetList))
        for csiRsSet in self.csiRsSetList:
            repStr += csiRsSet.print(indent+2, "CSI-RS Resource Set:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def addCsiResourceSets(self, csiRsSetList):
        r"""
        Adds one or more CSI-RS resource sets to this CSI-RS configuration.

        Parameters
        ----------
        csiRsSetList : list
            A list of :py:class:`CsiRsSet` objects to be added to this CSI-RS configuration.
        """
        for csiRsSet in csiRsSetList:
            self.csiRsSetList += [csiRsSet]

    # ******************************************************************************************************************
    def addCsiRs(self, setIndex=0, csiRs=None, **kwargs):
        r"""
        Adds the CSI-RS resource given in ``csiRs`` to the CSI-RS resource set specified by ``setIndex`` in this 
        CSI-RS configuration

        Parameters
        ----------
        setIndex : int
            The index of CSI-RS resource set in this CSI-RS configuration that receives the new CSI-RS resource.
            
        csiRs : :py:class:`CsiRs`
            If specified, the :py:class:`CsiRs` object is added to the CSI-RS resource set specified by ``setIndex``.
            Otherwise, a new CSI-RS resource object is created based on the information in ``kwargs`` and then it is
            added to the CSI-RS resource set specified by ``setIndex``.
            
        kwargs : dict
            These parameters are only used if this CSI-RS configuration is empty and/or if ``csiRs=None``.
            
            If this CSI-RS configuration is empty, the information in ``kwargs`` is first used to create a CSI-RS
            resource set.
            
            If ``csiRs`` is not specified, then the information in ``kwargs`` is first used to create a CSI-RS
            resource and add to the specified CSI-RS resource set.
        """
        if len(self.csiRsSetList)==0:
            # Empty config. Create a CsiRsSet.
            bwp = kwargs.get('bwp', None)
            if bwp is None:
                raise ValueError("You need to specify a bandwidth part 'bwp' when adding to an empty config!")
            csiType = kwargs.get('csiType', "NZP")
            if 'bwp' in kwargs:     del kwargs['bwp']
            if 'csiType' in kwargs: del kwargs['csiType']
            
            if csiRs is None:   csiRsSet = CsiRsSet(csiType, bwp, **kwargs)
            else:               csiRsSet = CsiRsSet(csiType, bwp, csiRsList=[csiRs])
            self.addCsiResourceSets( [csiRsSet] )
            return

        if setIndex >= len(self.csiRsSetList):
            raise ValueError(f"Invalid 'setIndex' value '{setIndex}'. setIndex < {len(self.csiRsSetList)}.")
            
        csiRsSet = self.csiRsSetList[setIndex]
        if csiRs is None:   csiRs = CsiRs(**kwargs)
        csiRsSet.addCsiRs( [ CsiRs(**kwargs) ] )

    # ******************************************************************************************************************
    def populateGrid(self, grid):
        r"""
        Uses the information in this CSI-RS configuration to calculate reference signal values and updates the 
        :py:class:`~neoradium.grid.Grid` object specified by ``grid``.

        Parameters
        ----------
        grid : :py:class:`~neoradium.grid.Grid`
            The :py:class:`~neoradium.grid.Grid` object that gets populated with CSI-RS based on this CSI-RS
            configuration.
        """
        if len(self.csiRsSetList)==0:
            raise ValueError("Cannot populate the grid because this 'CsiRsConfig' object is empty!")
        for csiRsSet in self.csiRsSetList:
            csiRsSet.populateGrid(grid)

    # ******************************************************************************************************************
    @property               # Not documented (Already documented above in the __init__ documentation)
    def numPorts(self): return max(csiRsSet.numPorts for csiRsSet in self.csiRsSetList)

    # ******************************************************************************************************************
    @property
    def bwp(self): return None if len(self.csiRsSetList)==0 else self.csiRsSetList[0].bwp

