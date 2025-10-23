# Copyright (c) 2024 InterDigital AI Lab
"""
The CSI Feedback implementation.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 06/05/2024    Shahab Hamidi-Rad       First version of the file.
# **********************************************************************************************************************
import numpy as np
from .antenna import AntennaElement, AntennaPanel, AntennaArray

# **********************************************************************************************************************
# This implementation is based on:
#   TS 38.214 V17.0.0 (2021-12)
# Also see:
#   https://www.sharetechnote.com/html/5G/5G_CSI_Report.html
#   The book: 5G NR The next generation wireless access technology, Sections 8.2, 11.2,


# Start with: (See hDLPMISelect.m)
#    1) getCodebook function
#    2) computeSINRPerRE function

def enumErrorMsg(var, valids):
    if type(valids)==list:
        formatStr = "'%s'" if type(valids[0])==str else "%s"
        return "Invalid '%s'! ('%s' ∈ {%s})"%(var, var, ", ".join([formatStr%str(x) for x in valids]))
    
    if type(valids)==tuple and len(valids)==2:
        formatStr = "'%s'" if type(valids[0])==str else "%s"
        return "Invalid '%s'! ('%s' ∈ {%s})"%(var, var, ",...,".join([formatStr%str(x) for x in valids]))
    
    formatStr = "'%s'" if type(valids)==str else "%s"
    return "Invalid '%s'! (It must be "%(var) + formatStr%str(valids) + ")"

def validateRange(var, valids, context="", varName=None):
    if varName is None:
        import inspect
        frame = inspect.getouterframes(inspect.currentframe())[1]
        string = inspect.getframeinfo(frame[0]).code_context[0].strip()
        varName = string[string.find('(') + 1:-1].split(',')[0].strip("self.")

    if type(valids)==list:
        if var in valids:                      return
        fStr = "'%s'" if type(valids[0])==str else "%s"
        raise ValueError("Invalid '%s'! ('%s' ∈ {%s}%s)"%(varName, varName,
                                                        ", ".join([fStr%str(x) for x in valids]), context))

    if type(valids)==tuple and len(valids)==2:
        if var in range(valids[0],valids[1]+1): return
        fStr = "'%s'" if type(valids[0])==str else "%s"
        raise ValueError("Invalid '%s'! ('%s' ∈ {%s}%s)"%(varName, varName,
                                                        ",...,".join([fStr%str(x) for x in valids]), context))

    if var==valids:                              return
    fStr = "'%s'" if type(valids)==str else "%s"
    raise ValueError("Invalid '%s'! (It must be "%(varName) + fStr%str(valids) + context + ")")

# Combinatorial coefficients: (See TS 38.214, Table 5.2.2.2.3-1)
#   y → 1   2    3    4         # x↓
cxy = [[0,  0,   0,   0],       #  0
       [1,  0,   0,   0],       #  1
       [2,  1,   0,   0],       #  2
       [3,  3,   1,   0],       #  3
       [4,  6,   4,   1],       #  4
       [5,  10,  10,  5],       #  5
       [6,  15,  20,  15],      #  6
       [7,  21,  35,  35],      #  7
       [8,  28,  56,  70],      #  8
       [9,  36,  84,  126],     #  9
       [10, 45,  120, 210],     #  10
       [11, 55,  165, 330],     #  11
       [12, 66,  220, 495],     #  12
       [13, 78,  286, 715],     #  13
       [14, 91,  364, 1001],    #  14
       [15, 105, 455, 1365]]    #  15

cqiTables = [None, # There is no table 0
             # TS 38.214, Table 5.2.2.1-2: 4-bit CQI Table 1
             # modulation  codeRate*1024   efficiency   CQI index
             [[None,       None,           None],       # 0: (Out of Range)
             [ 'QPSK',     78,             0.1523],     # 1
             [ 'QPSK',     120,            0.2344],     # 2
             [ 'QPSK',     193,            0.3770],     # 3
             [ 'QPSK',     308,            0.6016],     # 4
             [ 'QPSK',     449,            0.8770],     # 5
             [ 'QPSK',     602,            1.1758],     # 6
             [ '16QAM',    378,            1.4766],     # 7
             [ '16QAM',    490,            1.9141],     # 8
             [ '16QAM',    616,            2.4063],     # 9
             [ '64QAM',    466,            2.7305],     # 10
             [ '64QAM',    567,            3.3223],     # 11
             [ '64QAM',    666,            3.9023],     # 12
             [ '64QAM',    772,            4.5234],     # 13
             [ '64QAM',    873,            5.1152],     # 14
             [ '64QAM',    948,            5.5547]],    # 15

             # TS 38.214, Table 5.2.2.1-3: 4-bit CQI Table 2
             # modulation  codeRate*1024   efficiency   CQI index
             [[None,       None,           None],       # 0: (Out of Range)
             [ 'QPSK',     78,             0.1523],     # 1
             [ 'QPSK',     193,            0.3770],     # 2
             [ 'QPSK',     449,            0.8770],     # 3
             [ '16QAM',    378,            1.4766],     # 4
             [ '16QAM',    490,            1.9141],     # 5
             [ '16QAM',    616,            2.4063],     # 6
             [ '64QAM',    466,            2.7305],     # 7
             [ '64QAM',    567,            3.3223],     # 8
             [ '64QAM',    666,            3.9023],     # 9
             [ '64QAM',    772,            4.5234],     # 10
             [ '64QAM',    873,            5.1152],     # 11
             [ '256QAM',   711,            5.5547],     # 12
             [ '256QAM',   797,            6.2266],     # 13
             [ '256QAM',   885,            6.9141],     # 14
             [ '256QAM',   948,            7.4063]],    # 15
             
             # TS 38.214, Table 5.2.2.1-4: 4-bit CQI Table 3
             # modulation  codeRate*1024   efficiency   CQI index
             [[None,       None,           None],       # 0: (Out of Range)
             [ 'QPSK',      30,            0.0586],     # 1
             [ 'QPSK',      50,            0.0977],     # 2
             [ 'QPSK',      78,            0.1523],     # 3
             [ 'QPSK',      120,           0.2344],     # 4
             [ 'QPSK',      193,           0.3770],     # 5
             [ 'QPSK',      308,           0.6016],     # 6
             [ 'QPSK',      449,           0.8770],     # 7
             [ 'QPSK',      602,           1.1758],     # 8
             [ '16QAM',     378,           1.4766],     # 9
             [ '16QAM',     490,           1.9141],     # 10
             [ '16QAM',     616,           2.4063],     # 11
             [ '64QAM',     466,           2.7305],     # 12
             [ '64QAM',     567,           3.3223],     # 13
             [ '64QAM',     666,           3.9023],     # 14
             [ '64QAM',     772,           4.5234]],    # 15
             
             # TS 38.214, Table 5.2.2.1-5: 4-bit CQI Table 4
             # modulation  codeRate*1024   efficiency   CQI index
             [[None,       None,           None],       # 0: (Out of Range)
             [ 'QPSK',     78,             0.1523],     # 1
             [ 'QPSK',     193,            0.377],      # 2
             [ 'QPSK',     449,            0.877],      # 3
             [ '16QAM',    378,            1.4766],     # 4
             [ '16QAM',    616,            2.4063],     # 5
             [ '64QAM',    567,            3.3223],     # 6
             [ '64QAM',    666,            3.9023],     # 7
             [ '64QAM',    772,            4.5234],     # 8
             [ '64QAM',    873,            5.1152],     # 9
             [ '256QAM',   711,            5.5547],     # 10
             [ '256QAM',   797,            6.2266],     # 11
             [ '256QAM',   885,            6.9141],     # 12
             [ '256QAM',   948,            7.4063],     # 13
             [ '1024QAM',  853,            8.3301],     # 14
             [ '1024QAM',  948,            9.2578]]]    # 15

cqiTableBLERs = [None, 0.1, 0.1, 0.1, 0.00001, 0.1]     # See TS 38.214, Section 5.2.2.1

# **********************************************************************************************************************
class CsiReport:  # CSI-ReportConfig
    # ******************************************************************************************************************
    def __init__(self, csiRsConfig, **kwargs):
            
        self.reportId = kwargs.get('id', 0)
        
        # This is the same CSI-RS config as the one used for channel estimation. It must not have any ZP resources.
        self.csiRsConfig = csiRsConfig
        self.bwp = self.csiRsConfig.bwp             # Get the bandwidth part info from the CsiRsConfig object
        for csiRsSet in self.csiRsConfig.csiRsSetList:  # Make sure there are no ZP resources
            if csiRsSet.csiType=="ZP":  raise ValueError( "`ZP` resources are not allowed in 'csiRsConfig'." )
        
        # These are lists of CsiRsConfig objects
        self.measurementRes = kwargs.get('measurementRes', [])      # resourcesForChannelMeasurement
        self.interfereResIm = kwargs.get('interfereResIm', [])      # csi-IM-ResourcesForInterference
        self.interfereResNzp = kwargs.get('interfereResNzp', [])    # csi-IM-ResourcesForInterference

        self.reportType = kwargs.get('reportType', "Periodic")      # higher layer parameter "reportConfigType"
        validateRange(self.reportType, ["Periodic", "SpOnPUCCH", "SpOnPUSCH", "Aperiodic"])
        self.period = kwargs.get('period', 5)                       # Used for Periodic, SpOnPUCCH, and SpOnPUSCH cases
        self.offset = kwargs.get('offset', 0)                       # Used for Periodic, SpOnPUCCH, and SpOnPUSCH cases

        if self.reportType in ["Periodic", "SpOnPUCCH"]:    validateRange(self.period, [5, 10, 20, 40, 80, 160, 320])
        elif self.reportType == "SpOnPUSCH":    validateRange(self.period, [4, 5, 8, 10, 16, 20, 32, 40, 80, 160, 320])
        validateRange(self.offset, (0,self.period-1))

        # quantity (reportQuantity) indicates what to measure. The type of quatities can be CSI-related
        # or L1-RSRP-related See 5.2.1.4.2:
        #   'none', 'cri-RI-PMI-CQI ', 'cri-RI-i1', 'cri-RI-i1-CQI', 'cri-RI-CQI', 'cri-RSRP', 'cri-SINR',
        #   'ssb-Index-RSRP', 'ssb-Index-SINR', 'cri-RI-LI-PMI-CQI', 'cri-RSRP-Index', 'ssb-Index-RSRP-Index',
        #   'cri-SINR-Index', 'ssb-Index-SINR-Index' or 'tdcp'.
        # CRI: CSI-RS Resource Indicator
        self.quantity = kwargs.get('quantity', 'CriRiPmiCqi')   # See 3GPP TS 38.214, Section 5.2.1.4
        validateRange(self.quantity, ['CriRiPmiCqi', 'CriRiLiPmiCqi', 'CriRiI1', 'CriRiCqi', 'CriRiI1Cqi',
                                      'CriRsrp', 'SsbRIdxRsrp', 'CriSinr', 'SsbIdxSinr'])

        # See section 5.2.1.4.2 for more about the following quantity values:
        # cri-RI-i1 ->     Wideband PMI and typeI-SinglePanel
        # cri-RI-i1-CQI -> Wideband PMI and typeI-SinglePanel and use random i2 for CQI
        # cri-RI-CQI ->    




        # This enables/disables the group beam based reporting. If enabled, UE shall report different CRI or SSBRI for
        # each report setting in a single report, otherwise, UE shall report in a single reporting instance two
        # different CRI or SSBRI for each report setting, where CSI-RS and/or SSB resources can be received
        # simultaneously by the UE either with a single spatial domain receive filter, or with multiple simultaneous
        # spatial domain receive filters.
        self.groupBeams = kwargs.get('groupBeams', True)
        noOfRepRS = kwargs.get('noOfRepRS', 1) # (nrofReportedRS) Ignored when 'groupBeams' is True
        if not self.groupBeams:     validateRange(self.noOfRepRS, (1,4))

        self.codebookType = kwargs.get('codebookType', 'Type1SP')
        validateRange(self.codebookType, ['Type1SP', 'Type1MP', 'Type2', 'EnancedType2'])
        
        self.txAntenna = kwargs.get('txAntenna', None)
        if self.txAntenna is None:
            self.n1 = kwargs.get('n1', None)    # Number of antenna in horizontal direction
            self.n2 = kwargs.get('n2', None)    # Number of antenna in vertical direction
            self.ng = kwargs.get('ng', None)    # Number of antenna panels (Type1MP case only)
            if self.codebookType == 'Type1MP':
                if (self.n1 is None) or (self.n2 is None) or (self.ng is None):
                    raise ValueError("The antenna configuration is missing! (A 'txAntenna' or ng/n1/n2 values must be specified)")
            elif (self.n1 is None) or (self.n2 is None):
                raise ValueError("The antenna configuration is missing! (A 'txAntenna' or n1/n2 values must be specified)")
            if self.ng is None: self.ng = 1     # Set ng to 1 when it is not used
        else:
            if isinstance(self.txAntenna, AntennaPanel):
                self.ng = 1
                if self.codebookType == 'Type1MP':
                    raise ValueError("Single-Panel 'txAntenna' is configured with Multi-Panel 'codebookType' (Type1MP)!")
                self.n2, self.n1 = self.txAntenna.shape
                
            elif isinstance(self.txAntenna, AntennaArray):
                self.ng = np.prod(self.txAntenna.shape)
                if (self.ng>1) and (self.codebookType == 'Type1SP'):
                    raise ValueError("Multi-Panel 'txAntenna' is configured with Single-Panel 'codebookType' (Type1SP)!")
                self.n2, self.n1 = self.txAntenna.panel[0][0].shape
            
            else:
                raise ValueError("Unsupported antenna class '%s'!"%(self.txAntenna.__class__.__name__))

        if self.codebookType in ['Type1SP', 'Type2']:
            validN1N2Combs = ["1-1","2-1","2-2","4-1","3-2","6-1","4-2","8-1","4-3","6-2","12-1","4-4","8-2","16-1"]
            if "%d-%d"%(self.n1,self.n2) not in validN1N2Combs:
                raise ValueError("Invalid N1-N2 combination %d-%d. See TS 38.214, Table 5.2.2.2.1-2"%(self.n1,self.n2))
        elif self.codebookType == 'Type1MP':
            validNgN1N2Combs = ["2-2-1", "2-4-1", "4-2-1", "2-2-2", "2-8-1", "4-4-1", "2-4-2", "4-2-2"]
            if "%d-%d-%d"%(self.ng,self.n1,self.n2) not in validNgN1N2Combs:
                raise ValueError("Invalid Ng-N1-N2 combination %d-%d-%d. See TS 38.214, Table 5.2.2.2.2-1"%(self.ng,self.n1,self.n2))

        # 'codebookMode' is only used when codebookType is 'Type1SP' or 'Type1MP'
        if self.codebookType in ['Type1SP', 'Type1MP']:
            self.codebookMode = kwargs.get('codebookMode', 1)
            if self.ng==4:  validateRange(self.codebookMode, 1, " when Ng is 4")    # TS 38.214, Sec. 5.2.2.2.2
            else:           validateRange(self.codebookMode, [1,2])

        if self.codebookType in ['Type1SP', 'Type1MP']:
            # See TS 38.214, Tables 5.2.2.2.1-2 and 5.2.2.2.2-1)
            self.o1 = 4
            self.o2 = 4 if self.n2>1 else 1
        
        self.numPorts = 2 * self.ng * self.n1 * self.n2
        self.ac = self.n1 * self.o1 * self.n2 * self.o2
        
        # This is a bitmap with Ac = N1*O1*N2*O2 bits.
        # The following can be one of:
        #    n1-n2 parameter                    (Type1SP, length = Ac)
        #    ng-n1-n2 parameter                 (Type1MP, length = Ac)
        #    twoTX-CodebookSubsetRestriction    (2 Antenna ports, length = 6)
        # Note:
        # Spec says this should be "Ac" bits. But having "Ac" bits is not enough for some
        # cases (For example: Type 1 Single Panel with N2=1 and codebookMode=2 (See table 5.2.2.2.1-5, 3rd table), where:
        #   MaxIndex = N2O2*l+m = N2O2*(2i11+3)+0 = N2O2*(2(N1O1/2-1)+3) = N2O2*(N1O1-2+3) = N2O2*(N1O1+1) > Ac
        self.cbSubsetRestriction = kwargs.get('cbSubsetRestriction', max(8,2*self.ac)*'1')  # Having some extra 1's doesn't hurt!

        # This is used for "Type1SP" case only when quantity is "CriRiI1Cqi". See "typeI-SinglePanel-codebookSubsetRestriction-i2"
        self.cbSubsetRestrictionI2 = kwargs.get('cbSubsetRestrictionI2', 16*'1')    # This is always 16-bit

        # The following can be one of:
        #    typeI-SinglePanel-ri-Restriction   (for Single-Panel type1: 8-bits) - NOT USED YET
        #    ri-Restriction                     (for Multi-Panel type1:  4-bits) - NOT USED YET
        self.cbRiRestriction = kwargs.get('cbRiRestriction', 8*'1')

        if self.codebookType=='Type2':
            self.numBeams = kwargs.get('numBeams', 2)               # See numberOfBeams (L)
            if self.numPorts==4:  validateRange(self.numBeams, 2, " when 'numPorts' is 4")
            validateRange(self.numBeams, [2,3,4])

            self.pskSize = kwargs.get('pskSize', 4)                 # See phaseAlphabetSize (Npsk)
            validateRange(self.pskSize, [4,8])
            
            self.subbandAmp = kwargs.get('subbandAmp', False)       # See subbandAmplitude

        # The size of Precoding RB groups (PRGs). See 3GPP TS 38.214, Section 5.1.2.3
        # If this is provided, it will be used instead of subbandSizePmi below.
        # 0 means 'Wideband' which means a single precoding is used for all PRBs
        # This is the higher layer parameter "pdsch-BundleSizeForCSI" mentioned in 3GPP TS 38.214, Section 5.2.1.4.2
        self.prgSize = kwargs.get('prgSize', None)
        if self.prgSize is not None:
            if self.prgSize not in [0,2,4]:             raise ValueError("'prgSize' must be 0 (Wideband), 2, or 4)")

        # subbandSize: See 3GPP TS 38.214, Table 5.2.1.4-2
        if self.bwp.numRbs<24:      subbandSizeValues = [0]      # No subbands if BWP size is less than 24
        elif self.bwp.numRbs<73:    subbandSizeValues = [4, 8]
        elif self.bwp.numRbs<145:   subbandSizeValues = [8, 16]
        else:                       subbandSizeValues = [16, 32]
        subbandSize = kwargs.get('subbandSize', subbandSizeValues[0])
        validateRange(subbandSize, subbandSizeValues)
        self.subbandSizePmi = kwargs.get('subbandSizePmi', subbandSize)
        self.subbandSizeCqi = kwargs.get('subbandSizeCqi', subbandSize)
        validateRange(self.subbandSizePmi, subbandSizeValues)
        validateRange(self.subbandSizeCqi, subbandSizeValues)

        # See 3GPP TS 38.214, Section 5.2.2.1 for the cqi-table values:
        #   1 -> 'table1'     -> Table 5.2.2.1-2   Error Prob: 0.1
        #   2 -> 'table2'     -> Table 5.2.2.1-3   Error Prob: 0.1
        #   3 -> 'table3'     -> Table 5.2.2.1-4   Error Prob: 0.00001
        #   4 -> 'table4-r17' -> Table 5.2.2.1-5   Error Prob: 0.1
        self.cqiTable = kwargs.get('cqiTable', 1)
        validateRange(self.cqiTable, [1,2,3,4])


    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CsiReport object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If `None` (default), the text
            "CSI Report Properties:" is used for the title.

        getStr : Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "CSI Report Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  Report ID:            {self.reportId}\n"
        repStr += indent*' ' + f"  Report Type:          {self.reportType}\n"
        repStr += indent*' ' + f"  codebookType:         {self.codebookType}\n"
        if self.codebookType in ['Type1SP', 'Type1MP']:
            repStr += indent*' ' + f"  codebookMode:         {self.codebookMode}\n"
        if self.reportType in ["Periodic", "SpOnPUCCH", "SpOnPUSCH"]:
            repStr += indent*' ' + f"  period:               {self.period}\n"
            repStr += indent*' ' + f"  offset:               {self.offset}\n"
        repStr += indent*' ' + f"  quantity:             {self.quantity}\n"
        repStr += indent*' ' + f"  groupBeams:           {self.groupBeams}\n"
        if not self.groupBeams:
            repStr += indent*' ' + f"  noOfRepRS:            {self.noOfRepRS}\n"
        repStr += indent*' ' + f"  Ng x N1 x N2:         {self.ng} x {self.n1} x {self.n2}\n"
        if self.codebookType in ['Type1SP', 'Type1MP']:
            repStr += indent*' ' + f"  o1 x o2:              {self.o1} x {self.o2}\n"
        repStr += indent*' ' + f"  numPorts:             {self.numPorts}\n"
        repStr += indent*' ' + f"  cbSubsetRestriction:  {self.cbSubsetRestriction}\n"
        repStr += indent*' ' + f"  cbSubsetRestrictionI2:{self.cbSubsetRestrictionI2}\n"
        repStr += indent*' ' + f"  cbRiRestriction:      {self.cbRiRestriction}\n"

        if self.codebookType=='Type2':
            repStr += indent*' ' + f"  numBeams:             {self.numBeams}\n"
            repStr += indent*' ' + f"  pskSize:              {self.pskSize}\n"
            repStr += indent*' ' + f"  subbandAmp:           {self.subbandAmp}\n"

        repStr += indent*' ' + f"  prgSize:              {self.prgSize}\n"
        repStr += indent*' ' + f"  subbandSizePmi:       {self.subbandSizePmi}\n"
        repStr += indent*' ' + f"  subbandSizeCqi:       {self.subbandSizeCqi}\n"
        repStr += indent*' ' + f"  cqiTable:             {self.cqiTable}\n"

        if getStr: return repStr
        print(repStr)

    def getEffectiveSINR(self):
        pass

#    # ************************************************************************************************************************************************
#    def getAllowedRanks(self, numRxAntenna):
#        if self.codebookType == 'Type1SP':      numPorts = min(self.csiRsConfig.numPorts, numRxAntenna, 8)
#        elif self.codebookType == 'Type1MP':    numPorts = min(numRxAntenna, 4)
#        elif self.codebookType == 'Type2':      numPorts = min(numRxAntenna, 2)
#        else:
#            assert False, "Codebook type %s not supported yet!"%(self.codebookType)
#            
#        return [r for r in range(numPorts) if self.cbRiRestriction[-1-r]=='1']
#
#    # ************************************************************************************************************************************************
#    def findBestRank(self, h):
#        _, _ ,nr, nt = h.shape
#        allowedRanks = self.getAllowedRanks(nr)
#        
#        for rank in allowedRanks:

    # ******************************************************************************************************************
    def removeNeighbors(self, idx):
        # idx is the 2D indices (port removed). It is a tuple (x, y).
        # We have a CSI-RS RE at symbol x[i], and subcarrier y[i].
        # We want to use only one RE for a set of Neighboring REs (i.e. the ones in the same CDM group)
        bmp = np.ones((idx[0].max()+3,idx[1].max()+3),dtype=np.int8)*2
        bmp[(idx[0]+1,idx[1]+1)] = 1
        x,y = np.where(bmp[:,1:]-bmp[:,:-1]!=-1)
        bmp[ (x,y+1) ] = 2
        x,y = np.where(bmp[1:,:]-bmp[:-1,:]!=-1)
        bmp[ (x+1,y) ] = 2
        x,y = np.where(bmp==1)
        return (x-1,y-1)

    # ******************************************************************************************************************
    def getSINR(self, h, w, noiseVar):
        # h: L x K x Nr x Nt  or n x Nr x Nt
        # w: Ncb x Nt x Nl   (Note that Nl<= min(Nr,Nt)) (Ncb: codebook size)
        # Returns Ncb x n x nl
        h = h.reshape(-1, h.shape[-2], h.shape[-1])          # n x Nr x Nt   (n=L*K if h is 4D)
        heff = np.matmul(h[None,:,:,:],w[:,None,:,:],axes=[(2,3),(2,3),(2,3)])       # Ncb x n x Nr x Nl
        u, s, vH = np.linalg.svd(heff, full_matrices=True)  # Ncb x n x Nr x Nl , Ncb x n x Nl, Ncb x n x Nl x Nl
        noisyInvS = 1/(np.square(np.abs(s))+noiseVar)                # Ncb x n x Nl
        # Calculating (V . noisyInvS . VH) is the same as:
        #  1) expanding dimensions of 'noisyInvS',  n x Nl =>                  Ncb x n x Nl x 1
        #  2) calculating V.VH witch is the same square of magnitude of V      Ncb x n x Nl x Nl
        #  3) doing elementwise multiplication (with boadcasting on last dim)  Ncb x n x Nl x Nl
        #  4) summing on second axis                                           Ncb x n x Nl
        gamma = 1/(noiseVar*(noisyInvS[:,:,:,None] * np.square(np.abs(vH))).sum(2)) - 1
        return gamma.real         # Ncb x n x Nl
        
    # ******************************************************************************************************************
    def subbands(self, sbSize):
        rb = self.bwp.startRb
        endRb = rb + self.bwp.numRbs
        sb = 0
        while rb < endRb:
            # calculate rbsInSb: the number of RBs in this subband
            if sb==0:                   rbsInSb = sbSize - (rb % sbSize)
            elif (rb+sbSize)>endRb:     rbsInSb = endRb % sbSize
            else:                       rbsInSb = sbSize

            yield rbsInSb
            rb, sb = rb+rbsInSb, sb+1

    # ******************************************************************************************************************
    def bestPmiForRank(self, channel, numLayers, noiseVar):
        csiRsGrid = self.bwp.createGrid(self.numPorts)
        self.csiRsConfig.populateGrid(csiRsGrid)
        csiRsIndexes = csiRsGrid.getReIndexes("CSIRS_NZP")      # A tuple of (ports, symbols, subcarriers)
        p0Idx = np.where(csiRsIndexes[0]==0)[0]                 # Indexes in the csiRsIndexes corresponding to port 0
        csiRsIndexesP0 = (csiRsIndexes[1][p0Idx], csiRsIndexes[2][p0Idx]) # A tuple of (symbols, subcarriers) for to port 0
        csiRsIndexesP0 = self.removeNeighbors(csiRsIndexesP0)   # keeps only one of the Neighboring REs in a CDM group
        hAtCsiRs = channel[csiRsIndexesP0]              # Channel values at the CSI-RS REs -> Shape: numREs x Nr x Nt

        # Now get the codebook and calcualte the precoded SINR for each RE in each layer and each codebook entry
        cbIndexes, codebook = self.getCodebook(numLayers)           # Shape: Ncb x numPorts x numLayers
        sinrValues = self.getSINR(hAtCsiRs, codebook, noiseVar)     # Shape: Ncb x numREs x numLayers

        # First find best precoder for the whole bandwidth (Wideband precoder). The 'i1' PMI indices are the same for
        # all subbands. 'i2' can be different for each subband. If there is only one subband (=wideband), then there
        # is a single 'i2' which is the one for wideband.
        sumSinrs = sinrValues.sum((1,2))            # Sum over all numREs and numLayers one SINR per codebook entry
        maxSinrIdx = sumSinrs.argmax()              # Index of the max SINR corresponding to the best wideband precoder

        widebandI1, widebandI2 = cbIndexes[maxSinrIdx]      # Wideband PMI indices (i1 and i2)
        widebandW = [ codebook[maxSinrIdx] ]                # The wideband precoding Matrix

        # Getting the subband size used in the rest of this function:
        # If prgSize is not provided, use 0 (wideband) if BWP is smaller than 24 PRB or subbandSizePmi otherwise
        # Otherwise use prgSize (Regardless of BWP size)
        if self.prgSize is None:    sbSize = self.subbandSizePmi if self.bwp.numRbs>=24 else 0
        else:                       sbSize = self.prgSize
        if sbSize == 0:                             # If Wideband PMI is requested, we are done
            return [widebandI1, [widebandI2]], [ widebandW ], [ sinrValues[maxSinrIdx] ]
        
        # Now find an i2 and a precoder for each subband (i1 is the same for all)
        reIndexes = csiRsIndexesP0[1]
        
        # Indices of codebook entries with i1 = widebandI1:
        i1CodebookIndexes = [i for i, cbIdx in enumerate(cbIndexes) if np.all(cbIdx[0]==widebandI1)]

        subbandI2s, subbandWs, sbReSinr = [], [], []
        rb = 0
        for sb, rbsInSb in enumerate(self.subbands(sbSize)):
            # Indexes to the indices of REs in 'reIndexes' that are in this subband
            sbReIndexes = np.where( (reIndexes>=(rb*12)) & (reIndexes<(rb+rbsInSb)*12) )[0]
            if sbReIndexes.size == 0:
                raise ValueError(f"Invalid CSI-RS config. Subband {sb} does not have any CSI-RS REs!")

            # All SINR values for the CSI-RS REs in this subband for all precoders in the codebook
            sbSinrValues = sinrValues[:,sbReIndexes,:]              # Shape: Ncb x numSbREs x numLayers
            
            # All SINR values for the CSI-RS REs in this subband for the precoders in the codebook wiht i1 = widebandI1
            i1SbSinrValues = sbSinrValues[i1CodebookIndexes,:,:]    # Shape: NcbI1 x numSbREs x numLayers

            # Sum over REs and Layers => One SINR per codebook entry wiht i1 = widebandI1:
            i1SbSumSinrs = i1SbSinrValues.sum((1,2))                # Shape: NcbI1

            # Indexes of best precoder for this subband in i1CodebookIndexes
            i1SbMaxSinrIdx = i1SbSumSinrs.argmax()

            # Indices of the best precoder for this subband in the codebook
            sbMaxSinrIdx = i1CodebookIndexes[i1SbMaxSinrIdx]

            subbandI2s += [ cbIndexes[ sbMaxSinrIdx ][1] ]
            subbandWs  += [ codebook[sbMaxSinrIdx] ]
            sbReSinr   += [ i1SbSinrValues[i1SbMaxSinrIdx] ] # Subband SINRs per RE & Layer, Shape: numSbREs x numLayers
            rb += rbsInSb
                
        return [widebandI1, subbandI2s], subbandWs, sbReSinr

    # ******************************************************************************************************************
    def getBestRank(self, channel, noiseVar):
        l, k, nr, nt = channel.shape
        if nt != self.numPorts:
            raise ValueError("The given numver of transmit antenna from channel must mach the number of ports!")
        if self.codebookType == 'Type1SP':    maxRank = min(nr, nt, 8)
        elif self.codebookType == 'Type1MP':  maxRank = min(nr, 4)
        elif self.codebookType == 'Type2':    maxRank = min(nr, 2)

        # The leftmost bit in cbRiRestriction is for rank 1, bit value of 0 means the rank is restricted
        ranks = [r for r in range(1,maxRank+1) if self.cbRiRestriction[-r]=='1']

        bestSinr, bestRank, bestPmim, bestSbReSinr = -100000, 0, None, None
        for rank in ranks:
            pmi, ws, sbReSinr = self.bestPmiForRank(channel, rank, noiseVar)
            sbSinr = np.float64([sinr.mean(0) for sinr in sbReSinr])    # Subband SINRs, Shape: numSb x numLayers
            layerSinr = sbSinr.mean(0)*rank                             # SINR for each layer, Shape: numLayers
            rankSinr = layerSinr.sum()
            if rankSinr > bestSinr:   bestSinr, bestRank, bestPmi, bestSbReSinr = rankSinr, rank, pmi, sbReSinr

        return bestRank, bestPmi, bestSbReSinr
    
    # ******************************************************************************************************************
    def getCqiToPmiIdxes(self, pmiSbSize):
        cqiSizes = [self.bwp.numRbs] if self.subbandSizeCqi==0 else [s for s in self.subbands(self.subbandSizeCqi)]
        pmiSizes = [self.bwp.numRbs] if pmiSbSize==0 else [s for s in self.subbands(pmiSbSize)]
        
        cqiPmiIdxes = [[] for _ in cqiSizes]
        pmi = 0
        sumPmiSize = pmiSizes[0]
        sumCqiSize = 0
        for cqi,cqiSize in enumerate(cqiSizes):
            cqiPmiIdxes[cqi] += [pmi]
            sumCqiSize += cqiSize
            while 1:
                if sumPmiSize==sumCqiSize:
                    pmi += 1
                    if pmi<len(pmiSizes): sumPmiSize = pmiSizes[pmi]
                    sumCqiSize = 0
                    break
                if sumPmiSize>sumCqiSize:   break
                sumPmiSize += pmiSizes[pmi]
                pmi+=1
                cqiPmiIdxes[cqi] += [pmi]
        return cqiPmiIdxes

    # ******************************************************************************************************************
    def getCodebook(self, numLayers):
        indexes = []
        codebook = []
        if self.codebookType == 'Type1SP':
            for i1,i2 in self.type1SpIndexes(numLayers):
                indexes += [ [i1,i2] ]
                codebook += [ self.getType1SpPrecoder(numLayers, i1, i2) ]
            return indexes, np.array(codebook)
            
        assert self.codebookType == 'Type1MP'
        for i1,i2 in self.type1MpIndexes(numLayers):
            indexes += [ [i1,i2] ]
            codebook += [ self.getType1MpPrecoder(numLayers, i1, i2) ]
        return indexes, np.array(codebook)

    # ******************************************************************************************************************
    def v(self, l, m, tilde=False):
        if tilde in [True, '~']:    ul = np.exp( 4j*np.pi* l *np.arange(self.n1//2)/(self.n1*self.o1) )     # Shape: N1//2
        else:                       ul = np.exp( 2j*np.pi* l *np.arange(self.n1)/(self.n1*self.o1) )        # Shape: N1
        um = np.exp( 2j*np.pi* m *np.arange(self.n2)/(self.n2*self.o2) )                                    # Shape: N2
        return np.outer(ul, um)                                                                             # Shape: N1 x N2 or N1//2 x N2

    # ******************************************************************************************************************
    def getCombs(self, *argv):
        # Returns 2d NumPy array. Each row is one combination. The i'th element loops through possible values 2^i times.
        lists = []
        for listI in argv[::-1]:
            if type(listI)==list: lists += [listI]
            else:                 lists += [list(range(listI))]
                
        lists = [lists[1]] + [lists[0]] + lists[2:]
        n = len(lists)
        a = list(range(n-1,1,-1)) + [0,1]
        return np.int32(np.meshgrid(*lists)).T.reshape(-1,n)[:,a].tolist()

    # ******************************************************************************************************************
    def type1SpIndexes(self, numLayers):
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)
        
        if self.quantity=='CriRiI1Cqi':
            subsetRestrictionI2 = self.cbSubsetRestrictionI2    # See typeI-SinglePanel-codebookSubsetRestriction-i2
        else:
            subsetRestrictionI2 = 16*'1'
        
        context = " with %d layers, CB Mode %d, %dx%d Ant"%(numLayers, self.codebookMode, self.n1, self.n2)

        # ..............................................................................................................
        if self.numPorts == 2:          # See TS 38.214, Table 5.2.2.2.1-1
            # Note that in this case "self.cbSubsetRestriction" is "twoTX-CodebookSubsetRestriction" in the spec.
            validateRange(numLayers, [1,2], " when 'numPorts' is 2")
            if numLayers == 1:          n1, pmiAllowed = 4, self.cbSubsetRestriction[-4:]   # 1st Column, Bits 0,1,2,3 (From right/end/lsb)
            elif numLayers == 2:        n1, pmiAllowed = 2, self.cbSubsetRestriction[-6:-4] # 2nd Column, Bits 4 and 5 (From right/end/lsb)

            # In this case 'i1' is a single Scalar integer. In all other cases it is a list of 2 or 3 values.
            for i1 in range(4):
                if pmiAllowed[i1]:
                    yield [i1,0,0], 0
                    
        # ..............................................................................................................
        elif numLayers == 1:                                                # See TS 38.214, Table 5.2.2.2.1-5
            if self.codebookMode==1:
                combs = self.getCombs(bb1, bb2, 4)                          # 1st Table
                for i11, i12, i2 in combs:
                    l, m = i11, i12
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':            continue # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    if subsetRestrictionI2[ i2 ]=='0':                      continue
                    yield [i11,i12,0], i2
            elif self.n2>1:                                                 # codebookMode=2, N2>1
                combs = self.getCombs(bb1//2, bb2//2, 16)                   # 2nd Table
                for i11, i12, i2 in combs:
                    l, m = 2*i11 + (i2//4)%2, 2*i12 + i2//8
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':            continue # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    if subsetRestrictionI2[ i2 ]=='0':                      continue
                    yield [i11,i12,0], i2
            elif self.n2==1:                                                # codebookMode=2, N2=1
                combs = self.getCombs(bb1//2, 16)                           # 3rd Table
                for i11, i2 in combs:
                    l, m = 2*i11 + i2//4, 0
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':            continue # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    if subsetRestrictionI2[ i2 ]=='0':                      continue
                    yield [i11,0,0], i2
            else:
                raise ValueError( "Unsupported case" + context + "!" )

        # ..............................................................................................................
        elif numLayers == 2:                                                # See TS 38.214, Table 5.2.2.2.1-6
            i13Len = 2 if (self.n1==2 and self.n2==1) else 4                # See TS 38.214, Table 5.2.2.2.1-3
            if self.codebookMode==1:
                combs = self.getCombs(bb1, bb2, i13Len, 2)                  # 1st Table
                for i11, i12, i13, i2 in combs:
                    l, m = i11, i12
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':            continue # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    if subsetRestrictionI2[ i2 ]=='0':                      continue
                    yield [i11,i12,i13], i2
            elif self.n2>1:
                combs = self.getCombs(bb1//2, bb2//2, i13Len, 8)            # 2nd Table
                for i11, i12, i13, i2 in combs:
                    l, m = 2*i11 + (i2//2)%2, 2*i12 + i2//4
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':            continue # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    if subsetRestrictionI2[ i2 ]=='0':                      continue
                    yield [i11,i12,i13], i2
            elif self.n2==1:
                combs = self.getCombs(bb1//2, i13Len, 8)                    # 3rd Table
                for i11, i13, i2 in combs:
                    l, m = 2*i11 + i2//2, 0
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':            continue # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    if subsetRestrictionI2[ i2 ]=='0':                      continue
                    yield [i11,0,i13], i2
            else:
                raise ValueError( "Unsupported case" + context + "!" )

        # ..............................................................................................................
        elif numLayers in [3,4]:                                                        # See TS 38.214, Tables 5.2.2.2.1-7 and Table 5.2.2.2.1-8
            if self.numPorts>=16:               i13Len = 4                              # This is from 2nd table of 5.2.2.2.1-7/5.2.2.2.1-8
            elif (self.n1==2 and self.n2==1):   i13Len = 1                              # This is from Table 5.2.2.2.1-4
            elif (self.n1==4 and self.n2==1):   i13Len = 3                              # This is from Table 5.2.2.2.1-4
            elif (self.n1==2 and self.n2==2):   i13Len = 3                              # This is from Table 5.2.2.2.1-4
            else:                               i13Len = 4                              # This is from Table 5.2.2.2.1-4

            if self.numPorts<16:    combs = self.getCombs(bb1, bb2, i13Len, 2)          # 1st Table
            else:                   combs = self.getCombs(bb1//2, bb2, i13Len, 2)       # 2nd Table
            for i11, i12, i13, i2 in combs:
                l, m = i11, i12
                if self.numPorts in [16,24,32]:                                         # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                    bits = self.cbSubsetRestriction[ bb2*(2*l-1)+m ] + \
                           self.cbSubsetRestriction[ bb2*(2*l)  +m ] + \
                           self.cbSubsetRestriction[ bb2*(2*l+1)+m ]
                    if bits != '111':                                       continue    # If any of the 3 bits is zero, it is not allowed
                elif self.cbSubsetRestriction[ bb2*l+m ]=='0':              continue    # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                if subsetRestrictionI2[ i2 ]=='0':                          continue
                yield [i11,i12,i13], i2

        # ..............................................................................................................
        elif numLayers in [5, 6]:                                                       # See TS 38.214, Tables 5.2.2.2.1-9 and 5.2.2.2.1-10
            if self.n2>1:                       combs = self.getCombs(bb1, bb2, 2)      # 1st row
            elif (self.n1>2) and (self.n2==1):  combs = self.getCombs(bb1, 1, 2)        # 2nd row
            else:                               raise ValueError( "Unsupported case" + context + "!" )

            for i11, i12, i2 in combs:
                l, m = i11, i12
                if self.cbSubsetRestriction[ bb2*l+m ]=='0':                continue    # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                if subsetRestrictionI2[ i2 ]=='0':                          continue
                yield [i11,i12,0], i2

        # ..............................................................................................................
        elif numLayers in [7,8]:                                                        # See TS 38.214, Tables 5.2.2.2.1-11 and 5.2.2.2.1-12
            if (self.n1==4) and (self.n2==1):   combs = self.getCombs(bb1//2, 1, 2)     # 1st row
            elif (self.n1>4) and (self.n2==1):  combs = self.getCombs(bb1, 1, 2)        # 2nd row
            elif (self.n1==2) and (self.n2==2): combs = self.getCombs(bb1, bb2, 2)      # 3rd row
            elif (self.n1>2) and (self.n2==2):  combs = self.getCombs(bb1, bb2//2, 2)   # 4th row
            elif (self.n1>2) and (self.n2>2):   combs = self.getCombs(bb1, bb2, 2)      # 5th row
            else:                               raise ValueError( "Unsupported case" + context + "!" )
            for i11, i12, i2 in combs:
                l, m = i11, i12
                if self.cbSubsetRestriction[ bb2*l+m ]=='0':                continue    # See "bitmap parameter n1-n2" description in Sec. 5.2.2.2.1
                if subsetRestrictionI2[ i2 ]=='0':                          continue
                yield [i11,i12,0], i2
        else:
            raise ValueError( "Unsupported number of layers %d! (codebookType: Type1SP)"%(self.numLayers) )

    # ******************************************************************************************************************
    def getType1SpPrecoder(self, numLayers, i1=0, i2=0):
        # See TS 38.214, Section 5.2.2.2.1
        # i1 can be a number, [i11,i12] or [i11,i12,i13]
        # i2 is always a number.
        # retutns a complex matrix of shape: numPorts x numLayers
        assert self.numPorts >= 2
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)
        context = " with %d layers, CB Mode %d, %dx%d Ant"%(numLayers, self.codebookMode, self.n1, self.n2)
        if not ( (type(i1) in [tuple, list]) and (len(i1)==3) ):    raise ValueError( "'i1' must be a tuple or list of length 3!" )
        i11,i12,i13 = i1

        # ..............................................................................................................
        if self.numPorts == 2:  # See TS 38.214, Table 5.2.2.2.1-1
            # In this case only 'i11' is used (a single Scalar integer)
            if numLayers == 1:
                validateRange(i11, (0,3), context)
                codebook = np.array([ [[1], [1]], [[1], [1j]], [[1], [-1]], [[1], [-1j]] ])/np.sqrt(2)      # Shape: 4, 2, 1
                return codebook[i11]    # Shape: 2, 1

            if numLayers == 2:
                validateRange(i11, [0,1], context)
                codebook = np.array([ [[1, 1], [1, -1]], [[1, 1], [1j, -1j]] ])/2                           # Shape: 2, 2, 2
                return codebook[i11]     # Shape: 2, 2

            raise ValueError( "'numLayers' must be 1 or 2 when 'numPorts' is 2!" )
            
        # Now handling all the cases with more than 2 ports:
        # ..............................................................................................................
        if numLayers == 1:
            # Note: i13 is not used
            if self.codebookMode==1:                                    # See TS 38.214, Table 5.2.2.2.1-5 (1st Table)
                validateRange(i11, (0,bb1-1), context)
                validateRange(i12, (0,bb2-1), context)
                validateRange(i2,  (0,3),     context)
                l, m, n = i11, i12, i2
            else:
                assert self.codebookMode==2
                validateRange(i11, (0,bb1//2-1), context)
                validateRange(i2,  (0,15),       context)
            
                if self.n2>1:                                           # See TS 38.214, Table 5.2.2.2.1-5 (2nd Table)
                    validateRange(i12, (0,bb2//2-1), context)
                    if i2<4:        l, m, n = 2*i11,   2*i12,   i2
                    elif i2<8:      l, m, n = 2*i11+1, 2*i12,   i2-4
                    elif i2<12:     l, m, n = 2*i11,   2*i12+1, i2-8
                    else:           l, m, n = 2*i11+1, 2*i12+1, i2-12
                else:                                                   # See TS 38.214, Table 5.2.2.2.1-5 (3rd Table)
                    assert self.n2==1
                    validateRange(i12, 0, context)
                    if i2<4:        l, m, n = 2*i11,   0, i2
                    elif i2<8:      l, m, n = 2*i11+1, 0, i2-4
                    elif i2<12:     l, m, n = 2*i11+2, 0, i2-8
                    else:           l, m, n = 2*i11+3, 0, i2-12
            vlm = self.v(l,m)                                                   # Shape: N1 x N2
            phi = np.exp( 1j*np.pi*n/2 )
            return np.concatenate([vlm, phi*vlm])/np.sqrt(self.numPorts)        # Shape: 2N1 x N2

        # ..............................................................................................................
        if numLayers == 2:
            i13Len = 2 if (self.n1==2 and self.n2==1) else 4    # From Table 5.2.2.2.1-3
            validateRange(i13, (0,i13Len-1), context)
            
            # Getting k1,k2 from i13: (TS 38.214, Table 5.2.2.2.1-3)
            if i13==0:                                  k1,k2 = 0, 0
            elif i13==1:                                k1,k2 = self.o1, 0
            elif i13==2:
                if (self.n1>self.n2) and (self.n2>1):   k1,k2 = 0, self.o2
                elif self.n1==self.n2:                  k1,k2 = 0, self.o2
                elif (self.n1==2) and (self.n2==1):     validateRange(i1[2], [0,1], context)
                elif (self.n1>2) and (self.n2==1):      k1,k2 = 2*self.o1, 0
                else:                                   raiseError( "Unsupported N1/N2 combination (N1=%d, N2=%d)!"%(self.n1,self.n2))
            elif i13==3:
                if (self.n1>self.n2) and (self.n2>1):   k1,k2 = 2*self.o1, 0
                elif self.n1==self.n2:                  k1,k2 = self.o1, self.o2
                elif (self.n1==2) and (self.n2==1):     validateRange(i1[2], [0,1], context)
                elif (self.n1>2) and (self.n2==1):      k1,k2 = 3*self.o1, 0
                else:                                   raise ValueError( "Unsupported N1/N2 combination (N1=%d, N2=%d)!"%(self.n1,self.n2))

            if self.codebookMode==1:                                    # See TS 38.214, Table 5.2.2.2.1-6 (1st Table)
                validateRange(i11, (0,bb1-1), context)
                validateRange(i12, (0,bb2-1), context)
                validateRange(i2,  [0,1],     context)
                l, lp, m, mp, n = i11, i11+k1, i12, i12+k2, i2
                
            else:
                assert self.codebookMode==2
                validateRange(i11, (0,bb1//2-1), context)
                if self.n2>1:                                           # See TS 38.214, Table 5.2.2.2.1-6 (2nd Table)
                    validateRange(i12, (0,bb2//2-1), context)
                    validateRange(i2,  (0,7),        context)
                    if i2<2:        l, lp, m, mp, n = 2*i11,   2*i11+k1,   2*i12,   2*i12+k2,   i2
                    elif i2<4:      l, lp, m, mp, n = 2*i11+1, 2*i11+k1+1, 2*i12,   2*i12+k2,   i2-2
                    elif i2<6:      l, lp, m, mp, n = 2*i11,   2*i11+k1,   2*i12+1, 2*i12+k2+1, i2-4
                    else:           l, lp, m, mp, n = 2*i11+1, 2*i11+k1+1, 2*i12+1, 2*i12+k2+1, i2-6

                else:                                                   # See TS 38.214, Table 5.2.2.2.1-6 (3rd Table)
                    assert self.n2==1
                    validateRange(i12, 0,     context)
                    validateRange(i2,  (0,7), context)
                    if i2<2:        l, lp, m, mp, n = 2*i11,   2*i11+k1,   0, 0,   i2
                    elif i2<4:      l, lp, m, mp, n = 2*i11+1, 2*i11+k1+1, 0, 0,   i2-2
                    elif i2<6:      l, lp, m, mp, n = 2*i11+2, 2*i11+k1+2, 0, 0,   i2-4
                    else:           l, lp, m, mp, n = 2*i11+3, 2*i11+k1+3, 0, 0,   i2-6
                
            vlm = self.v(l,m)
            vlmp = self.v(lp,mp)
            phi  = np.exp( 1j*np.pi*n/2 )
            return np.concatenate([ np.concatenate( [vlm,     vlmp     ], axis=-1 ),
                                    np.concatenate( [phi*vlm, -phi*vlmp], axis=-1)])/np.sqrt(2*self.numPorts)

        # i13Len From Table 5.2.2.2.1-4
        if (self.n1==2 and self.n2==1):     i13Len = 1
        elif (self.n1==4 and self.n2==1):   i13Len = 3
        elif (self.n1==2 and self.n2==2):   i13Len = 3
        else:                               i13Len = 4

        def k12(i13):                                   # See TS 38.214, Table 5.2.2.2.1-4
            # This is used when numLayers∈[3,4] and numPorts<16
            if i13==0:                                  return (self.o1, 0)
            if i13==1:
                if (self.n1==4) and (self.n2==1):       return (2*self.o1, 0)
                if (self.n1==6) and (self.n2==1):       return (2*self.o1, 0)
                if (self.n1==2) and (self.n2==2):       return (0, self.o2)
                if (self.n1==3) and (self.n2==2):       return (0, self.o2)
            if i13==2:
                if (self.n1==4) and (self.n2==1):       return (3*self.o1, 0)
                if (self.n1==6) and (self.n2==1):       return (3*self.o1, 0)
                if (self.n1==2) and (self.n2==2):       return (self.o1, self.o2)
                if (self.n1==3) and (self.n2==2):       return (self.o1, self.o2)
            if i13==3:
                if (self.n1==6) and (self.n2==1):       return (4*self.o1, 0)
                if (self.n1==3) and (self.n2==2):       return (2*self.o1, 0)
            raise ValueError( "Unsupported N1/N2 combination (i1,3=%d, N1=%d, N2=%d)!"%(i13, self.n1,self.n2))

        # ..............................................................................................................
        if numLayers == 3:
            validateRange(i12, (0, bb2-1),    context)
            validateRange(i13, (0, i13Len-1), context)
            validateRange(i2,  [0,1],         context)

            if self.numPorts<16:                                        # See TS 38.214, Table 5.2.2.2.1-7 (1st Table)
                validateRange(i11, (0, bb1-1), context)
                k1,k2 = k12(i13)
                l, lp, m, mp, n = i11, i11+k1, i12, i12+k2, i2
                vlm = self.v(l,m)
                vlmp = self.v(lp,mp)
                phi = np.exp( 1j*np.pi*n/2 )
                return np.concatenate([ np.concatenate([vlm,     vlmp,     vlm     ], axis=-1),
                                        np.concatenate([phi*vlm, phi*vlmp, -phi*vlm], axis=-1)])/np.sqrt(3*self.numPorts)
            
            else:                                                       # See TS 38.214, Table 5.2.2.2.1-7 (2nd Table)
                validateRange(i11, (0, bb1//2-1), context)
                l, m, p, n = i11, i12, i13, i2
                vtlm = self.v(l,m,'~')
                phi   = np.exp( 1j*np.pi*n/2 )
                theta = np.exp( 1j*np.pi*p/4 )
                return np.concatenate([ np.concatenate([vtlm,           vtlm,            vtlm           ], axis=-1),
                                        np.concatenate([theta*vtlm,     -theta*vtlm,     theta*vtlm     ], axis=-1),
                                        np.concatenate([phi*vtlm,       phi*vtlm,        -phi*vtlm      ], axis=-1),
                                        np.concatenate([theta*phi*vtlm, -theta*phi*vtlm, -theta*phi*vtlm], axis=-1) ])/np.sqrt(3*self.numPorts)

        # ..............................................................................................................
        if numLayers == 4:
            if not ( (type(i1) in [tuple, list]) and (len(i1)==3) ):    raiseError( "'i1' must be a tuple or list of length three!" )
            i11,i12,i13 = i1
            validateRange(i12, (0, bb2-1),    context)
            validateRange(i13, (0, i13Len-1), context)
            validateRange(i2,  [0, 1],        context)

            if self.numPorts<16:                                        # See TS 38.214, Table 5.2.2.2.1-8 (1st Table)
                validateRange(i11, (0, bb1-1), context)
                k1,k2 = k12(i13)
                l, lp, m, mp, n = i11, i11+k1, i12, i12+k2, i2
                vlm = self.v(l,m)
                vlmp = self.v(lp,mp)
                phi = np.exp( 1j*np.pi*n/2 )
                return np.concatenate([ np.concatenate([vlm,     vlmp,     vlm,      vlmp     ], axis=-1),
                                        np.concatenate([phi*vlm, phi*vlmp, -phi*vlm, -phi*vlmp], axis=-1) ])/np.sqrt(4*self.numPorts)

            else:                                                       # See TS 38.214, Table 5.2.2.2.1-8 (2nd Table)
                validateRange(i11, (0, bb1//2-1), context)
                l, m, p, n = i11, i12, i13, i2
                vtlm = self.v(l,m,'~')
                phi   = np.exp( 1j*np.pi*n/2 )
                theta = np.exp( 1j*np.pi*p/4 )
                return np.concatenate([ np.concatenate([vtlm,           vtlm,            vtlm,            vtlm          ], axis=-1),
                                        np.concatenate([theta*vtlm,     -theta*vtlm,     theta*vtlm,      -theta*vtlm   ], axis=-1),
                                        np.concatenate([phi*vtlm,       phi*vtlm,        -phi*vtlm,       -phi*vtlm     ], axis=-1),
                                        np.concatenate([theta*phi*vtlm, -theta*phi*vtlm, -theta*phi*vtlm, theta*phi*vtlm], axis=-1)
                                      ])/np.sqrt(4*self.numPorts)

        # ..............................................................................................................
        if numLayers == 5:                                              # See TS 38.214, Table 5.2.2.2.1-9
            validateRange(i11, (0, bb1-1), context)
            validateRange(i2,  [0,1],      context)
            phi = np.exp( 1j*np.pi*i2/2 )   # n=i2

            if self.n2>1:
                validateRange(i12, (0, bb2-1), context)
                l, lp, ls = i11, i11+self.o1, i11+self.o1
                m, mp, ms = i12, i12,         i12+self.o2
            elif (self.n1>2) and (self.n2==1):
                validateRange(i12, 0, context)
                l, lp, ls = i11, i11+self.o1, i11+2*self.o1
                m, mp, ms = 0, 0, 0
            else:
                raise ValueError( "Unsupported case for numLayers=%d: N1=%d, N2=%d"%(self.numLayers, self.n1, self.n2) )
            
            vlm = self.v(l,m)
            vlmp = self.v(lp,mp)
            vlms = self.v(ls,ms)
            return np.concatenate([ np.concatenate([vlm,     vlm,      vlmp, vlmp,  vlms], axis=-1),
                                    np.concatenate([phi*vlm, -phi*vlm, vlmp, -vlmp, vlms], axis=-1) ])/np.sqrt(5*self.numPorts)

        # ..............................................................................................................
        if self.numLayers == 6:                                         # See TS 38.214, Table 5.2.2.2.1-10
            validateRange(i11, (0, bb1-1), context)
            validateRange(i2,  [0,1],      context)
            phi = np.exp( 1j*np.pi*i2/2 )   # n=i2

            if self.n2>1:
                validateRange(i12, (0, bb2-1), context)
                l, lp, ls = i11, i11+self.o1, i11+self.o1
                m, mp, ms = i12, i12,         i12+self.o2
            elif (self.n1>2) and (self.n2==1):
                validateRange(i12, 0, context)
                l, lp, ls = i11, i11+self.o1, i11+2*self.o1
                m, mp, ms = 0, 0, 0
            else:
                raise ValueError( "Unsupported case for numLayers=%d: N1=%d, N2=%d"%(self.numLayers, self.n1, self.n2) )

            vlm = self.v(l,m)
            vlmp = self.v(lp,mp)
            vlms = self.v(ls,ms)
            return np.concatenate([ np.concatenate([vlm,     vlm,      vlmp,     vlmp,      vlms, vlms ], axis=-1),
                                    np.concatenate([phi*vlm, -phi*vlm, phi*vlmp, -phi*vlmp, vlms, -vlms], axis=-1) ])/np.sqrt(6*self.numPorts)

        # ..............................................................................................................
        if numLayers == 7:                                              # See TS 38.214, Table 5.2.2.2.1-11
            validateRange(i2,    [0,1],      context)
            phi = np.exp( 1j*np.pi*i2/2 )       # n=i2

            if (self.n1==4) and (self.n2==1):   # 1st row
                validateRange(i11, (0, bb1//2-1), context)
                if i12 != 0:                                            raiseError( "'i1[1]' must be zero!" )
                l, l1, l2, l3 = i11, i11+self.o1, i11+2*self.o1, i11+3*self.o1
                m, m1, m2, m3 = 0, 0, 0, 0
            else:
                validateRange(i11, (0, bb1-1), context)
                if (self.n1>4) and (self.n2==1):  # 2nd row
                    validateRange(i12, 0, context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11+2*self.o1, i11+3*self.o1
                    m, m1, m2, m3 = 0, 0, 0, 0
                elif (self.n1==2) and (self.n2==2):   # 3nd row
                    validateRange(i12, (0, bb2-1), context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11,         i11+self.o1
                    m, m1, m2, m3 = i12, i12,         i12+self.o2, i12+self.o2
                elif (self.n1>2) and (self.n2==2):    # 4th row
                    validateRange(i12, (0, bb2//2-1), context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11,         i11+self.o1
                    m, m1, m2, m3 = i12, i12,         i12+self.o2, i12+self.o2
                elif (self.n1>2) and (self.n2>2):     # 5th row
                    validateRange(i12, (0, bb2-1), context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11,         i11+self.o1
                    m, m1, m2, m3 = i12, i12,         i12+self.o2, i12+self.o2
                else:
                    raise ValueError( "Unsupported case for numLayers=%d: N1=%d, N2=%d"%(self.numLayers, self.n1, self.n2) )
                
            vlm = self.v(l,m)
            vlm1 = self.v(l1,m1)
            vlm2 = self.v(l2,m2)
            vlm3 = self.v(l3,m3)
            return np.concatenate([ np.concatenate([vlm,     vlm,      vlm1,     vlm2, vlm2,  vlm3, vlm3 ], axis=-1),
                                    np.concatenate([phi*vlm, -phi*vlm, phi*vlm1, vlm2, -vlm2, vlm3, -vlm3], axis=-1) ])/np.sqrt(7*self.numPorts)

        # ..............................................................................................................
        if numLayers == 8:                                              # See TS 38.214, Table 5.2.2.2.1-12
            validateRange(i2, [0,1], context)
            phi = np.exp( 1j*np.pi*i2/2 )       # n=i2

            if (self.n1==4) and (self.n2==1):   # 1st row
                validateRange(i11, (0, bb1//2-1), context)
                validateRange(i12, 0,             context)
                l, l1, l2, l3 = i11, i11+self.o1, i11+2*self.o1, i11+3*self.o1
                m, m1, m2, m3 = 0, 0, 0, 0
            else:
                validateRange(i11, (0, bb1-1), context)
                if (self.n1>4) and (self.n2==1):    # 2nd row
                    validateRange(i12, 0, context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11+2*self.o1, i11+3*self.o1
                    m, m1, m2, m3 = 0, 0, 0, 0
                elif (self.n1==2) and (self.n2==2):   # 3rd row
                    validateRange(i12, (0, bb2-1), context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11,         i11+self.o1
                    m, m1, m2, m3 = i12, i12,         i12+self.o2, i12+self.o2
                elif (self.n1>2) and (self.n2==2):    # 4th row
                    validateRange(i12, (0, bb2//2-1), context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11,         i11+self.o1
                    m, m1, m2, m3 = i12, i12,         i12+self.o2, i12+self.o2
                elif (self.n1>2) and (self.n2>2):   # 5th row
                    validateRange(i12, (0, bb2-1), context)
                    l, l1, l2, l3 = i11, i11+self.o1, i11,         i11+self.o1
                    m, m1, m2, m3 = i12, i12,         i12+self.o2, i12+self.o2
                else:
                    raise ValueError( "Unsupported case for numLayers=%d: N1=%d, N2=%d"%(self.numLayers, self.n1, self.n2) )

            vlm = self.v(l,m)
            vlm1 = self.v(l1,m1)
            vlm2 = self.v(l2,m2)
            vlm3 = self.v(l3,m3)
            return np.concatenate([ np.concatenate([vlm,     vlm,      vlm1,     vlm1,      vlm2, vlm2,  vlm3, vlm3 ], axis=-1),
                                    np.concatenate([phi*vlm, -phi*vlm, phi*vlm1, -phi*vlm1, vlm2, -vlm2, vlm3, -vlm3], axis=-1) ])/np.sqrt(8*self.numPorts)
            
        raise ValueError( "Unsupported number of layers %d! (codebookType: Type1SP)"%(self.numLayers) )

    # ******************************************************************************************************************
    def type1MpIndexes(self, numLayers):
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)
        if self.numPorts<8:         raise ValueError( "Need at least 8 ports for Codebook Type 1 Multi-Panel!")

        context = " with %d layers, CB Mode %d, %dx%d Ant"%(numLayers, self.codebookMode, self.n1, self.n2)

        # ..............................................................................................................
        if numLayers == 1:                                                              # TS 38.214, Table 5.2.2.2.2-3
            if self.codebookMode==1:                                                    # 1st Table
                validateRange(self.ng, [2,4], context)
                if self.ng==2:      combs = self.getCombs(bb1, bb2, 4, 4)               # i11, i12, i141, i2
                else:               combs = self.getCombs(bb1, bb2, 4, 4, 4, 4)         # i11, i12, i141, i142, i143, i2
                for comb in combs:
                    l, m = i11, i12 = comb[:2]
                    if self.n2==1 and i12>0:                            continue        # See TS 38.214, section 5.2.2.2.2 descriptions
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':        continue        # See "bitmap parameter ng-n1-n2" in Sec. 5.2.2.2.2
                    i14 = comb[2:-1]
                    i2 = comb[-1]
                    yield [i11, i12, 0, i14], [i2]
            else:                                                                       # 2nd Table
                assert self.codebookMode==2
                validateRange(self.ng, 2, context)
                combs = self.getCombs(bb1, bb2, 4, 4, 4, 2, 2)                          # i11, i12, i141, i142, i20, i21, i22
                for i11, i12, i141, i142, i21, i22 in combs:
                    l, m = i11, i12
                    if self.n2==1 and i12>0:                            continue        # See TS 38.214, section 5.2.2.2.2 descriptions
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':        continue        # See "bitmap parameter ng-n1-n2" in Sec. 5.2.2.2.2
                    yield [i11, i12, 0, [i141, i142]], [i20, i21, i22]
        
        # ..............................................................................................................
        elif numLayers in [2,3,4]:                                                      # TS 38.214, Tables 5.2.2.2.2-4, 5.2.2.2.2-5, 5.2.2.2.2-6
            # For 'i13Len', See TS 38.214, Table 5.2.2.2.2-2
            if numLayers==2:                        i13Len = 2 if (self.n1==2 and self.n2==1) else 4    # From Table 5.2.2.2.1-3
            elif (self.n1==2 and self.n2==1):       i13Len = 1                                          # From Table 5.2.2.2.2-2
            elif (self.n1==4 and self.n2==1):       i13Len = 3                                          # From Table 5.2.2.2.2-2
            elif (self.n1==2 and self.n2==2):       i13Len = 3                                          # From Table 5.2.2.2.2-2
            else:                                   i13Len = 4                                          # From Table 5.2.2.2.2-2
            if self.codebookMode==1:                                                    # 1st Table
                validateRange(self.ng, [2,4], context)
                if self.ng==2:  combs = self.getCombs(bb1, bb2, i13Len, 4, 2)           # i11, i12, i13, i141, i2
                else:           combs = self.getCombs(bb1, bb2, i13Len, 4, 4, 4, 2)     # i11, i12, i13, i141, i142, i143, i2
                for comb in combs:
                    l, m, _ = i11, i12, i13 = comb[:3]
                    if self.n2==1 and i12>0:                            continue        # See TS 38.214, section 5.2.2.2.2 descriptions
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':        continue        # See "bitmap parameter ng-n1-n2" in Sec. 5.2.2.2.2
                    i14 = comb[2:-1]
                    i2 = comb[-1]
                    yield [i11, i12, i13, i14], [i2]
            else:                                                                       # 2nd Table
                assert self.codebookMode==2
                validateRange(self.ng, 2, context)
                combs = self.getCombs(bb1, bb2, i13Len, 4, 4, 2, 2, 2)                  # i11, i12, i13, i141, i142, i20, i21, i22
                for i11, i12, i13, i141, i142, i20, i21, i22 in combs:
                    l, m = i11, i12
                    if self.n2==1 and i12>0:                            continue        # See TS 38.214, section 5.2.2.2.2 descriptions
                    if self.cbSubsetRestriction[ bb2*l+m ]=='0':        continue        # See "bitmap parameter ng-n1-n2" in Sec. 5.2.2.2.2
                    yield [i11, i12, i13, [i141, i142]], [i20, i21, i22]

    # ******************************************************************************************************************
    def getType1MpPrecoder(self, numLayers, i1=0, i2=0):
        # See TS 38.214, Section 5.2.2.2.2
        # i1 can be a number, [i11,i12,i14] (when numLayers=1) or [i11,i12,i13,i14] (when numLayers∈{2,3,4})
        # When codebookMode=1, i14 can be the Scalar i141 (when ng=2) or [i141, i142, i143] (when ng=4)
        # When codebookMode=2, i14 is [i141, i142] and i2 is [i20, i21, i22]
        if self.numPorts<8:         raise ValueError( "Need at least 8 ports for Codebook Type 1 Multi-Panel!")
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)
        context = " with %d layers, CB Mode %d, %dx%d Ant"%(numLayers, self.codebookMode, self.n1, self.n2)

        if not ( (type(i1) in [tuple, list]) and (len(i1)==4) ):    raise ValueError( "'i1' must be a tuple or list of length 4!" )
        i11,i12,i13,i14 = i1

        if not (type(i14) in [tuple, list]):    raise ValueError( "'i14' must be a tuple or list!" )
        if not (type(i2) in [tuple, list]):     raise ValueError( "'i2' must be a tuple or list!" )

        # ..............................................................................................................
        def w(col,l,m,p,n):
            # This calculates: w^(col,ng,codebookMode)_(l,m,p,n)
            s = 1 if col==1 else -1                         # The sign used based on column
            vlm = self.v(l,m)                               # Shape: (n1, n2)
            if self.codebookMode == 1:
                phiN = np.exp( 1j*np.pi*n/2 )               # A scaler
                if self.ng == 2:  # p = p1
                    phiP1 = np.exp( 1j*np.pi*p/2 )
                    return np.concatenate([vlm, s*phiN*vlm, phiP1*vlm, s*phiN*phiP1*vlm], axis=-1)/np.sqrt(self.numPorts)
                
                assert self.ng == 4     # p = [p1, p2, p3]
                phiP1 = np.exp( 1j*np.pi*p[0]/2 )
                phiP2 = np.exp( 1j*np.pi*p[1]/2 )
                phiP3 = np.exp( 1j*np.pi*p[2]/2 )
                return np.concatenate([vlm,       s*phiN*vlm,       phiP1*vlm, s*phiN*phiP1*vlm,
                                       phiP2*vlm, s*phiN*phiP2*vlm, phiP3*vlm, s*phiN*phiP3*vlm], axis=-1)/np.sqrt(self.numPorts)
                
            # p = [p1, p2], n = [n0, n1, n2]
            p1, p2 = p
            n0, n1, n2 = n
            phiN0 = np.exp( 1j*np.pi*n0/2 )           # A Scalar
            aP1 = np.exp( 1j*np.pi*(p1/2 + 1/4) )
            aP2 = np.exp( 1j*np.pi*(p2/2 + 1/4) )     # A Scalar
            bN1 = np.exp( 1j*np.pi*(n1/2 - 1/4) )
            bN2 = np.exp( 1j*np.pi*(n2/2 - 1/4) )
            return np.concatenate([vlm, s*phiN0*vlm, aP1*bN1*vlm, s*aP2*bN2*vlm], axis=-1)/np.sqrt(self.numPorts)

        # ..............................................................................................................
        if numLayers == 1:
            # Note that i13 is not used.
            validateRange(i11, (0,bb1-1), context)
            validateRange(i12, (0,bb2-1), context)
                
            if self.codebookMode==1:            # See TS 38.214, Table 5.2.2.2.2-3 (1st Table)
                # i14 must be a tuple or list: [i141] (when ng=2) or [i141, i142, i143] (when ng=4)
                if len(i14) != (self.ng-1):
                    raise ValueError("'i14' must be a tuple or list of length %d"%(self.ng-1) + context + "!")
                
                validateRange(i14[0], (0,3), context, "i141")
                if self.ng==4:
                    validateRange(i14[1], (0,3), context, "i142")
                    validateRange(i14[2], (0,3), context, "i143")

                validateRange(i2[0], (0,3), context, "i2")
                
            else:   # self.codebookMode==2      # See TS 38.214, Table 5.2.2.2.2-3 (2nd Table)
                # i14 must be a tuple or list: [i141, i142]
                if len(i14) != 2:
                    raise ValueError("'i14' must be a tuple or list of length 2" + context + "!")
                validateRange(i14[0], (0,3), context, "i141")
                validateRange(i14[1], (0,3), context, "i142")

                # i2 must be a tuple or list: [i20, i21, i22]
                if len(i2) != 3:
                    raise ValueError("'i2' must be a tuple or list of length 3" + context + "!")
                validateRange(i2[0], (0,3), context, "i20")
                validateRange(i2[1], [0,1], context, "i21")
                validateRange(i2[2], [0,1], context, "i22")

            l, m, p, n = i11, i12, i14, i2
            return w(1, l, m, p, n)
        
        # ..............................................................................................................
        if numLayers == 2:
            validateRange(i11, (0,bb1-1), context)
            validateRange(i12, (0,bb2-1), context)
            
            i13Len = 2 if (self.n1==2 and self.n2==1) else 4    # From Table 5.2.2.2.1-3
            validateRange(i13, (0,i13Len-1), context)

            # Getting k1,k2 from i13: (TS 38.214, Table 5.2.2.2.1-3)
            if i13==0:                                  k1,k2 = 0, 0
            elif i13==1:                                k1,k2 = self.o1, 0
            elif i13==2:
                if (self.n1>self.n2) and (self.n2>1):   k1,k2 = 0, self.o2
                elif self.n1==self.n2:                  k1,k2 = 0, self.o2
                elif (self.n1==2) and (self.n2==1):     validateRange(i1[2], [0,1], context)
                elif (self.n1>2) and (self.n2==1):      k1,k2 = 2*self.o1, 0
                else:                                   raiseError( "Unsupported N1/N2 combination (N1=%d, N2=%d)!"%(self.n1,self.n2))
            elif i13==3:
                if (self.n1>self.n2) and (self.n2>1):   k1,k2 = 2*self.o1, 0
                elif self.n1==self.n2:                  k1,k2 = self.o1, self.o2
                elif (self.n1==2) and (self.n2==1):     validateRange(i1[2], [0,1], context)
                elif (self.n1>2) and (self.n2==1):      k1,k2 = 3*self.o1, 0
                else:                                   raise ValueError( "Unsupported N1/N2 combination (N1=%d, N2=%d)!"%(self.n1,self.n2))

            if self.codebookMode==1:            # See TS 38.214, Table 5.2.2.2.2-4 (1st Table)
                # i14 must be a tuple or list: [i141] (when ng=2) or [i141, i142, i143] (when ng=4)
                if len(i14) != (self.ng-1):
                    raise ValueError("'i14' must be a tuple or list of length %d"%(self.ng-1) + context + "!")
                
                validateRange(i14[0], (0,3), context, "i141")
                if self.ng==4:
                    validateRange(i14[1], (0,3), context, "i142")
                    validateRange(i14[2], (0,3), context, "i143")

                validateRange(i2[0], [0,1], context, "i2")

            else:   # self.codebookMode==2      # See TS 38.214, Table 5.2.2.2.2-4 (2nd Table)
                # i14 must be a tuple or list: [i141, i142]
                if len(i14) != 2:
                    raise ValueError("'i14' must be a tuple or list of length 2" + context + "!")
                validateRange(i14[0], (0,3), context, "i141")
                validateRange(i14[1], (0,3), context, "i142")

                # i2 must be a tuple or list: [i20, i21, i22]
                if len(i2) != 3:
                    raise ValueError("'i2' must be a tuple or list of length 3" + context + "!")
                validateRange(i2[0], [0,1], context, "i20")
                validateRange(i2[1], [0,1], context, "i21")
                validateRange(i2[2], [0,1], context, "i22")

            l, lp, m, mp, p, n = i11, i11+k1, i12, i12+k2, i14, i2
            return np.concatenate([ w(1, l,  m,  p, n),  w(2, lp, mp, p, n) ], axis=-1)/np.sqrt(2)

        if (self.n1==2 and self.n2==1):     i13Len = 1  # From Table 5.2.2.2.2-2
        elif (self.n1==4 and self.n2==1):   i13Len = 3  # From Table 5.2.2.2.2-2
        elif (self.n1==2 and self.n2==2):   i13Len = 3  # From Table 5.2.2.2.2-2
        else:                               i13Len = 4  # From Table 5.2.2.2.2-2

        def k12(i13):   # See TS 38.214, Table 5.2.2.2.2-2
            # This is used when numLayers∈[3,4]
            if i13==0:                                  return (self.o1, 0)
            if i13==1:
                if (self.n1==4) and (self.n2==1):       return (2*self.o1, 0)
                if (self.n1==8) and (self.n2==1):       return (2*self.o1, 0)
                if (self.n1==2) and (self.n2==2):       return (0, self.o2)
                if (self.n1==4) and (self.n2==2):       return (0, self.o2)
            if i13==2:
                if (self.n1==4) and (self.n2==1):       return (3*self.o1, 0)
                if (self.n1==8) and (self.n2==1):       return (3*self.o1, 0)
                if (self.n1==2) and (self.n2==2):       return (self.o1, self.o2)
                if (self.n1==4) and (self.n2==2):       return (self.o1, self.o2)
            if i13==3:
                if (self.n1==8) and (self.n2==1):       return (4*self.o1, 0)
                if (self.n1==4) and (self.n2==2):       return (2*self.o1, 0)
            assert False

        # ..............................................................................................................
        if numLayers == 3:
            validateRange(i11, (0,bb1-1), context)
            validateRange(i12, (0,bb2-1), context)
            
            validateRange(i13, (0,i13Len-1), context)
            k1,k2 = k12(i13)

            if self.codebookMode==1:            # See TS 38.214, Table 5.2.2.2.2-5 (1st Table)
                # i14 must be a tuple or list: [i141] (when ng=2) or [i141, i142, i143] (when ng=4)
                if len(i14) != (self.ng-1):
                    raise ValueError("'i14' must be a tuple or list of length %d"%(self.ng-1) + context + "!")
                
                validateRange(i14[0], (0,3), context, "i141")
                if self.ng==4:
                    validateRange(i14[1], (0,3), context, "i142")
                    validateRange(i14[2], (0,3), context, "i143")

                validateRange(i2[0], [0,1], context, "i2")

            else:   # self.codebookMode==2      # See TS 38.214, Table 5.2.2.2.2-5 (2nd Table)
                # i14 must be a tuple or list: [i141, i142]
                if len(i14) != 2:
                    raise ValueError("'i14' must be a tuple or list of length 2" + context + "!")
                validateRange(i14[0], (0,3), context, "i141")
                validateRange(i14[1], (0,3), context, "i142")

                # i2 must be a tuple or list: [i20, i21, i22]
                if len(i2) != 3:
                    raise ValueError("'i2' must be a tuple or list of length 3" + context + "!")
                validateRange(i2[0], [0,1], context, "i20")
                validateRange(i2[1], [0,1], context, "i21")
                validateRange(i2[2], [0,1], context, "i22")

            l, lp, m, mp, p, n = i11, i11+k1, i12, i12+k2, i14, i2
            return np.concatenate([ w(1, l,  m,  p, n),  w(1, lp, mp, p, n),  w(2, l,  m,  p, n) ], axis=-1)/np.sqrt(3)

        # ..............................................................................................................
        if numLayers == 4:
            validateRange(i11, (0,bb1-1), context)
            validateRange(i12, (0,bb2-1), context)
            
            validateRange(i13, (0,i13Len-1), context)
            k1,k2 = k12(i13)

            if self.codebookMode==1:            # See TS 38.214, Table 5.2.2.2.2-6 (1st Table)
                # i14 must be a tuple or list: [i141] (when ng=2) or [i141, i142, i143] (when ng=4)
                if len(i14) != (self.ng-1):
                    raise ValueError("'i14' must be a tuple or list of length %d"%(self.ng-1) + context + "!")
                
                validateRange(i14[0], (0,3), context, "i141")
                if self.ng==4:
                    validateRange(i14[1], (0,3), context, "i142")
                    validateRange(i14[2], (0,3), context, "i143")

                validateRange(i2[0], [0,1], context, "i2")

            else:   # self.codebookMode==2      # See TS 38.214, Table 5.2.2.2.2-6 (2nd Table)
                # i14 must be a tuple or list: [i141, i142]
                if len(i14) != 2:
                    raise ValueError("'i14' must be a tuple or list of length 2" + context + "!")
                validateRange(i14[0], (0,3), context, "i141")
                validateRange(i14[1], (0,3), context, "i142")

                # i2 must be a tuple or list: [i20, i21, i22]
                if len(i2) != 3:
                    raise ValueError("'i2' must be a tuple or list of length 3" + context + "!")
                validateRange(i2[0], [0,1], context, "i20")
                validateRange(i2[1], [0,1], context, "i21")
                validateRange(i2[2], [0,1], context, "i22")

            l, lp, m, mp, p, n = i11, i11+k1, i12, i12+k2, i14, i2
            return np.concatenate([ w(1, l,  m,  p, n),  w(1, lp, mp, p, n),  w(2, l,  m,  p, n),  w(2, lp, mp, p, n) ], axis=-1)/np.sqrt(4)
        
        raise ValueError( "Unsupported number of layers %d! (codebookType: Type1MP)"%(self.numLayers) )
        
    # ******************************************************************************************************************
    def getType2n12(self, i12):
        # See TS 38.214, Section 5.2.2.2.3
        s = 0
        n1n2 = self.n1*self.n2
        n1, n2 = [], []
        for i in range(self.numBeams):
            y = self.numBeams-i
            xStar,e = -1,-1
            for x in range(y-1, n1n2-1-i):
                if (i12-s>=cxy[x][y]) and (x>xStar):
                    xStar,e = x,cxy[x][y]
            assert (xStar>=0) and (e>=0)
            s += e
            n = n1n2-1-xStar
            n1 += [ n%self.N1 ]
            n2 += [ (n-n1[-1])//self.N1 ]
        return np.int32(n1), np.int32(n2)
        
    # ******************************************************************************************************************
    def getType2I12(self, n1, n2):
        n1n2 = self.n1*self.n2
        n = self.n1*n2 + n1
        return np.sum(cxy[n1n2-1-n[i], self.numBeams-i] for i in range(self.numBeams))

#    # ************************************************************************************************************************************************
#    def getType2Precoder(self, numLayers, i1=0, i2=0):
#        # See TS 38.214, Section 5.2.2.2.3
#        # i1 can be a number, [i11,i12,i131,i141] (when numLayers=1) or [i11,i12,i131,i141,i132,i142] (when numLayers=2)
#        #   i11 is [q1,q2] (q1∈[0...self.o1-1] and q2∈[0...self.o2-1])
#        #   i12∈[0...C-1] where C is the number of all combination of choosing 'numBeams' from self.n1*self.n2
#        #   i13l∈[0...2*numBeams-1] where l∈[1,...,numLayers]
#        #   i14l = [k1l0...k1lm] where m=2*self.numBeams-1 and l∈[1,...,numLayers] and k1li∈[0,1,...,7]
#        # i2 can be:
#        #   [i211]                      When subbandAmp=False and numLayers=1
#        #   [i211, i212]                When subbandAmp=False and numLayers=2
#        #   [i211, i221]                When subbandAmp=True and numLayers=1
#        #   [i211, i221, i212, i222]    When subbandAmp=True and numLayers=2
#        #   i22l = [k2l0...k2lm] where m=2*self.numBeams-1 and l∈[1,...,numLayers] and k2li∈[0,1]
#
#        # Strongest coefficient on layer 'l':   i13l∈[0,1,...,2*numBeams-1] (l∈[1,...,numLayers])
#        # Amplitude coefficient indicators:     i14l and i22l (l∈[1,...,numLayers])
#        # Amplitude coefficient:                p1l and p2l (l∈[1,...,numLayers])
#        # Phase coefficient indicators:         i21l (l∈[1,...,numLayers])
#        
#        def kToP1(k):   return 0 if k==0 else np.sqrt(1/(1<<(7-k)))     # See TS 38.214, Table 5.2.2.2.3-2
#        def kToP2(k):   return [np.sqrt(1/2), 1][k]                     # See TS 38.214, Table 5.2.2.2.3-3
#
#        assert self.numPorts >= 2
#        assert self.numLayers in [1, 2]
#
#        # In the following: i ∈ {0,...,2L-1} is the array index, l ∈ {1, 2} is the layer index, and L=self.numBeams
#        # Decompose i1:
#        i132, i142 = 0,0
#        if self.numLayers==1:   i11, i12, i131, i141 = i1
#        else:                   i11, i12, i131, i141, i132, i142 = i1
#        q1, q2 = i11                    # q1 and q2 are numbers                                 q1 ∈ {0,...,self.o1-1} and q2 ∈ {0,...,self.o2-1}
#        i13 = np.int32([i131, i132])    # Strongest coefficient on layers 1 and 2                               i131, i132 ∈ {0, 1,...,L}
#        i14 = np.int32([i141, i142])    # Amplitude coefficient indicators for layers 1 and 2 (k1li values)     Shape: 2 x 2L, k1li ∈ {0, 1,...,7}
#
#        # Decompose i2:
#        i211, i221, i212, i222 = 0,0,0,0
#        if subbandAmp:
#            if self.numLayers==1:   i211, i221 = i2                 # Contain c1i and k21i values correspondingly
#            else:                   i211, i221, i212, i222 = i2     # Contain c1i, k21i, c2i, k22i values correspondingly
#        elif self.numLayers==1:     i211 = i2[0]                    # Contains c1i values
#        else:                       i211, i212 = i2                 # Contain c1i and c2i values correspondingly
#        i21 = np.int32([i211, i212])  # Phase coefficient indicators for layers 1 and 2 (cli values)                Shape: 2 x 2L, cli ∈ {0,...,7}
#        i22 = np.int32([i221, i222])  # Amplitude coefficient indicators for layers 1 and 2 (k2li values)           Shape: 2 x 2L, k2li ∈ {0, 1}
#
#        numCombs = scipy.scipy.special.comb(self.n1*self.n2,self.numBeams)
#        if i12 not in range(numCombs):                  raise ValueError( "'i12' must be in {0, 1,...,%d}"%(numCombs-1) )
#        
#        for l in range(self.numLayers):
#            if i13[l] not in range(2*self.numBeams):    raise ValueError( "'i13%d' must be in {0, 1,...,%d}"%(l+1, 2*self.numBeams-1) )
#
#            if len(i14[l]) != (2*self.numBeams):        raise ValueError( "'i14%d' must contain %d values"%(l+1, 2*self.numBeams-1) )
#            for k1li in i14[l]:
#                if k1li not in range(8):                raise ValueError( "'i14%d' values must be in {0, 1,...,7}"%(l+1) )
#            if i14[l,i13[l]] != 7:                      raise ValueError( "'K1,%d,i13%d must be 7!"%(l+1,l+1) )
#
#            if len(i21[l]) != (2*self.numBeams):        raise ValueError( "'i21%d' must contain %d values"%(l+1, 2*self.numBeams-1) )
#            for i,cli in enumerate(i21[l]):
#                if cli not in range(self.pskSize):      raise ValueError( "'i14%d' values must be in {0, 1,...,%d}"%(l+1, self.pskSize-1) )
#                if subbandAmplitude == False:
#                    if (i14[l,i]<=0) and (cli!=0):      raise ValueError( "'C%d,%d must be 0 if k(1)%d,%d is not positive."%(l+1,i, l+1,i) )
#            if i21[l,i13[l]] != 0:                      raise ValueError( "'C%d,i13%d must be 0!"%(l+1,l+1) )
#
#            if len(i22[l]) != (2*self.numBeams):        raise ValueError( "'i22%d' must contain %d values"%(l+1, 2*self.numBeams-1) )
#            if not self.subbandAmp:  # If 'subbandAmplitude' is false, all K2li values must be 1 and the whole i22l is not reported.
#                if np.any(i22[l]!=1):                   raise ValueError( "'i22%d' values must all be 1 when 'subbandAmp' is False."%(l+1) )
#            else:
#                for k2li in i22[l]:
#                    if k2li not in [0,1]:               raise ValueError( "'i22%d' values must be in {0,1}"%(l+1) )
#                if i22[l,i13[l]] != 1:                  raise ValueError( "'K2,%d,i13%d must be 1!"%(l+1,l+1) )
#
#        m = (i14>0).sum(1)      # Ml (=m[l]) is the number of elements of i14l (K1li values) that are positive. (Always 2 values)
#
#        # n1 and n2 contain n1i and n2i values correspondingly, where i ∈ {0,...,L-1}, n1i ∈ {0,...,N1-1}, and n2i ∈ {0,...,N2-1}
#        n1, n2 = self.getType2n12(i12)
#        m1 = self.o1*n1 + q1            # Same shape as n1 (array of length L)
#        m2 = self.o2*n2 + q2            # Same shape as n2 (array of length L)
#        
#        p1 = np.array([ [kToP1(k1li) for k1li in i14[l-1]] for l in [1,2] ])   # p1li values where i ∈ {0,...,2L-1} and l ∈ {1,2}, shape: 2 x 2L
#        p2 = np.array([ [kToP2(k2li) for k2li in i22[l-1]] for l in [1,2] ])   # p2li values where i ∈ {0,...,2L-1} and l ∈ {1,2}, shape: 2 x 2L
#
#        def w(q1, q2, n1, n2, p1, p2, c):
#            vm1m2 = np.array([v(m1i,m2i) for i in range(self.numBeams)])    # Shape: numBeams x N1 x N2
#            phi =
#            w1 = np.sum(vm1m2*p1[:,None,None]*p2[:,None,None]*
#            return np.concatenate([ vlm(m1[i], m2[i]) * p1[i] * p2[i] *  ])
#
#        if self.numLayers==1:
#            
#        
#        
#        
#        if self.n2==1:
#            # q2 is not reported
#            n2i = np.int32(self.numBeams*[0])
#            q2 = 0
#            
#            if self.n1==2:                                          n1i, n2i = np.int32([0,1]),    np.int32([0,0])      # i12 is not reported
#            if (self.n1==4) and (self.numBeams==4):                 n1i, n2i = np.int32(range(4)), np.int32(4*[0])      # i12 is not reported
#        
#        if (self.n1==2) and (self.n2==2) and (self.numBeams==4):    n1i, n2i = np.int32([0,1,0,1]), np.int32([0,0,1,1]) # i12 is not reported
#
#
#
#    # ************************************************************************************************************************************************
#    def getCodebook(self, numPorts, numLayers):
#        # See https://www.sharetechnote.com/html/5G/5G_CSI_RS_Codebook.html
#        # A precoding matrix has 𝛎*P values. (𝛎=numLayers, P=numPorts=2*n1*n2). A codebook has B=b1*b2=n1*o1*n2*o2 entries for each
#        # one B beams. There can be C codebooks to choose from for each configuration.
#        
#        if self.codebookType == 'Type1SP':  return self.getType1SpCodebook(numPorts, numLayers)     # Type 1 Single-Panel
#            
#        if self.codebookType == 'Type1MP':  assert False, "Not Implemented yet!"                    # Type 1 Multi-Panel
#        
#        return None     # No actual codebooks for Type-2 and enhanced type-2
#
#
#    # ************************************************************************************************************************************************
#    def getType1SpCodebook(self, numLayers):
#        if numPorts == 1:   return [1] # when numPorts is 1, the codebook contains a single value equal to 1
#        
#        # codebooks.shape: (numCodebooks, numPorts, numLayers)
#        # codebooks.shape: (C, B2, B1, P, 𝛎)
#        # where:
#        #   P=2*N1*N2 (numPorts),
#        #   B=B1*B2=N1*O1*N2*O2 (Number of beams),
#        #   C=max(i2)+1 i2∈[0,C) (Number of codebooks)
#
#        if numPorts == 2:  # See 3GPP TS 38.214, Table 5.2.2.2.1-1
#            if numLayers == 1:          # C=4, n1=n2=1, o1=o2=1, P=2, 𝛎=1, B=1
#                codebooks = np.array([ [[1], [1]], [[1], [1j]], [[1], [-1]], [[1], [-1j]] ])/np.sqrt(2)     # Shape: 4, 1, 1, 2, 1
#                for i in range(4):      # Bits 0 to 3 are used for numLayers=1 (See 3GPP TS 38.214, Section 5.2.2.2.1)
#                    if codebooksAllowed[-i-1]=='0': codebooks[i] = 0        # Codebook i is restricted
#                        
#            elif numLayers == 2:        # C=2, n1=n2=1, o1=o2=1, P=2, 𝛎=2
#                codebooks = np.array([ [[1, 1], [1, -1]], [[1, 1], [1j, -1j]] ])/2                          # Shape: 2, 1, 1, 2, 2
#                for i in [4,5]:         # Bits 4 and 5 are used for numLayers=2 (See 3GPP TS 38.214, Section 5.2.2.2.1)
#                    if codebooksAllowed[-i-1]=='0': codebooks[i-4] = 0     # Codebook i-4 is restricted
#        
#        else: # numPorts>2
#            nn2, nn1 = self.txAntenna.shape     # N1 is horizontal elements (=y axis)
#            o2, o1 = self.overSampling          # O1 is horizontal elements (=y axis)
#            bb1, bb2 = nn1*o1, nn2*o2           # B1, B2 number of beams (horizontal and vertical)
#            
#            def get4Codebook(bn1, bn2):
#                bbl1, bbl2 = bn1.shape[0], bn2.shape[0]
#                u1 = np.exp( 2*np.pi*1j*bn1/(bb1) )                             # Same shape as bn1 (bbl1 x N1)
#                u2 = np.exp( 2*np.pi*1j*bn2/(bb2) )                             # Same shape as bn2 (bbl2 x N2)
#                
#                vml = np.outer(u2, u1).reshape(bbl2, nn2, bbl1, nn1)            # Shape: bbl2 x N2 x bbl1 x N1
#                vml = np.transpose(vml, (0,2,1,3)).reshape(bbl2, bbl1, -1)      # Shape: bbl2 x bbl1 x N2*N1
#                
#                phi = np.exp(1j*np.pi*np.arange(4)/2)                           # Shape: 4
#                phiVml = np.outer(phi,vml).reshape(4, bb2, bb1, -1)             # Shape: 4 x bbl2 x bbl1 x N2*N1
#                return np.concatenate([4*[vml], phiVml], axis=-1)               # Shape: 4 x bbl2 x bbl1 x P
#
#            if numLayers == 1:
#                # See 3GPP TS 38.214, Table 5.2.2.2.1-5
#                if self.codebookMode == 1:
#                    # See 3GPP TS 38.214, Table 5.2.2.2.1-5  (1st Table)
#                    cc = 4
#                    bn1 = np.outer(np.arange(bb1), np.arange(nn1))              # Shape: B1 x N1
#                    bn2 = np.outer(np.arange(bb2), np.arange(nn2))              # Shape: B2 x N2
#                    codebooks = get4Codebook(bn1, bn2)                          # Shape: 4 x bb2 x bb1 x P
#                    codebooks = codebooks.reshape(cc, bb2, bb1, numPorts, 1)    # Shape: 4 x B2 x B1 x P x 𝛎  (𝛎=1)
#                    
#                else:   # self.codebookMode = 2
#                    # See 3GPP TS 38.214, Table 5.2.2.2.1-5  (2nd and 3rd Tables for N2>1 and N2=1 cases)
#                    codebooks = []
#                    llmm = [[0,0],[1,0],[0,1],[1,1]] if nn2>1 else [[0,0],[1,0],[2,0],[3,0]]
#                    for ll,mm in llmm:
#                        bn1 = np.outer(2*np.arange(bb1//2)+ll, np.arange(nn1))              # Shape: B1/2 x N1
#                        bn2 = np.outer(2*np.arange(max(bb2//2,1))+mm, np.arange(nn2))       # Shape: B2/2 x N2 (or 1 x N2 when N2 is 1)
#                        codebooks += [ get4Codebook(bn1, bn2) ]                             # Shape: 4 x bb2 x bb1 x P
#                    codebooks = np.array(codebooks).reshape(16, -1, bb1//2, numPorts, 1)    # Shape: 16 x B2 x B1 x P x 𝛎  (𝛎=1)
#
#        return codebooks
