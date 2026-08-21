# Copyright (c) 2026, InterDigital AI Lab
"""
The module ``csireport.py`` implements the :py:class:`CsiReport` and :py:class:`CsiReportMan` classes used 
to process CSI information on the UE side and create CSI-feedback reports such as CRI, RI, PMI, and CQI.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 03/12/2026    Shahab Hamidi-Rad       Support started in NeoRadium version 0.5.0:
#                                       * Implemented the first version of the code.
# 08/07/2026    Shahab                  Changes in NeoRadium version 0.5.1:
#                                       * Added support for `CriRiPmiCqi` quantity.
#                                       * 'CsiReport' class now accepts the 'rxAntenna' and 'ldpcIter' parameters
#                                         used with CQI lookup tables.
#                                       * Renamed `makeTypicalReports()` to `beamformingReports()` in `CsiReportMan`
#                                         and added improved configuration controls.
# **********************************************************************************************************************
# This implementation is based on:
#   - 3GPP TS 38.214
#   - https://www.sharetechnote.com/html/5G/5G_CSI_Report.html
#   - The book: 5G NR: The next generation wireless access technology, Sections 8.2, 11.2,
# Other reads:
#   * New Radio Physical Layer Abstraction for System-Level Simulations of 5G Networks
#   * BLER-SNR Curves for 5G NR MCS under AWGN Channel with Optimum Quantization
#   * https://github.com/QiuYukang/5G-LENA/blob/master/model/nr-eesm-t1.cc

import numpy as np
from dataclasses import dataclass

from .pdsch import PDSCH
from .antenna import AntennaArray, AntennaPanel
from .csirs import CsiRsSet, CsiRsConfig
from .utils import herm, toLinear, toDb, validateRange, deprecated
from .cbtype1sp import PmiCbT1Sp
from .cqidata import CqiData

docFile = "CsiReport"          # Used by the 'deprecated' decorators

# **********************************************************************************************************************
# CSI-Feedback structure:
NpComplex = np.typing.NDArray[np.complexfloating]
@dataclass
class CriFeedback:
    cri:        int | list[int]                     # The CSI-RS ID(s) corresponding to the best beam(s)
    rsrp:       float | list[float] | None = None   # RSRP(s) of the best beam(s) (in dB). Set when quantity="Cri".
    score:      float | None = None                 # Spectral-efficiency score of the best beam.
                                                    # Set when quantity="CriRiPmiCqi".

@dataclass
class RiFeedback:
    ri:         int                                 # The rank (number of layers)
    score:      float                               # The score of the selected rank

@dataclass
class PmiIndex:
    i1:         list[int]                           # First part of PMI index: [i11,i12,i13]
    i2:         int                                 # Second part of PMI index
    
    def __repr__(self): return f"(I1:{self.i1}, I2:{self.i2})"
    
@dataclass
class PmiFeedback:
    wbPMI:      PmiIndex                                            # PMI index [[i11,i12,i13],i2]
    wbW:        NpComplex                                           # The precoding matrix
    sbPMIs:     list[PmiIndex] | None = None                        # List of PMI indices for each subband
    sbWs:       list[ tuple[list[int], NpComplex] ] | None = None   # Subband precoder structure

@dataclass
class CqiFeedback:
    cqis:       list[int]                           # One CQI per codeword
    blers:      list[float]                         # The BLER values for each codeword

    @property
    def cqi(self):  return self.cqis[0]
    
    @property
    def bler(self): return self.blers[0]

@dataclass
class CsiFeedback:
    cri:        CriFeedback | None = None
    ri:         RiFeedback | None = None
    pmi:        PmiFeedback | None = None
    cqi:        CqiFeedback | None = None
    
    @property
    def hasInfo(self):
        return ((self.cri is not None) or (self.ri is not None) or (self.pmi is not None) or (self.cqi is not None))

# **********************************************************************************************************************
class CsiReport:
    r"""
    Represents one CSI feedback report configuration (referred to as a `CSI-ReportConfig` in **3GPP TS 38.214**) and  
    the associated UE-side processing state.

    This class consumes CSI-RS observations from a received resource grid and generates CSI feedback
    information according to the configured report quantity. Supported feedback quantities include:

        :Cri: CSI-RS Resource Indicator, channel/beam selection based on CSI-RS RSRP
        :RiPmiCqi: Rank indicator (RI), precoding matrix indicator (PMI), and channel quality indicator (CQI), jointly 
            referred to as channel-state information (CSI)
        :RiPmi: Rank indicator (RI) and precoding matrix indicator (PMI)

    A report may be configured as `periodic`, `semi-persistent-on-PUCCH`, `semi-persistent-on-PUSCH`,
    or `aperiodic`. The object maintains internal measurement state across slots when needed
    (for example, when CRI requires measurements from multiple CSI-RS resources before selecting
    the best one).
    """
    # ******************************************************************************************************************
    def __init__(self, csiRsSets, **kwargs):
        """
        Parameters
        ----------
        csiRsSets : :py:class:`~neoradium.csirs.CsiRsSet` or list of :py:class:`~neoradium.csirs.CsiRsSet`
            One CSI-RS resource set or a list of CSI-RS resource sets used by this report. All resource sets are 
            assumed to belong to the same bandwidth part.

        kwargs : dict
            Optional configuration parameters:

            :reportId:
                Report identifier. If not specified, ``csiRsSets[0].rsId + 10`` is used.

            :reportType:
                CSI reporting type. Supported values are ``"periodic"`` (default), ``"spOnPUCCH"``,
                ``"spOnPUSCH"``, and ``"aperiodic"``.

            :period:
                Reporting period in slots for periodic and semi-persistent reporting modes. The
                default is ``5``.

            :offset:
                Slot offset for periodic and semi-persistent reporting modes. The default is ``0``.

            :active:
                Initial active state for semi-persistent reporting modes. The default is 
                ``0`` (inactive)

            :quantity:
                Type of CSI quantity to report. Supported values are ``"Cri"`` (default),
                ``"RiPmiCqi"``, ``"CriRiPmiCqi"``, and ``"RiPmi"``.

            :numCri: 
                Number of ``(resourceId, RSRP)`` pairs reported when ``quantity`` is set to 
                ``"Cri"``. If ``1`` (default), ``CriFeedback.cri`` is the resource ID corresponding
                to the highest RSRP, and ``CriFeedback.rsrp`` is that RSRP value (in dB). If 
                ``K > 1``, ``CriFeedback.cri`` is a list of top-K resource IDs, and ``CriFeedback.rsrp``
                is a list of their corresponding RSRP values (in dB, descending order). This 
                Corresponds to ``nrofReportedRS`` in **3GPP TS 38.214 and TS 38.331**.
                
            :txAntenna:
                Transmit antenna configuration used for PMI/RI/CQI processing (An 
                :py:class:`~neoradium.antenna.AntennaPanel` object). This is required when 
                quantity is set to ``"RiPmiCqi"`` or ``"RiPmi"``.
                 
            :codebookType:
                Precoder codebook type. Currently only ``"Type1SP"`` is supported.

            :allowedRanks:
                List of allowed transmission ranks considered during RI/PMI selection. The
                default is ``[1, 2, 3, 4]``.

            :prgSize:
                Size of precoding resource groups (PRGs). A value of 0 (default) means wideband PMI.

            :pmiGranularity:
                PMI granularity, either ``"wideband"`` or ``"subband"``.

            :cqiTable:
                CQI table index as defined by **3GPP TS 38.214**. The default is ``1``. 
        """
        if isinstance(csiRsSets, CsiRsSet): self.csiRsSets = [csiRsSets]
        elif isinstance(csiRsSets, list):   self.csiRsSets = csiRsSets
        else:  raise ValueError( "'csiRsSets' must be a list of 'CsiRsSet' objects." )
        self.reportId = kwargs.get('reportId', self.csiRsSets[0].rsId+10)
        self.bwp = self.csiRsSets[0].bwp    # Assume all CSI-RS resource sets are associated with the same BWP.
        for csiRsSet in self.csiRsSets:     # Make sure there are no ZP resources
            if csiRsSet.csiType=="ZP":
                raise ValueError( "'ZP' resources are not currently supported for CSI-RS reports." )
        
        self.reportType = kwargs.get('reportType', "periodic")      # Higher-layer parameter "reportConfigType"
        validateRange(self.reportType, ["periodic", "spOnPUCCH", "spOnPUSCH", "aperiodic"])
        self.period = kwargs.get('period', 5)                       # Used for periodic, spOnPUCCH, and spOnPUSCH cases
        self.offset = kwargs.get('offset', 0)                       # Used for periodic, spOnPUCCH, and spOnPUSCH cases
        
        # For 'spOnPUCCH' and 'spOnPUSCH', active is 1 (active) or 0 (inactive)
        # For 'aperiodic', active is set to 1 when triggered and reset to 0 when the report is sent.
        # Note: The behavior of 'aperiodic' here differs from that of CSI-RS resource sets, because the slotNo is
        #       different when the report is triggered vs when we want to create the report. The report remains active
        #       until it is actually sent.
        if self.reportType=='aperiodic':    self.active = 0
        elif self.reportType=='periodic':   self.active = 1
        else:                               self.active = kwargs.get('active', 0)
        
        if self.reportType in ["periodic", "spOnPUCCH"]:    validateRange(self.period, [5, 10, 20, 40, 80, 160, 320])
        elif self.reportType == "spOnPUSCH":    validateRange(self.period, [4, 5, 8, 10, 16, 20, 32, 40, 80, 160, 320])
        validateRange(self.offset, (0,self.period-1))
        
        self.quantity = kwargs.get('quantity', 'Cri')               # See 3GPP TS 38.214, Section 5.2.1.4
        validateRange(self.quantity, ['Cri', 'RiPmiCqi', 'RiPmi', 'CriRiPmiCqi'])

        self.numCri = kwargs.get('numCri', 1)   # Number of (resourceId, RSRP) pairs when quantity is set to "Cri"

        # PMI/Ri/CQI settings ------------------------------------------------------------------------------------------
        if ('Pmi' in self.quantity) or ('Ri' in self.quantity) or ('Cqi' in self.quantity):
            for csiRsSet in self.csiRsSets:
                for csiRs in csiRsSet.csiRsList:
                    # CSI-RS with density 3 is mostly used for TRS.
                    if csiRs.density == 3:
                        raise ValueError("CSI-RS with density=3 is not supported for PMI/RI/CQI reports!")
                
            self.txAntenna = kwargs.get('txAntenna', None)
            if self.txAntenna is None:
                raise ValueError("The TX antenna configuration is missing!")
            
            self.codebookType = kwargs.get('codebookType', 'Type1SP')
            validateRange(self.codebookType, ['Type1SP'])  # TODO: Add other types later (Type1MP, Type2, EnhancedType2)
            if isinstance(self.txAntenna, AntennaPanel):
                self.ng = 1
                self.codebook = PmiCbT1Sp(**kwargs)
            elif isinstance(self.txAntenna, AntennaArray):
                self.ng = np.prod(self.txAntenna.shape)
                if self.ng == 1:
                    self.codebook = PmiCbT1Sp(self.txAntenna.panels[0][0], cbMode=2, **kwargs)
                else:
                    raise ValueError("Multi-Panel codebooks are not supported yet!")
            else:
                raise ValueError("Unsupported antenna class '%s'!"%(self.txAntenna.__class__.__name__))

            self.allowedRanks = kwargs.get('allowedRanks', [1,2,3,4])   # The allowed ranks
            
            # The size of precoding resource-block groups (PRGs). See 3GPP TS 38.214, Section 5.1.2.3
            # If this is provided, it will be used as the subband size for PMI.
            # 0 means 'wideband' which means a single precoding is used for all PRBs.
            self.prgSize = kwargs.get('prgSize', 0)
            validateRange(self.prgSize, [0,2,4])
            if (self.prgSize>0) and (self.bwp.numRbs % self.prgSize)!=0:
                raise ValueError(f"Number of RBs in the bandwidth part must be a multiple of {self.prgSize}!")
            self.pmiGranularity = kwargs.get('pmiGranularity', 'wideband' if self.prgSize==0 else 'subband').lower()
            validateRange(self.pmiGranularity, ['wideband', 'subband'])
            if self.pmiGranularity=='subband' and self.prgSize==0:
                raise ValueError("'prgSize' cannot be zero when pmiGranularity='subband'")

            # CQI Settings ---------------------------------------------------------------------------------------------
            # See 3GPP TS 38.214, Section 5.2.2.1 for the cqi-table values:
            #   1 -> 'table1'     -> Table 5.2.2.1-2   Error Prob: 0.1
            #   2 -> 'table2'     -> Table 5.2.2.1-3   Error Prob: 0.1
            #   3 -> 'table3'     -> Table 5.2.2.1-4   Error Prob: 0.00001
            #   4 -> 'table4-r17' -> Table 5.2.2.1-5   Error Prob: 0.1
            self.cqiTable = kwargs.get('cqiTable', 1)
            self.cqiData = CqiData()
            self.rxAntenna = kwargs.get('rxAntenna', None)
            self.ldpcIter = kwargs.get('ldpcIter', 5)
            # TODO: Add subband CQI support
            # Currently we only support wideband CQI.
            # self.cqiGranularity = kwargs.get('cqiGranularity', 'wideband' if self.prgSize==0 else 'subband').lower()
            # validateRange(self.cqiGranularity, ['wideband', 'subband'])
            # if self.cqiGranularity=='subband' and self.prgSize==0:
            #     raise ValueError("'prgSize' cannot be zero when cqiGranularity='subband'")

        if self.quantity == 'Cri':
            # CRI settings/initialization ------------------------------------------------------------------------------
            self.criRsrpValues = {rs.resourceId: None for s in self.csiRsSets for rs in s.csiRsList}

        if self.quantity == 'CriRiPmiCqi':
            # For this case we need to store RI/PMI information until we have them for all resources
            # in the set. Then we can choose the best, send the report, and reset this temporary storage.
            self.riPmiInfo = {s.rsId: {rs.resourceId:None for rs in s.csiRsList} for s in self.csiRsSets}

        self.csiFeedback = CsiFeedback()

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CSI report.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "CSI Report Properties:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "CSI Report Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  reportId:           {self.reportId}\n"
        repStr += indent*' ' + f"  reportType:         {self.reportType}\n"
        if self.reportType in ["periodic", "spOnPUCCH", "spOnPUSCH"]:
            repStr += indent*' ' + f"  period:             {self.period}\n"
            repStr += indent*' ' + f"  offset:             {self.offset}\n"
        repStr += indent*' ' + f"  quantity:           {self.quantity}\n"

        if ('Pmi' in self.quantity) or ('Ri' in self.quantity):
            repStr += indent*' ' + f"  codebookType:       {self.codebookType}\n"
            repStr += indent*' ' + f"  allowedRanks:       {self.allowedRanks}\n"
            repStr += indent*' ' + f"  prgSize:            {self.prgSize}\n"
            repStr += indent*' ' + f"  pmiGranularity:     {self.pmiGranularity}\n"
        if ('Cqi' in self.quantity):
            repStr += indent*' ' + f"  cqiTable:           {self.cqiTable}\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def __len__(self):  return len(self.csiRsSets)                              # Undocumented

    # ******************************************************************************************************************
    def trigger(self):
        r"""
        Trigger an `aperiodic` CSI report. For reports configured with ``reportType="aperiodic"``, this method 
        activates the report so that feedback can be generated the next time the reporting condition is met. For 
        other report types, this method has no effect.
        """
        if self.reportType == 'aperiodic':
            self.active = 1

    # ******************************************************************************************************************
    @property                           # Undocumented (Already documented above)
    def isActive(self):
        r"""
        Whether this CSI report is currently active.

        Returns
        -------
        bool
            `True` if the report is currently active; otherwise ``False``.

        Notes
        -----
        - Periodic reports are always considered active.
        - Aperiodic and semi-persistent reports depend on the current internal activation state.
        """
        if self.reportType in ['aperiodic', "spOnPUCCH", "spOnPUSCH"]:  return (self.active==1)
        return True     # periodic is always active

    # ******************************************************************************************************************
    def anythingForCurSlot(self, upDelay):                                      # Undocumented
        # Return True if this CSI report has any allocations in the current slot in the bandwidth part
        # upDelay: Uplink reporting delay, in slots. This delay is subtracted from the current slot timing
        #          when checking periodic and semi-persistent reporting opportunities.
        if self.reportType == 'aperiodic':                  return self.isActive
        if self.reportType in ["spOnPUCCH", "spOnPUSCH"]:
            if self.isActive == False:
                return False

        if (self.bwp.slotNo - upDelay - self.offset)<0:     return False
        return ((self.bwp.slotNo - upDelay - self.offset)%self.period)==0
    
    # ******************************************************************************************************************
    def getRsrp(self, rxCsiRsValues, csiRsValues, noiseVar=0, mrc=False):       # Undocumented
        # Estimate CSI-RS RSRP from received CSI-RS resource elements. This method first forms an LS estimate per
        # CSI-RS RE and then averages the resulting power metric.
        # rxCsiRsValues: Received CSI-RS values with shape ``Nr x M``, where ``Nr`` is the number of receive antennas
        #                and ``M`` is the number of CSI-RS resource elements.
        # csiRsValues: Reference CSI-RS values with shape ``1 x M`` or an equivalent broadcast-compatible shape.
        # noiseVar: Noise variance used for simple noise-power compensation.
        # mrc: If True, maximum-ratio combining is used across receive antennas before RSRP estimation.
        nr,m = rxCsiRsValues.shape
        h = rxCsiRsValues/(csiRsValues[0] + 1e-12)  # LS effective channel estimate per RE per Rx antenna, shape: nr x m
        if mrc:                                             # Maximum Ratio Combining
            hNorm = np.linalg.norm(h, axis=0)               # This is the norm across RX antennas, shape: (m,)
            u = h/(hNorm + 1e-12)                           # The MRC combiner. shape: nr x m
            z = (np.conj(u) * rxCsiRsValues).sum(0)         # combined values per RE, shape: (m,)
            rsrp = np.square(np.abs(z)).mean()              # Average power of MRC values over all CSI-RS REs
        else:
            csiRsRePowers = np.sum(np.abs(h)**2, axis=0)    # Sum over all RX antenna, shape: (m,)
            rsrp = csiRsRePowers.mean()                     # Average of channel powers over all CSI-RS REs
        
        ps = np.square(np.abs(csiRsValues)).mean()          # Average power of CSI-RS REs (this is usually 1)
        rsrp = np.maximum(rsrp - noiseVar/ps, 0)            # Correct by removing the noise power
        return rsrp

    # ******************************************************************************************************************
    def processCri(self, rxGrid, csiRsSet, setResources):                       # Undocumented
        # Process CSI-RS measurements for CRI selection. This method collects CSI-RS RSRP measurements for all CSI-RS
        # resources in the configured set and, once all required measurements are available, selects the resource
        # with the strongest RSRP as the current CRI. This is used only when the 'quantity' is set to "Cri"; when
        # 'quantity' is set to "CriRiPmiCqi" this function is not used.
        # rxGrid: Received resource grid.
        # csiRsSet: CSI-RS resource set being processed.
        # setResources: Resource information for the CSI-RS resources in ``csiRsSet``. Each entry maps a resource ID
        #               to a tuple (lIdx, kIdx, csiRsValues)
        # The selected CRI and the corresponding RSRP are stored in ``self.csiFeedback``.
        for resourceId, (lIdx, kIdx, csiRsValues) in setResources.items():
            rxCsiRsValues = rxGrid[:,lIdx, kIdx]                       # Get received CSI-RS REs at each RX antenna
            rsrp = self.getRsrp(rxCsiRsValues, csiRsValues, rxGrid.noiseVar)   # No noise power compensation
            self.criRsrpValues[ resourceId ] = rsrp

        rsrps = list(self.criRsrpValues.values())
        if None in rsrps:   return          # We don't have RSRP for all resources yet -> Cannot get CRI
        idx = np.argsort(rsrps)[::-1][:self.numCri]     # Top-K RSRP indices (K=self.numCri)
        ids = np.array(list(self.criRsrpValues.keys()))[idx].tolist()

        if self.numCri == 1:    self.csiFeedback.cri = CriFeedback(ids[0], toDb(rsrps[ idx[0] ]))       # Best
        else:                   self.csiFeedback.cri = CriFeedback(ids, toDb(rsrps)[idx].tolist() )     # Top-K
        
        # Now reset all RSRPs in 'criRsrpValues' to prepare for the next measurement round.
        for resourceId in self.criRsrpValues:
            self.criRsrpValues[resourceId] = None
       
    # ******************************************************************************************************************
    def getModRate(self, cqi):
        mod, rate, _ = self.cqiData.cqiTables[self.cqiTable][cqi]
        return mod, rate

    # ******************************************************************************************************************
    def processCqi(self, sinrs, numLayers):                                     # Undocumented
        # Determine CQI from per-resource-block SINR values. This method evaluates each candidate entry in the
        # configured CQI table and selects the one whose estimated BLER is closest to, but not worse than, the target
        # BLER threshold.
        # sinrs: SINR values with shape M x numLayers, where M is the number of RBs carrying
        #        usable CSI-RS channel information.
        # numLayers: Number of transmission layers assumed when evaluating CQI.
        # Returns a tuple (cqi, modulation, rate, bler) containing:
        #   - CQI index
        #   - modulation name
        #   - code rate scaled by 1024
        #   - estimated BLER in percent

        # For each modulation-and-rate entry in the CQI table:
        #    a) Create a reference PDSCH and get TBS from that
        #    b) Estimate BLER using sinrs
        # Then select the table entry that gives the closest BLER that is smaller than the threshold value.
        blers = np.ones(16)*100.0       # There are 16 entries in each CQI table
        for i, (mod,rate,_) in enumerate( self.cqiData.cqiTables[self.cqiTable] ):
            if mod is None:  continue
            modRate = f"{mod}~{rate}"
            coderate = rate/1024
            refPdsch = PDSCH(self.bwp, numLayers=numLayers, modulation=mod)
            refPdsch.setDMRS()   # Use all default values
            tbs = refPdsch.getTxBlockSize(coderate)[0]              # Assuming one codeword for now (numLayers<5)
            # Get the BLER in percent:
            rxAntShape = (1,2) if self.rxAntenna is None else self.rxAntenna.shape
            blers[i] = self.cqiData.getBler(sinrs.flatten(), modRate, tbs, self.bwp.spacing, self.ldpcIter,
                                            numLayers, rxAntShape, self.txAntenna.shape)

        blerThreshold = 0.001 if self.cqiTable==3 else 10   # BLER Threshold (in percent) - TS 38.214 §5.2.2.1
        cqi=0                                               # Default is "Out of Range" (cqi=0), if all BLERs>threshold.
        for i, bler in enumerate(blers):
            if bler<blerThreshold: cqi = i                  # CQI is the last entry with BLER lower than the threshold.
            
        return CqiFeedback([cqi], [blers[cqi]])

    # ******************************************************************************************************************
    @classmethod
    def getSINR(cls, h, noiseVar):                                              # Undocumented
        # Compute post-equalization SINR from an effective channel matrix. This method forms an MMSE equalizer from
        # ``h`` and ``noiseVar`` and computes the resulting signal, interference, and noise powers per layer.
        # h: Effective channel matrix with shape ... x Nr x Nl, where Nr is the number of
        #    receive antennas and Nl is the number of layers.
        # noiseVar: Noise variance.
        # Returns: SINR values with shape ... x Nl.
        u, s, vH = np.linalg.svd(h, full_matrices=False)       # ... x Nr x Nl, ... x Nl, ... x Nl x Nl
    
        def diagonalize(y): return np.eye(y.shape[-1])*y[...,None]
        g = herm(vH) @ diagonalize(s/(s*s+noiseVar)) @ herm(u) # MMSE Equalizer:              ... x Nl x Nr
        noisePower = noiseVar * ((np.abs(g)**2).sum(-1))       # Noise Power:                 ... x Nl
        
        a2 = np.abs(g @ h)**2                                  # A² (Gain power), A = G.H,    ... x Nl x Nl
        sigPower = np.diagonal(a2, axis1=-2, axis2=-1)         # Signal Power                 ... x Nl
        intPower = a2.sum(-1) - sigPower                       # Interference Power           ... x Nl
        return sigPower/(intPower+noisePower+1e-20)            # Gamma (SINR)                 ... x Nl

    # ******************************************************************************************************************
    def getSbScore(self, h, sbWs, noiseVar):
        # Calculates the overall score for the selected rank and subband PMIs
        m, nr, nt = h.shape
        _, nl = sbWs[0][1].shape
        
        effChan = np.zeros((m, nr, nl), dtype=np.complex128)    # Shape: M x Nr x Nl
        for rbs, w in sbWs: effChan[rbs] = h[rbs] @ w
        sinrs = self.getSINR(effChan, noiseVar)                 # M x Nl
        sbScore = np.log2(1+sinrs).sum(-1).mean(-1)
        return sbScore

    # ******************************************************************************************************************
    def getPmiForRank(self, rbs, h, noiseVar, numLayers):                       # Undocumented
        # Evaluate the precoder codebook for a given rank and select PMI candidates. The codebook is searched by
        # maximizing the average spectral-efficiency score derived from the estimated SINR values.
        # rbs: Resource-block indices corresponding to the channel matrices in 'h'.
        # h: Channel matrices with shape M x Nr x Nt, where M is the number of RBs,
        #    Nr is the number of receive antennas, and Nt is the number of transmit ports.
        # noiseVar: Noise variance used in SINR evaluation.
        # numLayers: Candidate transmission rank to evaluate.
        # Returns: A tuple (score, wbInfo, sbInfo, rbSinrs) where:
        #   - score is the overall score of the selected PMI for this rank (numLayers)
        #   - wbInfo is (pmiIndex, W) for the best wideband precoder
        #   - sbInfo contains subband PMI information when subband PMI is enabled
        #   - rbSinrs contains the resulting RB-wise SINR values for the selected precoder(s)
        m, nr, nPorts = h.shape
        cbIdx, cb = self.codebook.getCodebookInfo(numLayers)    # codebook shape: numCb x numPorts x numLayers
        
        effChan = h[None, ...] @ cb[:,None,...]         # numCb x m x nr x numLayers
        sinrs = self.getSINR(effChan, noiseVar)         # numCb x m x numLayers
        # The score is sum of spectral efficiency over all layers
        scores = np.log2(1+sinrs).sum(-1)               # numCb x m
        wbScores = scores.mean(-1)                      # numCb
        wbIdx = wbScores.argmax()                       # Index of the codebook entry corresponding to the highest score
        wbInfo = (cbIdx[wbIdx], cb[wbIdx], wbScores[wbIdx])     # (PMI Index, W, score)

        if self.pmiGranularity != 'subband':
            # If PMI is wideband, we are done. The overall score is the wideband score.
            return wbScores[wbIdx], (cbIdx[wbIdx], cb[wbIdx]), (None, None), sinrs[ wbIdx ]

        # Subband PMI:
        # The I1 in the 'wbIdx' is shared for all subband precoders. So, we first filter
        # codebook entries and keep only the ones with the wideband I1 found above. Then
        # we find the best I2 for each subband in the loop below.
        wbI1 = cbIdx[wbIdx][0]
        filteredCbIdx = [i for i, idx in enumerate(cbIdx) if np.all(idx[0]==wbI1)]  # Indices of CB entries with wbI1
        filteredScores = scores[ filteredCbIdx,: ]          # Filtered scores, len(filteredCbIdx) x m
        
        # sbPmiIdx is a list of PMI values for each subband. sbWs is a list of tuples of (sbRbIdx, sbW) for each
        # subband. This list is ready to be applied as precoder using PDSCH.precodeTo function. sbScores is the
        # score of each best subband precoder for each subband.
        sbPmiIdx, sbWs, sbScores = [], [], []

        # Find the highest score for each subband:
        sbRbIdx = []    # Indices of RBs in scores and SINR for each subband (e.g. [0,1,2,3], [4,5,6,7], etc.)
        sbSinrs = []    # SINRs for each subband
        for i,rb in enumerate(rbs):
            if (rb%self.prgSize == 0) and len(sbRbIdx)>0:
                sbScores = filteredScores[:,sbRbIdx]    # numFilteredCb x prgSize (or numFilteredCb x prgSize/2)
                sbfIdx = sbScores.mean(-1).argmax()     # Index of best score in the filtered codebook
                sbIdx = filteredCbIdx[ sbfIdx ]         # Index of best score in the codebook
                sbPmiIdx += [ cbIdx[ sbIdx ] ]          # Add precoder index for this subband
                sbWs += [ (sbRbIdx,cb[sbIdx]) ]         # Add the PRB indices and the corresponding subband precoder
                sbSinrs += [ sinrs[sbIdx][sbRbIdx] ]    # prgSize x numLayers (or prgSize/2 x numLayers)
                sbRbIdx = []                            # Reset
            sbRbIdx += [i]                              # Add current index to sbRbIdx

        if len(sbRbIdx)>0:                          # The last subband
            sbScores = filteredScores[:,sbRbIdx]        # numFilteredCb x prgSize (or numFilteredCb x prgSize/2)
            sbfIdx = sbScores.mean(-1).argmax()         # Index of best score in the filtered codebook
            sbIdx = filteredCbIdx[ sbfIdx ]             # Index of best score in the codebook
            sbPmiIdx += [ cbIdx[ sbIdx ] ]              # Add precoder index for this subband
            sbWs += [ (sbRbIdx,cb[sbIdx]) ]             # Add the PRB indices and the corresponding subband precoder
            sbSinrs += [ sinrs[sbIdx][sbRbIdx] ]        # prgSize x numLayers (or prgSize/2 x numLayers)

        rbSinrs = np.concatenate(sbSinrs)               # m x numLayers
        sbScore = self.getSbScore(h, sbWs, noiseVar)    # Overall score for the selected rank and PMI
        return sbScore, (cbIdx[wbIdx], cb[wbIdx]), (sbPmiIdx, sbWs), rbSinrs
    
    # ******************************************************************************************************************
    def processRiPmiCqi(self, rxGrid, csiRsSet, setResources):                  # Undocumented
        # Process CSI-RS observations to generate RI, PMI, and optionally CQI. The method derives one channel matrix
        # per RB from CSI-RS CDM bundles, evaluates all allowed ranks, selects the best rank and PMI, and stores the
        # resulting feedback in 'self.csiFeedback'. If CQI is requested by 'self.quantity', CQI is computed from the
        # selected SINR values.
        # rxGrid: Received resource grid.
        # csiRsSet: CSI-RS resource set used for PMI/RI/CQI estimation.
        # setResources: Resource information for the CSI-RS resources in 'csiRsSet'.
        noiseVar = max(1e-12, rxGrid.noiseVar)   # Avoid a zero noiseVar

        for csiRs in csiRsSet.csiRsList:
            if csiRs.resourceId not in setResources: continue   # This resource is not in current rxGrid
            cdms = csiRs.getCdms()
            # Calculate channel info per RB: works for all cases except density=3
            assert csiRs.density != 3, "density=3 is not supported for RI/PMI/CQI reporting!"
            # rbChannels is a numRbs x nr x numPorts which will contain one channel matrix per RB.
            # For density=0.5, some of the RBs will contain zeros. We keep the indices of the RBs with
            # non-zero channel info in "rbs".
            rbChannels = np.zeros((self.bwp.numRbs, rxGrid.shape[0], csiRs.numPorts), dtype=np.complex128)
            rbs = set()         # The set of RBs with CSI-RS resources.
            for cdmLs, cdmKs, cdmXs in cdms:                # Shape of cdmXs: cdmSize x numPorts
                cdmYs = rxGrid[:,cdmLs, cdmKs].T            # Shape:          cdmSize x nr
                # Since np.inv(herm(cmdXs) @ cmdXs)) = 1/csiRs.cdmSize and for QPSK, |r|²=1,
                cdmH = (herm(cdmXs) @ cdmYs)/csiRs.cdmSize  # Shape: numPorts * nr

                ports = np.where(np.abs(cdmXs).sum(0)>1e-8)[0]  # Ports covered by this CDM group
                rb = cdmKs[0]//12                               # The resource block containing this CDM group
                rbChannels[rb][:,ports] = cdmH[ports,:].T       # nr x numPorts
                rbs.add(rb.item())
            rbs = sorted(rbs)                       # List of RBs containing CSI-RS info
            nonZeroChannels = rbChannels[rbs,:,:]   # m x nr x nPorts (m: number of RBs with non-zero channel info)
            ranks = [ r for r in range(1, min(rxGrid.shape[0], csiRs.numPorts, 8)+1) if r in self.allowedRanks ]
            bestScore, bestRank = -np.inf, 0
            bestWbPmi, bestWbW, bestSbPmis, bestSbWs, bestSinrs = None, None, None, None, None
            for rank in ranks:
                score, wbInfo, sbInfo, rbSinr = self.getPmiForRank(rbs, nonZeroChannels, noiseVar, rank)
                if score > bestScore:
                    wbPmi, wbW = wbInfo
                    sbPmis, sbWs = sbInfo
                    bestScore = score
                    bestRank = rank
                    bestWbPmi = wbPmi                   # Wideband PMI
                    bestWbW = wbW                       # Wideband W
                    bestSbPmis = sbPmis                 # Subband PMIs
                    bestSbWs = sbWs                     # Subband Ws
                    bestSinrs = rbSinr
        
            if bestSbPmis is not None:      bestSbPmis = [PmiIndex(i[0], i[1]) for i in bestSbPmis]
            
            if self.quantity == 'CriRiPmiCqi':
                # Store this RI/PMI info in our temporary storage 'riPmiInfo'
                self.riPmiInfo[csiRsSet.rsId][csiRs.resourceId] = [
                    bestScore,
                    bestSinrs,
                    RiFeedback(bestRank, bestScore),
                    PmiFeedback(PmiIndex(bestWbPmi[0], bestWbPmi[1]), bestWbW, bestSbPmis, bestSbWs) ]

        if self.quantity == 'CriRiPmiCqi':
            # For this case we need to make sure we have the RI/PMI information for all the resources
            # in the set before choosing the best resource.
            cri, bestRiPmi = None, None
            for resourceId, riPmiInfo in self.riPmiInfo[csiRsSet.rsId].items():
                if riPmiInfo is None:                       return  # We don't have RI/PMI info for all resources yet
                if cri is None:                             cri, bestRiPmi = resourceId, riPmiInfo  # First info
                elif riPmiInfo[0]>bestRiPmi[0]:             cri, bestRiPmi = resourceId, riPmiInfo  # Better score
                
            # Update self.csiFeedback with the best resource information
            self.csiFeedback.cri = CriFeedback(cri, score=bestRiPmi[0])
            self.csiFeedback.ri = bestRiPmi[2]
            self.csiFeedback.pmi = bestRiPmi[3]
            bestSinrs, bestRank = bestRiPmi[1], bestRiPmi[2].ri
            self.csiFeedback.cqi = self.processCqi(bestSinrs, bestRiPmi[2].ri)
            # Reset the RI/PMI temporary storage
            self.riPmiInfo[csiRsSet.rsId] = { rs.resourceId:None for rs in csiRsSet.csiRsList }
        else:
            # Update self.csiFeedback
            self.csiFeedback.ri = RiFeedback(bestRank, bestScore)
            self.csiFeedback.pmi = PmiFeedback(PmiIndex(bestWbPmi[0], bestWbPmi[1]), bestWbW, bestSbPmis, bestSbWs)
            self.csiFeedback.cqi = self.processCqi(bestSinrs, bestRank) if 'Cqi' in self.quantity else None

    # ******************************************************************************************************************
    def processRxGrid(self, rxGrid, csiRsResources):                            # Undocumented
        # Process a received grid using the CSI-RS resources present in that grid. Depending on 'self.quantity', this
        # method dispatches processing to CRI or PMI/RI/CQI logic.
        # rxGrid: Received resource grid.
        # csiRsResources: CSI-RS resource information organized by CSI-RS resource-set ID.
        for csiRsSet in self.csiRsSets:
            if csiRsSet.rsId not in csiRsResources: continue    # No resources in this RX grid for this set.
            setResources = csiRsResources[csiRsSet.rsId]
            if self.quantity == 'Cri':                          # Beam sweeping/probing
                self.processCri(rxGrid, csiRsSet, setResources)
            elif ('Pmi' in self.quantity) or ('Ri' in self.quantity) or ('Cqi' in self.quantity):
                self.processRiPmiCqi(rxGrid, csiRsSet, setResources)

    # ******************************************************************************************************************
    def getFeedback(self, upDelay):
        # Return feedback for the current slot if reporting conditions are met. For aperiodic reports, the report is
        # deactivated after the feedback is returned.
        # upDelay: Uplink reporting delay in slots.
        # Returns a dictionary of the form {reportId: csiFeedback} if feedback is available for the
        # current slot; otherwise an empty dictionary.
        if self.anythingForCurSlot(upDelay):
            if self.csiFeedback.hasInfo:                # Ignore empty feedback
                if self.reportType=='aperiodic':
                    self.active = 0                     # Deactivate when the report is sent
                return { self.reportId: self.csiFeedback }
        return {}

# **********************************************************************************************************************
class CsiReportMan:
    r"""
    Manages a set of CSI reports and collects CSI feedback across slots. This class acts as a container and dispatcher 
    for multiple :py:class:`CsiReport` objects. It forwards received-grid processing to each report and aggregates the
    resulting CSI feedback.
    """
    # ******************************************************************************************************************
    def __init__(self, csiReports, **kwargs):
        """
        Parameters
        ----------
        csiReports : list of :py:class:`CsiReport`
            CSI reports managed by this object.

        kwargs : dict
            Optional configuration parameters:

            :upDelay:
                Uplink delay, in slots, between CSI measurement time and CSI report transmission time.
                The default is 2.
        """
        self.csiReports = csiReports
        self.upDelay = kwargs.get('upDelay', 2)     # Uplink delay for feedback, in number of slots
        
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CSI report manager.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "CSI Report Manager Properties:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "CSI Report Manager Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  upDelay:              {self.upDelay}\n"
        repStr += indent*' ' + f"  Num Reports:          {len(self)}\n"
        for csiReport in self.csiReports:
            repStr += csiReport.print(indent+2, f"CSI Report {csiReport.reportId}:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def __len__(self):  return len(self.csiReports)                             # Undocumented

    # ******************************************************************************************************************
    def processRxGrid(self, rxGrid, csiRsResources):
        """
        Forward a received grid to all managed CSI reports for processing. Refer to the notebook
        :doc:`../Playground/Notebooks/CSI-Feedback/CSI-Feedback1` for an example of using this method.

        Parameters
        ----------
        rxGrid : :py:class:`~neoradium.grid.Grid`
            Received resource grid.

        csiRsResources : dict
            CSI-RS resource information organized by CSI-RS resource-set ID.
        """
        for csiReport in self.csiReports:
            csiReport.processRxGrid(rxGrid, csiRsResources)

    # ******************************************************************************************************************
    def getFeedback(self):
        """
        Returns the CSI feedback generated for each configured CSI report. The returned value is a 
        dictionary that maps each CSI-Report ID to a CsiFeedback object. Each entry corresponds to one 
        CSI report configuration and contains the feedback quantities computed for that report, such as CRI,
        RI, PMI, and/or CQI.

        Refer to the notebook :doc:`../Playground/Notebooks/CSI-Feedback/CSI-Feedback1` for an example of using 
        this method.

        Returns
        -------
        dict
            Dictionary of the form: ``{ reportId: csiFeedback }``, where:

            reportId : int
                CSI-Report ID associated with a CSI report object. This ID is used  as the dictionary key 
                so that the feedback for each configured CSI report can be accessed independently.

            csiFeedback : CsiFeedback
                Dataclass containing the CSI feedback generated for the CSI report identified by reportId. 
                Depending on the CSI report configuration, some feedback components may be present while
                others may be ``None``. The CsiFeedback object has the following fields:

                cri : CriFeedback or None
                    CRI feedback. This field is populated when CSI-RS resource indication feedback is 
                    requested (e.g. the ``quantity`` of the :py:class:`CsiReport` object corresponding to 
                    ``reportId`` is set to ``"Cri"``). Otherwise, it is set to ``None``. CriFeedback 
                    contains:

                        cri : int or list of ints
                            CSI-RS resource indicator(s). If ``numCri`` is ``1``, this identifies
                            the resource ID corresponding to the highest RSRP. If ``numCri`` is ``K > 1``,
                            this is a list of top-K resource IDs.

                        rsrp : float, list of floats, or None 
                            If ``numCri`` is ``1``, this is the RSRP value (in dB) associated with 
                            the CSI-RS resource identified by ``cri``. If ``numCri`` is ``K > 1``, this
                            is a list of RSRP values corresponding to the cri list (in dB, descending order). 
                            This is set only when quantity="Cri".

                        score : float or None
                            Spectral-efficiency score of the best beam. Set when quantity="CriRiPmiCqi".

                ri : RiFeedback or None
                    RI feedback. This field is populated when rank indication feedback is requested (e.g. the 
                    ``quantity`` of the :py:class:`CsiReport` object corresponding to ``reportId`` is set 
                    to ``"RiPmiCqi"`` or ``"RiPmi"``). Otherwise, it is set to ``None``. RiFeedback contains:

                        ri : int
                            Selected rank, i.e., the preferred number of PDSCH transmission layers.

                        score : float
                            Score associated with the selected rank which is the average of spectral
                            efficiency over all layers.

                pmi : PmiFeedback or None
                    PMI feedback. This field is populated when precoder matrix indication feedback is 
                    requested (e.g. the ``quantity`` of the :py:class:`CsiReport` object corresponding to
                    ``reportId`` is set to ``"RiPmiCqi"`` or ``"RiPmi"``). Otherwise, it is set to ``None``.

                    PmiFeedback contains wideband PMI information and, when configured, subband PMI 
                    information. It contains:

                        wbPMI : PmiIndex
                            Selected wideband PMI index. This describes the preferred wideband precoder 
                            over the CSI reporting bandwidth. PmiIndex contains:

                                i1 : list[int]
                                    First part of the PMI index. It contains the three parts of I1 
                                    PMI index (e.g. [I11, I12, I13]).

                                i2 : int
                                    Second part of the PMI index.

                        wbW : complex numpy array
                            Wideband precoding matrix corresponding to the selected wideband PMI/codebook 
                            entry. This matrix is the precoder associated with wbPMI.

                        sbPMIs : list[PmiIndex] or None
                            List of selected subband PMI indices, one entry per reported subband. This field 
                            is ``None`` when subband PMI feedback is not configured (See ``pmiGranularity``).

                            Each element of sbPMIs is a PmiIndex object with the same structure as wbPMI above.

                        sbWs : list of tuples or None
                            A list of tuples of the form (``groupRBs``, ``groupW``). For each entry in the list,
                            the ``Nt x Nl`` precoding matrix ``groupW`` is used for all subcarriers of the 
                            resource blocks listed in ``groupRBs``. This is the same format used by the 
                            :py:meth:`~neoradium.pdsch.PDSCH.getPrecodingMatrix` and
                            :py:meth:`~neoradium.pdsch.PDSCH.precodeTo` methods of the 
                            :py:class:`~neoradium.pdsch.PDSCH` class.

                cqi : CqiFeedback or None
                    CQI feedback. This field is populated when channel quality indication feedback is 
                    requested (e.g. the ``quantity`` of the :py:class:`CsiReport` object corresponding to
                    ``reportId`` is set to ``"RiPmiCqi"``). Otherwise, it is set to ``None``. CqiFeedback
                    contains:

                        cqis : list[int]
                            CQI values reported for the corresponding CSI report. The list contains one 
                            CQI value per codeword. For single-codeword transmission, this list
                            has length 1. For dual-codeword transmission, this list has length 2.

                        blers : list[float]
                            The predicted BLER values corresponding to the CQI values in cqis. The length 
                            should match cqis. Each element represents the predicted BLER associated with 
                            the corresponding codeword CQI.

                        cqi : int
                            Convenience property returning the first CQI value, equivalent to cqis[0]. This
                            is useful for the common single-codeword case.

                        bler : float
                            Convenience property returning the first BLER value, equivalent to blers[0]. This
                            is useful for the common single-codeword case.
        """
        feedback = {}
        for csiReport in self.csiReports:
            feedback.update( csiReport.getFeedback(self.upDelay) )
        return feedback

    # ******************************************************************************************************************
    @classmethod
    @deprecated("CsiReportMan.beamformingReports", docFile)
    def makeTypicalReports(cls, csiRsConfig, txAntenna, prgSize=0, allowedRanks=[1]):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`~neoradium.csireport.CsiReportMan.beamformingReports` method instead.
        """
        return cls.beamformingReports(csiRsConfig, txAntenna, prgSize=prgSize, allowedRanks=allowedRanks)
        
    # ******************************************************************************************************************
    @classmethod
    def beamformingReports(cls, csiRsConfig, txAntenna, **kwargs):
        r"""
        Creates the typical CSI report configuration for a beamforming CSI-RS
        configuration created by :py:meth:`~neoradium.csirs.CsiRsConfig.beamformingConfig`.
        The returned report manager contains three CSI reports:

            #. A periodic CRI report for beam sweeping.
            #. An aperiodic CRI report for beam probing.
            #. A semi-persistent RI/PMI/CQI report transmitted on the PUSCH.

        The report IDs are derived from the corresponding CSI-RS resource-set IDs
        by adding 10. Unless explicitly overridden, the periodic reports use the same reporting
        periods as their corresponding CSI-RS resource sets. Default report offsets
        allow two slots between CSI-RS reception and CSI reporting to accommodate
        receiver processing.

        Parameters
        ----------
        csiRsConfig : :py:class:`~neoradium.csirs.CsiRsConfig`
            A CSI-RS configuration created by :py:meth:`~neoradium.csirs.CsiRsConfig.beamformingConfig`.
            It must contain three resource sets in the following order:

                #. Periodic beam-sweeping resource set.
                #. Aperiodic beam-probing resource set.
                #. Semi-persistent RI/PMI/CQI resource set.

        txAntenna : :py:class:`~neoradium.antenna.AntennaPanel`
            The transmit antenna array used when computing PMI, RI, and CQI for
            the multi-port CSI report.

        kwargs : dict
            Optional configuration parameters.

                :sweepPeriod: Reporting period for the periodic beam-sweeping CRI
                    report. By default, the period of the beam-sweeping CSI-RS
                    resource set is used.

                :sweepOffset: Slot offset of the periodic beam-sweeping CRI report.
                    The default is two slots after the last beam-sweeping CSI-RS
                    resource in the sweep sequence.

                :pmiPeriod: Reporting period for the RI/PMI/CQI report. By default,
                    the period of the RI/PMI/CQI CSI-RS resource set is used.

                :pmiOffset: Slot offset of the RI/PMI/CQI report. The default is two
                    slots after the corresponding CSI-RS transmission.

                :rxAntenna: The receive antenna array used for PMI, RI, and CQI
                    computation. If omitted, the underlying :py:class:`CsiReport`
                    implementation assumes its default receiver configuration.

                :prgSize: Precoding Resource Group (PRG) size used for PMI
                    computation. A value of ``0`` selects wideband PMI. The default is ``0``.

                :allowedRanks: List of transmission ranks considered during RI/PMI
                    optimization. The default is ``[1, 2]``.

                :cqiTable: CQI table index as defined by **3GPP TS 38.214**. The default is ``1``. 

        Returns
        -------
        :py:class:`CsiReportMan`
            A CSI report manager containing the periodic beam-sweeping CRI report,
            the aperiodic beam-probing CRI report, and the semi-persistent
            RI/PMI/CQI report.


        .. Note:: The beam-probing CRI report is configured as an aperiodic report and must
            be triggered explicitly before it is transmitted. The beam-sweeping CRI
            report and the RI/PMI/CQI report are configured as periodic and
            semi-persistent reports, respectively, and follow their configured periods
            and offsets.
        
        Please refer to the notebook :doc:`../Playground/Notebooks/CSI-Feedback/CSI-Feedback3` for an example of using 
        this function.
        """
        bwp = csiRsConfig.bwp
        # Get CSI resource sets for Sweeping, Probing, and RI/PMI/CQI
        sweepSet, probeSet, pmiSet = csiRsConfig.csiRsSetList

        # By default, use the same sweeping period as the CSI resource set
        sweepPeriod = kwargs.get('sweepPeriod', sweepSet.period)

        # By default, send sweeping CRI two slots after last sweeping resources are received
        maxSweepOffset = max([csiRs.offset for csiRs in sweepSet.csiRsList])
        sweepOffset = kwargs.get('sweepOffset', maxSweepOffset+2)

        # Create the CSI report object for sweeping
        sweepReport = CsiReport(sweepSet, reportId=sweepSet.rsId+10, quantity="Cri", reportType="periodic",
                                period=sweepPeriod*(bwp.u+1), offset=sweepOffset)
        
        probeReport = CsiReport(probeSet, reportId=probeSet.rsId+10, quantity="Cri", reportType="aperiodic")

        # By default, use the same RI/PMI/CQI period as the CSI resource set
        pmiPeriod = kwargs.get('pmiPeriod', pmiSet.period)
        pmiOffset = kwargs.get('pmiOffset', pmiSet.csiRsList[0].offset + 2) # Allow 2 slots for precessing
        # rxAntenna is used to obtain a more accurate CQI. If not provided, a 1 x 2 antenna panel is assumed.
        rxAntenna = kwargs.get('rxAntenna', None)
        prgSize = kwargs.get('prgSize', 0)
        allowedRanks = kwargs.get('allowedRanks', [1,2])
        cqiTable = kwargs.get('cqiTable', 1)
        pmiReport = CsiReport(pmiSet, reportId=pmiSet.rsId+10, quantity="RiPmiCqi", reportType="spOnPUSCH",
                              period=pmiPeriod*(bwp.u+1), offset=pmiOffset,
                              txAntenna=txAntenna, rxAntenna=rxAntenna,
                              prgSize=prgSize, allowedRanks=allowedRanks, cqiTable=cqiTable)
        return CsiReportMan([sweepReport, probeReport, pmiReport])

# **********************************************************************************************************************
class OLLA:
    r"""
    Outer Loop Link Adaptation (OLLA) controller for CQI adjustment.
    
    This class maintains a CQI offset for each codeword and updates the offset based on ACK/NACK 
    feedback. The adjusted CQI can then be used by the link adaptation or scheduler to select a 
    more conservative or more aggressive MCS than the one indicated directly by CSI feedback.
    
    The OLLA update targets the nominal BLER associated with the selected CQI table. For CQI 
    table 3, the default target BLER is 1e-5. For other CQI tables, the default target BLER is 0.1. 
    
    If a HARQ object is provided, this OLLA object is registered with the 
    :py:class:`~neoradium.harq.HarqEntity`, so that the HARQ object can update it 
    when ACK/NACK feedback becomes available.
    
    Please refer to the notebook :doc:`../Playground/Notebooks/RayTracing/LinkAdaptation` for an 
    example of using this class.
    """
    # ******************************************************************************************************************
    def __init__(self, harq=None, cqiTable=1, step=None, minCqiOffset=None, maxCqiOffset=None, fixedOffset=0):
        """
        Parameters
        ----------
        harq : :py:class:`~neoradium.harq.HarqEntity`, optional
            HARQ entity associated with this OLLA object. The number of codewords is taken
            from ``harq.numCW``, and ``harq.setLA(self)`` is called to register this OLLA 
            object for ACK/NACK updates.

        cqiTable : int, default=1
            CQI table index. CQI table 3 uses a default target BLER of 1e-5 and
            more conservative CQI-offset limits. Other values use a default target
            BLER of 0.1.

        step : float, optional
            CQI-offset update step. If not specified, the default is 0.01 for CQI
            table 3 and 0.05 for other CQI tables.

        minCqiOffset : float, optional
            Minimum allowed CQI offset. If not specified, the default is -6 for CQI
            table 3 and -4 for other CQI tables.

        maxCqiOffset : float, optional
            Maximum allowed CQI offset. If not specified, the default is 2 for CQI
            table 3 and 4 for other CQI tables.

        fixedOffset : int, default=0
            Fixed CQI offset added to the dynamic OLLA offset when adjusting CQI 
            values. This can be used to model known calibration mismatch or 
            implementation loss between the conditions used to generate the CQI 
            report and the conditions used for PDSCH transmission. This offset is 
            not modified by OLLA updates and is not reset by :meth:`reset`.
        """
        if cqiTable == 3:
            self.targetBler = 0.00001
            self.step = 0.01 if step is None else step          # CQI step used to update the CQI offset
            self.minCqiOffset = -6 if minCqiOffset is None else minCqiOffset
            self.maxCqiOffset = 2  if maxCqiOffset is None else maxCqiOffset
        else:
            self.targetBler = 0.1
            self.step = 0.05 if step is None else step          # CQI step used to update the CQI offset
            self.minCqiOffset = -4 if minCqiOffset is None else minCqiOffset
            self.maxCqiOffset = 4  if maxCqiOffset is None else maxCqiOffset
        
        self.fixedOffset = fixedOffset
        if harq is not None:
            self.cqiOffset = harq.numCW*[0.0]   # Current offset, one for each codeword
            harq.setLA(self)    # Register this OLLA object so the HARQ object can update it with ACK/NACK feedback
        else:
            self.cqiOffset = None               # Determine the number of codewords on the first adjustCqi/update call

    # ******************************************************************************************************************
    def reset(self):
        r"""
        Reset all CQI offsets to zero.

        If the CQI offsets have already been initialized, this method resets the offset of each codeword
        to zero. If the offsets have not yet been initialized, this method has no effect.
        """
        if self.cqiOffset is not None:
            for i in range(len(self.cqiOffset)): self.cqiOffset[i] = 0.0
        
    # ******************************************************************************************************************
    def update(self, ack):
        r"""
        Update the CQI offset using ACK/NACK feedback.

        The update follows the OLLA rule. ACK feedback increases the CQI offset by ``step * targetBler``, 
        making future transmissions slightly more aggressive. NACK feedback decreases the CQI offset by
        ``step * (1 - targetBler)``, making future transmissions more conservative. Each offset is 
        clipped to the range ``[minCqiOffset, maxCqiOffset]``.

        An ACK/NACK value of ``-1`` is treated as "no update" and can be used for events that should 
        not affect OLLA, such as HARQ retransmissions.

        Parameters
        ----------
        ack : int, list, or tuple
            ACK/NACK feedback for one or more codewords. A scalar value is used for a single 
            codeword. A list or tuple must contain one value per codeword. Supported values are:

            - ``1``: ACK
            - ``0``: NACK
            - ``-1``: no update
        """
        if not isinstance(ack, (list, tuple)):  ack = [ack]
        if self.cqiOffset is None:              self.cqiOffset = len(ack)*[0.0] # First call: initialize cqiOffset
        numCW = len(self.cqiOffset)
        if len(ack) != numCW:
            raise ValueError(f"Expected ACK/NACK flags for {numCW} codewords, received {len(ack)}!")

        # ack values: -1 -> no updates, 1 -> ACK, 0 -> NACK
        for cw in range(numCW):
            if ack[cw]==-1:     continue        # No updates to the offset (e.g. retransmissions)
            elif ack[cw]==1:    self.cqiOffset[cw] += self.step * self.targetBler           # First-transmission ACK
            else:               self.cqiOffset[cw] -= self.step * (1.0 - self.targetBler)   # First-transmission NACK
            self.cqiOffset[cw] = max( self.minCqiOffset, min(self.maxCqiOffset, self.cqiOffset[cw]) )

    # ******************************************************************************************************************
    def adjustCqi(self, cqi):
        r"""
        Apply the current OLLA offset to one or more reported CQI values.

        The adjusted CQI is obtained by adding the current CQI offset and the ``fixedOffset`` to 
        the reported CQI and clipping the result to the valid CQI range from 1 to 15. A scalar input 
        produces a scalar output, while a list or tuple input produces a list output with one 
        adjusted CQI per codeword.

        Parameters
        ----------
        cqi : int, list, or tuple
            Reported CQI value or values. A scalar value is used for a single codeword. A list or 
            tuple must contain one CQI value per codeword.

        Returns
        -------
        int or list of int
            Adjusted CQI value or values clipped to the range 1 to 15. The return type matches 
            the input: scalar input returns a scalar, and list or tuple input returns a list.
        """
        arr = isinstance(cqi, (list, tuple))
        if not arr:                     cqi = [cqi]
        if self.cqiOffset is None:      self.cqiOffset = len(cqi)*[0.0]         # First call: initialize cqiOffset
        numCW = len(self.cqiOffset)
        if len(cqi) != numCW:
            raise ValueError(f"Expected CQI for {numCW} codewords, received {len(cqi)}!")
        
        newCqi = [ int( np.clip( cqi[cw] + self.fixedOffset + np.round(self.cqiOffset[cw]), 1, 15) )
                        for cw in range(numCW) ]
        
        if not arr: return newCqi[0]        # If a scalar CQI was provided, return a scalar CQI
        return newCqi                       # Return a list with the same length as the input
