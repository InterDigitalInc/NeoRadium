# Copyright (c) 2026, InterDigital AI Lab
"""
The module ``cqidata.py`` implements the :py:class:`CqiData` class, used to obtain the CQI values for
CSI feedback. It includes the CQI tables based on **3GPP TS 38.214**.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 03/12/2026    Shahab Hamidi-Rad       Support started in NeoRadium version 0.5.0:
#                                       * Implemented the first version of the code.
# 08/07/2026    Shahab                  Changes in NeoRadium version 0.5.1:
# 07/16/2026    Shahab                  * Cleaned up the CQI calibration data handling. All related data is now
#                                         consolidated into a single dictionary, ‘cqiInfoDic’, which is loaded from
#                                         a JSON file the first time a CqiData object is created.
# **********************************************************************************************************************
import numpy as np
import json, os

from .utils import toLinear
from .ldpccodec import LdpcCodec

#**********************************************************************************************************************
# ‘cqiInfoDic’ is a hierarchical configuration dictionary loaded from a JSON file the first time a CqiData object is
# created. The dictionary is indexed as: cqiInfoDic[modRate][ldpcIter][spacing][numLayers]
# Each entry provides the AWGN SNR-BLER curves and beta/delta values for the corresponding configuration. See
# ‘getAwgnSnrBlerCurve’ and ‘getBeta’ for details on how this data is used.
#
# Each configuration dictionary contains the following keys:
#   tbses: A sorted list of Transport Block Size (TBS) values.
#   bgs: A list of LDPC base graph numbers mapped 1:1 to the values in ‘tbses’.
#   snrs: A list of lists containing SNR values in dB. Each inner list contains the SNR data points for the
#       AWGN SNR-BLER curve corresponding to the matching TBS in ‘tbses’.
#   blers: A list of lists containing BLER values. Each inner list contains the BLER data points for the AWGN SNR-BLER
#       curve corresponding to the matching TBS in ‘tbses’.
#   exConfig: An extended configuration string based on RX/TX antenna shapes. This allows different beta/delta
#       values to be stored for different antenna configurations. Additional configuration parameters, such as
#       mapping type, channel-estimation method, and prgSize, may be included in the future.
#       The value associated with each ‘exConfig’ key is a dictionary containing:
#           betas: A list of beta values for this extended configuration, mapped 1:1 to the values in ‘tbses’.
#           deltas: A list of delta values for this extended configuration, mapped 1:1 to the values in ‘tbses’.
#           minLosses: A list of minLoss values for this extended configuration, mapped 1:1 to the values in
#               ‘tbses’. These values are informational and are not used by the code.
cqiInfoDic = None

# **********************************************************************************************************************
class CqiData:
    # Provides CQI table definitions and helper methods for CQI-related BLER estimation. This class stores the
    # standardized 4-bit CQI tables from **3GPP TS 38.214** and provides utilities to:
    #   - obtain AWGN SNR-BLER curves for a given modulation/rate and transport-block size
    #   - compute effective SINR using Exponential Effective SINR Mapping (EESM)
    #   - retrieve beta values used by EESM
    #   - estimate BLER for a given set of resource-element SINRs
    # The class relies on lookup CSV files containing:
    #   - beta values for EESM mapping
    #   - AWGN SNR-BLER curves versus SNR
    # The CQI tables implemented here correspond to the CQI tables defined in **3GPP TS 38.214, Section 5.2.2.1**.
    cqiTables = [None, # There is no table 0
                 # TS 38.214, Table 5.2.2.1-2: 4-bit CQI Table 1
                 # modulation  coderate*1024   efficiency   CQI index
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
                 # modulation  coderate*1024   efficiency   CQI index
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
                 # modulation  coderate*1024   efficiency   CQI index
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
                 
                 # TS 38.214, Table 5.2.2.1-5: 4-bit CQI Table 4 (1024QAM / Rel-17)
                 # modulation  coderate*1024   efficiency   CQI index
                 [[None,       None,           None],       # 0: (Out of Range)
                 [ 'QPSK',     78,             0.1523],     # 1
                 [ 'QPSK',     193,            0.3770],     # 2
                 [ 'QPSK',     449,            0.8770],     # 3
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
    
    # ******************************************************************************************************************
    def __init__(self, **kwargs):
        # cqiInfoFileName: Path to the JSON file containing the CQI information (e.g. AWGN SNR-BLER curves
        # and beta/delta values).
        dataPath = os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data" )
        self.betaLookupFileName = kwargs.get("betaLookupFileName", os.path.join(dataPath, "BetaValues.csv"))
        self.awgnBlerCurvesFileName = kwargs.get("awgnBlerCurvesFileName", os.path.join(dataPath, "AwgnBlerCurves.csv"))
        self.cqiInfoFileName = kwargs.get("cqiInfoFileName", os.path.join(dataPath, "CqiInfo.json"))
        global cqiInfoDic
        if cqiInfoDic is None:
            with open(self.cqiInfoFileName, "r") as jsFile: cqiInfoDic = json.load(jsFile)
    
    # ******************************************************************************************************************
    def getAwgnSnrBlerCurve(self, modRate, tbs, spacing, ldpcIter):
        # Return the AWGN SNR-BLER curve for the specified transmission configuration. This method first looks up the
        # BLER-curve by modulation/rate, LDPC iteration count, and subcarrier spacing, then selects or interpolates the
        # appropriate curve based on the given transport-block size and LDPC base graph. If the requested TBS falls
        # outside the stored range, the nearest available curve is used. If the requested TBS lies between two curves
        # with the same base graph, the BLER values are interpolated.
        # modRate: Modulation and code-rate string in the form ``"Modulation~Rate"``, for example "QPSK~449" or
        #          "64QAM~666".
        # tbs: Transport-block size in bits.
        # spacing: Subcarrier spacing in kHz.
        # ldpcIter: The number of iterations for LDPC decoding
        # Returns a tuple (snrs, blers) where:
        #   - 'snrs' is a 1-D NumPy array of linear SNR values
        #   - 'blers' is a 1-D NumPy array of BLER values in percent
        
        # AWGN SNR-BLER curves are available in 1-layer configuration only
        try:
            dic = cqiInfoDic[modRate][str(ldpcIter)][str(spacing)]['1']
            tbses = np.int32(dic['tbses'])
            bgs = np.int32(dic['bgs'])
            snrses  = [ np.float32(x) for x in dic['snrs'] ]
            blerses = [ np.float32(x) for x in dic['blers'] ]
        except KeyError:
            raise ValueError(f"Could not find AWGN SNR-BLER curves for the following configuration:\n"
                             f"  modRate={modRate}\n"
                             f"  spacing={spacing}\n"
                             f"  ldpcIter={ldpcIter}\n")
    
        coderate = float(modRate.split("~")[1])/1024
        baseGraphNo = LdpcCodec.getBaseGraphNo(tbs, coderate)
        if tbs<=tbses[0]:           # Use smallest TBS in the list if 'tbs' is too small
            assert baseGraphNo==bgs[0], "Base Graph mismatch (tbs<=tbses[0])!"
            return toLinear(snrses[0]), blerses[0]
            
        if tbs>=tbses[-1]:          # Use the largest TBS in the list if 'tbs' is too large
            assert baseGraphNo==bgs[-1], "Base Graph mismatch (tbs>=tbses[-1])!"
            return toLinear(snrses[-1]), blerses[-1]

        i0 = np.searchsorted(tbses, tbs, side='right') - 1
        if tbses[i0]==tbs:          # Exact match
            assert baseGraphNo==bgs[i0], "Base Graph mismatch (tbses[i0]==tbs)!"
            return toLinear(snrses[i0]), blerses[i0]

        i1 = i0+1
        assert i0>=0 and i1>=0
        assert tbses[i0] <= tbs and tbs < tbses[i1]
        if bgs[i0] != bgs[i1]:
            # The tbs is between 2 TBS values in the list that belong to different base graphs
            # Choose the TBS based on the correct base graph
            if baseGraphNo == bgs[i0]: return toLinear(snrses[i0]), blerses[i0]
            return toLinear(snrses[i1]), blerses[i1]

        # i0 and i1 have same base graph, we can interpolate safely.
        snrs0 = snrses[i0]
        snrs1 = snrses[i1]
        blers0 = blerses[i0]
        blers1 = blerses[i1]
        snrStep = snrs0[1]-snrs0[0]

        # Extend the BLER values to match the union of SNRs by appending 100's or 0's
        deltaS = int(np.round( (snrs1[0] - snrs0[0])/snrStep ) )
        if deltaS>0:   blers1 = np.append([100]*deltaS, blers1)
        elif deltaS<0: blers0 = np.append([100]*(-deltaS), blers0)
        
        deltaE = int(np.round( (snrs1[-1] - snrs0[-1])/snrStep ) )
        if deltaE>0:   blers0 = np.append(blers0, [0]*deltaE)
        elif deltaE<0: blers1 = np.append(blers1, [0]*(-deltaE))
        
        # Now return the SNR/BLER curve.
        # SNRs are the union of snrs0 and snrs1 converted to linear scale (from dB)
        snrs = toLinear(np.sort(np.unique(np.concatenate((snrs0, snrs1)))))     # The union of Linear SNRs
        # BLERs are linear interpolations between blers0 and blers1. (clipped between 0 and 100)
        blers = np.clip([ (b1 - b0)*(tbs-tbses[i0])/(tbses[i1]-tbses[i0]) + b0 for b0,b1 in zip(blers0,blers1)], 0, 100)
        return snrs, blers          # SNRs are linear, BLERs are in percent
 
    # ******************************************************************************************************************
    @classmethod
    def getGammaEff(cls, reSinrs, beta, delta=0):
        # Compute effective SINR using Exponential Effective SINR Mapping (EESM).
        # reSinrs: A ... x numPdschREs array containing resource-element SINR values in linear scale.
        # beta: EESM beta parameter.
        # delta: Optional additive offset applied after EESM mapping. The default is 0.
        # Returns NumPy array with the shape of 'reSinrs' minus last dimension containing the effective SINR values.
   
        # Using Exponential Effective SINR Mapping (EESM) to calculate effective
        # gamma (effective SINR):   Effective SINR = -beta * ln(mean(exp(-reSinrs/beta))) + delta
        effSinr = -beta*np.log( np.maximum( np.exp(-reSinrs/beta).mean(-1), 1e-300 ) ) + delta   # Shape: (numSamples,)
        
        # Clamp the effective SINR to a small positive value for numerical stability.
        return np.maximum(effSinr, 1e-12)                                                       # Shape: (numSamples,)

    # ******************************************************************************************************************
    def getBeta(self, modRate, tbs, spacing, ldpcIter, numLayers, rxPanelShape, txPanelShape):
        # Retrieve the EESM beta and delta values for the specified transmission configuration. The lookup is
        # filtered by:
        #   - modulation and rate
        #   - LDPC iteration count
        #   - Subcarrier spacing
        #   - Number of transmission layers
        #   - LDPC base graph
        # The beta and delta corresponding to the closest transport-block size are returned.
        # modRate: Modulation and code-rate string in the form "Modulation~Rate".
        # tbs: Transport-block size in bits.
        # spacing: Subcarrier spacing (int, in kHz)
        # ldpcIter: The number of iterations for LDPC decoding
        # numLayers: Number of transmission layers.
        # rxPanelShape: The shape of RX antenna panel (numRows x numColumns)
        # txPanelShape: The shape of TX antenna panel (numRows x numColumns)
        # Returns the beta and delta values corresponding to the closest matching TBS in the lookup table.
        try:
            dic = cqiInfoDic[modRate][str(ldpcIter)][str(spacing)][str(numLayers)]
            tbsValues = dic['tbses']
            bgValues = dic['bgs']
        except KeyError:
            raise ValueError(f"Could not find beta/delta values for the following configuration:\n"
                             f"  modRate={modRate}\n"
                             f"  numLayers={numLayers}\n"
                             f"  spacing={spacing}\n"
                             f"  ldpcIter={ldpcIter}\n")
        
        exConfig = f"{rxPanelShape[0]}_{rxPanelShape[1]}_{txPanelShape[0]}_{txPanelShape[1]}"
        try:
            betaValues = dic[exConfig]['betas']
            deltaValues = dic[exConfig]['deltas']
        except KeyError:
            raise ValueError(f"Could not find beta/delta values for the following antenna configuration:\n"
                             f"  rxPanelShape={rxPanelShape}\n"
                             f"  txPanelShape={txPanelShape}\n")

        # Filter out the items with invalid 'baseGraphNo'
        n = len(tbsValues)
        coderate = float(modRate.split("~")[1])/1024
        baseGraphNo = LdpcCodec.getBaseGraphNo(tbs, coderate)
        tbsValues = [ tbsValues[i] for i in range(n) if bgValues[i]==baseGraphNo ]
        betaValues = [ betaValues[i] for i in range(n) if bgValues[i]==baseGraphNo ]
        deltaValues = [ deltaValues[i] for i in range(n) if bgValues[i]==baseGraphNo ]
        
        if len(tbsValues)==0:
            raise ValueError(f"Could not find beta matching 'baseGraphNo={baseGraphNo}'!")

        tbsIdx = np.abs(np.array(tbsValues)-tbs).argmin()   # Index of the TBS in the lookup table closest to given tbs.
        return betaValues[tbsIdx], deltaValues[tbsIdx]

    # ******************************************************************************************************************
    def getBler(self, reSinrs, modRate, tbs, spacing=15, ldpcIter=5,
                numLayers=1, rxPanelShape=(1,2), txPanelShape=(2,4)):
        # Estimate BLER from resource-element SINR values. This method performs the following steps:
        #   1. Retrieve the EESM beta and delta values for the specified configuration
        #   2. Compute the effective SINR using EESM
        #   3. Retrieve the AWGN SNR-BLER curve for the specified configuration
        #   4. Interpolate the BLER value corresponding to the effective SINR
        # The returned BLER is expressed in percent.
        # reSinrs: A 1-D array of length numPdschREs containing resource-element SINR values in linear scale.
        # modRate: Modulation and code-rate string in the form "Modulation~Rate".
        # tbs: Transport-block size in bits.
        # spacing: Subcarrier spacing (int, in kHz)
        # ldpcIter: The number of iterations for LDPC decoding
        # numLayers: Number of transmission layers.
        # rxPanelShape: The shape of RX antenna panel (numRows x numColumns)
        # txPanelShape: The shape of TX antenna panel (numRows x numColumns)
        # Returns estimated BLER in percent.

        # Get the beta and delta values for the given "modRate", "tbs", and "numLayers"
        beta, delta = self.getBeta(modRate, tbs, spacing, ldpcIter, numLayers, rxPanelShape, txPanelShape)

        # Calculate the effective SINR using Exponential Effective SINR Mapping (EESM)
        effSinr = self.getGammaEff(reSinrs, beta, delta)        # getGammaEff returns numpy array of length 1

        # Get AWGN SNR/BLER Curves:
        snrs, blers = self.getAwgnSnrBlerCurve(modRate, tbs, spacing, ldpcIter)
        
        # Obtain the BLER value for the given 'effSinr' using linear interpolation
        return np.interp(effSinr, snrs, blers)      # SNR -> AWGN BLER (in percent)
