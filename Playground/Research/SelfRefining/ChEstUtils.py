# Copyright (c) 2025 InterDigital AI Lab
# Author: Shahab Hamidi-Rad
# Part of the code for the paper "A Self-Refining Multi-Layer Receiver Pipeline"
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import time, os
from neoradium import Carrier, PDSCH, CdlChannel, AntennaPanel, Grid, random, LdpcEncoder

# ----------------------------------------------------------------------------------------------------------------------
def toComplex(x):
    # Converts a real-valued tensor of shape (n, 2*rr, ll, kk) to a complex tensor of shape (n, rr, ll, kk)
    n, rrx2, ll, kk = x.shape
    x = x.reshape((n,2,rrx2//2,ll,kk))
    return x[:,0,:,:,:] + 1j*x[:,1,:,:,:]  # n x 2*rr//2 x ll x kk
    
# ----------------------------------------------------------------------------------------------------------------------
def toReal(x):
    # Converts a complex tensor of shape (n, rr, ll, kk) to a real-valued tensor of shape (n, 2*rr, ll, kk)
    return np.concatenate([x.real, x.imag], axis=1)

# ----------------------------------------------------------------------------------------------------------------------
def getRandomPilotInfo(pdsch, txGrid, cbSizes, numGoodCBs=None, markPseodoPilots=False):
    # Returns the indices of all pilots and pseudo-pilots. This includes the indices
    # of DMRS REs plus numGoodCBs sets of indices for REs corresponding to each "good" code block
    # cbSizes is a list of code block sizes.
    # txGrid is the transmitted grid.
    layerMappedIdx = pdsch.getLayerMapIndexes(pdsch.dataIndices)[0]
    numCBs = len(cbSizes)
    if numGoodCBs is None: # If number of Good CBs is not given pick one randomly
        numGoodCBs = random.integers(0, numCBs)  # 0 (DMRS only), 1, 2, ..., numCBs-1
    qm =  pdsch.modems[0].qm
    if numGoodCBs==0:
        # Using only DMRS
        pilotIdx = txGrid.getReIndexes("DMRS")
    else:
        # Using DMRS and 'numGoodCBs' code blocks
        # Get a random list of good CBs. It has 'numGoodCBs' values each ranging from 0 to numCBs-1
        goodCBs = np.sort(random.choice(np.arange(numCBs), numGoodCBs, replace=False))
        goodIdxIdx = []
        offset = 0
        for s in range(numCBs):
            if s in goodCBs: goodIdxIdx += [ np.arange(cbSizes[s]//qm) + offset ]
            offset += cbSizes[s]//qm
        goodIdxIdx = np.concatenate(goodIdxIdx)
        
        pilotIdx = tuple(x[goodIdxIdx] for x in layerMappedIdx)
        dmrsIdx = txGrid.getReIndexes("DMRS")
        if markPseodoPilots:
            # NOTE: Do not set 'markPseodoPilots' to True if this function is called in a loop.
            txGrid.reTypeIds[pilotIdx] = Grid.retNameToId["PSEUDO_PILOT"]
        pilotIdx = tuple(np.append(pilotIdx[i],dmrsIdx[i]) for i in [0,1,2])  # The indexes to all pilots
    
    return pilotIdx

# ----------------------------------------------------------------------------------------------------------------------
def getPseudoPilotIndices(pdsch, ldpcEncoder, rsGrid, decodedTxBlockWithCRC, crcMatch):
    numBits = pdsch.getBitSizes(rsGrid)[0]  # Actual number of bits available in the resource grid
    rateMatchedCodeWords = ldpcEncoder.getRateMatchedCodeBlocks(decodedTxBlockWithCRC, numBits, concatCBs=False, addCrc=False)
    pdsch.populateGrid(rsGrid, np.concatenate(rateMatchedCodeWords))
      
    goodIdx = [] # This is the indexes of modulated symbols corresponding to the bits in the sub-blocks
                 # decoded with correct CRC.
    s = 0
    for i in range(len(rateMatchedCodeWords)):
        numCodeBlockSyms = len(rateMatchedCodeWords[i])//pdsch.modems[0].qm
        if crcMatch[i]: goodIdx += list(range(s,s+numCodeBlockSyms))
        s += numCodeBlockSyms

    layerMappedIndexes = pdsch.getLayerMapIndexes(pdsch.dataIndices)
    goodLayerMappedIndexes = tuple(x[goodIdx] for x in layerMappedIndexes[0])
    return goodLayerMappedIndexes

# ----------------------------------------------------------------------------------------------------------------------
def getModelIn(pilotIdx, txGrid, rxGrid, unknownValue=0):
    # Creates rr samples that can be feed to the model for each one of rx antennas.
    # pilotIdx is the location of known REs, txGrid and rxGrid are the transmitted and received grids
    rr, ll, kk = rxGrid.shape           # Number of RX antenna, Number of symbols, Number of subcarriers
    pp, ll2, kk2 = txGrid.shape         # Number of Ports (Layers), Number of symbols, Number of subcarriers
    assert (ll==ll2) and (kk==kk2)      # Ensure same time/freq. dimensions

    samples = []  # Each sample is: (pp+1) x ll x kk  (The TxGrid for 'pp' ports plus RxGrid for 1 Rx Antenna)
    for r in range(rr):
        # Initialize with Unkown values
        sample = np.ones((pp+1,ll,kk), dtype=np.complex128)*(unknownValue + 1j*unknownValue)
        sample[pilotIdx] = txGrid[pilotIdx]

        # Choose the port with min number of pilots and use that to get the pilot indexes for RX
        bestPort = np.argmin([(pilotIdx[0]==p).sum() for p in range(pp)])
        rxPilotIdx = tuple(pilotIdx[x][pilotIdx[0]==bestPort] for x in [1,2])
        sample[pp][rxPilotIdx] = rxGrid[r][rxPilotIdx]
        samples += [ sample ]

    samples = np.float32(toReal(np.stack(samples)))         # rr x 2*(pp+1) x ll x kk   (Real)
    return samples

# ----------------------------------------------------------------------------------------------------------------------
def getLabels(gtChannel):
    # gtChannel is the groundtruth channel (with precoding effect):  ll x kk x rr x pp  (pp: number of layers)
    # Create rr labels from the given channel matrix used for each one of rx antennas.
    rr = gtChannel.shape[2]
    labels = [ np.transpose(gtChannel[:,:,r,:], (2,0,1)) for r in range(rr)]    # rr x pp x ll x kk     (Complex)
    return np.float32(toReal(np.stack(labels)))                                 # rr x 2*pp x ll x kk   (Real)

# ----------------------------------------------------------------------------------------------------------------------
def estimateChannelML(model, pilotIdx, txGrid, rxGrid):
    modelIn = getModelIn(pilotIdx, txGrid, rxGrid)
    modelOut = model.infer(modelIn)                 # modelOut: rr x 2*pp x ll x kk
    estChan = toComplex( modelOut )                 # rr x pp x ll x kk
    return np.transpose( estChan, (2,3,0,1) )       # ll x kk x rr x pp

