# Copyright (c) 2025 InterDigital AI Lab
# Author: Shahab Hamidi-Rad
# Part of the code for the paper "A Self-Refining Multi-Layer Receiver Pipeline"
# ----------------------------------------------------------------------------------------------------------------------
import numpy as np
import time, os

import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------------------------------------
class ChEstDataset():
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, dataFile, batchSize=64):
        self.batchSize = batchSize
        sampleAndLabels = np.load(dataFile)
        x = sampleAndLabels.shape[1]//2+1
        self.samples, self.labels = np.float32(sampleAndLabels[:,:x,:,:]), np.float32(sampleAndLabels[:,x:,:,:])
        self.samples[ self.samples==10000 ] = 0     # Set unknown values to 0
        numPilots = set()
        for sample in self.samples: numPilots.add( np.sum(sample[:-2,:,:]!=0)//2)

        self.numPilots = sorted(numPilots)
        self.numSamples = self.samples.shape[0]

    # ------------------------------------------------------------------------------------------------------------------
    def batches(self, device=None, shuffle=False):
        numBatches = self.numSamples//self.batchSize
        if numBatches*self.batchSize < self.numSamples: numBatches += 1
        
        sampleOrder = np.arange(self.numSamples)
        if shuffle: np.random.shuffle(sampleOrder)
            
        for batch in range(numBatches):
            batchIndexes = sampleOrder[batch*self.batchSize : (batch+1)*self.batchSize]
            batchSamples, batchLabels = ( torch.from_numpy(self.samples[batchIndexes]),
                                          torch.from_numpy(self.labels[batchIndexes]) )
            if device is not None:  
                batchSamples, batchLabels = batchSamples.to(device), batchLabels.to(device)
            yield batchSamples, batchLabels

# ----------------------------------------------------------------------------------------------------------------------
# Define the residual block
class ResBlock(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, inDepth, midDepth, outDepth, kernel=(3,3), stride=(1,1)):
        super().__init__()
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(kernel, int): kernel = (kernel, kernel)

        self.path1 = nn.Sequential(
            nn.Conv2d(inDepth, midDepth, 1, stride, padding='valid'),  # 1x1 conv.
            nn.BatchNorm2d(midDepth),
            nn.ReLU(True),
            nn.Conv2d(midDepth, midDepth, kernel, padding='same'),
            nn.BatchNorm2d(midDepth),
            nn.ReLU(True),
            nn.Conv2d(midDepth, outDepth, 1, stride, padding='valid'), # 1x1 conv.
            nn.BatchNorm2d(outDepth))
        
        self.path2 = None
        if ((stride != (1,1)) or (inDepth!=outDepth)):
            self.path2 = nn.Sequential(nn.Conv2d(inDepth, outDepth, 1, stride),  # 1x1 conv.
                                       nn.BatchNorm2d(outDepth) )

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        out = (self.path1(x) + x) if self.path2 is None else (self.path1(x) + self.path2(x))
        out = nn.ReLU(True)(out)
        return out

# ----------------------------------------------------------------------------------------------------------------------
# Now the actual ChEstNet model
class ChEstNet(nn.Module):
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, device):
        super().__init__()
        self.inShape = (6, 14, 300)
        self.res1 = ResBlock(6, 48, 192, (9,9))     # Res Block 9x9 kernel
        self.res2 = ResBlock(192, 48, 192, (7,7))   # Res Block 7x7 kernel
        self.res3 = ResBlock(192, 48, 192, (3,3))   # Res Block 3x3 kernel
        self.conv = nn.Conv2d(192, 4, 3, padding='same')
        self.to(device)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.conv(out)
        return out

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def numParams(self):    return sum( p.numel() for p in self.parameters())
    @property
    def device(self):       return next(self.parameters()).device

    # ------------------------------------------------------------------------------------------------------------------
    def trainEpoch(self, trainDS, lossFunction, optimizer):
        self.train() # Set the model to training mode
        lossMin, lossSum, lossMax = torch.inf, 0, -torch.inf

        n = 0
        for batchSamples, batchLabels in trainDS.batches(self.device, shuffle=True):
            # Compute prediction and loss
            batchPredictions = self( batchSamples )
            loss = lossFunction(batchPredictions, batchLabels)

            loss.backward()  # Back propagation
            optimizer.step()
            optimizer.zero_grad()

            lossValue = loss.item()
            
            batchSize = batchSamples.shape[0]
            lossSum += lossValue * batchSize
            if lossValue>lossMax: lossMax = lossValue
            if lossValue<lossMin: lossMin = lossValue
            n += batchSamples.shape[0]
            
        return lossMin, lossSum/n, lossMax
        
    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation loop:
    def evaluate(self, evalDS, lossFunction=None):
        self.eval()  # Set the model to evaluation mode
        
        lossSum, n = 0, 0
        with torch.no_grad():
            for batchSamples, batchLabels in evalDS.batches(self.device):
                batchSize = batchSamples.shape[0]
                batchPredictions = self( batchSamples ).to(batchLabels.dtype)
                # Sum Loss for the whole batch
                batchLoss = lossFunction(batchPredictions, batchLabels).item() * batchSize
                lossSum += batchLoss
                n += batchSize

        return lossSum/n

    # ------------------------------------------------------------------------------------------------------------------
    def infer(self, samples, toNumpy=True):
        self.eval()  # Set the model to evaluation mode
        if type(samples) is np.ndarray: samples = torch.from_numpy(samples)
        with torch.no_grad():
            if toNumpy: return self(samples.to(self.device)).cpu().numpy()
            return self(samples.to(self.device))

    # ------------------------------------------------------------------------------------------------------------------
    def saveParams(self, fileName):
        torch.save(self.state_dict(), fileName)

    # ------------------------------------------------------------------------------------------------------------------
    def loadParams(self, fileName):
        if type(fileName)==str:
            self.load_state_dict( torch.load(fileName, weights_only=True, map_location=self.device) )
        else:
            self.load_state_dict(fileName)
