import numpy as np


# See TS 38.212 - Section 5.1
strToPoly = {'6':   [1, 1, 0, 0, 0, 0, 1],
             '11':  [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             '16':  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
             '24A': [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
             '24B': [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1],
             '24C': [1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1]}

def getCrc(bits, poly):
    if poly in strToPoly:   poly = np.uint8(strToPoly[poly])
    polyLen = len(poly)
    numPad = polyLen-1
    paddedBitArray = np.append(bits, [0]*(polyLen-1))
    n = len(bits)
    for d in range(n):
        if paddedBitArray[d]:
            paddedBitArray[d:d+polyLen] ^= poly
    return paddedBitArray[n:]

def randomBits(n):
    return np.random.randint(0,2,n,dtype=np.uint8)
