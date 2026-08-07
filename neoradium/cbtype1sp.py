# Copyright (c) 2026, InterDigital AI Lab
"""
The module ``cbtype1sp.py`` implements the :py:class:`PmiCbT1Sp` class, which processes PMI codebook type 1 (single
panel) based on **3GPP TS 38.214**.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 03/12/2026    Shahab Hamidi-Rad       Support started in NeoRadium version 0.5.0:
#                                       * Implemented the first version of the code.
# 08/07/2026    Shahab                  Changes in NeoRadium version 0.5.1:
#                                       * Added support for 'cbMask2Tx' for the case of 2-ports with 1 or 2 layers.
#                                       * Fixed the bitmap handling of 'cbMaskI1'.
# **********************************************************************************************************************
import numpy as np
from .antenna import AntennaPanel
from .utils import validateRange

# **********************************************************************************************************************
class PmiCbT1Sp:
    # Implements the 3GPP Type-1 single-panel PMI codebook defined in **3GPP TS 38.214**.
    # This class generates valid PMI index combinations and their corresponding precoding
    # matrices for a dual-polarized single-panel transmitter antenna configuration. The
    # generated codebook entries can be used for CSI-based PMI selection at the UE.
    
    # The implementation supports the valid ``N1 x N2`` panel dimensions listed in
    # **3GPP TS 38.214, Table 5.2.2.2.1-2**, and supports rank values from 1 to 8,
    # subject to the constraints imposed by the number of antenna ports and the codebook
    # tables in the standard.
    # ******************************************************************************************************************
    def __init__(self, **kwargs):
        # The transmitter antenna configuration. This must be an instance of 'AntennaPanel' and must represent a
        # dual-polarized single panel.
        self.txAntenna = kwargs.get('txAntenna', None)
        if self.txAntenna is None:
            raise ValueError("The antenna configuration is missing!")
            
        if isinstance(self.txAntenna, AntennaPanel):
            self.ng = 1         # Number of panel groups. For Type-1 single-panel codebooks, this is always 1.
            self.n2, self.n1 = self.txAntenna.shape     # Vertical and horizontal panel dimensions, respectively.
            if self.txAntenna.polarization not in ['x', '+']:
                raise ValueError("Type1SP codebook can only be used with a dual-polarized transmitter antenna panel.")
        else:
            raise ValueError(f"Unsupported antenna class '{self.txAntenna.__class__.__name__}' for 'Type1SP' codebook!")

        validN1N2Combs = ["1-1","2-1","2-2","4-1","3-2","6-1","4-2","8-1","4-3","6-2","12-1","4-4","8-2","16-1"]
        if "%d-%d"%(self.n1,self.n2) not in validN1N2Combs:
            raise ValueError(f"Invalid panel dimension N1xN2={self.n1}x{self.n2}. See TS 38.214, Table 5.2.2.2.1-2")

        self.cbMode = kwargs.get('cbMode', 1)   # Codebook mode. Supported values are 1 and 2.
        validateRange(self.cbMode, [1,2])

        # Oversampling factors defined by **3GPP TS 38.214** for the horizontal and vertical panel dimensions. See
        # TS 38.214, Table 5.2.2.2.1-2
        self.o1 = 4
        self.o2 = 4 if self.n2>1 else 1
        
        self.numPorts = 2 * self.ng * self.n1 * self.n2 # Number of CSI-RS antenna ports represented by the codebook
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)

        # Optional mask for first-stage beam index combinations for the special case of 2-ports with 1 or 2 layers.
        # Entries set to a nonzero value are excluded from the generated codebook.
        self.cbMask2Tx = kwargs.get('cbMask2Tx', np.int8(6*[0]) )
        if len(self.cbMask2Tx) != 6:            raise ValueError("'cbMask2Tx' must have length 6.")

        # Optional mask for first-stage beam index combinations. Entries set to a nonzero value are excluded from
        # the generated codebook.
        self.cbMaskI1 = kwargs.get('cbMaskI1', np.int8((bb1*bb2)*[0]) )
        if len(self.cbMaskI1) != (bb1 * bb2):   raise ValueError(f"'cbMaskI1' must have length {bb1 * bb2}.")
            
        # Optional implementation-level filter for i2 values. Entries set to a nonzero value are excluded from
        # the generated codebook.
        # This is not, by itself, a general implementation of the typeI-SinglePanel-codebookSubsetRestriction-i2
        # RRC procedure, whose normative use depends on reportQuantity.
        self.cbMaskI2 = kwargs.get('cbMaskI2', np.int8(16*[0]) )
        if len(self.cbMaskI2) != 16:            raise ValueError("'cbMaskI2' must have length 16.")
        
    # ******************************************************************************************************************
    @classmethod
    def getCombs(cls, *argv):
        # Generate all combinations of index values from the provided ranges or lists. Each returned row is one
        # combination. The ordering is arranged to match the PMI index enumeration used by the codebook logic in
        # this class. This helper is used internally to generate candidate PMI index combinations such as
        # (i11, i12, i13, i2).
        # *argv: Each argument is either:
        #   - an integer 'n', meaning the range '0, 1, ..., n-1', or
        #   - a list of explicit values
        # Returns a list of combinations. Each inner list contains one complete index tuple.
        lists = []
        for listI in argv[::-1]:
            if isinstance(listI, (list, tuple, np.ndarray)):    lists += [listI]
            else:                                               lists += [list(range(listI))]
                
        lists = [lists[1]] + [lists[0]] + lists[2:]
        n = len(lists)
        a = list(range(n-1,1,-1)) + [0,1]
        return np.int32(np.meshgrid(*lists)).T.reshape(-1,n)[:,a].tolist()

    # ******************************************************************************************************************
    def getIndexCombs(self, numLayers):
        # Return all valid PMI index combinations for the specified transmission rank. The supported combinations
        # follow the relevant tables in **3GPP TS 38.214, Section 5.2.2.2.1**.
        # numLayers: Transmission rank, i.e., the number of layers. Supported values are determined by the
        # antenna configuration and the 3GPP Type-1 single-panel codebook tables.
        # Returns a list of index combinations of the form [i11, i12, i13, i2].
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)
        configStr = f"{numLayers} layers, {self.numPorts} ports, cbMode {self.cbMode}, {self.n1}x{self.n2} panel"

        if self.numPorts == 2:          # See TS 38.214, Table 5.2.2.2.1-1
            if numLayers not in [1,2]:  raise ValueError(f"Unsupported case: {configStr}")
            return  self.getCombs(4, 1, 1, 1) if numLayers == 1 else self.getCombs(2, 1, 1, 1)
        
        if numLayers == 1:                                                      # See TS 38.214, Table 5.2.2.2.1-5
            if self.cbMode==1:  return self.getCombs(bb1, bb2, 1, 4)            # 1st Table
            if self.n2>1:       return self.getCombs(bb1//2, bb2//2, 1, 16)     # 2nd Table
            if self.n2==1:      return self.getCombs(bb1//2, 1, 1, 16)          # 3rd Table
            raise ValueError(f"Unsupported case: {configStr}")

        if numLayers == 2:                                                      # See TS 38.214, Table 5.2.2.2.1-6
            i13Len = 2 if (self.n1==2 and self.n2==1) else 4                    # See TS 38.214, Table 5.2.2.2.1-3
            if self.cbMode==1:  return self.getCombs(bb1, bb2, i13Len, 2)       # 1st Table of 5.2.2.2.1-6
            if self.n2>1:       return self.getCombs(bb1//2, bb2//2, i13Len, 8) # 2nd Table of 5.2.2.2.1-6
            if self.n2==1:      return self.getCombs(bb1//2, 1, i13Len, 8)      # 3rd Table of 5.2.2.2.1-6
            raise ValueError(f"Unsupported case: {configStr}")

        if numLayers in [3, 4]:                 # See TS 38.214, Tables 5.2.2.2.1-7 and 5.2.2.2.1-8
            if self.numPorts>=16:               return self.getCombs(bb1//2, bb2, 4, 2) # From 2nd table
            elif (self.n1==2 and self.n2==1):   return self.getCombs(bb1, bb2, 1, 2)    # These 4 lines are from 1st
            elif (self.n1==4 and self.n2==1):   return self.getCombs(bb1, bb2, 3, 2)    #    table of 5.2.2.2.1-7 and
            elif (self.n1==2 and self.n2==2):   return self.getCombs(bb1, bb2, 3, 2)    #    5.2.2.2.1-8 and Table
            else:                               return self.getCombs(bb1, bb2, 4, 2)    #    5.2.2.2.1-4 for i13.

        if numLayers in [5, 6]:                 # See TS 38.214, Tables 5.2.2.2.1-9 and 5.2.2.2.1-10
            if self.n2>1:                       return self.getCombs(bb1, bb2, 1, 2)    # 1st row
            if (self.n1>2) and (self.n2==1):    return self.getCombs(bb1, 1, 1, 2)      # 2nd row
            raise ValueError(f"Unsupported case: {configStr}")

        if numLayers in [7,8]:                  # See TS 38.214, Tables 5.2.2.2.1-11 and 5.2.2.2.1-12
            if (self.n1==4) and (self.n2==1):   return self.getCombs(bb1//2, 1, 1, 2)   # 1st row
            if (self.n1>4) and (self.n2==1):    return self.getCombs(bb1, 1, 1, 2)      # 2nd row
            if (self.n1==2) and (self.n2==2):   return self.getCombs(bb1, bb2, 1, 2)    # 3rd row
            if (self.n1>2) and (self.n2==2):    return self.getCombs(bb1, bb2//2, 1, 2) # 4th row
            if (self.n1>2) and (self.n2>2):     return self.getCombs(bb1, bb2, 1, 2)    # 5th row
        raise ValueError(f"Unsupported case: {configStr}")

    # ******************************************************************************************************************
    def isVectorMasked(self, l, m, tilde=False):
        bb1,bb2 = self.n1*self.o1, self.n2*self.o2  # B1/B2 number of beams (horizontal/vertical)

        if tilde in [True, '~']:
            # Rank 3/4, 16/24/32-port reduced-vector rule
            ac = bb1 * bb2
            l %= bb1 // 2
            m %= bb2
            indices = [(bb2*(2*l - 1) + m) % ac,
                       (bb2*(2*l)     + m) % ac,
                       (bb2*(2*l + 1) + m) % ac]
            return any(self.cbMaskI1[q] for q in indices)

        l %= bb1
        m %= bb2
        return bool(self.cbMaskI1[bb2*l + m])

    # ******************************************************************************************************************
    def v(self, l, m, tilde=False):
        # Generate the beamforming basis vector :math:`v_{l,m}` for the specified beam indices. The generated vector
        # is the Kronecker-style combination of the horizontal and vertical beam components defined by the Type-1
        # single-panel codebook.
        # l: Horizontal beam index.
        # m: Vertical beam index.
        # tilde : If 'True' or "~", generate the reduced-size vector used in the large-panel codebook cases
        # that require :math:`\\tilde{v}`. Otherwise, generate the full vector :math:`v`.
        # Returns a column vector with shape (N, 1), where 'N' depends on the panel size and the 'tilde' setting.
        # Returns 'None' if the (l, m) combination is masked by 'cbMaskI1'.
        bb1,bb2 = self.n1*self.o1, self.n2*self.o2  # B1/B2 number of beams (horizontal/vertical)
        if self.isVectorMasked(l, m, tilde):    return None

        if tilde in [True, '~']:    ul = np.exp( 4j*np.pi* l *np.arange(self.n1//2)/(bb1) ) # Shape: N1//2
        else:                       ul = np.exp( 2j*np.pi* l *np.arange(self.n1)/(bb1) )    # Shape: N1
        um = np.exp( 2j*np.pi* m *np.arange(self.n2)/(bb2) )                        # Shape: N2
        return (ul[:,None] @ um[None,:]).reshape(-1,1)                              # Shape: N1*N2 x 1 or N1*N2//2 x 1
        
    # ******************************************************************************************************************
    def getVecAngles(self, numLayers, i11, i12, i13, i2):
        # Convert PMI indices into codebook basis vectors and phase coefficients. This method implements the
        # index-to-vector/phase mapping from the Type-1 single-panel codebook tables in **3GPP TS 38.214**.
        # numLayers: Transmission rank.
        # i11, i12, i13, i2: PMI index components as defined by the Type-1 single-panel codebook.
        # The returned tuple depends on 'numLayers' and the applicable 3GPP table. It contains one or more beamforming
        # vectors and the corresponding phase coefficients needed to assemble the final precoding matrix.
        # Typical return forms include:
        #   - (v, phi) for rank 1
        #   - (v0, v1, phi) for rank 2
        #   - (vt, phi, theta) for some large-panel rank-3/4 cases
        #   - (v0, v1, v2, phi) for rank 5/6
        #   - (v0, v1, v2, v3, phi) for rank 7/8
        configStr = f"{numLayers} layers, {self.numPorts} ports, cbMode {self.cbMode}, {self.n1}x{self.n2} panel"

        if numLayers == 1:
            # Note: i13 is not used
            # See TS 38.214, Table 5.2.2.2.1-5
            if self.cbMode==1:      l, m, n = i11, i12, i2                              # 1st Table
            elif self.n2>1:         l, m, n = 2*i11 + (i2//4)%2, 2*i12 + i2//8, i2%4    # 2nd Table
            elif self.n2==1:        l, m, n = 2*i11 + i2//4, 0, i2%4                    # 2nd Table
            else:                   raise ValueError( f"Unsupported case: {configStr}")

            v = self.v(l,m)
            phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*n/2 )
            return v, phi

        if numLayers == 2:
            # Getting k1,k2 from i13: (TS 38.214, Table 5.2.2.2.1-3)
            k1,k2 = -1,-1
            if i13==0:                                  k1, k2 = 0, 0
            elif i13==1:                                k1, k2 = self.o1, 0
            elif i13==2:
                if (self.n1>self.n2) and (self.n2>1):   k1, k2 = 0, self.o2
                elif self.n1==self.n2:                  k1, k2 = 0, self.o2
                elif (self.n1>2) and (self.n2==1):      k1, k2 = 2*self.o1, 0
            elif i13==3:
                if (self.n1>self.n2) and (self.n2>1):   k1, k2 = 2*self.o1, 0
                elif self.n1==self.n2:                  k1, k2 = self.o1, self.o2
                elif (self.n1>2) and (self.n2==1):      k1, k2 = 3*self.o1, 0
            if k1<0 or k2<0:                            raise ValueError( f"Unsupported case: i13={i13}, {configStr}")

            # See TS 38.214, Table 5.2.2.2.1-6
            if self.cbMode==1:                                                          # 1st Table
                v0 = self.v(l=i11,      m=i12)
                v1 = self.v(l=i11 + k1, m=i12 + k2)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )            # n = i2
                return v0, v1, phi
                
            if self.n2>1:                                                               # 2nd Table
                v0 = self.v(l=2*i11 + (i2//2)%2,      m=2*i12 + i2//4)
                v1 = self.v(l=2*i11 + (i2//2)%2 + k1, m=2*i12 + i2//4 + k2)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*(i2%2)/2 )        # n = i2%2
                return v0, v1, phi
                
            if self.n2==1:                                                              # 3rd Table
                v0 = self.v(l=2*i11 + i2//2,      m=0)
                v1 = self.v(l=2*i11 + i2//2 + k1, m=0)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*(i2%2)/2 )        # n = i2%2
                return v0, v1, phi
                
        if numLayers in [3, 4]:
            # See TS 38.214, Tables 5.2.2.2.1-7 and Tables 5.2.2.2.1-8
            if self.numPorts<16:                                            # 1st Table
                # Getting k1,k2 from i13: (TS 38.214, Table 5.2.2.2.1-4)
                k1,k2 = -1,-1
                if i13==0:                                  k1, k2 = self.o1, 0
                if i13==1:
                    if (self.n1==4) and (self.n2==1):       k1, k2 = 2*self.o1, 0
                    if (self.n1==6) and (self.n2==1):       k1, k2 = 2*self.o1, 0
                    if (self.n1==2) and (self.n2==2):       k1, k2 = 0, self.o2
                    if (self.n1==3) and (self.n2==2):       k1, k2 = 0, self.o2
                if i13==2:
                    if (self.n1==4) and (self.n2==1):       k1, k2 = 3*self.o1, 0
                    if (self.n1==6) and (self.n2==1):       k1, k2 = 3*self.o1, 0
                    if (self.n1==2) and (self.n2==2):       k1, k2 = self.o1, self.o2
                    if (self.n1==3) and (self.n2==2):       k1, k2 = self.o1, self.o2
                if i13==3:
                    if (self.n1==6) and (self.n2==1):       k1, k2 = 4*self.o1, 0
                    if (self.n1==3) and (self.n2==2):       k1, k2 = 2*self.o1, 0
                if k1<0 or k2<0:                            raise ValueError( f"Unsupported case: i13={i13}, {configStr}")

                v0 = self.v(l=i11,    m=i12)
                v1 = self.v(l=i11+k1, m=i12+k2)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )# n = i2
                return v0, v1, phi
            
            vt = self.v(l=i11, m=i12, tilde=True)                           # 2nd Table
            phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )    # n = i2
            theta = np.exp( 1j*np.pi*i13/4 )                                # p = i13
            return vt, phi, theta
            
        if numLayers in [5, 6]:
            # See TS 38.214, Tables 5.2.2.2.1-9 and 5.2.2.2.1-10
            if self.n2>1:                                                       # 1st row
                v0 = self.v(l=i11,         m=i12)
                v1 = self.v(l=i11+self.o1, m=i12)
                v2 = self.v(l=i11+self.o1, m=i12+self.o2)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )    # n=i2
                return v0, v1, v2, phi
            
            if (self.n1>2) and (self.n2==1):                                    # 2nd row
                v0 = self.v(l=i11,           m=0)
                v1 = self.v(l=i11+self.o1,   m=0)
                v2 = self.v(l=i11+2*self.o1, m=0)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )    # n=i2
                return v0, v1, v2, phi

        if numLayers in [7, 8]:
            # See TS 38.214, Tables 5.2.2.2.1-11 and 5.2.2.2.1-12
            if (self.n1>=4) and (self.n2==1):                                   # 1st and 2nd rows
                v0 = self.v(l=i11,           m=0)
                v1 = self.v(l=i11+self.o1,   m=0)
                v2 = self.v(l=i11+2*self.o1, m=0)
                v3 = self.v(l=i11+3*self.o1, m=0)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )    # n=i2
                return v0, v1, v2, v3, phi
                
            if (self.n1==2) and (self.n2>2):            # Not in the table
                raise ValueError( f"Unsupported case: {configStr}")
            
            if (self.n1>=2) and (self.n2>=2):                                   # 3rd, 4th, and 5th row
                v0 = self.v(l=i11,           m=i12)
                v1 = self.v(l=i11+self.o1,   m=i12)
                v2 = self.v(l=i11,           m=i12+self.o2)
                v3 = self.v(l=i11+self.o1,   m=i12+self.o2)
                phi = None if self.cbMaskI2[i2] else np.exp( 1j*np.pi*i2/2 )    # n=i2
                return v0, v1, v2, v3, phi

        raise ValueError( f"Unsupported case: {configStr}")

    # ******************************************************************************************************************
    def getCodebookInfo(self, numLayers):
        # Generate all valid PMI entries and their corresponding precoding matrices. The returned PMI list and
        # codebook matrices are aligned: entry 'pmi[k]' corresponds to precoding matrix 'codebook[k]'. The codebook
        # construction follows the appropriate 3GPP Type-1 single-panel table for the given number of layers and panel
        # size.
        # numLayers: Transmission rank, i.e., the number of layers.
        # Returns:
        #   - A list of PMI entries. Each entry is stored as [[i11, i12, i13], i2].
        #   - A stacked array of precoding matrices with shape (numCodewords, numPorts, numLayers).
        bb1, bb2 = self.n1*self.o1, self.n2*self.o2     # B1, B2 number of beams (horizontal and vertical)
        configStr = f"{numLayers} layers, {self.numPorts} ports, cbMode {self.cbMode}, {self.n1}x{self.n2} panel"
        pmi = []
        codebook = []

        # Get all Indices (i11, i12, i13, i2)
        idxCombs = self.getIndexCombs(numLayers)

        if self.numPorts == 2:                          # See TS 38.214, Table 5.2.2.2.1-1
            if numLayers==1:
                codebook = np.array([ [[1], [1]], [[1], [1j]], [[1], [-1]], [[1], [-1j]] ])/np.sqrt(2)  # Shape: 4, 2, 1
                keep = [i for i in range(4) if self.cbMask2Tx[i]==0]    # Mask based on the 6-bit bitmap 'cbMask2Tx'
            else:
                codebook = np.array([ [[1, 1], [1, -1]], [[1, 1], [1j, -1j]] ])/2                       # Shape: 2, 2, 2
                keep = [i for i in range(2) if self.cbMask2Tx[4+i]==0]  # Mask based on the 6-bit bitmap 'cbMask2Tx'

            codebook = [codebook[i] for i in keep]
            pmi = [ [[idxCombs[i][0],0,0],0] for i in keep ]

        # ..............................................................................................................
        elif numLayers == 1:                            # See TS 38.214, Table 5.2.2.2.1-5
            for i11, i12, i13, i2 in idxCombs:
                v, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                if any(va is None for va in [v,phi]): continue                  # Masked based on cbMaskI1/cbMaskI2
                pmi += [ [[i11, i12, i13], i2] ]
                codebook += [ np.vstack([v, phi*v])/np.sqrt(self.numPorts) ]                            # Shape: Nt, 1
 
        # ..............................................................................................................
        elif numLayers == 2:                            # See TS 38.214, Table 5.2.2.2.1-6
            for i11, i12, i13, i2 in idxCombs:
                v0, v1, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                if any(va is None for va in [v0,v1,phi]): continue              # Masked based on cbMaskI1/cbMaskI2
                pmi += [ [[i11, i12, i13], i2] ]
                codebook += [ np.vstack([ np.hstack( [v0,     v1     ]),
                                          np.hstack( [phi*v0, -phi*v1]) ])/np.sqrt(2*self.numPorts) ]

        # ..............................................................................................................
        elif numLayers == 3:
            if self.numPorts<16:                        # See TS 38.214, Table 5.2.2.2.1-7 (1st Table)
                for i11, i12, i13, i2 in idxCombs:
                    v0, v1, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                    if any(va is None for va in [v0,v1,phi]): continue          # Masked based on cbMaskI1/cbMaskI2
                    pmi += [ [[i11, i12, i13], i2] ]
                    codebook += [ np.vstack([ np.hstack( [v0,     v1,     v0     ]),
                                              np.hstack( [phi*v0, phi*v1, -phi*v0]) ])/np.sqrt(3*self.numPorts) ]
            else:                                       # See TS 38.214, Table 5.2.2.2.1-7 (2nd Table)
                for i11, i12, i13, i2 in idxCombs:
                    vt, phi, theta = self.getVecAngles(numLayers, i11, i12, i13, i2)
                    if any(va is None for va in [vt, phi, theta]): continue     # Masked based on cbMaskI1/cbMaskI2
                    pmi += [ [[i11, i12, i13], i2] ]
                    codebook += [ np.vstack([ np.hstack([vt,           vt,            vt           ]),
                                              np.hstack([theta*vt,     -theta*vt,     theta*vt     ]),
                                              np.hstack([phi*vt,       phi*vt,        -phi*vt      ]),
                                              np.hstack([theta*phi*vt, -theta*phi*vt, -theta*phi*vt]) ])/np.sqrt(3*self.numPorts) ]
            
        # ..............................................................................................................
        elif numLayers == 4:
            if self.numPorts<16:                        # See TS 38.214, Table 5.2.2.2.1-8 (1st Table)
                for i11, i12, i13, i2 in idxCombs:
                    v0, v1, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                    if any(va is None for va in [v0, v1, phi]): continue        # Masked based on cbMaskI1/cbMaskI2
                    pmi += [ [[i11, i12, i13], i2] ]
                    codebook += [ np.vstack([ np.hstack( [v0,     v1,     v0,      v1     ]),
                                              np.hstack( [phi*v0, phi*v1, -phi*v0, -phi*v1]) ])/np.sqrt(4*self.numPorts) ]
            else:                                       # See TS 38.214, Table 5.2.2.2.1-8 (2nd Table)
                for i11, i12, i13, i2 in idxCombs:
                    vt, phi, theta = self.getVecAngles(numLayers, i11, i12, i13, i2)
                    if any(va is None for va in [vt, phi, theta]): continue     # Masked based on cbMaskI1/cbMaskI2
                    pmi += [ [[i11, i12, i13], i2] ]
                    codebook += [ np.vstack([ np.hstack([vt,           vt,            vt,            vt          ]),
                                              np.hstack([theta*vt,     -theta*vt,     theta*vt,      -theta*vt   ]),
                                              np.hstack([phi*vt,       phi*vt,        -phi*vt,       -phi*vt     ]),
                                              np.hstack([theta*phi*vt, -theta*phi*vt, -theta*phi*vt, theta*phi*vt]) ])/np.sqrt(4*self.numPorts) ]
            
        # ..............................................................................................................
        elif numLayers == 5:                            # See TS 38.214, Table 5.2.2.2.1-9
            for i11, i12, i13, i2 in idxCombs:
                v0, v1, v2, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                if any(va is None for va in [v0, v1, v2, phi]): continue        # Masked based on cbMaskI1/cbMaskI2
                pmi += [ [[i11, i12, i13], i2] ]
                codebook += [  np.vstack([ np.hstack([v0,     v0,      v1, v1,  v2]),
                                           np.hstack([phi*v0, -phi*v0, v1, -v1, v2]) ])/np.sqrt(5*self.numPorts) ]
            
        # ..............................................................................................................
        elif numLayers == 6:                            # See TS 38.214, Table 5.2.2.2.1-10
            for i11, i12, i13, i2 in idxCombs:
                v0, v1, v2, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                if any(va is None for va in [v0, v1, v2, phi]): continue        # Masked based on cbMaskI1/cbMaskI2
                pmi += [ [[i11, i12, i13], i2] ]
                codebook += [  np.vstack([ np.hstack([v0,     v0,      v1,     v1,      v2, v2 ]),
                                           np.hstack([phi*v0, -phi*v0, phi*v1, -phi*v1, v2, -v2]) ])/np.sqrt(6*self.numPorts) ]
            
        # ..............................................................................................................
        elif numLayers == 7:                            # See TS 38.214, Table 5.2.2.2.1-11
            for i11, i12, i13, i2 in idxCombs:
                v0, v1, v2, v3, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                if any(va is None for va in [v0, v1, v2, v3, phi]): continue    # Masked based on cbMaskI1/cbMaskI2
                pmi += [ [[i11, i12, i13], i2] ]
                codebook += [  np.vstack([ np.hstack([v0,     v0,      v1,     v2, v2,  v3, v3 ]),
                                           np.hstack([phi*v0, -phi*v0, phi*v1, v2, -v2, v3, -v3]) ])/np.sqrt(7*self.numPorts) ]
            
        # ..............................................................................................................
        elif numLayers == 8:                            # See TS 38.214, Table 5.2.2.2.1-12
            for i11, i12, i13, i2 in idxCombs:
                v0, v1, v2, v3, phi = self.getVecAngles(numLayers, i11, i12, i13, i2)
                if any(va is None for va in [v0, v1, v2, v3, phi]): continue    # Masked based on cbMaskI1/cbMaskI2
                pmi += [ [[i11, i12, i13], i2] ]
                codebook += [  np.vstack([ np.hstack([v0,     v0,      v1,     v1,      v2, v2,  v3, v3 ]),
                                           np.hstack([phi*v0, -phi*v0, phi*v1, -phi*v1, v2, -v2, v3, -v3]) ])/np.sqrt(8*self.numPorts) ]

        if len(codebook) == 0:
            raise ValueError("No valid codebook entries remain after applying codebook restrictions.")
        return pmi, np.stack(codebook)
        
