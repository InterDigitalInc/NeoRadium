# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``grid.py`` implements the :py:class:`Grid` class, which encapsulates the functionality of a Resource Grid,
including:

- Keeping the Resource Element (RE) values for a specified resource grid size.
- Providing easy access to a specific type of data in the resource grid (e.g., DMRS values, CSI-RS values, PDSCH 
  data, etc.)
- Providing statistics and visualizing the resource grid map.
- Applying the *precoding* process which results in a new *precoded* :py:class:`Grid` object.
- Applying `OFDM <https://en.wikipedia.org/wiki/Orthogonal_frequency-division_multiplexing>`_ modulation to the 
  resource grid which results in a :py:class:`~neoradium.waveform.Waveform` object.
- Applying a :doc:`Channel Model <./Channels>` to the resource grid in frequency domain.
- Applying *Additive White Gaussian Noise (AWGN)* to the resource grid in frequency domain.
- Performing *Synchronization* based on the correlation between the configured :doc:`reference signals <./RefSig>` 
  and a received :py:class:`~neoradium.waveform.Waveform`.
- Performing *Channel Estimation* based on a received resource grid and the configured 
  :doc:`reference signals <./RefSig>`.
- Performing *Equalization* using the estimated channel.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 12/08/2023    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np
import scipy.io
from scipy.interpolate import RBFInterpolator, interp1d

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches

from .utils import polarInterpolate, interpolate, herm, toLinear, toDb
from .random import random
from .waveform import Waveform

# **********************************************************************************************************************
class Grid:
    r"""
    This class implements the functionality of a resource grid. It stores the complex frequency-domain values of 
    resource elements (REs) in the grid.
    """
    # ******************************************************************************************************************
    # ReTypes:
    # See https://matplotlib.org/stable/gallery/color/named_colors.html for more colors.
    # predefined content types and colors
    retIdToName, retColors = zip(*[ ("UNASSIGNED",    "white"),
                                    ("RESERVED",      "gray"),
                                    ("NO_DATA",       "lightgray"),
                                    ("DMRS",          "pink"),
                                    ("PTRS",          "yellow"),
                                    ("CSIRS_NZP",     "red"),
                                    ("CSIRS_ZP",      "orange"),
                                    ("DATA",          "cyan"),
                                    ("PDSCH",         "cornflowerblue"),
                                    ("PDCCH",         "lime"),
                                    ("PUSCH",         "lightblue"),
                                    ("PUCCH",         "peachpuff"),
                                    ("PRECODED_MIX",  "violet"),
                                    ("RX_DATA",       "sienna")])
    retMaxPredefine, retMaxCustom = 50, 20
    retIdToName = list(retIdToName) + (retMaxPredefine+retMaxCustom-len(retIdToName))*[None]
    
    # Fill unused colors with the color of "UNASSIGNED"
    retColors = list(retColors) + (retMaxPredefine+retMaxCustom-len(retColors))*[retColors[0]]
    
    retNameToId = {n:i for i,n in enumerate(retIdToName)}
    retNumCustom = 0

    # ******************************************************************************************************************
    def __init__(self, bwp, numPlanes=1, contents="DATA", useReDesc=False, numSlots=1):
        r"""
        Parameters
        ----------            
        bwp : :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part object based on which this resource grid is created.
            
        numPlanes : int (default: 1)
            A resource grid can be considered as a 3-dimensional ``P x L x K`` complex tensor where ``L`` is the 
            number of OFDM symbols, ``K`` is the number of subcarriers (based on ``bwp``), and ``P`` is the number 
            of *planes*. In different contexts, ``P`` can be equivalent to the number of layers, number of transmitter
            antennas, or number of receiver antennas. To avoid any confusion, the resource grid implementation in 
            **NeoRadium** uses the term *"Plane"* for the first dimension of the resource grid.
        
        contents : str
            The default content type of this resource grid. Each resource element (RE) in the resource grid has an
            associated content type. When some data is assigned to some REs in this resource grid without a specified
            content type, the default value is used. The following content types are currently defined:
            
            :DATA: A Generic content type used when the type of data in the resource grid is unknown or not specified.
            :PDSCH: The content type used for the data carried in a Physical Downlink Shared Channel (PDSCH)
            :PDCCH: The content type used for the data carried in a Physical Downlink Control Channel (PDCCH)
            :PUSCH: The content type used for the data carried in a Physical Uplink Shared Channel (PUSCH)
            :PUCCH: The content type used for the data carried in a Physical Uplink Control Channel (PUCCH)
            :RX_DATA: The content type used for the received resource grid. (Created by the OFDM demodulation process)
        
        useReDesc : Boolean
            If ``True``, the resource grid will also include additional fields that describe the content of each
            resource element (RE). This can be used during the debugging to make sure the resources are allocated
            correctly.
            
        numSlots : int
            The number of time slots to include in the resource grid. The number of time symbols ``L`` (the second
            dimension of the resource grid tensor) is equal to ``numSlots * bwp.symbolsPerSlot``.
            
            
        **Other Read-Only Properties:**
        
        Here is a list of additional properties:
        
            :shape: Returns the shape of the 3-dimensional resource grid tensor.
            :numPlanes: The number of antenna ports. (The same as ``numPlanes``)
            :numPorts: The number of antenna ports. (The same as ``numPlanes``)
            :numLayers: The number of layers. (The same as ``numPlanes``)
            :numSubcarriers: The number of subcarriers in this resource grid.
            :numRBs: The number of resource blocks (RBs) in this resource grid. This is equal to ``numSubcarriers/12``.
                The number of subcarriers in a resource grid is always a multiple of 12.
            :numSymbols: The number of time symbols in this resource grid. This is equal to 
                ``numSlots*bwp.symbolsPerSlot``.
            :size: The size of resource grid tensor.
            :noiseVar: The variance of the AWGN noise present in this resource grid. This is usually initialized to
                zero. When an AWGN noise is applied to the grid using the :py:meth:`addNoise` function, the variance
                of the noise stored in this property. Also, if a noisy :py:class:`~neoradium.waveform.Waveform` is
                OFDM-demodulated using the :py:meth:`~neoradium.waveform.Waveform.ofdmDemodulate` method, then the
                amount of noise is transferred to the new :py:class:`~neoradium.grid.Grid` object created.
            
        Additionally, you can access (Read-Only) the following :py:class:`~neoradium.carrier.BandwidthPart` class 
        properties directly: ``startRb``, ``numRbs``, ``nFFT``, ``symbolsPerSlot``, ``slotsPerSubFrame``, 
        ``slotsPerFrame``, ``symbolsPerSubFrame``.
        
        **Resource Grid Indexing:**
        
        a) *Reading*: You can directly access the contents of the resource grid using indices. Here are a few
        examples of accessing the RE values in the resource grid:
        
        .. code-block:: python
        
            myREs = myGrid[0,2:5,:]     # instead of using myGrid.grid[0,2:5,:]
            print(myREs.shape)          # Assuming 612 subcarriers, this will print: "(3, 612)"
            
            indexes = myGrid.getReIndexes("DMRS")   # Get the indices of all DMRS values
            dmrsValues = myGrid[indexes]            # Get all DMRS values as a 1-D array.
            
        
        b) *Writing*: You can assign different values to different REs in the resource grid. Here are a few examples:
        
        .. code-block:: python
        
            # Set the RE at layer 1, symbol 2, and subcarrier 3 to the value
            # 0.707 - 0.707j and RE type "DMRS".
            myGrid[1,2,3] = (0.707 - 0.707j, "DMRS")
                                                      
            # Mark all REs in the time symbol 5 as "RESERVED" for layer 1. The
            # RE values are set to 0 in this case.
            myGrid[1,5,:] = "RESERVED"

            # Update the 3 RE values at layer 0, subcarrier 5, and symbols [1, 4, 7]
            # and set their RE content type to the grid's default content type.
            myGrid[0,1:10:3,5] = [-0.948 - 0.948j, -0.316+0.316j, 0.316-0.948j]
        """
        self.bwp = bwp

        if type(contents)==str:
            if contents not in ["DATA", "PDSCH", "PDCCH", "PUSCH", "PUCCH"]:
                raise ValueError("Unsupported grid content type \"%s\"!"%(contents))
            self.defaultReType = self.retNameToId[contents]
        elif self.retValid(contents)==False:
            raise ValueError("Unsupported grid content type \"%d\"!"%(contents))
        else:
            self.defaultReType = contents
            if self.retIdToName[self.defaultReType] not in ["DATA", "PDSCH", "PDCCH", "PUSCH", "PUCCH"]:
                raise ValueError("Unsupported grid content type \"%s\"!"%(self.retIdToName[self.defaultReType]))

        self.numSlots = numSlots
        gridShape = ( numPlanes, numSlots*self.symbolsPerSlot, 12*self.numRbs )
        
        self.grid = np.zeros(gridShape, dtype=np.complex128)
        self.reTypeIds = np.ones(gridShape, dtype=np.uint8)*self.retNameToId["UNASSIGNED"]

        self.reDesc = None
        if useReDesc:
            self.reDesc = np.array(np.prod(gridShape) * ["UNASSIGNED"], dtype=np.dtype('<U20')).reshape(gridShape)
        self.noiseVar = 0

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this resource grid object.

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
        if title is None:   title = "Resource Grid Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  startRb: %d\n"%(self.startRb)
        repStr += indent*' ' + "  numRbs: %d\n"%(self.numRbs)
        repStr += indent*' ' + "  numSlots: %d\n"%(self.numSlots)
        repStr += indent*' ' + "  Data Contents: %s\n"%(self.retIdToName[self.defaultReType])
        repStr += indent*' ' + "  Size: %d\n"%(self.size)
        repStr += indent*' ' + "  Shape: %s\n"%(str(self.shape))
        if self.noiseVar>0:
            repStr += indent*' ' + "  Noise Var.: %s\n"%(str(self.noiseVar))

        repStr += self.bwp.print(indent+2, "Bandwidth Part:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @classmethod
    def retValid(cls, key):
        if type(key)==str: return (key in cls.retNameToId)
        return (key in cls.retNameToId.values())

    # ******************************************************************************************************************
    @classmethod
    def retRegister(cls, name, color):
        if name in cls.retNameToId:             return cls.retNameToId[name]
        if color in cls.retColors:              raise ValueError("RE Color \"%s\" is already taken!"%(color))
        if cls.retNumCustom>=cls.retMaxCustom:  raise ValueError("Too many Custom RE types!")

        newId = cls.retMaxPredefine + cls.retNumCustom
        cls.retNumCustom += 1
        cls.retNameToId[name] = newId
        cls.retIdToName[newId] = name
        cls.retColors[newId] = color
        return newId
        
    # ******************************************************************************************************************
    def getStats(self):
        r"""
        Returns some statistics about the allocation of resources in the resource grid.

        Returns
        -------
        dict
            A dictionary of items containing the number of resource elements allocated for different types of data in
            this resource grid.
        """
        stats = {"GridSize": self.grid.size}
        for retName, retId in self.retNameToId.items(): # Go through all RE Types
            numREs = len(np.where(self.reTypeIds==retId)[0])
            if numREs==0:   continue
            stats[ retName ] = numREs

        return stats

    # ******************************************************************************************************************
    # These properties are documented above in the __init__ function.
    @property
    def shape(self):            return self.grid.shape
    @property
    def numPlanes(self):        return self.grid.shape[0]
    @property
    def numPorts(self):         return self.grid.shape[0]
    @property
    def numLayers(self):        return self.grid.shape[0]
    @property
    def numSubcarriers(self):   return self.grid.shape[2]
    @property
    def numRBs(self):           return self.grid.shape[2]//12
    @property
    def numSymbols(self):       return self.grid.shape[1]
    @property
    def size(self):             return self.grid.size

    # ******************************************************************************************************************
    def __getattr__(self, property):        # Not documented (Already mentioned in the __init__ documentation)
        # Get these properties from the 'bwp' object
        if property not in ["startRb", "numRbs", "nFFT", "symbolsPerSlot", "slotsPerSubFrame",
                            "slotsPerFrame", "symbolsPerSubFrame"]:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.bwp, property)

    # ******************************************************************************************************************
    def __getitem__(self, key): # Not documented (Already mentioned in section "Resource Grid Indexing")
        # This allows directly indexing the resource grid. (Reeding)
        return self.grid[key]   # For example you can use a = grid[1,2,3]  instead of grid.grid[1,2,3]
                    
    # ******************************************************************************************************************
    def __setitem__(self, key, values):     # Not documented (Already mentioned in section "Resource Grid Indexing")
        # This allows directly indexing the resource grid. (Writing)
        if type(values) == tuple:           # For example: grid[1,2,3] = (123, "DMRS") -> Assigning DMRS values
            values, retName = values
            if self.retValid(retName) == False:
                raise ValueError("Unknown content type \"%s\"!"%(retName))
            retId = self.retNameToId[retName]
            if self.reDesc is not None: self.reDesc[key] = retName
        elif type(values) == str:           # For example: grid[1,2,3] = "RESERVED" -> Setting as reserved (value is 0)
            values, retName = 0, values
            if self.retValid(retName) == False:
                raise ValueError("Unknown content type \"%s\"!"%(retName))
            retId = self.retNameToId[retName]
            if self.reDesc is not None: self.reDesc[key] = retName
        else:
            values, retId = values, self.defaultReType  # For example: grid[1,2,3] = 123 -> Assigning data (e.g., PDSCH)
            if self.reDesc is not None: self.reDesc[key] = self.retIdToName[retId]

        self.grid[key] = values
        self.reTypeIds[key] = retId

    # ******************************************************************************************************************
    def reTypeAt(self, p, l, k):
        r"""
        Returns the content type (as a string) of the resource element at the position specified by ``p``, ``l``, and
        ``k``.

        Parameters
        ----------
        p: int
            The *plane* number. It can be the layer or antenna index depending on the context.
            
        l: int
            The time symbol index.

        k: int
            The subcarrier index.
        
        Returns
        -------
        str
            The content type of the resource element specified by ``p``, ``l``, and ``k``.
        """
        return self.retIdToName[ self.reTypeIds[p,l,k] ]
        
    # ******************************************************************************************************************
    def getReIndexes(self, reTypeStr=None):
        r"""
        Returns the indices of all resource elements in the resource grid with the content type specified by the 
        ``reTypeStr``. For example, the code below gets the indices of all DMRS values in the resource grid and uses
        the returned indices to retrieve these values.

       
        .. code-block:: python
                    
            dmrsIdx = myGrid.getReIndexes("DMRS")   # Get the indices of all DMRS values
            dmrsValues = myGrid[dmrsIdx]            # Get all DMRS values as a 1-D array.


        Parameters
        ----------
        reTypeStr: str or None
            If ``reTypeStr`` is ``None``, the default content type of this resource grid is used as the key. For 
            example if this resource grid was created with ``contents="PDSCH"``, then the indices of all resource 
            elements with content type "PDSCH" are returned.
            
            Otherwise, this function returns the indices of all resource elements in the resource grid with the 
            content type specified by ``reTypeStr``. Here is a list of values that can be used:
            
                :"UNASSIGNED": The *un-assigned* resource elements.
                :"RESERVED": The reserved resource elements. This includes the REs reserved by ``reservedRbSets``
                    or ``reservedReMap`` parameters of this :py:class:`PDSCH`.
                :"NO_DATA": The resource elements that should not contain any data. For example when the corresponding
                    REs in a different layer is used for transmission of data for a different UE. (See 
                    ``otherCdmGroups`` parameter of :py:class:`~neoradium.dmrs.DMRS` class)
                :"DMRS": The resource elements used for :py:class:`~neoradium.dmrs.DMRS`.
                :"PTRS": The resource elements used for :py:class:`~neoradium.dmrs.PTRS`.
                :"CSIRS_NZP": The resource elements used for Non-Zero-Power (NZP) CSI-RS (See 
                    :py:mod:`~neoradium.csirs`).
                :"CSIRS_ZP": The resource elements used for Zero-Power (ZP) CSI-RS (See :py:mod:`~neoradium.csirs`).
                :"PDSCH": The resource elements used for user data in a Physical Downlink Shared Channel 
                    (:py:class:`PDSCH`)
                :"PDCCH": The resource elements used for user data in a Physical Downlink Control Channel 
                    (:py:class:`~neoradium.pdcch.PDCCH`)
                :"PUSCH": The resource elements used for user data in a Physical Uplink Shared Channel 
                    (:py:class:`~neoradium.pdcch.PUSCH`)
                :"PUCCH": The resource elements used for user data in a Physical Uplink Control Channel 
                    (:py:class:`~neoradium.pdcch.PUCCH`)

                    
        Returns
        -------
        3-tuple
            A tuple of three 1-D numpy arrays specifying a list of locations in the resource grid. This value can be
            used directly to access REs at the specified locations. (See the above example)
        """
        if reTypeStr is None:   reTypeStr = self.retIdToName[self.defaultReType]
        if self.retValid(reTypeStr)==False:
            raise ValueError("Unknown RE Content type \"%s\"!"%(reTypeStr))
        return np.where(self.reTypeIds==self.retNameToId[reTypeStr])
                
    # ******************************************************************************************************************
    def getReValues(self, reTypeStr=None):
        r"""
        Returns the values of all resource elements in the resource grid with the content type specified by the
        ``reTypeStr``. This is a shortcut method that allows accessing all the values in one step. For example, the
        following two methods are equivalent.
        
        .. code-block:: python
                    
            dmrsValues1 = myGrid[ myGrid.getReIndexes("DMRS") ] # Get indices, then access values
            dmrsValues2 = myGrid.getReValues("DMRS")            # Using this method
            assert np.all(dmrsValues1==dmrsValues2)             # The results are the same


        Parameters
        ----------
        reTypeStr: str or None
            If ``reTypeStr`` is ``None``, the default content type of this resource grid is used as the key. For
            example, if this resource grid was created with ``contents="PDSCH"``, then the values of all resource
            elements with content type "PDSCH" are returned.
            
            Otherwise, this function returns the values of all resource elements in the resource grid with the content
            type specified by ``reTypeStr``. See :py:meth:`getReIndexes` for a list of values that could be used
            for ``reTypeStr``.
                    
        Returns
        -------
        1-D numpy array
            A 1-D complex numpy array containing the values for all REs with the content type specified by
            ``reTypeStr``.
        """
        return self.grid[self.getReIndexes(reTypeStr)]
    
    # ******************************************************************************************************************
    def precode(self, f):
        r"""
        Applies the specified precoding matrix to this grid object and returns a new *precoded* grid. This function
        supports *Precoding RB groups (PRGs)* which means different precoding matrices could be applied to different
        groups of subcarriers in the resource grid. See **3GPP TS 38.214, Section 5.1.2.3** for more details.

        Parameters
        ----------
        f: numpy array or list of tuples
            This function supports two types of precoding:
        
            :Wideband: ``f`` is an ``Nt x Nl`` matrix where ``Nt`` is the number of transmitter antennas and ``Nl``
                is the number of layers which **must** match the number of layers in the resource grid. In this case
                the same precoding is applied to all subcarriers of the resource grid.
            
            :Using PRGs: ``f`` is a list of tuples of the form (``groupRBs``, ``groupF``).
                For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all subcarriers
                of the resource blocks listed in ``groupRBs``.

        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A new Grid object of shape ``Nt x L x K`` where ``Nt`` is the number of transmitter antennas, ``L`` is
            the number of OFDM symbols, and ``K`` is the number of subcarriers.
        """
        # The precoder matrix "f" is an Nt x Nl matrix or a list of tuples of the form (groupRBs, groupF)
        if type(f)==list:
            # The precoder matrix "f" is a list of tuples of the form (groupRBs, groupF)
            nt, nl = f[0][1].shape
            newF = np.zeros((self.numSubcarriers, nt, nl), dtype=np.complex128)         # Shape: K, Nt, Nl
            for groupRBs, groupF in f:
                groupREs = np.int32([rb*12+re for rb in groupRBs for re in range(12)])
                newF[groupREs] = groupF
            f = newF
            #     f       . self.grid   ->      precodedGrid        <--- Tensors
            # (K, Nt, Nl) . (Nl, L, K)  ->      (Nt, L, k)          <--- Shapes
            #     1   2      0   1               0   1              <--- Axes
            axes = [(1,2), (0,1), (0,1)]

        else:
            # f is a 2D matrix of shape Nt x Nl
            if type(f) != np.ndarray:
                raise ValueError("'f' must be a 2D numpy array or a list of tuples.")
            if f.shape[1] != self.numLayers:
                raise ValueError("The last dimension of 'f' (%d) must match the first dimension of the grid (%d)"%
                                 (f.shape[-1],self.shape[0]))
            #      f   . self.grid      ->      precodedGrid        <--- Tensors
            # (Nt, Nl) . (Nl, L, K)     ->      (Nt, L, K)          <--- Shapes
            #  0   1      0   1                  0   1              <--- Axes
            axes = [(0,1), (0,1), (0,1)]
        
        precodedGrid = Grid(self.bwp, f.shape[0], self.defaultReType)   # Precoded Grid Shape: Nt x L x K
        precodedGrid.grid = np.matmul(f, self.grid, axes=axes)
        
        newReTypeIds = self.reTypeIds[0]
        for p in range(1,self.numPlanes):
            diffIdx = np.where( self.reTypeIds[p] != self.reTypeIds[0] )
            if len(diffIdx[0])>0:
                newReTypeIds[diffIdx] = self.retNameToId["PRECODED_MIX"]
        
        precodedGrid.reTypeIds[:] = newReTypeIds

        return precodedGrid

    # ******************************************************************************************************************
    def ofdmModulate(self, f0=0, windowing="STD"):
        r"""
        Applies OFDM Modulation to the resource grid which results in a :py:class:`~neoradium.waveform.Waveform`
        object. This function is based on **3GPP TS 38.211, Section 5.3.1**.

        Parameters
        ----------
        f0: float
            The carrier frequency of the generated waveform. If it is 0 (default), then a baseband waveform is
            generated and the *up-conversion* process explained in **3GPP TS 38.211, Section 5.4** is not applied.

        windowing: str
            A text string indicating what type of windowing should be applied to the waveform after OFDM modulation. 
            The default value ``"STD"`` means the windowing should be applied based on **3GPP TS 38.104, Sections B.5.2
            and C.5.2**. For more information see :py:meth:`~neoradium.waveform.Waveform.applyWindowing` method of the
            :py:class:`~neoradium.waveform.Waveform` class.
            
        Returns
        -------
        :py:class:`~neoradium.waveform.Waveform`
            A :py:class:`~neoradium.waveform.Waveform` object containing the OFDM-modulated waveform information.
        """
        pp, ll, kk = self.shape
        assert (ll%self.bwp.symbolsPerSlot) == 0
            
        l0 = self.bwp.slotNoInSubFrame * self.symbolsPerSlot    # Number of symbols from start of this subframe
        maxL = self.symbolsPerSubFrame - l0                 # Max number of remaining symbols in this subframe from l0
        if ll > maxL:
            raise ValueError("Cannot modulate across subframe boundary! (At most %d symbols)"%(maxL))

        numPad = ((self.nFFT-kk+1)//2,(self.nFFT-kk)//2)    # Number of 0's to pad (beginning and end of subcarriers)
        paddedGrid = np.pad(self.grid, ((0,0),(0,0),numPad))        # Shape: pp, ll, nFFT
        shiftedPaddedGrid = np.fft.ifftshift(paddedGrid, axes=2)    # Shifted for IFFT
        waveform = np.fft.ifft(shiftedPaddedGrid, axis=2)           # Time-Domain waveforms:  Shape: pp, ll, nFFT
        
        symLens = self.bwp.symbolLens[l0:l0+ll]         # Symbol lengths in samples for each symbol in the next numSlots
        cpLens = symLens-self.nFFT                      # CP lengths in samples for each symbol in the next numSlots
        maxSymLen = symLens.max()
        
        # Indexes used to insert the CP-Len elements from the end of symbol waveforms to the beginning:
        indexes = (np.arange(maxSymLen) - cpLens[:,None])%self.nFFT

        waveformWithCPs = np.zeros((pp,ll, maxSymLen), dtype=np.complex128)     # Shape: pp, ll, maxSymLen
        
        # Insert the CP-Len elements from the end of symbol waveforms to the beginning
        for l in range(ll): waveformWithCPs[:,l,:] = waveform[:,l,indexes[l]]
        
        # Up-conversion. See 3GPP TS 38.211 V17.0.0 (2021-12), Section 5.4
        if f0>0:
            n0 = self.bwp.symbolLens[:l0].sum()                     # Number of samples from start of current subframe
            
            # Start sample index of each symbol in the next numSlots from the start of current subframe:
            startIndexes = np.cumsum(np.append(n0,symLens[:-1]))
            
            phaseFactors = np.exp( 2j * np.pi * f0 * (-startIndexes-cpLens)/self.bwp.sampleRate )   # ll values
            waveformWithCPs *= phaseFactors[None,:,None]                                            # Up-conversion

        # Now stitch the symbol waveforms back to back keeping only the first (symLens[l]) samples for each symbol 'l'
        waveform = Waveform(np.concatenate([waveformWithCPs[:,l,:symLen] for l,symLen in enumerate(symLens)], axis=1))
        
        if windowing.upper()!='NONE':   waveform = waveform.applyWindowing(cpLens, windowing, self.bwp)
        return waveform

    # ******************************************************************************************************************
    # Not documented. Will be removed in future. Use Waveform::ofdmDemodulate.
    @classmethod
    def ofdmDemodulate(cls, bwp, waveform, **kwargs):
        if type(waveform)!=Waveform:    return Waveform(waveform).ofdmDemodulate(bwp,**kwargs)
        return waveform.ofdmDemodulate(bwp,**kwargs)

    # ******************************************************************************************************************
    def estimateTimingOffset(self, rxWaveform):
        r"""
        Estimates the timing offset of a received waveform. This method first applies OFDM modulation to the 
        resource grid and then calculates the correlation of this waveform with the given ``rxWaveform``. The timing
        offset is the index of where the correlation is at its maximum. The output of this function can be used by the
        :py:meth:`~neoradium.waveform.Waveform.sync` method of the :py:class:`~neoradium.waveform.Waveform` class
        to synchronize a received waveform.

        Parameters
        ----------
        rxWaveform: :py:class:`~neoradium.waveform.Waveform`
            The :py:class:`~neoradium.waveform.Waveform` object containing the received waveform.
            
        Returns
        -------
        int
            The timing offset in number of time-domain samples. This is the number of samples that should be ignored
            from the beginning of the ``rxWaveform``.
        """
        # Here "self" is the grid created for the CSI-RS symbols only
        rsWaveForm = self.ofdmModulate(windowing="NONE")
        numRxAnt, numRxSamples = rxWaveform.shape
        numPorts, numCsiRsSamples = rsWaveForm.shape
    
        xCors = np.float64(numRxSamples * [0])
        rsWaveForm = rsWaveForm.pad(numRxSamples-numCsiRsSamples)
        for r in range(numRxAnt):
            for p in range(numPorts):
                xCor = scipy.signal.correlate(rxWaveform[r], rsWaveForm[p], 'full')
                xCors += np.abs(xCor[numRxSamples-1:])

        return np.argmax(xCors)

    # ******************************************************************************************************************
    def equalize(self, hf, noiseVar=None):
        r"""
        Equalizes a received resource grid using the estimated channel ``hf``. The estimated channel is assumed to
        include the effect of precoding matrix, therefore, its shape is ``L x K x Nr x Nl`` where ``L`` is the
        number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver antennas, and
        ``Nl`` is the number of layers. The output of the equalization process is a new :py:class:`Grid` object
        of shape ``Nl x L x K``.
        
        This function also outputs Log-Likelihood Ratio (LLR) scaling factors which are used by the demodulation 
        process when extracting Log-Likelihood Ratios (LLRs) from the equalized resource grid.
        
        This method uses the Minimum Mean Squared Error (MMSE) algorithm for the equalization.

        Parameters
        ----------
        hf: 4-D complex numpy array
            This is an ``L x K x Nr x Nl`` numpy array representing the estimated channel matrix, where ``L`` is
            the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver antennas,
            and ``Nl`` is the number of layers.
            
        noiseVar: float or None
            The variance of the noise applied to the received resource grid. If this is not provided, this method
            tries to use the noise variance of the resource grid obtained by the OFDM demodulation process for
            the time-domain case or the variance of the noise applied to the received resource grid by the
            :py:meth:`addNoise` method for the frequency domain case (See the ``noiseVar`` property of :py:class:`Grid`
            class).
            
        Returns
        -------
        eqGrid: :py:class:`Grid`
            The equalized grid object of shape ``Nl x L x K`` where ``Nl`` is the number of layers, ``L`` is the
            number of OFDM symbols, and ``K`` is the number of subcarriers.
            
        llrScales: 3-D numpy array
            The Log-Likelihood Ratios (LLR) scaling factors which are used by the demodulation process when extracting
            Log-Likelihood Ratios (LLRs) from the equalized resource grid. The shape of this array is ``Nl x L x K``
            which is similar to ``eqGrid`` above.
        """
        # Here self is the rxGrid with shape: nr,ll,kk
        # hf is an estimate of h.f (Channel matrix including the effect of precoding). Shape: ll,kk,nr,pp
        if (self.shape[0] != hf.shape[2]) or (self.shape[1] != hf.shape[0]) or (self.shape[2] != hf.shape[1]):
            raise ValueError("Mismatch in the number of receiver antennas, OFDM symbols, or subcarriers!")

        if noiseVar is None:
            # This works if the noise was added to this grid using addNoise function, or if
            # the noise was added to a waveform using the addNoise function and the
            # OFDM demodulation was called on that noisy waveform. In these 2 cases, we already
            # have the noise variance.
            noiseVar = self.noiseVar
        noiseVar = max(1e-8, noiseVar)          # Avoid division by zero
        
        ll,kk,rr,pp = hf.shape
        if rr<=pp:
            hhNoiseInv = np.linalg.pinv(herm(hf) @ hf + noiseVar*np.eye(pp), hermitian=True)
        else:
            # When rr>pp, it makes more sense to do SVD
            u, s, vH = np.linalg.svd(hf)
            s2NoiseInv = np.eye(s.shape[-1])/(np.square(s)+noiseVar)[...,None]
            hhNoiseInv = herm(vH) @ s2NoiseInv @ vH
            
        llrScale = (1/np.diagonal(hhNoiseInv,0,-2,-1)).real
        wMMSE = hhNoiseInv @ herm(hf)
        eq = np.matmul(wMMSE, self.grid[:,None,:,:], axes=[(2,3),(0,1),(0,1)])[:,0,:,:]
        
        eqGrid = Grid(self.bwp, numPlanes=rr, numSlots=self.numSlots)
        eqGrid.grid = eq
        eqGrid.reTypeIds = np.ones(eqGrid.shape, dtype=np.uint8)*self.retNameToId["RX_DATA"]
        eqGrid.noiseVar = noiseVar
        return eqGrid, np.transpose(llrScale,(2,0,1))   # Same Shape: pp x ll x kk

    # ******************************************************************************************************************
    def scaleNoiseVar(self, rawNoiseVar, numTx, lCdm, kCdm, numVar):    # Not documented
        # This method uses the raw noise variance calculated in the "estimateChannelLsEx"
        # function together with additional parameters to create an input vector. The
        # input vector "x" is then fed to a small neural network to get the actual
        # noise variance.
        rr, _, kk = self.shape             # Number of RX antennas, Number of subcarriers

        rawSnrDb = toDb( 1/(rawNoiseVar * rr) )
        if rawSnrDb>20: return rawNoiseVar
        
        # NN Model Params:
        w1 = [[6.25861, -0.22737, -8.51406, -0.25593, 0.08617, 0.54746, -10.5016, -0.0075 ],
              [0.05773, -0.08806, 0.03222, 0.65573, -1.05669, -0.00781, 0.01074, -0.02898],
              [-11.48739, -18.84534, 9.54569, -0.02089, 9.92439, 0.07408, 11.41916, -34.07344],
              [0.71498, 4.52607, -0.35023, 0.05907, 2.24553, 0.06049, 0.47961, 0.44182],
              [0.84015, 0.14097, 0.20389, -0.45147, 0.12305, -0.51977, 0.37225, 0.12104],
              [0.41917, 10.52318, 3.35156, 0.58207, -24.37617, 0.33745, -1.11957, 1.07133],
              [-0.12522, -1.82239, 0.90271, -0.06134, 10.43859, 0.37885, 1.36096, -0.70045],
              [0.00109, -0.00328, -0.00657, -0.16279, -0.00351, -0.28476, 0.00053, -0.00117]]
        b1 =  [0.60641, 0.06111, 0.24848, 0., 0.32098, 0., -0.21224, 0.007]
        w2 = [[0.10102, 0.22608, 0.32803, -0.11752],  [-0.01549, 0.39246, -0.30703, 0.12527],
              [-0.02698, 0.09462, -0.31409, 0.03994], [-0.08645, -0.00781, 0.52137, 0.45963],
              [0.07151, -0.27656, 0.23206, -0.06437], [-0.0154, 0.07408, -0.15198, -0.4007 ],
              [-0.17055, -0.06038, -0.8417, 0.43372], [-3.12708, 2.03716, -3.90529, 1.21203]]
        b2 = [0.54406, 0.36443, -0.21105, 0.35659]
        w3 = [[ 0.04271], [ 0.07268], [ 0.0702 ], [-0.16217]]
        b3 = [0.72121]

        # We assume the actual noise variance is a function of the following 8 values:
        #   1) Raw SNR
        #   2) subcarrier spacing
        #   3) Number of layers (or Tx antennas)
        #   4) Number of RX antennas
        #   5) Number of subcarriers
        #   6) lCdm
        #   7) kCdm
        #   8) length of estimates at pilot locations
        x = np.float64([ rawSnrDb, self.bwp.spacing, numTx, rr, kk, lCdm, kCdm, numVar ] )
        snrDb = (np.maximum(np.maximum(x.dot(w1)+b1, 0).dot(w2)+b2,0).dot(w3)+b3)[0]
        noisVar = 1/(toLinear(snrDb)*rr)
        return noisVar

    # ******************************************************************************************************************
    def estimateChannelLsEx(self, rsInfo, meanCdm=True, polarInt=True, int2d=True,
                            kernel='thin_plate_spline', neighbors=12, smoothing=0.0, degree=None):  # Not documented
        # This is the more flexible method for channel estimation with more control over the interpolation
        # parameters. The function "estimateChannelLS" is the official channel estimation method publicly
        # visible.
        # Here self is the rxGrid
        # rsInfo can be a "CsiRsConfig" object or a "DMRS" object
#        if rsInfo.__class__.__name__ == "CsiRsConfig":
        if isinstance(rsInfo, CsiRsConfig):
            csiRsConfig = rsInfo
            lCdm, kCdm = {1: (1,1), 2: (1,2), 4: (2,2), 8:(4,2) }[csiRsConfig.csiRsSetList[0].csiRsList[0].cdmSize]
            rsGrid = self.bwp.createGrid(csiRsConfig.numPorts)
            csiRsConfig.populateGrid(rsGrid)
            rsIndexes = rsGrid.getReIndexes("CSIRS_NZP")

#        elif rsInfo.__class__.__name__ == "DMRS":
        elif isinstance(rsInfo, DMRS):
            # For the case of DMRS, the returned channel (Heff) includes the effect of precoding. If 'V' is the
            # precoding matrix, we have y = H.V.x + n. This function returns Heff = H.V.
            dmrs = rsInfo
            lCdm, kCdm = dmrs.symbols, dmrs.configType
            rsGrid = self.bwp.createGrid( len(dmrs.pxxch.portSet) )
            dmrs.populateGrid(rsGrid)
            rsIndexes = rsGrid.getReIndexes("DMRS")
        
        cdmSize = lCdm * kCdm
        rr, ll, kk = self.shape     # Number of RX antennas, Number of symbols, Number of subcarriers
        pp, ll2, kk2 = rsGrid.shape # Number of Ports/Layers, Number of symbols, Number of subcarriers (from rsGrid)
        if (ll!=ll2) or (kk!=kk2):
            raise ValueError("The Grid size (%dx%d) does not match Reference Signals (%dx%d)."%(ll,kk,ll2,kk2))
        
        hEstAtPilots = []           # Channel Estimates at pilot locations. A list of 'numLs x numKs x rr' tensors
        hEstAtPilotSyms = []        # Channel Estimates at pilot symbols interpolated along the subcarriers. A list of
                                    # 'numLs x kk x rr' tensors one for each port
                                    
        for p in range(pp):
            portLs = rsIndexes[1][(rsIndexes[0]==p)]    # Indexes of symbols containing pilots in this port
            portKs = rsIndexes[2][(rsIndexes[0]==p)]    # Indexes of subcarriers containing pilots in this port

            ls = np.unique(portLs)                  # Unique Indexes of symbols containing pilots in this port
            ks = portKs[portLs==ls[0]]              # Unique Indexes of subcarriers containing pilots in this port
            numLs, numKs = len(ls), len(ks)

            pilotValues = rsGrid[p,ls,:][:,ks]   # Pilot values in this port. Shape: numLs x numKs
            rxValues = self.grid[:,ls,:][:,:,ks] # Received values for pilot signals in this port, (rr x numLs x numKs)
            
            # Channel estimates at pilot locations transposed to Shape: numLs x numKs x rr:
            hEst = np.transpose(rxValues/pilotValues[None,:,:], (1,2,0))
            
            hEstAtPilots += [ hEst ]                # Saving this to be used in the noise estimation

            if meanCdm:                                                     # Do CDM averaging
                # The number of pilots along symbols and subcarriers must be a multiple of 'lCdm' and 'kCdm'
                if (numKs%kCdm>0) or (numLs%lCdm>0):
                    raise ValueError("Partial CDMs are not supported in this version.")
                
                # Calculate the mean of all CDM groups; Shape: numLs/lCdm x numKs/kCdm x rr
                hEst = np.transpose(hEst.reshape(numLs,-1,kCdm,rr),
                                    (0,2,1,3)).reshape(numLs//lCdm, cdmSize, -1, rr).mean(1)

                if kCdm>1:  # Set the k values to the average subcarrier indices in the CDM group
                    ks = ks.reshape(-1,kCdm).mean(1)                            # Shape: numKs/kCdm

            # Interpolate along subcarriers:
            vs = np.transpose(hEst,(1,0,2))                                     # Shape: numKs/kCdm x numLs/lCdm x rr
            if polarInt:
                newVals = polarInterpolate(ks, vs, np.arange(kk), kernel, neighbors, smoothing) # kk x numLs/lCdm x rr
            else:
                newVals = interpolate(ks, vs, np.arange(kk), kernel, neighbors, smoothing)      # kk x numLs/lCdm x rr
            hEstInt = np.transpose(newVals,(1,0,2))                                             # numLs/lCdm x kk, rr
            hEstAtPilotSyms += [hEstInt]

        # Noise Estimation:
        riseLen = (min(self.bwp.symbolLens)-self.bwp.nFFT)*kk//self.bwp.nFFT
        
        # This is a sequence of 'riseLen' values monotonically and sinusoidally increasing from 0 to 1
        raisedCosine = (.5*(1-np.sin(np.pi*np.arange(riseLen-1,-riseLen,-2)/(2*riseLen))))
        
        # A window of shape: \__/ of len kk
        win = np.concatenate([ raisedCosine[::-1], np.float64((kk-2*riseLen)*[0]), raisedCosine])
        
        hEstDeltas = [] # A list of difference vectors between original pilot estimates and the denoised values
        for p in range(pp):
            portLs = rsIndexes[1][(rsIndexes[0]==p)]    # Indexes of symbols containing pilots in this port
            ls = np.unique(portLs)                      # Unique Indexes of symbols containing pilots in this port
            ks = portKs[portLs==ls[0]]                  # Unique Indexes of subcarriers containing pilots in this port
            estCirs = np.fft.ifft( hEstAtPilotSyms[p], axis=1)  # Channel Impulse Responses (CIR) (numLs/lCdm x kk x rr)
            estCirsWin = estCirs * win[None,:,None]        # The CIR after applying the window. (numLs/lCdm x kk x rr)
            hEstDenoised = np.fft.fft(estCirsWin, axis=1)  # Frequency domain (Denoised estimate) (numLs/lCdm x kk x rr)
            if lCdm>1:  # Repeat the hEstDenoised values for all the symbols of each CDM group
                hEstDenoised = np.repeat(hEstDenoised, lCdm, axis=0)            # Shape: numLs x kk x rr

            # Calculate the differences and flatten to a vector of length 'numLs*numKs*rr' for each port.
            hEstDeltas += [ (hEstAtPilots[p]-hEstDenoised[:,ks,:]).flatten() ]              # Shape: numLs*numKs*rr

        hEstDeltas = np.concatenate(hEstDeltas)                                             # Shape: numLs*numKs*rr*pp
        estNoiseVar = self.scaleNoiseVar( hEstDeltas.var(), pp, lCdm, kCdm, len(hEstDeltas))

        # Now doing interpolation along symbols
        # TODO: To support polar interpolation, we need to implement a reliable 2D unwrap function for the angles.
        hEst = []
        for p in range(pp):
            portLs = rsIndexes[1][(rsIndexes[0]==p)]        # Indexes of symbols containing pilots in this port
            portKs = rsIndexes[2][(rsIndexes[0]==p)]        # Indexes of subcarriers containing pilots in this port

            ls = np.unique(portLs)                          # Unique Indexes of symbols containing pilots in this port
            numLs = len(ls)

            if hEstAtPilotSyms[p].shape[0] == 1:
                hEst += [ np.repeat(hEstAtPilotSyms[p], ll, axis=0) ]
                continue

            if meanCdm:     # Set the 'l' values to the average symbol indices in the CDM group
                ls = ls.reshape(-1,lCdm).mean(1)                                    # Shape: numLs/lCdm

            if int2d:
                # Do 2D interpolation to get all channel values
                ks = np.arange(kk)
                pilotLKs = np.float64(np.meshgrid(ks, ls)).reshape(2, -1).T             # Shape: (numLs/lCdm)*kk x 2
                pilotValues = hEstAtPilotSyms[p].reshape(-1, rr)                        # Shape: (numLs/lCdm)*kk x rr

                f = RBFInterpolator(pilotLKs, pilotValues, neighbors, smoothing, kernel, degree=degree) # Interpolant
                allLKs = np.float64(np.meshgrid(range(kk), range(ll))).reshape(2, -1).T # Shape: ll*kk x 2
                allValues = f(allLKs).reshape(ll,kk,rr)                                 # Shape: ll x kk x rr
            else:
                # Do 1D interpolation along symbols
                vs = hEstAtPilotSyms[p]                                                 # Shape: numLs/lCdm x kk, rr
                # Note: Polar interpolation does not work here because of the wrapping mess with angles
                allValues = interpolate(ls, vs, np.arange(ll), kernel, neighbors, smoothing)    # Shape: ll x kk x rr

            hEst += [ allValues ]
            
        hEst = np.stack(hEst, axis=3)                                                   # Shape: ll x kk x rr x pp
        return hEst, estNoiseVar, hEstAtPilotSyms

    # ******************************************************************************************************************
    def estimateChannelLS(self, rsInfo, meanCdm=True, polarInt=False, kernel='linear'):
        r"""
        Performs channel estimation based on this received grid and the reference signal information in the 
        ``rsInfo``. Here is a list of steps taken by this function to calculated the estimated channel and noise
        variance:
        
        1) First the channel information is calculated at each pilot location using least squared method based on
        the following equations:
        
        .. math::

            Y_p = h_p \odot P + n_p

        where :math:`Y_p` is a vector of received values at the pilot locations which are the values in this
        :py:class:`Grid` object at the pilot locations indicated in ``rsInfo``, :math:`h_p` is the vector of channel
        values at pilot locations, :math:`P` is the vector of pilot values extracted from ``rsInfo``, and
        :math:`n_p` is the noise at pilot locations. The least square estimate of the channel values at pilot
        locations :math:`h_p` is then calculated by:
        
        .. math::

            h_p = \frac {Y_p} P  \qquad \qquad \qquad \text{(element-wise division)}

        2) If ``meanCdm`` is ``True``, the :math:`h_p` values in each CDM group are averaged which results in a new
        smaller set of :math:`h_p` values located at centers of CDM groups.
        
        3) Frequency interpolation along subcarriers is applied to :math:`h_p` values at all OFDM symbols containing 
        pilots based on ``polarInt`` and ``kernel`` values.
        
        4) A *raise-cosine* low-pass filter is applied to the Channel Impulse Response (CIR) values to get a
        *de-noised* version of CIRs. The noise variance is estimated using the difference between the noisy and
        de-noised versions of the CIRs.
        
        5) Finally another interpolation is applied along OFDM symbols to estimate the channel information for the 
        whole channel matrix.

        Parameters
        ----------
        rsInfo: :py:class:`~neoradium.csirs.CsiRsConfig` or :py:class:`~neoradium.pdsch.DMRS`
            This object contain reference signal information for the channel estimation. If it is a 
            :py:class:`~neoradium.csirs.CsiRsConfig` object, the channel matrix is estimated based on the CSI-RS
            signals which does not include the precoding effect.
            
            If this is a :py:class:`~neoradium.pdsch.DMRS` object, the channel matrix is estimated based on the 
            demodulation reference signals which includes the precoding effect.
            
        meanCdm: Boolean
            If ``True``, the :math:`h_p` values at pilot locations for each CDM group are averaged before applying 
            subcarrier interpolation. Otherwise interpolation is applied directly on the :math:`h_p` values.
            
        polarInt: Boolean
            If ``True``, the interpolation along the subcarriers is applied in polar coordinates. This means all
            :math:`h_p` values are converted to the polar coordinates and then the type of interpolation specified by
            ``kernel`` is applied to magnitudes and angles of these values. The results are then converted back to the
            cartesian coordinates. Otherwise (default), the interpolation is applied in the cartesian coordinates.
            
            Doing polar interpolation provides slightly better results at the cost of longer execution time.
            
        kernel: str
            The type of interpolation used for channel estimation process. The same type of 1-D interpolations are 
            applied along subcarriers and then OFDM symbols. Here is a list of supported values:
            
            :linear: A linear interpolation is applied to the values using extrapolation at both ends of the arrays.
                This uses the function 
                `interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_ with 
                ``kind`` set to ``linear``.

            :nearest: A nearest neighbor interpolation is applied to the values using extrapolation at both ends of
                the arrays. This uses the function
                `interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_
                with ``kind`` set to ``nearest``.

            :quadratic: A quadratic interpolation is applied to the values using extrapolation at both ends of the 
                arrays. This uses the function
                `interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_
                with ``kind`` set to ``quadratic``.

            :thin_plate_spline: An RBF interpolation is applied with a ``thin_plate_spline``
                kernel. This uses the
                `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
                class.

            :multiquadric: An RBF interpolation is applied with a ``multiquadric``
                kernel. This uses the
                `RBFInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html>`_
                class.
            
        Returns
        -------
        hEst: a 4-D complex numpy array
            If ``rsInfo`` is a :py:class:`~neoradium.csirs.CsiRsConfig` object, an ``L x K x Nr x Nt`` complex numpy
            array is returned where ``L`` is the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is
            the number of receiver antennas, and ``Nt`` is the number of transmitter antennas.
            
            If ``rsInfo`` is a :py:class:`~neoradium.pdsch.DMRS` object, an ``L x K x Nr x Nl`` complex numpy array
            is returned where ``L`` is the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is
            the number of receiver antennas, and ``Nl`` is the number of layers.
            
        estNoiseVar: float
            The estimated noise variance.
        """
        return self.estimateChannelLsEx(rsInfo, meanCdm, polarInt, False, kernel)[:2]

    # ******************************************************************************************************************
    def applyChannel(self, channelMatrix):
        r"""
        Applies a channel to this grid in frequency domain which results in a new *received* :py:class:`Grid` object.
        This function performs a matrix multiplication where this grid of shape ``Nt x L x K`` is multiplied by the
        channel matrix of shape ``L x K x Nr x Nt`` and results in the *received* grid of shape ``Nr x L x K``, where
        ``L`` is the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver
        antennas, and ``Nt`` is the number of transmitter antennas.
        
        This method can be used as a shortcut method to get the received resource grid faster compared to time domain 
        process of doing OFDM modulation, applying the channel, performing synchronization, and doing OFDM demodulation.
        
        Please note that the results are slightly different when a channel is applied in time domain vs frequency 
        domain.
        
        Parameters
        ----------
        channelMatrix: 4-D complex numpy array
            This is an ``L x K x Nr x Nt`` numpy array representing the estimated channel matrix, where ``L`` is the
            number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver antennas,
            and ``Nt`` is the number of transmitter antennas.
                        
        Returns
        -------
        :py:class:`Grid`
            The received grid of shape ``Nr x L x K``, where ``Nr`` is the number of receiver antennas, ``L`` is the
            number of OFDM symbols, and ``K`` is the number of subcarriers.
        """
        ll, kk, nr, nt = channelMatrix.shape
        if nt != self.numPorts:
            raise ValueError("Mismatch in the number of transmitter antennas (%d vs %d)!"%(nt, self.numPorts))
            
        # channelMatrix     grid           rxgrid
        #  ll,kk,nr,nt   nt,1,ll,kk  ->  nr,1,ll,kk
        #        2  3    0  1            0  1
        axes = [(2,3), (0,1), (0,1)]
        rxGrid = np.matmul(channelMatrix, self.grid[:,None,...], axes=axes)[:,0,:,:]        # Shape: nr,ll,kk

        grid = Grid(self.bwp, numPlanes=nr, numSlots=self.numSlots)
        grid.grid = rxGrid
        grid.reTypeIds = np.ones(grid.shape, dtype=np.uint8)*self.retNameToId["RX_DATA"]
        return grid

    # ******************************************************************************************************************
    def addNoise(self, **kwargs):
        r"""
        Adds Additive White Gaussian Noise (AWGN) to this resource grid based on the given noise properties. The
        *noisy* grid is then returned in a new :py:class:`Grid` object. The ``noiseVar`` property of the returned grid
        contains the variance of the noise applied by this function.
        
        Parameters
        ----------
        kwargs: dict
            One of the following parameters **must** be specified. They specify how the noise signal is generated.
            
            :noise: A numpy array with the same shape as this :py:class:`Grid` object containing the noise information.
                If the noise information is provided by ``noise`` it is added directly to the grid. In this case all
                other parameters are ignored.
            
            :noiseStd: The standard deviation of the noise. An AWGN complex noise signal is generated with zero-mean
                and the specified standard deviation. If ``noiseStd`` is specified, ``noiseVar`` and ``snrDb`` values
                below are ignored.

            :noiseVar: The variance of the noise. An AWGN complex noise signal is generated with zero-mean and the
                specified variance. If ``noiseVar`` is specified, the value of ``snrDb`` is ignored.

            :snrDb: The signal to noise ratio in dB. First the noise variance is calculated using the given SNR value
                and then an AWGN complex noise signal is generated with zero-mean and the calculated variance. This
                function uses the following formula to calculate the noise variance :math:`\sigma^2_{AWGN}` from
                :math:`snrDb`:
                
                .. math::

                    \sigma^2_{AWGN} = \frac 1 {N_r.10^{\frac {snrDb} {10}}}

                where :math:`N_r` is the number of receiver antennas.
                
            :ranGen: If provided, it is used as the random generator
                for the AWGN generation. Otherwise, **NeoRadium**'s :doc:`global random generator <./Random>` is used.

        Returns
        -------
        :py:class:`Grid`
            A new grid object containing the *noisy* version of this grid. The ``noiseVar`` property of the returned
            grid contains the variance of the noise applied by this function.
        """
        noise = kwargs.get('noise', None)
        if noise is not None:
            if self.shape != noise.shape:
                raise ValueError("Shape Mismatch: Grid: %s vs Noise: %s"%(str(self.shape), str(noise.shape)))
            grid = Grid(self.bwp, numPlanes=self.shape[0], numSlots=self.numSlots)
            grid.grid = self.grid + noise
            grid.reTypeIds = self.reTypeIds.copy()
            grid.noiseVar = noiseStd*noiseStd
            return grid
        
        ranGen = kwargs.get('ranGen', random)       # The Random Generator
        noiseStd = kwargs.get('noiseStd', None)
        if noiseStd is not None:
            noise = ranGen.awgn(self.shape, noiseStd)
            grid = Grid(self.bwp, numPlanes=self.shape[0], numSlots=self.numSlots)
            grid.grid = self.grid + noise
            grid.reTypeIds = self.reTypeIds.copy()
            grid.noiseVar = noiseStd*noiseStd
            return grid

        noiseVar = kwargs.get('noiseVar', None)
        if noiseVar is not None:
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        snrDb = kwargs.get('snrDb', None)
        if snrDb is not None:
            # SNR is the average SNR per RE per RX antenna
            snr = toLinear(snrDb)
            noiseVar = 1/(snr * self.shape[0])  # Note: self.shape[0] is the number of RX antennas
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        raise ValueError("You must specify the noise power using 'snrDb', 'noiseVar', or 'noiseStd'!")
        
    # ******************************************************************************************************************
    def drawMap(self, ports=[0], reRange=(0,12), title=None):
        r"""
        Draws a color-coded map of this grid object. Each ``plane`` of the grid is drawn separately with subcarriers
        in horizontal direction and OFDM symbols in vertical direction.
        
        Parameters
        ----------
        ports: list
            Specifies the list of ports (or ``planes``) to draw. Each port is drawn separately. By default this
            function draws only the first plane of the resource grid.
            
        reRange: tuple
            Specifies the range of subcarriers (REs) to draw. By default this function only draws the first resource
            block of the grid (subcarriers 0 to 12). The tuple ``(a, b)`` means the first RE drawn is the one at ``a``
            and last one is at ``b-1``.
           
        title: str or None
            If specified, it is used as the title for the drawn grid. Otherwise, this function automatically creates 
            a title based on the given parameters.
        """
        colorMap = colors.ListedColormap(self.retColors)

        if title is None:   title = "Slot Map for subcarriers %d to %d"%(reRange[0],reRange[1]-1)
        
        usedDataTypes = set()  # This is used for the Legend
        for p in ports:
            subGrid = self.reTypeIds[p,:,reRange[0]:reRange[1]]
            
            maxRetId = 0
            for retId in range(self.retMaxPredefine+self.retMaxCustom):
                if self.retIdToName[ retId ] is None: continue
                maxRetId = retId
                idx = np.where(subGrid==retId)
                if len(idx[0])>0: usedDataTypes.add( retId )

            plt.figure(figsize=(min(subGrid.shape[1]/2,12),6))
            x = np.arange(subGrid.shape[1]+1)-.5
            y = np.arange(subGrid.shape[0]+1)-.5
            plt.pcolormesh(x, y, subGrid, cmap=colorMap, edgecolors='black',
                           linewidths=(1 if subGrid.shape[1]<=48 else 0), vmin=0, vmax=len(self.retColors))
            
            if subGrid.shape[1]<=48:    plt.xticks(np.arange(subGrid.shape[1]))
            elif subGrid.shape[1]<=120: plt.xticks(np.arange(0,subGrid.shape[1],12))
            else:                       plt.xticks(np.arange(0,subGrid.shape[1],24))
            plt.yticks(np.arange(14))
            plt.xlabel("Subcarriers", fontsize=16)
            plt.ylabel("Symbols", fontsize=16)
            
            plt.title(title + " (Layer %d)"%(p), fontsize=18)
            
            if p == ports[-1]:  # Draw the legend only for the last port
                usedDataTypes = sorted(list(usedDataTypes))
                plt.legend(
                    [patches.Patch(facecolor=self.retColors[dataType],edgecolor='black') for dataType in usedDataTypes],
                    [self.retIdToName[dataType] for dataType in usedDataTypes],
                    loc='lower left', ncol=len(usedDataTypes), bbox_to_anchor=(0, -0.3), fontsize=12)
            plt.show()
