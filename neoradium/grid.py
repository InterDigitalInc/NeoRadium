# Copyright (c) 2024-2026, InterDigital AI Lab
"""
The module ``grid.py`` implements the :py:class:`Grid` class, which encapsulates the functionality of a resource grid,
including:

- Keeping the Resource Element (RE) values for a specified resource grid size.
- Providing easy access to a specific type of data in the resource grid (e.g., DM-RS values, CSI-RS values, PDSCH 
  data, etc.)
- Providing statistics and visualization for the resource grid map.
- Applying `OFDM <https://en.wikipedia.org/wiki/Orthogonal_frequency-division_multiplexing>`_ modulation to the 
  resource grid which results in a :py:class:`~neoradium.waveform.Waveform` object.
- Applying a :doc:`Channel Model <./Channels>` to the resource grid in the frequency domain.
- Applying *Additive White Gaussian Noise (AWGN)* to the resource grid in the frequency domain.
- Performing *Channel Estimation* based on a received resource grid and the configured 
  :doc:`reference signals <./RefSig>`.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 12/08/2023    Shahab Hamidi-Rad       Completed the documentation.
# 10/13/2025    Shahab Hamidi-Rad       * Added the calculation of noise power based on the signal power and SNR. See
#                                         the new 'getNoiseStd' and 'getRePower' functions and the updates to the
#                                         'addNoise' function.
#                                       * Added the 'clone' function.
# 03/12/2026    Shahab                  Changes in NeoRadium version 0.5.0:
#                                       * The grid now supports an object ID as well as the RE type assigned to each
#                                         resource element. For example, for CSI-RS REs, the CSI-RS resource ID is
#                                         available for each resource element belonging to that CSI-RS. See
#                                         the functions 'typeId', 'objId', 'makeTypeObjId', '__setitem__', and
#                                         'reTypeAndObjIdAt' (New) for more details.
#                                       * You can use the 'getReIndexes' function to get indices of REs for a specific
#                                         type and object ID.
#                                       * A grid can now have different (smaller) number of resource blocks than its
#                                         bandwidth part (e.g. when associated with a PDSCH that does not cover the
#                                         whole BWP).
#                                       * Improved the 'drawMap' method.
#                                       * Deprecated functions: 'precode', 'equalize', and 'estimateChannelLS'. New
#                                         versions are now available in the PDSCH class.
#                                       * Updated the 'addNoise' function's documentation. 'getNoiseStd' is not
#                                         documented anymore.
#                                       * Removed the 'reDesc' from the Grid class. It was originally defined only for
#                                         debugging purposes.
# **********************************************************************************************************************
import numpy as np
import os, scipy.io
from scipy.interpolate import RBFInterpolator, interp1d

from .utils import polarInterpolate, interpolate, herm, toLinear, toDb, deprecated, warnOnce
from .random import random
from .waveform import Waveform
from .csirs import CsiRsConfig
from .dmrs import DMRS

docFile = "Grid"         # Used by the 'deprecated' decorators

# **********************************************************************************************************************
class Grid:
    r"""
    This class implements the functionality of a resource grid. It stores the complex frequency-domain values of
    resource elements (REs) in the grid.

    All transformation methods (:py:meth:`ofdmModulate`, :py:meth:`applyChannel`, :py:meth:`addNoise`,
    :py:meth:`clone`, etc.) return a new :py:class:`Grid` or :py:class:`~neoradium.waveform.Waveform` object;
    the source grid is never mutated in place.
    """
    # ******************************************************************************************************************
    # ReTypes:
    # See https://matplotlib.org/stable/gallery/color/named_colors.html for more colors.
    # Predefined content types and colors:
    # The items are ordered such that the later items override the type of earlier items
    # at the same time/frequency locations for multi-port grids. For example, if the RE type at a specific OFDM symbol
    # and subcarrier in the first and second layers are "NO_DATA" and "DMRS", then when mixed (e.g., precoding)
    # the resulting RE will have type "DMRS" because its retId is larger.
    retIdToName, retColors = zip(*[ ("UNASSIGNED",    "white"),
                                    ("RESERVED",      "gray"),
                                    ("NO_DATA",       "lightgray"),
                                    ("PDSCH",         "cornflowerblue"),
                                    ("PDCCH",         "lime"),
                                    ("PUSCH",         "cornflowerblue"),    # Same color as PDSCH
                                    ("PUCCH",         "lime"),              # Same color as PDCCH
                                    ("DMRS",          "pink"),
                                    ("PTRS",          "yellow"),
                                    ("CSIRS_NZP",     "red"),
                                    ("CSIRS_ZP",      "orange") ])
    # Other colors available: "cyan", "lightblue", "peachpuff", "sienna", "violet"
    retMaxPredefine, retMaxCustom = 50, 20
    retIdToName = list(retIdToName) + (retMaxPredefine+retMaxCustom-len(retIdToName))*[None]
    
    # Fill unused colors with the color of "UNASSIGNED"
    retColors = list(retColors) + (retMaxPredefine+retMaxCustom-len(retColors))*[retColors[0]]
    
    retNameToId = {n:i for i,n in enumerate(retIdToName)}
    retNumCustom = 0

    # ******************************************************************************************************************
    def __init__(self, bwp, numPlanes=1, contents="UNASSIGNED", numSlots=1, numRbs=None):
        r"""
        Parameters
        ----------            
        bwp : :py:class:`~neoradium.carrier.BandwidthPart`
            The bandwidth part object based on which this resource grid is created.
            
        numPlanes : int (default: 1)
            A resource grid can be considered as a three-dimensional ``P x L x K`` complex tensor where ``L`` is the 
            number of OFDM symbols, ``K`` is the number of subcarriers (based on ``bwp``), and ``P`` is the number 
            of *planes*. In different contexts, ``P`` can be equivalent to the number of layers, number of transmitter
            antenna ports, or number of receiver antennas. To avoid confusion, the resource grid implementation in 
            **NeoRadium** uses the term *"plane"* for the first dimension of the resource grid.
        
        contents : str
            The default content type of this resource grid. Each resource element (RE) in the resource grid has an
            associated content type. When data is assigned to REs in this resource grid without a specified
            content type, the default value is used. The following content types are currently defined:
            
            :UNASSIGNED: A generic content type used when the type of data in the resource grid is unknown.
            :PDSCH: The content type used for the data carried in a Physical Downlink Shared Channel (PDSCH)
            :PDCCH: The content type used for the data carried in a Physical Downlink Control Channel (PDCCH)
            :PUSCH: The content type used for the data carried in a Physical Uplink Shared Channel (PUSCH)
            :PUCCH: The content type used for the data carried in a Physical Uplink Control Channel (PUCCH)
        
        numSlots : int
            The number of time slots to include in the resource grid. The number of time symbols ``L`` (the second
            dimension of the resource grid tensor) is equal to ``numSlots * bwp.symbolsPerSlot``.
            
        numRbs : int or None
            If this is specified, the resource grid will contain this many resource blocks. Otherwise (default), 
            the resource grid will have the same number of resource blocks as the bandwidth part.
            
        
        **Other Read-Only Properties:**
        
        Here is a list of additional properties:
        
            :shape: Returns the shape of the 3-dimensional resource grid tensor.
            :numPorts: The number of antenna ports. (The same as ``numPlanes``)
            :numLayers: The number of layers. (The same as ``numPlanes``)
            :numSubcarriers: The number of subcarriers in this resource grid.
            :numSymbols: The number of time symbols in this resource grid. This is equal to 
                ``numSlots*bwp.symbolsPerSlot``.
            :size: The size of the resource grid tensor.
            :noiseVar: The variance of AWGN noise present in this resource grid. This is usually initialized to
                zero. When AWGN noise is applied to the grid using the :py:meth:`addNoise` function, the variance
                of the noise is stored in this property. Also, if a noisy :py:class:`~neoradium.waveform.Waveform` is
                OFDM-demodulated using the :py:meth:`~neoradium.waveform.Waveform.ofdmDemodulate` method, then the
                amount of noise is transferred to the new :py:class:`~neoradium.grid.Grid` object created.
            
        
        Additionally, you can access the following read-only :py:class:`~neoradium.carrier.BandwidthPart` class 
        properties directly: ``nFFT``, ``symbolsPerSlot``, ``slotsPerSubFrame``, ``slotsPerFrame``, and 
        ``symbolsPerSubFrame``.
        
        **Resource Grid Indexing:**
        
        a) *Reading*: You can directly access the contents of the resource grid using indices. Here are a few
        examples of accessing the RE values in the resource grid:
        
        .. code-block:: python
        
            myREs = myGrid[0,2:5,:]     # instead of using myGrid.grid[0,2:5,:]
            print(myREs.shape)          # Assuming 612 subcarriers, this will print: "(3, 612)"
            
            indexes = myGrid.getReIndexes("DMRS")   # Get the indices of all DM-RS REs
            dmrsValues = myGrid[indexes]            # Get all DM-RS values as a 1-D array.
            
        
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
        self.numRbs = bwp.numRbs if numRbs is None else numRbs
        if type(contents)==str:
            if contents not in ["UNASSIGNED", "PDSCH", "PDCCH", "PUSCH", "PUCCH"]:
                raise ValueError("Unsupported grid content type \"%s\"!"%(contents))
            self.defaultReType = self.retNameToId[contents]
        elif self.retValid(contents)==False:
            raise ValueError("Unsupported grid content type \"%d\"!"%(contents))
        else:
            self.defaultReType = contents
            if self.retIdToName[self.defaultReType] not in ["UNASSIGNED", "PDSCH", "PDCCH", "PUSCH", "PUCCH"]:
                raise ValueError("Unsupported grid content type \"%s\"!"%(self.retIdToName[self.defaultReType]))

        self.numSlots = numSlots
        gridShape = ( numPlanes, numSlots*self.symbolsPerSlot, 12*self.numRbs )
        
        self.grid = np.zeros(gridShape, dtype=np.complex128)
        # reTypeObjIds for each RE is: reType*256 + objId. objId is set to 255 by default (e.g., unassigned object ID)
        self.reTypeObjIds = np.ones(gridShape, dtype=np.uint16)*self.makeTypeObjId(self.retNameToId["UNASSIGNED"])

        self.noiseVar = 0

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this resource grid object.

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
        if title is None:   title = "Resource Grid Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  Shape:                {' x '.join(str(x) for x in self.shape)}\n"
        repStr += indent*' ' + f"  numRbs:               {self.numRbs}\n"
        repStr += indent*' ' + f"  numSlots:             {self.numSlots}\n"
        repStr += indent*' ' + f"  Data Contents:        {self.retIdToName[self.defaultReType]}\n"
        repStr += indent*' ' + f"  Size:                 {self.size}\n"
        if self.noiseVar>0:
            repStr += indent*' ' + f"  Noise Var.:           {self.noiseVar}\n"

        repStr += self.bwp.print(indent+2, "Bandwidth Part:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def clone(self):
        r"""
        Creates a copy of this resource grid object.

        Returns
        -------
        :py:class:`Grid`
            A copy of this resource grid object.
        """
        grid = Grid(self.bwp, self.numPlanes, self.defaultReType, self.numSlots, self.numRbs)
        grid.grid = np.copy(self.grid)
        grid.reTypeObjIds = np.copy(self.reTypeObjIds)
        grid.noiseVar = self.noiseVar
        return grid

    # ******************************************************************************************************************
    @classmethod
    def makeTypeObjId(cls, reType, reObj=255):  return reType*256 + reObj
    @classmethod
    def typeId(cls, reTypeObj):                 return reTypeObj//256
    @classmethod
    def objId(cls, reTypeObj):                  return reTypeObj%256

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
        for retName, retId in self.retNameToId.items():         # Go through all RE types
            reIdx = np.where(self.typeId(self.reTypeObjIds)==retId)
            if len(reIdx[0])==0: continue
            objIds = set(self.objId(self.reTypeObjIds[reIdx]))
            if (len(objIds)==1) and (255 in objIds):            # No object IDs specified
                stats[ retName ] = len(reIdx[0])
            else:
                for objId in objIds:
                    reIdx = np.where(self.reTypeObjIds==self.makeTypeObjId(retId,objId))
                    stats[ f"{retName}({objId})" ] = len(reIdx[0])

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
    def numSymbols(self):       return self.grid.shape[1]
    @property
    def size(self):             return self.grid.size

    # ******************************************************************************************************************
    def __getattr__(self, attrName):        # Undocumented (Already mentioned in the __init__ documentation)
        # Get these attributes from the 'bwp' object
        if attrName not in ["nFFT", "symbolsPerSlot", "slotsPerSubFrame", "slotsPerFrame", "symbolsPerSubFrame"]:
            raise AttributeError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, attrName))
        return getattr(self.bwp, attrName)

    # ******************************************************************************************************************
    def __getitem__(self, key): # Undocumented (Already mentioned in section "Resource Grid Indexing")
        # This allows directly indexing the resource grid. (Reading)
        return self.grid[key]   # For example, you can use a = grid[1,2,3] instead of grid.grid[1,2,3]
                    
    # ******************************************************************************************************************
    def __setitem__(self, key, values):     # Undocumented (Already mentioned in section "Resource Grid Indexing")
        # This allows directly indexing the resource grid. (Writing)
        objectId = 255
        if type(values) == tuple:
            # e.g.: grid[1,2,3] = (123, "CSIRS_NZP", 1) -> Assigns the value, RE type CSIRS_NZP, and ID
            if len(values)==3:                  values, retName, objectId = values
            else:                               values, retName = values
            if self.retValid(retName) == False: raise ValueError("Unknown content type \"%s\"!"%(retName))
        elif type(values) == str:
            # For example: grid[1,2,3] = "RESERVED" -> Marks the RE as reserved (value is 0)
            values, retName = 0, values
            if self.retValid(retName) == False:
                raise ValueError("Unknown content type \"%s\"!"%(retName))
        else:
            # For example: grid[1,2,3] = 123 -> Assigns data (e.g., PDSCH)
            values, retName = values, self.retIdToName[self.defaultReType]

        self.grid[key] = values
        self.reTypeObjIds[key] = self.makeTypeObjId(self.retNameToId[retName],objectId)

    # ******************************************************************************************************************
    def reTypeAt(self, p, l, k):
        r"""
        Returns the content type (as a string) of the resource element at the position specified by ``p``, ``l``, and
        ``k``.

        Parameters
        ----------
        p : int
            The *plane* number. It can be the layer or antenna index depending on the context.
            
        l : int
            The time symbol index.

        k : int
            The subcarrier index.
        
        Returns
        -------
        str or tuple
            The content type of the resource element specified by ``p``, ``l``, and ``k``.
        """
        return self.retIdToName[self.typeId(self.reTypeObjIds[p,l,k])]

    # ******************************************************************************************************************
    def reTypeAndObjIdAt(self, p, l, k):
        r"""
        Returns the content type and object ID of the resource element at the position specified by ``p``, ``l``, and
        ``k``. For example, for a resource element assigned to non-zero-power CDI-RS, the RE type would be 
        ``"CSIRS_NZP"`` and the object ID would be the resource ID of the CSI-RS resource.

        Parameters
        ----------
        p : int
            The *plane* number. It can be the layer or antenna index depending on the context.
            
        l : int
            The time symbol index.

        k : int
            The subcarrier index.
        
        Returns
        -------
        tuple
            The content type and object ID associated with the resource element specified by ``p``, ``l``, and ``k``.
        """
        reTypeAndObjId = self.reTypeObjIds[p,l,k]
        return self.retIdToName[self.typeId(reTypeAndObjId)], self.objId(reTypeAndObjId)

    # ******************************************************************************************************************
    def getReIndexes(self, reTypeStr=None, objectId=255):
        r"""
        Returns the indices of all resource elements in the resource grid with the content type specified by the 
        ``reTypeStr``. For example, the code below gets the indices of all DM-RS resource elements in the resource grid 
        and uses the returned indices to retrieve these values. Here are some examples:
       
        .. code-block:: python
                    
            dmrsIdx = myGrid.getReIndexes("DMRS")   # Get the indices of all DM-RS resource elements
            dmrsValues = myGrid[dmrsIdx]            # Get all DM-RS values as a 1-D array.

            # Get indices of all CSI-RS resource elements
            allCsiRsIdx = myGrid.getReIndexes("CSIRS_NZP")  
            
            # Get indices of CSI-RS resource elements corresponding to a CsiRs with resourceId=1
            csiRsIdx1 = myGrid.getReIndexes("CSIRS_NZP",1)  # Get indices of CSI-RS resource elements for the CsiRs

        Parameters
        ----------
        reTypeStr : str or None
            If ``reTypeStr`` is `None`, the default content type of this resource grid is used as the key. For 
            example, if this resource grid was created with ``contents="PDSCH"``, then the indices of all resource 
            elements with content type "PDSCH" are returned.
            
            Otherwise, this function returns the indices of all resource elements in the resource grid with the 
            content type specified by ``reTypeStr``. Here is a list of values that can be used:
            
                :"UNASSIGNED": The *un-assigned* resource elements.
                :"RESERVED": The reserved resource elements. This includes the resource blocks reserved by the 
                    :py:class:`~neoradium.carrier.ReservedPrbSet` class.
                :"NO_DATA": The resource elements that should not contain any data. See ``numCdmGroupsWithoutData`` 
                    parameter of :py:class:`~neoradium.dmrs.DMRS` class for more information.
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

        objectId : int
            Specifies the object ID corresponding to the resource elements of type ``reTypeStr``. For example, if there 
            are many CSI-RS resources with different resourceId values in this grid, you could use this parameter to 
            specify the :py:class:`~neoradium.csirs.CsiRs` ``resourceId`` and only return the RE indices for the 
            specified resource ID. (See the above examples)
            
        Returns
        -------
        3-tuple
            A tuple of three 1-D NumPy arrays specifying a list of locations in the resource grid. This value can be
            used directly to access REs at the specified locations. (See the above example)
        """
        if reTypeStr is None:   reTypeStr = self.retIdToName[self.defaultReType]

        if type(reTypeStr) in [list, tuple]:
            reIndexes = [ self.getReIndexes(s) for s in reTypeStr ]
            return tuple( np.concatenate(idx) for idx in zip(*reIndexes) )

        if self.retValid(reTypeStr)==False:
            raise ValueError("Unknown RE Content type \"%s\"!"%(reTypeStr))
        
        if objectId==255:   return np.where(self.typeId(self.reTypeObjIds)==self.retNameToId[reTypeStr])
        return np.where(self.reTypeObjIds==self.makeTypeObjId(self.retNameToId[reTypeStr], objectId))
                
    # ******************************************************************************************************************
    def getReValues(self, reTypeStr=None, objectId=255):
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
        reTypeStr : str or None
            If ``reTypeStr`` is `None`, the default content type of this resource grid is used as the key. For
            example, if this resource grid was created with ``contents="PDSCH"``, then the values of all resource
            elements with content type "PDSCH" are returned.
            
            Otherwise, this function returns the values of all resource elements in the resource grid with the content
            type specified by ``reTypeStr``. See :py:meth:`getReIndexes` for a list of values that could be used
            for ``reTypeStr``.
                    
        objectId : int
            Specifies the object ID corresponding to the resource elements of type ``reTypeStr``. For example, if there 
            are many CSI-RS resources with different resourceId values in this grid, you could use this parameter to 
            specify the :py:class:`~neoradium.csirs.CsiRs` ``resourceId`` and only return the RE values for the 
            specified resource ID.

        Returns
        -------
        1-D NumPy array
            A 1-D complex NumPy array containing the values for all REs with the content type specified by
            ``reTypeStr``.
        """
        return self.grid[self.getReIndexes(reTypeStr, objectId)]
    
    # ******************************************************************************************************************
    @deprecated("PDSCH.precodeTo", docFile)
    def precode(self, f, reIndices=None):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`~neoradium.pdsch.PDSCH.precodeTo` method instead.

        Applies the specified precoding matrix to this grid object and returns a new *precoded* grid. This function
        supports *precoding resource block groups (PRGs)* which means different precoding matrices could be applied to 
        different groups of subcarriers in the resource grid. See **3GPP TS 38.214, Section 5.1.2.3** for more details.

        Parameters
        ----------
        f : NumPy array or list of tuples
            This function supports two types of precoding:
        
            :Wideband: ``f`` is an ``Nt x Nl`` matrix where ``Nt`` is the number of transmitter antenna ports and ``Nl``
                is the number of layers which **must** match the number of layers in the resource grid. In this case
                the same precoding is applied to all subcarriers of the resource grid.
            
            :Using PRGs: ``f`` is a list of tuples of the form (``groupRBs``, ``groupF``).
                For each entry in the list, the ``Nt x Nl`` precoding matrix ``groupF`` is applied to all subcarriers
                of the resource blocks listed in ``groupRBs``.

        reIndices : 3-tuple or None
            A tuple of three 1-D NumPy arrays specifying a list of locations in this resource grid where the precoding
            is applied. If this is `None` (default), the precoding is applied to the whole grid.
            
        Returns
        -------
        :py:class:`~neoradium.grid.Grid`
            A new :py:class:`~neoradium.grid.Grid` object of shape ``Nt x L x K`` where ``Nt`` is the number of 
            transmitter antenna ports, ``L`` is the number of OFDM symbols, and ``K`` is the number of subcarriers.
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
            # (K, Nt, Nl) . (Nl, L, K)  ->      (Nt, L, K)          <--- Shapes
            #     1   2      0   1               0   1              <--- Axes
            axes = [(1,2), (0,1), (0,1)]
        else:
            # f is a 2D matrix of shape Nt x Nl
            if type(f) != np.ndarray:
                raise ValueError("'f' must be a 2D NumPy array or a list of tuples.")
            nt, nl = f.shape
            if nl != self.numLayers:
                raise ValueError("The last dimension of 'f' (%d) must match the first dimension of the grid (%d)"%
                                 (f.shape[-1],self.shape[0]))
            #      f   . self.grid      ->      precodedGrid        <--- Tensors
            # (Nt, Nl) . (Nl, L, K)     ->      (Nt, L, K)          <--- Shapes
            #  0   1      0   1                  0   1              <--- Axes
            axes = [(0,1), (0,1), (0,1)]
            
        precodedGrid = Grid(self.bwp, nt, numRbs=self.numRbs)
        
        precodedGrid.reTypeObjIds = np.stack(nt*[self.reTypeObjIds.max(0)])
        precodedGrid.grid = np.matmul(f, self.grid, axes=axes)  # Precoded Grid Shape: Nt x L x K
        return precodedGrid

    # ******************************************************************************************************************
    def ofdmModulate(self, f0=0, windowing="STD"):
        r"""
        Applies OFDM modulation to the resource grid which results in a :py:class:`~neoradium.waveform.Waveform`
        object. This function is based on **3GPP TS 38.211, Section 5.3.1**.

        Parameters
        ----------
        f0 : float
            The carrier frequency of the generated waveform. If it is 0 (default), then a baseband waveform is
            generated and the *up-conversion* process explained in **3GPP TS 38.211, Section 5.4** is not applied.

        windowing : str
            A string indicating which type of windowing should be applied to the waveform after OFDM modulation. 
            The default value ``"STD"`` means that windowing is applied based on **3GPP TS 38.104, Sections B.5.2
            and C.5.2**. For more information, see :py:meth:`~neoradium.waveform.Waveform.applyWindowing` method of the
            :py:class:`~neoradium.waveform.Waveform` class.
            
        Returns
        -------
        :py:class:`~neoradium.waveform.Waveform`
            A :py:class:`~neoradium.waveform.Waveform` object containing the OFDM-modulated waveform information.
        """
        pp, ll, kk = self.shape
        assert (ll%self.bwp.symbolsPerSlot) == 0
            
        l0 = self.bwp.slotNoInSubFrame * self.symbolsPerSlot    # Number of symbols from the start of this subframe
        maxL = self.symbolsPerSubFrame - l0             # Maximum number of remaining symbols in this subframe from l0
        if ll > maxL:
            raise ValueError("Cannot modulate across subframe boundary! (At most %d symbols)"%(maxL))

        numPad = ((self.nFFT-kk+1)//2,(self.nFFT-kk)//2)    # Number of zeros to pad (beginning and end of subcarriers)
        paddedGrid = np.pad(self.grid, ((0,0),(0,0),numPad))        # Shape: pp, ll, nFFT
        shiftedPaddedGrid = np.fft.ifftshift(paddedGrid, axes=2)    # Shifted for IFFT
        waveform = np.fft.ifft(shiftedPaddedGrid, axis=2)           # Time-Domain waveforms:  Shape: pp, ll, nFFT
        
        symLens = self.bwp.symbolLens[l0:l0+ll]         # Symbol lengths in samples for each symbol in the next numSlots
        cpLens = symLens-self.nFFT                      # CP lengths in samples for each symbol in the next numSlots
        maxSymLen = symLens.max()
        
        # Indices used to insert the CP-length elements from the end of symbol waveforms to the beginning:
        indexes = (np.arange(maxSymLen) - cpLens[:,None])%self.nFFT

        waveformWithCPs = np.zeros((pp,ll, maxSymLen), dtype=np.complex128)     # Shape: pp, ll, maxSymLen
        
        # Insert the CP-length elements from the end of symbol waveforms to the beginning
        for l in range(ll): waveformWithCPs[:,l,:] = waveform[:,l,indexes[l]]
        
        # Up-conversion. See 3GPP TS 38.211, Section 5.4
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
    def estimateTimingOffset(self, rxWaveform):
        r"""
        Estimates the timing offset of a received waveform. This method first applies OFDM modulation to the
        resource grid and then calculates the correlation of this waveform with the given ``rxWaveform``. The timing
        offset is the index where the correlation reaches its maximum. The output of this function can be used by the
        :py:meth:`~neoradium.waveform.Waveform.sync` method of the :py:class:`~neoradium.waveform.Waveform` class
        to synchronize a received waveform.

        Parameters
        ----------
        rxWaveform : :py:class:`~neoradium.waveform.Waveform`
            The :py:class:`~neoradium.waveform.Waveform` object containing the received waveform.

        Returns
        -------
        int
            The timing offset, in number of time-domain samples. This is the number of samples that should be ignored
            from the beginning of the ``rxWaveform``.

        Notes
        -----
        The correlation is computed independently for each (RX antenna, TX port) pair using
        ``scipy.signal.correlate``. The magnitudes of those per-pair correlations are summed across all
        pairs, and the timing offset is the index of the peak of this aggregated magnitude. Magnitude is used
        (rather than the raw complex correlation) so that random per-pair phase offsets do not cancel out.
        """
        # Here, "self" is the grid created only for the CSI-RS symbols.
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
    @deprecated("PDSCH.equalize", docFile)
    def equalize(self, hf, noiseVar=None):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`~neoradium.pdsch.PDSCH.equalize` method instead.

        Equalizes a received resource grid using the estimated channel ``hf``. The estimated channel is assumed to
        include the effect of the precoding matrix, therefore, its shape is ``L x K x Nr x Nl`` where ``L`` is the
        number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver antennas, and
        ``Nl`` is the number of layers. The output of the equalization process is a new :py:class:`Grid` object
        of shape ``Nl x L x K``.
        
        This function also outputs Log-Likelihood Ratio (LLR) scaling factors which are used by the demodulation 
        process when extracting Log-Likelihood Ratios (LLRs) from the equalized resource grid.
        
        This method uses the Minimum Mean Squared Error (MMSE) algorithm for the equalization.

        Parameters
        ----------
        hf : 4-D complex NumPy array
            This is an ``L x K x Nr x Nl`` NumPy array representing the estimated channel matrix, where ``L`` is
            the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver antennas,
            and ``Nl`` is the number of layers.
            
        noiseVar : float or None
            The variance of noise applied to the received resource grid. If this is not provided, this method
            tries to use the noise variance of the resource grid obtained by the OFDM demodulation process for
            the time-domain case or the variance of the noise applied to the received resource grid by the
            :py:meth:`addNoise` method for the frequency domain case (See the ``noiseVar`` property of :py:class:`Grid`
            class).
            
        Returns
        -------
        eqGrid : :py:class:`Grid`
            The equalized grid object of shape ``Nl x L x K`` where ``Nl`` is the number of layers, ``L`` is the
            number of OFDM symbols, and ``K`` is the number of subcarriers.
            
        llrScales : 3-D NumPy array
            The Log-Likelihood Ratios (LLR) scaling factors which are used by the demodulation process when extracting
            Log-Likelihood Ratios (LLRs) from the equalized resource grid. The shape of this array is ``Nl x L x K``
            which is similar to ``eqGrid`` above.
        """
        # Here self is the rxGrid with shape: nr,ll,kk
        # hf is an estimate of h.f (Channel matrix including the effect of precoding). Shape: ll,kk,nr,pp
        if (self.shape[0] != hf.shape[2]) or (self.shape[1] != hf.shape[0]) or (self.shape[2] != hf.shape[1]):
            raise ValueError("Mismatch in the number of receiver antennas, OFDM symbols, or subcarriers!")

        if noiseVar is None:
            # This works if the noise was added to this grid using the addNoise function, or if
            # the noise was added to a waveform using the addNoise function and the
            # OFDM demodulation was called on that noisy waveform. In these two cases, we already
            # have the noise variance.
            noiseVar = self.noiseVar
        noiseVar = max(1e-8, noiseVar)          # Avoid division by zero
        
        ll,kk,rr,pp = hf.shape
        if rr<=pp:
            hhNoiseInv = np.linalg.pinv(herm(hf) @ hf + noiseVar*np.eye(pp), hermitian=True)
        else:
            # When rr > pp, it makes more sense to use SVD
            u, s, vH = np.linalg.svd(hf)
            s2NoiseInv = np.eye(s.shape[-1])/(np.square(s)+noiseVar)[...,None]
            hhNoiseInv = herm(vH) @ s2NoiseInv @ vH
            
        llrScale = (1/np.diagonal(hhNoiseInv,0,-2,-1)).real
        wMMSE = hhNoiseInv @ herm(hf)
        eq = np.matmul(wMMSE, self.grid[:,None,:,:], axes=[(2,3),(0,1),(0,1)])[:,0,:,:]
        
        eqGrid = Grid(self.bwp, numPlanes=pp, numSlots=self.numSlots, numRbs=self.numRbs)
        eqGrid.grid = eq
        eqGrid.reTypeObjIds = np.stack(pp*[self.reTypeObjIds[0]])
        eqGrid.noiseVar = noiseVar
        return eqGrid, np.transpose(llrScale,(2,0,1))   # Same shape: pp x ll x kk

    # ******************************************************************************************************************
    def scaleNoiseVar(self, rawNoiseVar, numTx, lCdm, kCdm, numVar):    # Undocumented
        # This method uses the raw noise variance calculated in the "estimateChannelLsEx"
        # function together with additional parameters to create an input vector. The
        # input vector "x" is then fed to a small neural network to obtain the actual
        # noise variance.
        rr, _, kk = self.shape             # Number of RX antennas, Number of subcarriers

        rawSnrDb = toDb( 1/(rawNoiseVar * rr) )
        if rawSnrDb>20: return rawNoiseVar
        
        # NN model parameters:
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
        #   2) Subcarrier spacing
        #   3) Number of layers (or Tx antennas)
        #   4) Number of RX antennas
        #   5) Number of subcarriers
        #   6) lCdm
        #   7) kCdm
        #   8) Length of the estimates at pilot locations
        x = np.float64([ rawSnrDb, self.bwp.spacing, numTx, rr, kk, lCdm, kCdm, numVar ] )
        snrDb = (np.maximum(np.maximum(x.dot(w1)+b1, 0).dot(w2)+b2,0).dot(w3)+b3)[0]
        noiseVar = 1/(toLinear(snrDb)*rr)
        return noiseVar

    # ******************************************************************************************************************
    def estimateChannelLsEx(self, rsInfo, meanCdm=True, polarInt=True, int2d=True,
                            kernel='thin_plate_spline', neighbors=12, smoothing=0.0, degree=None):  # Undocumented
        # This is the more flexible method for channel estimation with more control over the interpolation
        # parameters. The function "estimateChannelLS" is the official publicly visible channel-estimation method.
        # Here, self is the rxGrid
        # rsInfo can be a "CsiRsConfig" object or a "DMRS" object.
        if isinstance(rsInfo, CsiRsConfig):
            csiRsConfig = rsInfo
            lCdm, kCdm = {1: (1,1), 2: (1,2), 4: (2,2), 8:(4,2) }[csiRsConfig.csiRsSetList[0].csiRsList[0].cdmSize]
            rsGrid = Grid(self.bwp, csiRsConfig.numPorts, numRbs=self.numRbs)
            csiRsConfig.populateGrid(rsGrid)
            rsIndexes = rsGrid.getReIndexes("CSIRS_NZP")

        elif isinstance(rsInfo, DMRS):
            # For the case of DMRS, the returned channel (Heff) includes the effect of precoding. If 'V' is the
            # precoding matrix, we have y = H @ V @ x + n. This function returns Heff = H.V.
            dmrs = rsInfo
            lCdm, kCdm = dmrs.symbols, (4 if dmrs.enhanced else 2)
            rsGrid = Grid(self.bwp, len(dmrs.pxsch.portSet), numRbs=self.numRbs)
            dmrs.populateGrid(rsGrid)
            rsIndexes = rsGrid.getReIndexes("DMRS")
        
        cdmSize = lCdm * kCdm
        rr, ll, kk = self.shape     # Number of RX antennas, Number of symbols, Number of subcarriers
        pp, ll2, kk2 = rsGrid.shape # Number of ports/layers, number of symbols, number of subcarriers (from rsGrid)
        if (ll!=ll2) or (kk!=kk2):
            raise ValueError("The Grid size (%dx%d) does not match Reference Signals (%dx%d)."%(ll,kk,ll2,kk2))
        
        hEstAtPilots = []           # Channel estimates at pilot locations. A list of 'numLs x numKs x rr' tensors
        hEstAtPilotSyms = []        # Channel Estimates at pilot symbols interpolated along the subcarriers. A list of
                                    # 'numLs x kk x rr' tensors one for each port
                                    
        for p in range(pp):
            portLs = rsIndexes[1][(rsIndexes[0]==p)]    # Indices of symbols containing pilots in this port
            portKs = rsIndexes[2][(rsIndexes[0]==p)]    # Indices of subcarriers containing pilots in this port

            ls = np.unique(portLs)                  # Unique Indices of symbols containing pilots in this port
            ks = portKs[portLs==ls[0]]              # Unique Indices of subcarriers containing pilots in this port
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
            hEstInt = np.transpose(newVals,(1,0,2))                                             # numLs/lCdm x kk x rr
            hEstAtPilotSyms += [hEstInt]

        # Noise estimation:
        riseLen = (min(self.bwp.symbolLens)-self.bwp.nFFT)*kk//self.bwp.nFFT
        
        # This is a sequence of 'riseLen' values that increase monotonically and sinusoidally from 0 to 1
        raisedCosine = (.5*(1-np.sin(np.pi*np.arange(riseLen-1,-riseLen,-2)/(2*riseLen))))
        
        # A window of shape: \__/ of length kk
        win = np.concatenate([ raisedCosine[::-1], np.float64((kk-2*riseLen)*[0]), raisedCosine])
        
        hEstDeltas = [] # A list of difference vectors between the original pilot estimates and the denoised values
        for p in range(pp):
            portLs = rsIndexes[1][(rsIndexes[0]==p)]    # Indices of symbols containing pilots in this port
            ls = np.unique(portLs)                      # Unique Indices of symbols containing pilots in this port
            ks = portKs[portLs==ls[0]]                  # Unique Indices of subcarriers containing pilots in this port
            estCirs = np.fft.ifft( hEstAtPilotSyms[p], axis=1)  # Channel Impulse Responses (CIR) (numLs/lCdm x kk x rr)
            estCirsWin = estCirs * win[None,:,None]        # The CIR after applying the window. (numLs/lCdm x kk x rr)
            hEstDenoised = np.fft.fft(estCirsWin, axis=1)  # Frequency domain (Denoised estimate) (numLs/lCdm x kk x rr)
            if lCdm>1:  # Repeat the hEstDenoised values for all the symbols of each CDM group
                hEstDenoised = np.repeat(hEstDenoised, lCdm, axis=0)            # Shape: numLs x kk x rr

            # Calculate the differences and flatten them into a vector of length 'numLs*numKs*rr' for each port.
            hEstDeltas += [ (hEstAtPilots[p]-hEstDenoised[:,ks,:]).flatten() ]              # Shape: numLs*numKs*rr

        hEstDeltas = np.concatenate(hEstDeltas)                                             # Shape: numLs*numKs*rr*pp
        estNoiseVar = self.scaleNoiseVar( hEstDeltas.var(), pp, lCdm, kCdm, len(hEstDeltas))

        # Now perform interpolation along symbols
        # TODO: To support polar interpolation, implement a reliable 2D phase-unwrapping function for the angles.
        hEst = []
        for p in range(pp):
            portLs = rsIndexes[1][(rsIndexes[0]==p)]        # Indices of symbols containing pilots in this port
            portKs = rsIndexes[2][(rsIndexes[0]==p)]        # Indices of subcarriers containing pilots in this port

            ls = np.unique(portLs)                          # Unique Indices of symbols containing pilots in this port
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
                vs = hEstAtPilotSyms[p]                                                 # Shape: numLs/lCdm x kk x rr
                # Note: Polar interpolation does not work here because of the wrapping mess with angles
                allValues = interpolate(ls, vs, np.arange(ll), kernel, neighbors, smoothing)    # Shape: ll x kk x rr

            hEst += [ allValues ]
            
        hEst = np.stack(hEst, axis=3)                                                   # Shape: ll x kk x rr x pp
        return hEst, estNoiseVar, hEstAtPilotSyms

    # ******************************************************************************************************************
    @deprecated("PDSCH.estimateChannel", docFile)
    def estimateChannelLS(self, rsInfo, meanCdm=True, polarInt=False, kernel='linear'):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`~neoradium.pdsch.PDSCH.estimateChannel` method instead.

        Performs channel estimation based on this received grid and the reference signal information in the 
        ``rsInfo``. Here is a list of steps taken by this function to calculate the estimated channel and noise
        variance:
        
        1) First the channel information is calculated at each pilot location using the least-squares method based on
        the following equations:
        
        .. math::

            Y_p = h_p \odot P + n_p

        where :math:`Y_p` is a vector of received values at the pilot locations which are the values in this
        :py:class:`Grid` object at the pilot locations indicated in ``rsInfo``, :math:`h_p` is the vector of channel
        values at pilot locations, :math:`P` is the vector of pilot values extracted from ``rsInfo``, and
        :math:`n_p` is the noise at pilot locations. The least-squares estimate of the channel values at pilot
        locations :math:`h_p` is then calculated by:
        
        .. math::

            h_p = \frac {Y_p} P  \qquad \qquad \qquad \text{(element-wise division)}

        2) If ``meanCdm`` is `True`, the :math:`h_p` values in each CDM group are averaged which results in a new
        smaller set of :math:`h_p` values located at centers of CDM groups.
        
        3) Frequency interpolation along subcarriers is applied to :math:`h_p` values at all OFDM symbols containing 
        pilots based on ``polarInt`` and ``kernel`` values.
        
        4) A *raised-cosine* low-pass filter is applied to the Channel Impulse Response (CIR) values to get a
        *de-noised* version of CIRs. The noise variance is estimated using the difference between the noisy and
        de-noised versions of the CIRs.
        
        5) Finally another interpolation is applied along OFDM symbols to estimate the channel information for the 
        whole channel matrix.

        Parameters
        ----------
        rsInfo : :py:class:`~neoradium.csirs.CsiRsConfig` or :py:class:`~neoradium.dmrs.DMRS`
            This object contains reference-signal information for the channel estimation. If it is a 
            :py:class:`~neoradium.csirs.CsiRsConfig` object, the channel matrix is estimated based on the CSI-RS
            signals, which do not include precoding effects.
            
            If this is a :py:class:`~neoradium.dmrs.DMRS` object, the channel matrix is estimated based on the 
            demodulation reference signals, which include the precoding effect.
            
        meanCdm : bool
            If `True`, the :math:`h_p` values at pilot locations for each CDM group are averaged before applying 
            subcarrier interpolation. Otherwise, interpolation is applied directly on the :math:`h_p` values.
            
        polarInt : bool
            If `True`, the interpolation along the subcarriers is applied in polar coordinates. This means all
            :math:`h_p` values are converted to polar coordinates and then the type of interpolation specified by
            ``kernel`` is applied to magnitudes and angles of these values. The results are then converted back to the
            cartesian coordinates. Otherwise (default), the interpolation is applied in the Cartesian coordinates.
            
            Doing polar interpolation provides slightly better results at the cost of longer execution time.
            
        kernel : str
            The type of interpolation used for channel-estimation process. The same type of 1-D interpolations are 
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
        hEst : 4-D complex NumPy array
            If ``rsInfo`` is a :py:class:`~neoradium.csirs.CsiRsConfig` object, an ``L x K x Nr x Nt`` complex NumPy
            array is returned where ``L`` is the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is
            the number of receiver antennas, and ``Nt`` is the number of transmitter antennas.
            
            If ``rsInfo`` is a :py:class:`~neoradium.dmrs.DMRS` object, an ``L x K x Nr x Nl`` complex NumPy array
            is returned where ``L`` is the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is
            the number of receiver antennas, and ``Nl`` is the number of layers.
            
        estNoiseVar : float
            The estimated noise variance.
        """
        return self.estimateChannelLsEx(rsInfo, meanCdm, polarInt, False, kernel)[:2]

    # ******************************************************************************************************************
    def applyChannel(self, channelMatrix):
        r"""
        Applies a channel to this grid in the frequency domain which results in a new *received* :py:class:`Grid`
        object. This function performs a matrix multiplication where this grid of shape ``Nt x L x K`` is multiplied by 
        the channel matrix of shape ``L x K x Nr x Nt`` and results in the *received* grid of shape ``Nr x L x K``, 
        where ``L`` is the number of OFDM symbols, ``K`` is the number of subcarriers, ``Nr`` is the number of receiver
        antennas, and ``Nt`` is the number of transmitter antennas.
        
        This method can be used as a shortcut method to get the received resource grid faster compared to the 
        time-domain process of performing OFDM modulation, applying the channel, performing synchronization, and 
        carrying out OFDM demodulation.
        
        Please note that the results are slightly different when a channel is applied in the time domain vs. the 
        frequency domain.
        
        Parameters
        ----------
        channelMatrix : 4-D complex NumPy array
            This is an ``L x K x Nr x Nt`` NumPy array representing the estimated channel matrix, where ``L`` is the
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
        if ll != self.numSymbols:
            raise ValueError("Mismatch in the number of OFDM symbols (%d vs %d)!"%(ll, self.numSymbols))
        if kk != self.numRbs*12:
            raise ValueError("Mismatch in the number of subcarriers (%d vs %d)!"%(kk, self.numRbs*12))

        # channelMatrix     grid           rxgrid
        #  ll,kk,nr,nt   nt,1,ll,kk  ->  nr,1,ll,kk
        #        2  3    0  1            0  1
        axes = [(2,3), (0,1), (0,1)]
        rxGrid = np.matmul(channelMatrix, self.grid[:,None,...], axes=axes)[:,0,:,:]        # Shape: nr,ll,kk

        grid = Grid(self.bwp, numPlanes=nr, numSlots=self.numSlots, numRbs=self.numRbs)
        grid.grid = rxGrid
        # Copy RE types and object IDs from the first port of txGrid (=self) for all nr antennas
        grid.reTypeObjIds = np.stack(nr*[self.reTypeObjIds[0,:,:]])
        return grid

    # ******************************************************************************************************************
    def getRePower(self):                                                       # Undocumented
        # Returns the average RE power across the entire grid (over one slot, the full bandwidth, and all ports).
        # This corresponds to S_{RE} as defined in the page "SNR, signal and noise power calculations" in
        # "Implementation Notes" slides
        return (self.grid.var()/(self.bwp.nFFT**2)).item()

    # ******************************************************************************************************************
    def getNoiseStd(self, snr):                                                 # Undocumented
        # This method is only used in the 'addNoise' function below.
        # See equation 7 in the page "SNR, signal and noise power calculations" in the "Implementation Notes" slides
        return np.sqrt(self.grid.var()/snr)

    # ******************************************************************************************************************
    def addNoise(self, **kwargs):
        r"""
        Adds Additive White Gaussian Noise (AWGN) to this resource grid and returns a new
        :py:class:`Grid` object. The ``noiseVar`` property of the returned grid contains the
        variance of the applied noise.

        You can provide the noise directly, or specify its standard deviation, variance, or
        a target SNR.

        If you already have a noise signal in a NumPy array, use the ``noise`` parameter to
        add it directly to this grid:

        .. code-block:: python
            :caption: Example

            myNoise = random.awgn(rxGrid.shape, 0.1)    # Create AWGN with σ = 0.1
            rxGrid.addNoise(noise=myNoise)

        If you already know the standard deviation or variance of the noise, use
        ``noiseStd`` or ``noiseVar`` respectively:

        .. code-block:: python
            :caption: Example

            rxGrid.addNoise(noiseStd=0.1)       # Same result as above
            rxGrid.addNoise(noiseVar=0.01)      # Same result as above

        If you specify ``snrDb``, this function supports two different interpretations of SNR,
        controlled by the ``useRxPower`` parameter.

        **1) Reference-power SNR** (``useRxPower=False``)

        In this mode, the noise power is computed using a fixed reference signal power,
        independent of the instantaneous received grid. This approach is closer to typical
        **3GPP-style link-level simulation methodology**, where the AWGN level is fixed for
        a given SNR point and channel effects such as fading, path loss, and beamforming
        affect the received signal power without changing the injected noise power.

        In NeoRadium, this corresponds to assuming a normalized received signal power of
        :math:`\frac{1}{N_r}`, where :math:`N_r` is the number of receive antennas:

        .. math::

            \sigma^2_{AWGN} = \frac{1}{N_r \cdot 10^{\frac{SNR_{dB}}{10}}}

        .. code-block:: python
            :caption: Example

            rxGrid.addNoise(snrDb=mySnrDb, useRxPower=False)

        This mode is recommended for link-level performance evaluation and for generating
        results comparable to standard BLER vs. SNR curves. It is also the convention used
        by MATLAB 5G Toolbox link-level simulations.

        **2) Received-power-based SNR** (``useRxPower=True``)

        In this mode, the noise power is derived from the actual received grid. The function
        first estimates the average received signal power per resource element (RE), and then
        applies noise to achieve the requested SNR relative to that measured power:

        .. math::

            \sigma^2_{AWGN} = \frac{\sigma^2_{RX}}{10^{\frac{SNR_{dB}}{10}}}

        where :math:`\sigma^2_{RX}` is the estimated average received power per RE.

        .. code-block:: python
            :caption: Example

            rxGrid.addNoise(snrDb=mySnrDb, useRxPower=True)

        This mode enforces a **post-channel SNR**, meaning that the resulting SNR is tied to
        the instantaneous received signal. As a result, variations caused by fading or other
        channel effects are partially normalized out, since both signal and noise scale
        together.

        This approach is useful for controlled algorithm evaluation, for example when
        benchmarking equalization, channel estimation, or decoding at a fixed received SNR.
        However, it is generally **less suitable for 3GPP-style link-level performance
        studies**, where channel variability is expected to directly impact the effective SNR.

        In summary:

        * ``useRxPower=False``:
          Reference-power SNR. Recommended for link-level simulation results that are closer
          to common 3GPP-style evaluation methodology.

        * ``useRxPower=True``:
          Received-power-based SNR. Useful when you intentionally want to control the SNR
          relative to the actual received grid power.

        Please refer to the notebook :doc:`../Playground/Notebooks/Others/SnrCalculations`
        for a more detailed discussion of SNR definitions and AWGN scaling in **NeoRadium**.

        Parameters
        ----------
        kwargs : dict
            The amount of noise must be specified by one of ``noise``, ``noiseStd``,
            ``noiseVar``, or ``snrDb``.

            :noise: NumPy array with the same shape as this :py:class:`Grid` object containing
                the noise values. If ``noise`` is provided, it is added directly to the grid
                and all other parameters are ignored.

            :noiseStd: Standard deviation of the AWGN. Complex zero-mean AWGN is generated
                using the specified standard deviation. If ``noiseStd`` is specified,
                ``noiseVar`` and ``snrDb`` are ignored.

            :noiseVar: Variance of the AWGN. Complex zero-mean AWGN is generated using the
                specified variance. If ``noiseVar`` is specified, ``snrDb`` is ignored.

            :snrDb: Signal-to-noise ratio in decibels (dB). When ``snrDb`` is provided, the
                noise standard deviation is calculated from the given SNR and the
                ``useRxPower`` setting, and AWGN is generated accordingly.

            :useRxPower: Controls how ``snrDb`` is interpreted.

                * ``False``: Use the reference-power SNR convention. A normalized received
                  power of :math:`\frac{1}{N_r}` is assumed, where :math:`N_r` is the number
                  of receive antennas. This mode is closer to common 3GPP-style link-level
                  evaluation practice and is the default.

                * ``True``: Use the actual received grid power to compute the AWGN level.
                  This sets the noise power relative to the measured received signal power.

                .. note::
                    The default is ``False``. This keeps the behavior closer to common
                    3GPP-style link-level simulations and to MATLAB 5G Toolbox conventions.
                    For reproducibility and clarity, it is recommended to always set
                    ``useRxPower`` explicitly in user code.

            :ranGen: If provided, this random-number generator is used for AWGN generation. Typically a
                :py:class:`~neoradium.random.RanGen` instance, or any object exposing an
                ``awgn(shape, noiseStd)`` method that returns complex Gaussian samples. Otherwise,
                **NeoRadium**'s :doc:`global random generator <./Random>` (the module-level
                random singleton) is used.

        Returns
        -------
        :py:class:`Grid`
            A new grid containing the noisy version of this grid. The ``noiseVar`` property of
            the returned grid contains the variance of the noise applied by this function.
        """
        noise = kwargs.get('noise', None)
        if noise is not None:
            if self.shape != noise.shape:
                raise ValueError(f"Shape Mismatch: Grid: {self.shape} vs Noise: {noise.shape}")
            grid = self.clone()
            grid.grid += noise
            grid.noiseVar = noise.var()
            return grid
        
        ranGen = kwargs.get('ranGen', random)       # The random-number generator
        noiseStd = kwargs.get('noiseStd', None)
        if noiseStd is not None:
            noise = ranGen.awgn(self.shape, noiseStd)
            grid = self.clone()
            grid.grid += noise
            grid.noiseVar = noiseStd*noiseStd
            return grid

        noiseVar = kwargs.get('noiseVar', None)
        if noiseVar is not None:
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        snrDb = kwargs.get('snrDb', None)
        if snrDb is not None:
            # SNR is the average SNR per RE per RX antenna
            useRxPower = kwargs.get('useRxPower', False)
            snr = toLinear(snrDb)
            if useRxPower:
                # Post-channel SNR mode:
                # Use the measured received-grid power to set the AWGN level.
                return self.addNoise(noiseStd=self.getNoiseStd(snr), ranGen=ranGen)
            
            # Reference-power SNR mode (MATLAB-style / 3GPP-style link simulation convention):
            # assume Rx power = 1/Nr and keep noise independent of the instantaneous channel realization
            noiseVar = 1/(snr * self.shape[0])  # Note: self.shape[0] is the number of RX antennas
            return self.addNoise(noiseStd=np.sqrt(noiseVar), ranGen=ranGen)

        raise ValueError("You must specify the noise power using 'snrDb', 'noiseVar', or 'noiseStd'!")
        
    # ******************************************************************************************************************
    def drawMap(self, ports=[0], rbRange=(0,0), title=None, figSize=6.0, axes=None, reRange=None):
        r"""
        Draws a color-coded map of this grid object. Each ``port`` is drawn separately with subcarriers
        in the horizontal direction and OFDM symbols in vertical direction.
        
        Parameters
        ----------
        ports : list
            Specifies the list of ports (or ``planes``) to draw. Each port is drawn separately. By default, this
            function draws only the first plane of the resource grid.
            
        rbRange : tuple or int
            If this is an integer, this function draws the map for the specified resource block. If this is a 
            tuple, it specifies the range of resource blocks (RBs) to draw. By default, this function only draws the 
            first resource block of the grid (subcarriers 0 to 12). The tuple ``(a, b)`` means draw resource blocks 
            ``a`` to ``b`` including both ``a`` and ``b``.
           
        title : str or None
            If specified, it is used as the title for the drawn resource-grid map. Otherwise, this function  
            automatically creates a title based on the given parameters.

        figSize : float
            The figure size. Use this to control size of the plot. The default is 6.0.
            
        ax : `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_ or None
            If specified, it must be a matplotlib ``Axis`` object on which the resource grid map is drawn. This can 
            be used to create a group of matplotlib subplots and draw the resource grid map in one of the subplots.
            
        reRange : tuple
            This parameter is :red:`deprecated`: and included only for backward compatibility. It will be removed
            in future releases. Please use ``rbRange`` instead.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors
        import matplotlib.patches as patches
        colorMap = colors.ListedColormap(self.retColors)
        try:
            val = int(rbRange)
            rbRange = (val, val)
            defaultTitle = f"Slot Map for resource block {rbRange[0]}"
        except (TypeError, ValueError):
            if isinstance(rbRange, tuple):
                if rbRange[0]==rbRange[1]:  defaultTitle = f"Slot Map for resource block {rbRange[0]}"
                else:                       defaultTitle = f"Slot Map for resource blocks {rbRange[0]} to {rbRange[1]}"
            else:
                raise ValueError("'rbRange' must be a tuple of the form (firstRB, lastRB)!")
        if len(ports)>1: defaultTitle += f" ({len(ports)} ports)"
        if title is None: title = defaultTitle

        if (reRange is not None) and (rbRange==(0,0)):
            warnOnce("The 'reRange' parameter is deprecated and will be removed in future releases. "+
                     "Please use 'rbRange' instead!")
        else:
            reRange = (rbRange[0]*12, rbRange[1]*12+12)
            
        numREs = reRange[1]-reRange[0]+1
        usedDataTypes = set()  # This is used for the legend

        def scaleFont(f): return f*figSize/6
        if axes is None:
            fig, axes = plt.subplots(len(ports), 1,
                                     figsize=(min(numREs*figSize/12, 2*figSize),
                                     len(ports)*figSize), layout='constrained')
        else:
            fig = axes[0].get_figure()
            
        if not isinstance(axes, (list, np.ndarray)):    axes = [axes]

        fig.suptitle(title, fontsize=18*figSize/6)
        for p,ax in zip(ports, axes):
            subGrid = self.typeId(self.reTypeObjIds[p,:,reRange[0]:reRange[1]])

            maxRetId = 0
            for retId in range(self.retMaxPredefine+self.retMaxCustom):
                if self.retIdToName[ retId ] is None: continue
                maxRetId = retId
                idx = np.where(subGrid==retId)
                if len(idx[0])>0: usedDataTypes.add( retId )
            
            x = np.arange(subGrid.shape[1]+1)-.5
            y = np.arange(subGrid.shape[0]+1)-.5
            ax.pcolormesh(x, y, subGrid, cmap=colorMap, edgecolors='black',
                          linewidths=(.5 if numREs<=180 else 0),
                          vmin=0, vmax=len(self.retColors))
            ax.hlines(y, xmin=x.min(), xmax=x.max(), colors='black', linewidths=1)
            ax.vlines(x=[xx for xx in x if (xx+.5)%12==0], ymin=y.min(), ymax=y.max(),
                      colors='black', linewidths=2 if numREs<=180 else 1)

            if subGrid.shape[1]<=48:    ax.set_xticks(np.arange(subGrid.shape[1]))
            elif subGrid.shape[1]<=480: ax.set_xticks(np.arange(0,subGrid.shape[1],12))
            else:                       ax.set_xticks(np.arange(0,subGrid.shape[1],24))
            ax.tick_params(axis='x', bottom=False, top=False)
            ax.set_yticks(np.arange(14))
            for label in ax.get_yticklabels(): label.set_fontsize(scaleFont(12))
            for label in ax.get_xticklabels(): label.set_fontsize(scaleFont(12))
            if p == ports[-1]:  ax.set_xlabel("Subcarriers", fontsize=scaleFont(14))
            ax.set_ylabel("Symbols", fontsize=scaleFont(14))
            if len(ports)>1: ax.set_title(f"Port {p}", fontsize=scaleFont(14), loc='left')
            
        usedDataTypes = sorted(list(usedDataTypes))
        ax.legend([patches.Patch(facecolor=self.retColors[dataType],edgecolor='black') for dataType in usedDataTypes],
                  [self.retIdToName[dataType] for dataType in usedDataTypes],
                  loc='lower left', ncol=len(usedDataTypes), bbox_to_anchor=(0, -.2 if figSize>=4 else -.25),
                  fontsize=scaleFont(12))
        return axes
