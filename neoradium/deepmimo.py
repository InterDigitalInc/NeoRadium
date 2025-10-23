# Copyright (c) 2025 InterDigital AI Lab
"""
This module introduces the :py:class:`~neoradium.deepmimo.DeepMimoData` class, which encapsulates data related to 
various scenarios in the `DeepMIMO <https://www.deepmimo.net>`_ framework. The 
:py:meth:`~neoradium.deepmimo.DeepMimoData.getRandomTrajectory` method within this class facilitates the generation of 
a random trajectory within the specified `DeepMIMO <https://www.deepmimo.net>`_ scenario. Additionally, the 
:py:meth:`~neoradium.deepmimo.DeepMimoData.interactiveTrajPoints` method enables you to define your own trajectory on 
an interactive map. A complete example of using the :py:class:`~neoradium.deepmimo.DeepMimoData` class is available at
:doc:`../Playground/Notebooks/RayTracing/DeepMimo` in the playground.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 10/02/2024    Shahab Hamidi-Rad       First version of the file.
# 12/23/2024    Shahab                  Added documentation.
# 03/17/2025    Shahab                  Added support for interactive trajectory generation.
# 06/20/2025    Shahab                  Added the method "getChanGen" which samples random points from the current
#                                       scenario and returns a generator object that can generate channel matrices
#                                       corresponding to those random points.
# **********************************************************************************************************************
import numpy as np
import os, time, json
import scipy
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, colorConverter
import matplotlib.patches as mpatches
import matplotlib.animation as animation

from .trjchan import TrjPoint, Trajectory, TrjChannel
from .carrier import SAMPLE_RATE
from .utils import freqStr
from .random import random

testedScenarios = ["O1_3P5B", "O1_3P5", "O1_3P4", "O1_28B", "O1_28", "O1_60", "O1_140",
                   "ASU_CAMPUS1", "ASU_CAMPUS_3P5", "BOSTON5G_3P5", "BOSTON5G_28",
                   "CITY_9_SANFRANCISCO", "CITY_11_SANTACLARA",
                   "I1_2P5", "I1_2P4", "I3_2P4", "I3_60_V1", "OFFICEFLOOR1"]
inDoorScenarios = [ "I1_2P5", "I1_2P4", "I3_2P4", "I3_60_V1", "OFFICEFLOOR1"]

# **********************************************************************************************************************
class DeepMimoData:
    r"""
    This class encapsulates all the ray-tracing data read from `DeepMIMO <https://www.deepmimo.net>`_ scenario 
    files. It can be used to create random trajectories of user (UE) movements by interpolating the ray-tracing 
    information at intermediate points on the trajectory.
    
    The generated trajectories can then be used by the :py:class:`~neoradium.trjchan.TrjChannel` class to generate 
    temporally and spatially consistent sequences of MIMO channels.    
    """
    # This is the default path. It can be overwritten by the "setScenariosPath" function.
    pathToScenarios = "/data/RayTracing/DeepMIMO/Scenarios/"
    
    def __init__(self, scenario, baseStationId=1, gridId=0):
        r"""
        Parameters
        ----------
        scenario: str
            The name of the `DeepMIMO <https://www.deepmimo.net>`_ scenario, which is also the name of the folder 
            containing the scenario files. 

        baseStationId: int or str
            The base station identifier. In the newer versions of `DeepMIMO <https://www.deepmimo.net>`_ scenario 
            files, the base station identifier is a text string. You can use the 
            :py:meth:`~neoradium.deepmimo.DeepMimoData.showScenarioInfo` class 
            method to print information about available base stations and corresponding ``baseStationId`` values. The
            default value is 1. In case of string base station identifiers, this default value results in selecting 
            the first base station (after sorting the base station identifiers).
            
        gridId: int or str
            For the scenarios with multiple user grids, this parameter determines which user grid data should be 
            loaded. The default value is 0 which results in loading the first user grid. In the newer versions of 
            `DeepMIMO <https://www.deepmimo.net>`_ scenario files, the base station identifier is a text string. You 
            can use the :py:meth:`~neoradium.deepmimo.DeepMimoData.showScenarioInfo` class method to print 
            information about available user grids and their corresponding identifiers. In case of string grid 
            identifiers, this default value of 0 results in selecting the first user grid (after sorting the 
            user grid identifiers).
            

        **Other Properties:**
        
        After reading the `DeepMIMO <https://www.deepmimo.net>`_ scenario files, this class sets internal properties 
        as follows:
        
            :gridSize: A NumPy array of 2 integers indicating the number of grid points  in ``x`` and ``y`` 
                directions.
                
            :numGridPoints: The total number of grid points with ray-tracing data in the specified scenario. Note 
                that :math:`numGridPoints = gridSize[0] * gridSize[1]`.
                
            :delta: The distance between 2 neighboring grid points (in ``x`` or ``y`` direction). It is assumed that 
                this value is the same along the ``X`` and ``Y`` axes.
                
            :bsXyz: A NumPy array containing the three dimensional coordinates of the base station.
         
            :xyMin, xyMax: NumPy arrays containing the coordinates of lower left and upper right points on the grid. 
                In other words, for the :math:`(x,y)` coordinates of any grid point, we have:
                
                .. math::

                    xyMin[0] \le x \le xyMax[0]
                    
                    xyMin[1] \le y \le xyMax[1]
                

            :carrierFreq: The carrier frequency for the specified scenario.
            
            :minPaths, avgPaths, maxPaths: Measured statistics representing the minimum, average, and maximum number of 
                paths between the UE and the base station for all grid points in the specified scenario.
         
            :numTotalBlockage: Total number of grid points with no paths between the UE and the base station.

            :numLOS: Total number of grid points where there is a Line-of-Sight (LOS) path between the UE and the base
                station.


        **Indexing:**
        
        This class supports direct indexing to the :py:class:`~neoradium.trjchan.TrjPoint` objects in the DeepMIMO 
        dataset. For example:
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("O1_3p5B", baseStationId=3) # Read and create dataset
            tenFirstPoints = deepMimoData[:10]      # Getting the first 10 points in the dataset
            
            
        **Iterating through points:**
        
        This class has a generator function (``__iter__``) which makes it easier to use it in a loop. For example, the 
        following code counts the number of points with LOS paths.
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("O1_3p5B", baseStationId=3) # Read and create dataset
            numLosPoints = 0
            for point in deepMimoData:  # Use "deepMimoData" directly with the "for" loop
                if point.hasLos==1:
                    numLosPoints += 1
        """
        if scenario.upper() not in testedScenarios:
            print(f"Warning: This implementation has not been tested with scenario \"{scenario}\"!")
            
        scenarioFolder = self.pathToScenarios + scenario + "/"
        if os.path.exists(scenarioFolder) == False: scenarioFolder = os.path.expanduser("~") + scenarioFolder
        if os.path.exists(scenarioFolder) == False:
            raise ValueError("Could not find the folder \"%s\"!"%(self.pathToScenarios + scenario + "/"))

        self.scenario = scenario
        self.baseStationId = baseStationId
        self.gridId = gridId

        if os.path.exists(scenarioFolder + 'params.mat'):
            # New version of scenario files
            self.loadV3(scenarioFolder)
            return

        if os.path.exists(scenarioFolder + 'params.json'):
            self.loadV4(scenarioFolder)
            return

        self.scenario = scenario
        self.version = 1
        scenarioInfo = scipy.io.loadmat(scenarioFolder + '%s.params.mat'%(scenario))
        self.carrierFreq = scenarioInfo['carrier_freq'][0][0]
        gridInfo = np.int32(scenarioInfo['user_grids'])  # Each gridInfo is [startRow, endRow, usersPerRow]
        numUserGrids = len(gridInfo)
        gridId = int(gridId)
        if gridId>=numUserGrids:
            raise ValueError("Invalid \"gridId\" value (%d)! It must be smaller than %d!"%(gridId, numUserGrids))
        usersPerGrid = [(gi[1]-gi[0]+1)*gi[2] for gi in gridInfo]
        
        # Use the grid at `gridId`
        startRow, endRow, usersPerRow = gridInfo[gridId][0], gridInfo[gridId][1], int(gridInfo[gridId][2])
        self.numGridPoints = usersPerGrid[gridId]

        cirInfo = scipy.io.loadmat(scenarioFolder+'%s.%d.CIR.mat'%(scenario, baseStationId))
        cirInfo = cirInfo['CIR_array_full'][0].tolist()
        assert cirInfo[0]>=self.numGridPoints, "%d vs %d"%(cirInfo[0], self.numGridPoints)

        dodInfo = scipy.io.loadmat(scenarioFolder+'%s.%d.DoD.mat'%(scenario, baseStationId))
        dodInfo = dodInfo['DoD_array_full'][0].tolist()
        assert dodInfo[0]>=self.numGridPoints, "%d vs %d"%(dodInfo[0], self.numGridPoints)

        doaInfo = scipy.io.loadmat(scenarioFolder+'%s.%d.DoA.mat'%(scenario, baseStationId))
        doaInfo = doaInfo['DoA_array_full'][0].tolist()
        assert doaInfo[0]>=self.numGridPoints, "%d vs %d"%(doaInfo[0], self.numGridPoints)

        locInfo = scipy.io.loadmat(scenarioFolder+'%s.Loc.mat'%(scenario))
        locInfo = locInfo['Loc_array_full']
        assert locInfo.shape[0]>=self.numGridPoints, "%d vs %d"%(locInfo[0], self.numGridPoints)
        
        txLocInfo = scipy.io.loadmat(scenarioFolder+'%s.TX_Loc.mat'%(scenario))
        txLocInfo = txLocInfo['TX_Loc_array_full']
        self.bsXyz = txLocInfo[baseStationId-1][1:4]

        # losInfo is an array of length 'numGridPoints'. Each element can be:
        #       1: Has LOS path
        #       0: Does not have LOS path (All paths are NLOS)
        #       -1: No Paths at all
        losInfo = scipy.io.loadmat(scenarioFolder+f'{scenario}.{baseStationId}.LoS.mat')['LOS_tag_array_full'][0][1:]

        pathLossInfo = scipy.io.loadmat(scenarioFolder+'%s.%d.PL.mat'%(scenario, baseStationId))['PL_array_full']
        distances = pathLossInfo[:,0]   # Distance between each UE and the specified base station in meters
        pathLosses = pathLossInfo[:,1]  # Path Loss between each UE and the specified base station

        self.allTrjPoints = []
        userIdx, fileIdx = 0, 1
        self.maxPaths, self.minPaths, sumPaths, self.numTotalBlockage, self.numLOS = -1, 10000, 0, 0, 0
        for g in range(gridId+1):
            for userId in range(1,usersPerGrid[g]+1):
                assert (userId == cirInfo[fileIdx]) and (userId == dodInfo[fileIdx]) and (userId == doaInfo[fileIdx])
                fileIdx += 1
                numPaths = int(cirInfo[fileIdx])
                assert (numPaths == dodInfo[fileIdx]) and (numPaths == doaInfo[fileIdx])
                if g != gridId:
                    fileIdx += 4*numPaths + 1
                    userIdx += 1
                    continue
                
                fileIdx += 1
                pathsInfo = []
                for pathIdx in range(numPaths):
                    pathNo = pathIdx + 1
                    assert (pathNo==cirInfo[fileIdx]) and (pathNo==dodInfo[fileIdx]) and (pathNo==doaInfo[fileIdx]), \
                        "%d, %d, %d, %d"%(pathNo, int(cirInfo[fileIdx]), int(dodInfo[fileIdx]), int(doaInfo[fileIdx]))
                    phase, delay, power = cirInfo[fileIdx+1:fileIdx+4]  # Doppler Phase, Propagation Delay (𝜏), power
                    aod, zod, dodPower = dodInfo[fileIdx+1:fileIdx+4]   # Azimuth of departure, Zenith of departure
                    aoa, zoa, doaPower = doaInfo[fileIdx+1:fileIdx+4]   # Azimuth of arrival, Zenith of arrival
                    assert (doaPower==power) and (dodPower==power)
                    # Path Values: 0:Phase, 1:delay, 2:RxPower, 3:aoa, 4:zoa, 5:aod, 6:zod
                    pathsInfo += [ [phase, delay*1e9, power, aoa, zoa, aod, zod] ]
                    fileIdx += 4

                pathsInfo = np.array(pathsInfo).reshape(-1,7)
                xyz = locInfo[userIdx][1:4]
                distance = np.sqrt(np.square(self.bsXyz-xyz).sum()) if distances is None else distances[userIdx]
                pathLoss = 0 if pathLosses is None else pathLosses[userIdx]
                self.allTrjPoints += [ TrjPoint(xyz, losInfo[userIdx], pathsInfo, distance, pathLoss) ]

                if numPaths > self.maxPaths:    self.maxPaths = numPaths
                if numPaths < self.minPaths:    self.minPaths = numPaths
                if numPaths == 0:               self.numTotalBlockage += 1
                if losInfo[userIdx]==1:         self.numLOS += 1
                sumPaths += numPaths
                userIdx += 1
            
        self.gridSize = np.array([usersPerRow, endRow-startRow+1])   # Number of grid points along the X and Y axis
        self.xyMin = self.allTrjPoints[0].xyz[:2]
        self.xyMax = self.allTrjPoints[-1].xyz[:2]
        self.delta = (self.xyMax-self.xyMin)/(self.gridSize-1)

        self.avgPaths = sumPaths/len(self.allTrjPoints)

    # ******************************************************************************************************************
    def loadV3(self, scenarioFolder):   # Not documented
        # This function loads the DeepMIMO data based on the V2/V3 format of files
        # unused params: 'transmit_power'
        # TODO: Test this with scenarios with multiple user grids
        params = scipy.io.loadmat(scenarioFolder + 'params.mat')
        self.carrierFreq = params['carrier_freq'][0][0];
        self.version = params['version'][0][0];
        
        gridInfo = params['user_grids']
        numUserGrids = len(gridInfo)
        assert numUserGrids==1, "This code has not been tested with numUserGrids>1!"
        gridId = self.gridId
        if gridId>=numUserGrids:
            raise ValueError("Invalid \"gridId\" value (%d)! It must be smaller than %d!"%(gridId, numUserGrids))
        usersPerGrid = [int((gi[1]-gi[0]+1)*gi[2]) for gi in gridInfo]
        startRow, endRow, usersPerRow = np.int32(gridInfo[gridId])         # Use the grid at `gridId`
        self.numGridPoints = usersPerGrid[gridId]

        numBS = params['num_BS'][0][0];
        if self.baseStationId > numBS:  raise ValueError("Invalid base station \"%d\"!"%(self.baseStationId))
        
        self.dualPolarAvailable = params['dual_polar_available'][0][0];
        self.dopplerAvailable = params['doppler_available'][0][0];
        
        # Load the information about all UEs and the specified base station
        ueInfo = scipy.io.loadmat(scenarioFolder + 'BS%d_UE_0-%d.mat'%(self.baseStationId, self.numGridPoints))
        self.allTrjPoints = []
        self.maxPaths, self.minPaths, sumPaths, self.numTotalBlockage, self.numLOS = -1, 10000, 0, 0, 0
        for i in range(self.numGridPoints):
            channels = ueInfo['channels'][0][i][0][0][0]  # Shape: (numPathFields, numPaths)
            if channels.size==0:
                pathsInfo = np.empty((0,7))
                los = -1
                numPaths = 0
            else:
                # Path Values: 0:Phase, 1:delay, 2:RxPower, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:los
                pathsInfo = channels.T
                numPaths = pathsInfo.shape[0]
                los = 1 if any(pathsInfo[:,7]==1) else 0
                pathsInfo = pathsInfo[:,:7] # We don't keep los flag per path. The los path is the one with lowest delay
                pathsInfo[:,1] *= 1e9       # Delays are kept in nanoseconds

            # Using only rx Location for the UE location. In most scenarios txLoc and rxLoc are the same.
            rxLocs = ueInfo['rx_locs'][i];      # xyz, dist, pathloss
            txLocs = ueInfo['tx_loc'][0];       # Base station xyz
            xyz = rxLocs[:3]
            distance = rxLocs[3]
            pathLoss = rxLocs[4]
            
            self.allTrjPoints += [ TrjPoint(xyz, los, pathsInfo, distance, pathLoss) ]

            if numPaths > self.maxPaths:    self.maxPaths = numPaths
            if numPaths < self.minPaths:    self.minPaths = numPaths
            if numPaths == 0:               self.numTotalBlockage += 1
            if los==1:                      self.numLOS += 1
            sumPaths += numPaths
        
        self.bsXyz = txLocs
        self.gridSize = np.array([usersPerRow, endRow])   # Number of grid points along the X and Y axis
        self.xyMin = self.allTrjPoints[0].xyz[:2]
        self.xyMax = self.allTrjPoints[-1].xyz[:2]
        self.delta = self.allTrjPoints[usersPerRow+1].xyz[:2]-self.allTrjPoints[0].xyz[:2]
        self.avgPaths = sumPaths/len(self.allTrjPoints)
    
    # ******************************************************************************************************************
    def findId(self, name, id, default, dic):   # Not documented
        # This function aims to ensure backward compatibility as much as possible. It returns the most suitable
        # match for a base station or user grid identifier in the scenario information for the provided identifier,
        # which can be an integer or a text string.
        if type(id)==str:
            if id in dic:                   return id       # "id" is a string that exists in the JSON dictionary
            
        keys = list(dic.keys())
        if len(dic) == 1:                   return keys[0]  # Only one item in dictionary: Return its key (Ignore "id")
        
        # Create a dictionary of number_id -> string_id (For example: 3: "grid_3")
        try:    numKeysToKey = {int("".join([c for c in key if c.isdigit()])):key for key in keys}
        except: numKeysToKey = None     # Happens when there is no digits in string_ids (keys) in the JSON dictionary
        if numKeysToKey is None:    keys = sorted(keys)     # Sort the string_ids alphabetically
        else:
            # Sort the string ids based on the digits in them. (For example: "grid_2" comes before "grid_10")
            keys = [numKeysToKey[k] for k in sorted(list(numKeysToKey.keys()))]
            if type(id) != str:
                if id in numKeysToKey:      return numKeysToKey[id]
        
        # If we are here, there was no match. Now try one last trick:
        if id == default:                   return keys[0]  # Return the first item if id is the default value
        
        # Couldn't match id to any identifier in the dictionary. Raise an exception with useful information:
        if len(keys)==2:    optionsStr = f"'{keys[0]}' or '{keys[1]}'"
        else:               optionsStr = "'" + "', '".join(keys[:-1]) + f"', or '{keys[-1]}'"
        raise ValueError(f"Invalid '{name}' value '{id}'! It must be one of: {optionsStr}.")

    # ******************************************************************************************************************
    def loadV4(self, scenarioFolder):   # Not documented
        # Loading the scenario information from the new version (V4) of the DeepMIMO files.
        # The biggest backward compatibility problem is the change in Base Station and User Group IDs; they are
        # now text string instead of integers.
        with open(scenarioFolder + 'params.json', 'r') as file: metadata = json.load(file)

        self.carrierFreq = metadata['rt_params']['frequency'];
        self.version = metadata['version'];
        rxGridInfo = {}
        txInfo = {}
        
        # Reading the "txrx_sets" section of the JSON file which contains information about base station ids
        # and user grid ids and how they are paired.
        for i in range(100):
            if f'txrx_set_{i}' in metadata['txrx_sets']:
                txrx = metadata['txrx_sets'][f'txrx_set_{i}']
                if txrx['is_rx'] and (not txrx['is_tx']):
                    rxGridInfo[ txrx['name'] ] = (txrx['id'], txrx['num_points'])
                if txrx['is_tx']:
                    fileName =  f"{scenarioFolder}tx_pos_t{txrx['id']:03d}_tx{0:03d}_r{txrx['id']:03d}.mat"
                    bsPos = scipy.io.loadmat(fileName)["tx_pos"][0]
                    txInfo[ txrx['name'] ] = (txrx['id'], bsPos)
            else:
                break
        
        # Parameter names used in the file names
        paramNames = ['phase', 'delay', 'power', 'aoa_az', 'aoa_el', 'aod_az', 'aod_el', 'inter', 'rx_pos']
        paramVals = {}

        # Get best matches in the "rxGridInfo" and "txInfo" for the given "gridId" and "baseStationId. See "findId"
        self.gridId = self.findId('gridId', self.gridId, 0, rxGridInfo)
        self.baseStationId = self.findId('baseStationId', self.baseStationId, 1, txInfo)

        # Now open the files and read all multipath information for the given "baseStationId" and "gridId".
        rxId, self.numGridPoints = rxGridInfo[self.gridId]
        txId, self.bsXyz = txInfo[self.baseStationId]
        for paramName in paramNames:
            fileName =  f"{scenarioFolder}{paramName}_t{txId:03d}_tx000_r{rxId:03d}.mat"
            if os.path.exists(fileName)==False:
                raise ValueError(f"File {fileName} does not exist!")
            paramVals[paramName] = scipy.io.loadmat(fileName)[paramName]
    
        # Create the TrjPoint objects:
        self.allTrjPoints = []
        self.maxPaths, self.minPaths, sumPaths, self.numTotalBlockage, self.numLOS = -1, 10000, 0, 0, 0
        prevPoint, dx, dy, nx = None, None, None, None
        for ueIdx in range(self.numGridPoints):
            numPaths = (np.isnan(paramVals['phase'][ueIdx])==False).sum()
            if numPaths > 0:
                pathsInfo = np.stack([paramVals[n][ueIdx][:numPaths] for n in paramNames[:-1]],axis=1)
                assert np.isnan(pathsInfo).sum()==0
            else:
                pathsInfo = np.empty((0,8))

            pathsInfo[:,1] *= 1e9       # Delays are kept in nanoseconds
            ueXyz = paramVals['rx_pos'][ueIdx]
            bsUeDistance = np.sqrt(np.square(self.bsXyz-ueXyz).sum())
            los, nlos = (paramVals['inter'][ueIdx]==0).sum(), (paramVals['inter'][ueIdx]>=1).sum()
            assert (los+nlos)==numPaths, f"{los}, {nlos}, [numPaths], {paramVals['inter'][ueIdx]}"
            losFlag = -1 if numPaths==0 else los
            self.allTrjPoints += [ TrjPoint(ueXyz, losFlag, pathsInfo, bsUeDistance) ]

            if prevPoint is not None:
                d = ueXyz - prevPoint
                assert d[2]==0, f"{ueIdx}: dz ({d[2]}) is not zero!"
                if d[1]==0:
                    assert d[0]>0, f"{ueIdx}: dx ({d[0]}) - {d}   is not positive while dy is zero!"
                    if dx is None: dx = d[0]
                    assert np.abs(dx - d[0])<0.01, f"{ueIdx}: {d[0]} - {dx} = {d[0]-dx>=0.01}"
                else:
                    if dy is None: nx, dy = ueIdx, d[1]
                    assert np.abs(d[1]-dy)<0.01, f"{ueIdx}: {d[1]} - {dy} = {d[1]-dy}"
                    assert (ueIdx%nx) == 0, f"{ueIdx}: Grid row size is not consistent!"
            prevPoint = ueXyz


            if numPaths > self.maxPaths:    self.maxPaths = numPaths
            if numPaths < self.minPaths:    self.minPaths = numPaths
            if numPaths == 0:               self.numTotalBlockage += 1
            if los==1:                      self.numLOS += 1
            sumPaths += numPaths

        self.avgPaths = sumPaths/self.numGridPoints

        assert (self.numGridPoints%nx)==0, f"'numGridPoints'({self.numGridPoints}) is not a multiple of GridRow({nx})"
        self.gridSize = np.array([nx, self.numGridPoints//nx])   # Number of grid points along the X and Y axis
        self.xyMin = self.allTrjPoints[0].xyz[:2]
        self.xyMax = self.allTrjPoints[-1].xyz[:2]
        self.delta = np.array([dx, dy])
        
    # ******************************************************************************************************************
    @classmethod
    def showScenarioInfo(cls, scenario):
        r"""
        This class method prints information about the specified `DeepMIMO <https://www.deepmimo.net>`_ scenario. It 
        can be used to find out the base stations and user grids available in the scenario.
        
        Parameters
        ----------
        scenario: str
            The name of DeepMIMO scenario, which is also the name of the folder containing the scenario files. 
        """
        if scenario.upper() not in testedScenarios:
            print(f"Warning: This implementation has not been tested with scenario \"{scenario}\"!")

        scenarioFolder = cls.pathToScenarios + scenario + "/"
        if os.path.exists(scenarioFolder) == False: scenarioFolder = os.path.expanduser("~") + scenarioFolder
        if os.path.exists(scenarioFolder) == False:
            raise ValueError("Could not find the folder \"%s\"!"%(cls.pathToScenarios + scenario + "/"))

        if os.path.exists(scenarioFolder + 'params.mat'):
            # New version of scenario files
            cls.showV3(scenarioFolder)
            return
            
        if os.path.exists(scenarioFolder + 'params.json'):
            cls.showV4(scenarioFolder)
            return

        cls.showV1(scenarioFolder)

    # ******************************************************************************************************************
    @classmethod
    def showV1(cls, scenarioFolder):    # Not documented
        # Prints the scenario information for the scenarios stored with the first version of DeepMIMO file format.
        scenario = scenarioFolder.split(os.path.sep)[-2]
        scenarioInfo = scipy.io.loadmat(scenarioFolder + '%s.params.mat'%(scenario))
        print(f"Scenario:          {scenario}")
        print(f"File Version:      {1}")
        print(f"Carrier Frequency: {freqStr(scenarioInfo['carrier_freq'][0][0])}")
        print(f"Data Folder:       {scenarioFolder}")

        locInfo = scipy.io.loadmat(scenarioFolder+'%s.Loc.mat'%(scenario))["Loc_array_full"]
        usXyz = locInfo[:,1:4]
        gridInfo = np.int32(scenarioInfo['user_grids'])  # Array of [gridRowStart, gridRowEnd, gridCols]
        print(f"\nUE Grids ({len(gridInfo)}):")
        for g, (gridRowStart, gridRowEnd, gridCols) in enumerate(gridInfo):
            s,e = (gridRowStart-1)*gridCols, gridRowEnd*gridCols
            print(f"  {g}: Num UEs:{e-s:,}, " +
                  f"xRange:{usXyz[s:e].min(0)[0]:.2f}..{usXyz[s:e].max(0)[0]:.2f}, "+
                  f"yRange:{usXyz[s:e].min(0)[1]:.2f}..{usXyz[s:e].max(0)[1]:.2f}")

        txLocInfo = scipy.io.loadmat(scenarioFolder+'%s.TX_Loc.mat'%(scenario))['TX_Loc_array_full']
        print(f"\nBase Stations: ({len(txLocInfo)})")
        for b, bsInfo in enumerate(txLocInfo):
            print(f"  {b+1}: Position:({','.join('%.2f'%(x) for x in bsInfo[1:4])})")

    # ******************************************************************************************************************
    @classmethod
    def showV3(cls,scenarioFolder):     # Not documented
        # Prints the scenario information for the scenarios stored with the V2/V3 of DeepMIMO file format.
        params = scipy.io.loadmat(scenarioFolder + 'params.mat')

        print(f"Scenario:          {scenarioFolder.split(os.path.sep)[-2]}")
        print(f"File Version:      {params['version'][0][0]}")
        print(f"Carrier Frequency: {freqStr(params['carrier_freq'][0][0])}")
        print(f"Data Folder:       {scenarioFolder}")

        gridInfo = params['user_grids']
        print(f"\nUE Grids ({len(gridInfo)}):")
        for g, (gridRowStart, gridRowEnd, gridCols) in enumerate(gridInfo):
            numUEs = int((gridRowEnd-gridRowStart+1)*gridCols)
            ueXYZs = scipy.io.loadmat(scenarioFolder + f"BS1_UE_0-{numUEs}.mat")['rx_locs'][:,:3]
            print(f"  {g}: Num UEs:{numUEs:,}, " +
                  f"xRange:{ueXYZs.min(0)[0]:.2f}..{ueXYZs.max(0)[0]:.2f}, " +
                  f"yRange:{ueXYZs.min(0)[1]:.2f}..{ueXYZs.max(0)[1]:.2f}")

        bsXYZs = scipy.io.loadmat(scenarioFolder + 'BS1_BS.mat')['rx_locs'][:,:3]
        print(f"\nBase Stations: ({params['num_BS'][0][0]})")
        for b in range(params['num_BS'][0][0]):
            print(f"  {b+1}: Position:({','.join('%.2f'%(x) for x in bsXYZs[b])})")

    # ******************************************************************************************************************
    @classmethod
    def showV4(cls,scenarioFolder):     # Not documented
        # Prints the scenario information for the scenarios stored with the V4 of DeepMIMO file format.
        with open(scenarioFolder + 'params.json', 'r') as file: metadata = json.load(file)

        rxGridInfo = {}
        txInfo = {}
        for i in range(100):
            if f'txrx_set_{i}' in metadata['txrx_sets']:
                txrx = metadata['txrx_sets'][f'txrx_set_{i}']
                if txrx['is_rx'] and (not txrx['is_tx']):
                    rxGridInfo[ txrx['name'] ] = (txrx['id'], txrx['num_points'])
                if txrx['is_tx']:
                    fileName =  f"{scenarioFolder}tx_pos_t{txrx['id']:03d}_tx{0:03d}_r{txrx['id']:03d}.mat"
                    bsPos = scipy.io.loadmat(fileName)["tx_pos"][0]
                    txInfo[ txrx['name'] ] = (txrx['id'], bsPos)
            else:
                break
        
        print(f"Scenario:          {scenarioFolder.split(os.path.sep)[-2]}")
        print(f"File Version:      {metadata['version']}")
        print(f"Carrier Frequency: {freqStr(metadata['rt_params']['frequency'])}")
        print(f"Data Folder:       {scenarioFolder}")

        keys = list(rxGridInfo.keys())
        try:
            numKeysToKey = {int("".join([c for c in key if c.isdigit()])):key for key in keys}
            keys = [numKeysToKey[k] for k in sorted(list(numKeysToKey.keys()))]
        except:
            keys = sorted(keys)
        
        txId = list(txInfo.values())[0][0]
        print(f"\nUE Grids: ({len(rxGridInfo)})")
        for ueGridName in keys:
            gridId, numUEs = rxGridInfo[ueGridName]
            fileName =  f"{scenarioFolder}rx_pos_t{txId:03d}_tx000_r{gridId:03d}.mat"
            pos =  scipy.io.loadmat(fileName)['rx_pos']
            print(f"  {ueGridName}: ID:{gridId}, Num UEs:{numUEs:,}, " +
                  f"xRange:{pos.min(0)[0]:.2f}..{pos.max(0)[0]:.2f}, yRange:{pos.min(0)[1]:.2f}..{pos.max(0)[1]:.2f}")
            
        keys = list(txInfo.keys())
        try:
            numKeysToKey = {int("".join([c for c in key if c.isdigit()])):key for key in keys}
            keys = [numKeysToKey[k] for k in sorted(list(numKeysToKey.keys()))]
        except:
            keys = sorted(keys)
        print(f"\nBase Stations: ({len(txInfo)})")
        for bsName in keys:
            bsId, bsPos = txInfo[bsName]
            print(f"  {bsName}: ID:{bsId}, Position:({','.join('%.2f'%(x) for x in bsPos)})")

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="DeepMimoData Properties:", getStr=False):
        r"""
        Prints the properties of this class.
        
        Parameters
        ----------
        indent: int
            Used internally to adjust the indentation of the printed info.
            
        title: str
            The title used for the information. By default the text "DeepMimoData Properties:" is used.

        getStr: boolean
            If this is `True`, the function returns a text string instead of printing the info. Otherwise when this 
            is `False` (default) the function prints the information.
            
        Returns
        -------
        str or None
            If "getStr" is true, this function returns a text string containing the information about the properties of 
            this class. Otherwise, nothing is returned (default).
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  Scenario:                   {self.scenario}\n"
        repStr += indent*' ' + f"  Version:                    {self.version}\n"
        repStr += indent*' ' + f"  UE Grid:                    {self.gridId}\n"
        repStr += indent*' ' + f"  Grid Size:                  {self.gridSize[0]} x {self.gridSize[1]}\n"
        repStr += indent*' ' + f"  Base Station:               {self.baseStationId} (at {np.round(self.bsXyz,2)})\n"
        repStr += indent*' ' + f"  Total Grid Points:          {self.numGridPoints:,}\n"
        repStr += indent*' ' + f"  UE Spacing:                 {np.round(self.delta,2)}\n"
        repStr += indent*' ' + f"  UE bounds (xyMin, xyMax)    {np.round(self.xyMin,2)}, {np.round(self.xyMax,2)}\n"
        repStr += indent*' ' + f"  UE Height:                  {self.allTrjPoints[0].xyz[2]:.2f}\n"
        repStr += indent*' ' + f"  Carrier Frequency:          {freqStr(self.carrierFreq)}\n"
        repStr += indent*' ' + f"  Num. paths (Min, Avg, Max): {self.minPaths}, {self.avgPaths:.2f}, {self.maxPaths}\n"
        repStr += indent*' ' + f"  Num. total blockage:        {self.numTotalBlockage}\n"
        repStr += indent*' ' + f"  LOS percentage:             {self.numLOS*100/self.numGridPoints:.2f}%\n"

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def validateScenarioInfo(self, quiet=False):    # Not documented
        # Validates the information loaded from current scenario prints any problems and/or inconsistency in the
        # dataset.
        mins, maxs = 7*[+np.inf], 7*[-np.inf]
        for p in self:
            allParams =  [p.phases, p.delays, p.powers, p.aoas, p.zoas, p.aods, p.zods]
            losParams =  [p.losPhase, p.losDelay, p.losPower, p.losAoa, p.losZoa, p.losAod, p.losZod]
            nlosParams = [p.nlosPhases, p.nlosDelays, p.nlosPowers, p.nlosAoas, p.nlosZoas, p.nlosAods, p.nlosZods]
            if p.hasLos == -1:
                assert p.numPaths == 0,                                                       f"{p}"
                assert p.numNlosPaths == 0,                                                   f"{p}"
                for a in allParams:                          assert a is None,                f"{p}"
                for a in losParams:                          assert a is None,                f"{p}"
                for a in nlosParams:                         assert a is None,                f"{p}"
            elif p.hasLos == 1:
                assert p.numNlosPaths == p.numPaths-1,                                        f"{p}"
                for a,b in zip(losParams, allParams):        assert a == b[0],                f"{p}"
                if p.numPaths>1:
                    assert p.losDelay <= p.nlosDelays.min(),                                  f"{p}"
                    for a,b in zip(nlosParams, allParams):   assert np.abs(a-b[1:]).max()==0, f"{p}"
                else:
                    assert p.numPaths==1,                                                     f"{p}"
                    for a in nlosParams:                     assert a is None,                f"{p}"
                   
            elif p.hasLos == 0:
                assert p.numNlosPaths == p.numPaths,                                          f"{p}"
                for a in losParams:                          assert a is None,                f"{p}"
                for a,b in zip(nlosParams, allParams):       assert np.abs(a-b).max()==0,     f"{p}"
                
            if p.numPaths > 0:
                assert np.all(p.phases<=180) and np.all(p.phases>=-180),                      f"{p.phases}"
                assert np.all(p.aoas<=180) and np.all(p.aoas>=-180),                          f"{p.aoas}"
                assert np.all(p.aods<=180) and np.all(p.aods>=-180),                          f"{p.aods}"
                assert np.all(p.zoas<=180) and np.all(p.zoas>=0),                             f"{p.zoas}"
                assert np.all(p.zods<=180) and np.all(p.zods>=0),                             f"{p.zods}"
                assert np.abs(p.delays.argsort()-np.arange(p.numPaths)).max()==0,             f"{p.delays}"
                for i,values in enumerate(allParams):
                    maxs[i] = max(maxs[i], values.max())
                    mins[i] = min(mins[i], values.min())
                
            assert np.abs(self.gridXyToXy(self.xyToGridXy(p.xyz[:2]))-p.xyz[:2]).max()==0, \
                        f"{p.xyz[:2]}, {deepMimoData.xyToGridXy(p.xyz[:2])}"
                        
        print("Successfully validated all the information in the dataset.")
        print("Range of values:")
        paramNames = ["phases (Degrees)", "delays (ns)", "powers (dB)", "aoas (Degrees)",
                      "zoas (Degrees)", "aods (Degrees)", "zods (Degrees)"]
        for i,values in enumerate(allParams):
            print(f"    {paramNames[i]:17s}: {mins[i]} .. {maxs[i]}")

    # ******************************************************************************************************************
    @classmethod
    def setScenariosPath(cls, newPath):
        r"""
        This class method establishes the path to a folder that contains the ray-tracing scenarios. Within this 
        folder, each scenario is organized into its own sub-folder, with the same name as the scenario itself.
        
        Parameters
        ----------
        newPath: str
            The new path to the ray-tracing scenario files.
        """
        cls.pathToScenarios = newPath
        if cls.pathToScenarios[-1] != '/': cls.pathToScenarios += '/'

    # ******************************************************************************************************************
    def __iter__(self):                                     # Not documented
        # The generator function used to iterate through the "TrjPoint" objects in this DeepMIMO dataset.
        for point in self.allTrjPoints:
            yield point
            
    # ******************************************************************************************************************
    def __getitem__(self, idx):                             # Not documented
        # Provides indexing functionality
        return self.allTrjPoints[idx]

    # ******************************************************************************************************************
    # Grid positions are increased in steps of one in the actual X and Y dimensions. The grid starts at xy=(0,0) which
    # corresponds with "self.xyMin" and ends at xy=gridSize which corresponds to "self.xyMax". The following functions
    # convert these different "coordinate systems". These are not documented since they don't need to be called
    # directly. These function can be called for a single item or an array of items and returns values corresponding to
    # the shape of received inputs.
    def gridXyToXy(self, gridXy):
        indexes = np.array([self.gridXyToIndex(gridXy)]).reshape(-1,)
        return np.array([self.allTrjPoints[i].xyz[:2] for i in indexes]).squeeze()
    def xyToGridXy(self, xy):               return np.int32((np.array(xy)-self.allTrjPoints[0].xyz[:2])/self.delta+.5)
    def gridXyToIndex(self, trajectory):    return trajectory[...,0] + trajectory[...,1]*self.gridSize[0]

    # ******************************************************************************************************************
    def getRandomGridTraj(self, xyBounds, segLen, trajLen, xyStart=None, prob=None, trajDir="All"):   # Not documented
        # Generates a random trajectory on the grid. To create a real trajectory, intermediate points need to be
        # calculated by interpolation. This function is called by the "getRandomTrajectory" function below.
        
        # xyBounds is a 2x2 matrix: [[minX, minY], [maxX, maxY]]. These are actual X and Y values. We need the bounds
        # in the grid coordinates.
        minXy = np.maximum(self.xyMin, xyBounds[0])
        maxXy = np.minimum(self.xyMax, xyBounds[1])
        bounds = np.array([self.xyToGridXy(minXy),self.xyToGridXy(maxXy)])  # Grid Bounds
        
        if type(trajLen) is int: trajDist = np.inf                       # The given "trajLen" is number of grid points
        else:                    trajDist, trajLen = trajLen, 100000000  # The given "trajLen" is total travel distance
        
        # "trajDir" can be "All", "+X", "-X", "+Y", "-Y". In the last 4 cases, the UE should not move in reverse or
        # orthogonal to the specified direction.
        if xyStart is None:
            # If the start point of trajectory is not provided, set it based on trajectory direction
            if trajDir == "+X":     start = np.int32([bounds[0,0],bounds.mean(0)[1]]) # start at middle left & go right
            elif trajDir == "-X":   start = np.int32([bounds[1,0],bounds.mean(0)[1]]) # start at middle right & go left
            elif trajDir == "+Y":   start = np.int32([bounds.mean(0)[0],bounds[0,1]]) # start at center bottom & go up
            elif trajDir == "-Y":   start = np.int32([bounds.mean(0)[0],bounds[1,1]]) # start at center top & go down
            else:                   start = np.int32([bounds.mean(0)[0],bounds[0,1]]) # start at center bottom & go up
        else:
            # Otherwise, make sure we are in the bounds and we don't start at corners
            start = np.minimum(np.maximum(bounds[0]+[2*segLen,2*segLen], self.xyToGridXy( xyStart )),
                               bounds[1]-[2*segLen,2*segLen])

        # The delta values added to current position when moving in different directions
        dirToDeltas = {0:(1,0), 45:(1,1), 90:(0,1), 135:(-1,1), 180:(-1,0), 225:(-1,-1), 270:(0,-1), 315:(1,-1)}
        trajectory = [ np.int32(start) ]        # Current X,Y

        # Set the starting direction and maximum trajectory length if trajDir is not "All"
        if trajDir == "+X":     trajLen, curDir = min(trajLen,bounds[1,0]-start[0]-segLen), 0
        elif trajDir == "-X":   trajLen, curDir = min(trajLen,start[0]-bounds[0,0]-segLen), 180
        elif trajDir == "+Y":   trajLen, curDir = min(trajLen,bounds[1,1]-start[1]-segLen), 90
        elif trajDir == "-Y":   trajLen, curDir = min(trajLen,start[1]-bounds[0,1]-segLen), 270
        else:                   curDir = 0
    
        # The "prob" is a tuple of 3 values for probability of turning right, going straight, and turning left.
        # [0,1,0] can be used to force it to always go straight. None means uniform distribution for each decision.
        if prob is None:
            probNoLeft = probNoRight = None
        elif (type(prob)==tuple) and (len(prob)==3):
            probNoLeft  = (prob[0]/(prob[0]+prob[1]), prob[1]/(prob[0]+prob[1]))  # Used when left turn is not possible
            probNoRight = (prob[1]/(prob[1]+prob[2]), prob[2]/(prob[1]+prob[2]))  # Used when right turn is not possible
        else:
            raise ValueError("'prob' must be a tuple of 3 values for probabilities " +
                             "of turning right, going straight, and turning left.")
        allowedTurns = {"+X": {45: "NoLeft", 315:"NoRight", 0:  "All"},
                        "-X": {225:"NoLeft", 135:"NoRight", 180:"All"},
                        "+Y": {135:"NoLeft", 45: "NoRight", 90: "All"},
                        "-Y": {315:"NoLeft", 225:"NoRight", 270:"All"}}
                        
        def isBadMove(newXY, newDir):
            # returns True is a movement in direction "newDir" which ends up in "newXY" grid position
            # results in crossing the specified bounds.
            corner = {0:(-1,-1), 45:(2,3), 90:(-1,-1), 135:(0,3), 180:(-1,-1), 225:(0,1), 270:(-1,-1), 315:(1,2)}
            border = {0:2, 45:-1, 90:3, 135:-1, 180:0, 225:-1, 270:1, 315:-1}
    
            # An array of 4 values indicating how close we are to each boarder
            borderCloseness = ((bounds - newXY)*[[-1],[1]]).flatten()
            if np.any(borderCloseness<0):      return True
            
            # An array of corner indices that are close to the "newXY" position
            closeCorners = tuple(np.where(borderCloseness<2*segLen)[0])
            if corner[newDir] == closeCorners: return True

            # An array of border indices that are close to the "newXY" position
            closeBorders = tuple(np.where(borderCloseness<segLen)[0])
            if border[newDir] in closeBorders: return True
    
            return False

        curTrajDist = 0
        while len(trajectory) < trajLen:
            if trajDir=="All":          action = random.choice([-1,0,1], p=prob)
            else:
                turns = allowedTurns[trajDir][curDir]
                if turns=="NoLeft":     action = random.choice([-1,0], p=probNoLeft) # action: -1 or 0 (No left turn)
                elif turns=="NoRight":  action = random.choice([0,1], p=probNoRight) # action: 0 or 1 (No right turn)
                else:                   action = random.choice([-1,0,1], p=prob)

            newDir = (curDir + action*45)%360
            newXY = trajectory[-1] + segLen*np.int32(dirToDeltas[newDir])
            if isBadMove(newXY, newDir):    # This is a bad move -> Do use in trajectory; Try a different direction.
                continue
        
            trajectory += [ trajectory[-1]+(s+1)*np.int32(dirToDeltas[newDir]) for s in range(segLen) ]
            curDir = newDir
            
            # Assuming delta is the same along the X and Y axis
            curTrajDist += self.delta[0] * segLen * np.sqrt(np.square(dirToDeltas[newDir]).sum())
            if curTrajDist>trajDist:        break
            
        return np.array(trajectory[:trajLen])   # trajLen x 2

    # ******************************************************************************************************************
    def getSamplePerPoints(self, speedMps, u=0, cpType="normal", timeRes="coherence"):  # Not documented
        # NOTE: This function is not used anymore and will be removed later
        # This function returns the number of samples (at SAMPLE_RATE) per trajectory step
        # based on the given timeRes, cpType, and u
        symPerSlot = 14 if cpType.lower()=='normal' else 12
        if timeRes.lower() in ["symbol","slot"]:
            # points per slot: 14 (or 12) for "symbol", 2 for "slot"
            subSegPerSec = 1000*(1<<u)*{"symbol":symPerSlot, "slot":2}[timeRes.lower()]
        elif timeRes.lower() == "coherence":
            # First calculate the coherence time based on speed
            c = 299792458                                       # speed of light
            dopplerShift = speedMps*self.carrierFreq/c     # Doppler Shift
            
            # Coherence Time. See: https://en.wikipedia.org/wiki/Coherence_time_(communications_systems)
            coherenceTime = np.sqrt(9/(16*np.pi))/dopplerShift
            subSegPerSec = 2/coherenceTime                      # we want to sample at least twice during coherenceTime
            if subSegPerSec < 2*1000*(1<<u):
                subSegPerSec = 2*1000*(1<<u)                    # One slot divided to at least 2 parts
            elif subSegPerSec < 4*1000*(1<<u):
                subSegPerSec = 4*1000*(1<<u)                    # One slot divided to 4 parts
            elif subSegPerSec < (symPerSlot//2)*1000*(1<<u):
                subSegPerSec = (symPerSlot//2)*1000*(1<<u)      # One slot divided to 6/7 parts
            else:
                subSegPerSec = symPerSlot*1000*(1<<u)           # One slot divided at most to 12/14 parts
        
        # Return the number of samples (at SAMPLE_RATE) per trajectory step
        return int(np.round(SAMPLE_RATE/subSegPerSec))

    # ******************************************************************************************************************
    def interpolateTrajectory(self, idxTrajectory, speedMps, bwp):    # Not documented
        # This function generates intermediate points on a trajectory by linearly interpolating between the
        # endpoints of each segment. The “idxTrajectory” list contains the indices of the points in the dataset that
        # are included in the trajectory. In the code in this function, a “segment” refers to the line connecting two
        # consecutive points on the grid along the trajectory, while a “step” denotes the line connecting two
        # consecutive trajectory points after interpolation. The desired duration of each step is precisely equal
        # to the corresponding slot length in the 3GPP standard sampling time (30,720,000 samples per second).
        slotLens = [bwp.getSlotLen(i) for i in range(bwp.slotsPerSubFrame)]
        slotStarts = np.cumsum([0]+slotLens)
        subFrameLen = int(bwp.sampleRate//1000)  # 1 ms in number of samples (at 30,720,000 samples per second)
        
        # We create a point for each slot and use the closest speed that divides a segment into a whole number of
        # steps.
        xyzs = np.array([self.allTrjPoints[i].xyz for i in idxTrajectory])  # Shape: (trajLen, 3)
        segLens = np.sqrt(np.square(xyzs[1:]-xyzs[:-1]).sum(-1))            # The lengths of all trajLen-1 segments
        intPoints=[]
        segStart = 0
        for i in range(1,len(idxTrajectory)):                           # Doing interpolation on one segment at a time
            # First get the 2 points of this segment
            p0 = self.allTrjPoints[idxTrajectory[i-1]]                  # The TrjPoint object for the start of segment
            p1 = self.allTrjPoints[idxTrajectory[i]]                    # The TrjPoint object for the end of segment
            numSubFrame = segLens[i-1]*bwp.sampleRate/(subFrameLen*speedMps) # Number of subFrames for this segment
            subFrameFrac = (numSubFrame%1)*subFrameLen                  # Number of samples in the fractional subframe
            slotIdx = np.abs(slotStarts-subFrameFrac).argmin()          # Number of additional slots
            numSubFrame = int(numSubFrame)                              # Use it as int from now on
            numSegSamples = numSubFrame*subFrameLen+slotStarts[slotIdx] # Total number of samples in this segment
            numSteps = numSubFrame*bwp.slotsPerSubFrame + slotIdx       # Total slots/steps for this segment

#            print(p0, p1, slotIdx, slotStarts, numSubFrame, subFrameLen,  numSegSamples)
            # Calculate the sample number at the begining of each step:
            # Note that slotIdx ∈ {0, ..., bwp.slotsPerSubFrame}
            if slotIdx == bwp.slotsPerSubFrame:
                # This is when fraction is close to 1 => Total duration (in samples) on this segment is the
                # next multiple of subFrameLen. We want all slots from 0 to (numSubFrame+1) inclusive
                stepStarts = slotStarts[None,:-1]+np.arange(numSubFrame+2,dtype=np.int32)[:,None]*subFrameLen
            else:
                stepStarts = slotStarts[None,:-1]+np.arange(numSubFrame+1,dtype=np.int32)[:,None]*subFrameLen
            stepStarts = stepStarts.flatten()[:numSteps+1]
            assert stepStarts[-1]==numSegSamples, f"{stepStarts}\n{numSegSamples}\n"

            if segStart > 0:
                # The starting point has already been included as the last point of previous segment.
                stepStarts = stepStarts[1:]

            seqLinSpeed = segLens[i-1]*bwp.sampleRate/numSegSamples # Actual linear speed. Close to the desired speedMps
            assert np.abs(seqLinSpeed-speedMps) < (0.1*speedMps)    # Ensure we are still close enough to original speed
            segSpeed = (p1.xyz-p0.xyz)*bwp.sampleRate/numSegSamples # The 3D speed vector on this segment
            
            if (p0.hasLos == -1) or (p1.hasLos == -1):
                # Total blockage -> No Interpolation needed. All points on this segment will have no paths.
                c = 0
            else:
                # If m'th path in 'p0' matches n'th path in 'p1', then curToNext[m]=n
                # We set the maxDiff to twice the maximum delay difference between neighboring points in
                # nanoseconds. The matchPathInfo function does not match paths is the difference of the 6D
                # path info (Delay, Power, and 4 angles) is more than this.
                maxDiff = 2*np.linalg.norm(self.delta)*1e9/299792458
                curToNext = p0.matchPathInfo(p1, maxDiff)                           # Match paths between p0 and p1
                commonIdxCur = np.where(curToNext>-1)[0]                            # Indexes of common paths in p0
                commonIdxNext = curToNext[curToNext!=-1]                            # Indexes of common paths in p1
                assert len(commonIdxCur)==len(commonIdxNext)
                c = len(commonIdxCur)                                               # c: Number of common paths

            if c == 0:
                endPointsInfo = np.concatenate(([p0.xyz],[p1.xyz]))                 # Only contains the xyz values
                los = -1
            else:
                pathsLost   = p0.numPaths-c
                pathsGained = p1.numPaths-c
                
                # Figuring out the los flag for the interpolated points:
                if p0.hasLos == 0:  los = 0 # We didn't have a los path. => Continue with no los path in this segment.
                elif pathsLost==0:  los = 1 # We had a los path, we didn't lose any path => We still have the los path
                elif p1.hasLos == 1:        # We had a los path and we have a los path now.
                    # In this case we assume that the los paths are the same. Can't consider a case that they may be
                    # different paths since there is only one los path at each point.
                    los = 1
                else:               los = 0 # We had los path but we don't have los paths anymore and we lost some paths

                # Path information at the segment endpoints. Shape: (2, c, 8)
                endPointsInfo = np.concatenate(([p0.pathInfo[commonIdxCur]], [p1.pathInfo[commonIdxNext]]))
                
                # unwrapping azimuth/phase angles.
                # Path Values: 0:Phase, 1:delay, 2:RxPower, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:bounces
                endPointsInfo[:,:,(0,3,5)] = np.unwrap(endPointsInfo[:,:,(0,3,5)],.5,0, period=360)
                
                # Endpoint coordinates. Shape: (2, 3)
                endPointsXyz = np.concatenate(([p0.xyz],[p1.xyz]))
                
                # All info at endpoints. Shape: (2, c*8+3)
                endPointsInfo = np.concatenate((endPointsInfo.reshape(2,-1), endPointsXyz), axis=1)
            
            # Do the linear interpolation to get the points between the endpoints. Shape: (numSteps+1, c*8+3)
            intPointInfo = endPointsInfo[0] + \
                           (endPointsInfo[1]-endPointsInfo[0])*(stepStarts.reshape(-1,1))/numSegSamples
            intXyzs = intPointInfo[:,-3:]                           # Interpolated coordinates, Shape(numSteps+1, 3)
            if c>0:
                intPathInfo = intPointInfo[:,:-3].reshape(-1,c,8)   # Interpolated path info, Shape(numSteps+1, c, 8)
                # Wrapping azimuth/phase angles
                intPathInfo[:,:,(0,3,5)] += (intPathInfo[:,:,(0,3,5)]<-180)*360 - (intPathInfo[:,:,(0,3,5)]>180)*360
                intPoints += [ TrjPoint(xyz, los, pathInfo, bsDist=np.sqrt(np.square(xyz-self.bsXyz).sum()),
                                        speed=segSpeed, sampleNo=pointSample+segStart)
                                    for xyz, pathInfo, pointSample in zip(intXyzs, intPathInfo, stepStarts) ]
            else:
                intPoints += [ TrjPoint(xyz, los, [], bsDist=np.sqrt(np.square(xyz-self.bsXyz).sum()),
                                        speed=segSpeed, sampleNo=pointSample+segStart)
                                    for xyz, pointSample in zip(intXyzs, stepStarts) ]
            segStart += numSegSamples
        
        return Trajectory(intPoints, self.carrierFreq)

    # ******************************************************************************************************************
    def getRandomTrajectory(self, xyBounds, segLen, bwp,
                            trajLen=None, trajTime=None, trajDist=None, xyStart=None, prob=None,
                            trajDir="All", speedMps=None):
        r"""
        Creates and returns a random trajectory in the area specified by ``xyBounds`` inside the grid of points in the
        given scenario. This function first creates a random "On-Grid" trajectory of points. It then interpolates 
        additional trajectory points between the grid points. See 
        :doc:`../Playground/Notebooks/RayTracing/DeepMimo` for a complete example.
        
        Parameters
        ----------
        xyBounds: 2-D list of integers
            A 2x2 matrix representing the bounds of the area where the random trajectory will be generated. The matrix 
            should be in the format ``[[minX, minY], [maxX, maxY]]``. All points in the returned trajectory will be 
            confined within these bounds. If the area defined by ``xyBounds`` overlaps with parts outside the grid area
            specified by ``xyMin`` and ``xyMax``, the boundaries are internally adjusted to ensure that the trajectory 
            falls within the intersection of the areas defined by ``xyBounds`` and the pair (``xyMin``, ``xyMax``).

        segLen: integer
            The number of grid points that the shortest segment of the trajectory traverses, excluding the starting 
            point. For instance, if ``segLen`` is set to 2, it implies that each segment of the generated trajectory 
            passes through at least three grid points (including the starting point). This parameter can be utilized 
            to control the frequency of turns in a trajectory. A larger ``segLen`` value results in a reduced number of
            turns in the trajectory.

        bwp: The bandwidth part used to decide the timing of the interpolated trajectory points. One interpolated 
            trajectory point is created for each slot of communication.
        
        trajTime: float or None
            If provided, it represents the total travel time (in seconds) along the trajectory. Note that the actual 
            travel time on the generated trajectory may not be precisely equal to this value due to the approximations
            in the calculations.
            
        trajDist: float or None
            If provided, it represents the total travel distance (in meters) along the trajectory. This parameter is
            ignored if ``trajTime`` is specified. Note that the actual travel distance on the generated trajectory may
            not be precisely equal to this value due to the approximations in the calculations.

        trajLen: integer or None
            If provided, it represents the total number of grid points on the trajectory (excluding the starting point).
            This parameter is ignored if one of ``trajTime`` or ``trajDist`` is specified.

            .. Important:: At least one of ``trajTime``, ``trajDist``, or ``trajLen`` must be specified.

        xyStart: list, tuple, NumPy array, or None
            The 2-D coordinates of the the trajectory’s initial position. If this parameter is set to `None` 
            (default), the trajectory’s starting point is automatically determined based on ``trajDir`` and 
            ``xyBounds``. Otherwise, the given value is first checked against the trajectory bounds (``xyBounds``) and
            modified if needed to ensure that the starting point falls within the specified boundaries.
            
        prob: tuple or None
            If provided, it must be a tuple containing three probability values for turning right, going straight, and 
            turning left. These three probability values must collectively sum up to 1. If not specified, all three
            probabilities are assumed to be equal: :math:`P_{right}=P_{straight}=P_{left}=\frac 1 3`
            
        trajDir: str
            This value can be used to restrict the direction of movement along the trajectory. At each step, the moving
            direction is the angle between the velocity vector and the X-axis. There are eight possible directions,
            corresponding to angles: 0, 45, 90, 135, 180, 225, 270, and 315 degrees. This parameter provides a general
            direction to the trajectory. It can take one of the following values:
        
                :All: No restriction in direction of movement in the trajectory. This is the default value.
                
                :+X: This forces the trajectory to move along the X-axis in positive direction. The only movement
                    directions allowed in the trajectory are 45, 0, and 315 degrees.

                :-X: This forces the trajectory to move along the X-axis in negative direction. The only movement
                    directions allowed in the trajectory are 135, 180, and 225 degrees.

                :+Y: This forces the trajectory to move along the Y-axis in positive direction. The only movement
                    directions allowed in the trajectory are 135, 90, and 45 degrees.

                :-Y: This forces the trajectory to move along the Y-axis in negative direction. The only movement
                    directions allowed in the trajectory are 225, 270, and 315 degrees.

        speedMps: float or None
            If provided, it specifies the trajectory speed in meters per second. If not provided, the speed is
            automatically determined based on the scenario type (indoor vs outdoor). The current implementation uses
            an average walking speed of 1.2 m/s for indoor scenarios and 14 m/s for outdoor scenarios (which 
            corresponds to a car moving at 31.32 miles per hour). Note that the actual linear speed on the trajectory
            may not be precisely equal to this value due to the approximations in the calculations.
            
        Returns
        -------
        :py:class:`~neoradium.trjchan.Trajectory`
            A :py:class:`~neoradium.trjchan.Trajectory` object containing all the information about the created
            trajectory
        """
        # If the speed is not specified, set the speed based on the scenario (Indoor vs outdoor).
        if speedMps is None:
            if self.scenario in inDoorScenarios:    speedMps = 1.2          # Walking speed
            else:                                   speedMps = 14           # A car at 31.32 miles per hour

        if trajTime is not None:    trajLen = float(trajTime*speedMps)      # trajLen is distance
        elif trajDist is not None:  trajLen = float(trajDist)               # trajLen is distance
        elif trajLen is None:
            raise ValueError("At least one of 'trajTime', `trajDist`, or `trajLen` must be specified!")
        else:                       trajLen = int(trajLen)                  # trajLen is number of grid points
            
        # First find a "grid" trajectory
        gridTrajectory = self.getRandomGridTraj(xyBounds, segLen, trajLen, xyStart, prob, trajDir)  # trajLen x 2
        
        # Get the indices of the points on the "grid" trajectory
        idxTrajectory = self.gridXyToIndex(gridTrajectory)                  # array of indices (len=trajLen)

        # Now interpolate to create intermediate points on each segment
        return self.interpolateTrajectory(idxTrajectory, speedMps, bwp)

    # ******************************************************************************************************************
    def drawMap(self, mapType="LOS-NLOS", overlay=None, figSize=6, ax=None):
        r"""
        This visualization function creates a map of the scenario, assigning different colors to the points 
        on the grid.

        Parameters
        ----------
        mapType: str
            This specifies the type of map to be drawn by this function:
            
                :LOS-NLOS: The color used for each point depends on whether it has a Line-of-Sight path, or if there
                    is a total blockage at that point.
                
                :1stPathDelays: The color used for each point depends the amount of delay for the strongest path at
                    that point.
                 
                :1stPathPowers: The color used for each point depends the path power of the strongest path at that
                    point.
                        
        overlay: :py:class:`~neoradium.trjchan.Trajectory` or NumPy array or None
            If this is a :py:class:`~neoradium.trjchan.Trajectory` object, then the trajectory will be drawn over the
            map. If this is a NumPy array, it must contain a list of indices of the points in the current scenario. In
            this case all the points in the list will be drawn (scatter plot) over the map.

        figSize: float
            This value determines the approximate size of the drawn map. If the maximum of the map’s width and height 
            is less than the specified value, the map is scaled to match the specified size. The default value is set 
            to ``6``.

        ax: `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_ or None
            If specified, it must be a matplotlib ``Axis`` object on which the Scenario Map is drawn. This can be used 
            if you want to have a group of matplotlib subplots and draw the map in one of them.
              
    
        **Example:**
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("asu_campus1", baseStationId=1, gridId=0)
            deepMimoData.drawMap("1stPathDelays")
            
        .. figure:: ../Images/DeepMimoMap.png
            :align: center
            :figwidth: 600px
        """
        bsGridXy = self.xyToGridXy(self.bsXyz[:2])
        gridMin, gridMax = np.int32([[0,0], self.gridSize])
        gridMin += bsGridXy*(bsGridXy<0)
        gridMax += (bsGridXy-self.gridSize)*((bsGridXy-self.gridSize)>0)
        margin = int(np.ceil(max(gridMax-gridMin)*0.01))
        gridMin, gridMax = gridMin-margin, gridMax+margin
        
        if ax is None:
            # The one additional inch in the width is for the legend/colorbar
            wInches, hInches = (gridMax-gridMin) / plt.rcParams['figure.dpi']
            scale = figSize/max(wInches, hInches)
            fig, ax = plt.subplots(figsize=((wInches+1)*scale, hInches*scale))

        if mapType == "LOS-NLOS":
            heatMap = np.zeros((gridMax-gridMin)[::-1]) # 0 is translated to white
            grid = np.array([p.hasLos+2 for p in self]).reshape(self.gridSize[::-1]) # -1,0,1 -> 1,2,3
            cmap = ListedColormap(['white', 'black', 'red', 'green',])
            categories = {0: 'BLOCKED', 1: 'NLOS', 2: 'LOS'}
            patches = [mpatches.Patch(color=cmap(i+1), label=categories[i]) for i in range(len(categories))]
            patches += [ mpatches.Patch(color='orange', label="BS") ]
            title = "Map of LOS/NLOS paths"
        elif mapType == "1stPathDelays":
            heatMap = np.ones((gridMax-gridMin)[::-1]) # 1 is white for all types of map
            grid = np.array([p.delays[0] if p.hasLos>-1 else 1 for p in self]).reshape(self.gridSize[::-1])
            cmap = 'viridis'
            title = "Delay of first path (ns)"
        elif mapType == "1stPathPowers":
            heatMap = np.ones((gridMax-gridMin)[::-1]) # 1 is white for all types of map
            grid = np.array([p.powers[0] if p.hasLos>-1 else 1 for p in self]).reshape(self.gridSize[::-1])
            cmap = 'viridis'
            title = "Power of first path (dB)"
        
        # heatMap includes the margins around the grid
        heatMap[-gridMin[1]:self.gridSize[1]-gridMin[1], -gridMin[0]:self.gridSize[0]-gridMin[0]] = grid

        # Draw the heatmap
        xyMin = self.xyMin + gridMin*self.delta
        xyMax = self.xyMax + (gridMax-self.gridSize)*self.delta
        im = ax.imshow(heatMap, cmap=cmap, interpolation='nearest', origin='lower',
                       extent=[xyMin[0],xyMax[0],xyMin[1],xyMax[1]])
        
        # Draw the base station
        bsGridXy = self.bsXyz[:2]
        ax.scatter(x=bsGridXy[0], y=bsGridXy[1], c='orange', s=50,
                   label=self.baseStationId if type(self.baseStationId)==str else "BS%d"%(self.baseStationId))

        # Axis labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)

        if overlay is not None:
            if type(overlay) is Trajectory:
                if mapType == "LOS-NLOS":
                    patches += [ mpatches.Patch(color='yellow', label="Trajectory"),
                                 mpatches.Patch(color='gray', label="Traj. Start") ]
                x = [tp.xyz[0] for tp in overlay.points]
                y = [tp.xyz[1] for tp in overlay.points]
                ax.plot(x, y, 'yellow')                             # Draw the trajectory
                ax.scatter(x=x[0], y=y[0], c='gray')                # Draw the starting point
            elif type(overlay)==np.ndarray:
                if mapType == "LOS-NLOS":
                    patches += [ mpatches.Patch(color='blue', label="Specified Points") ]
                x = [self[i].xyz[0] for i in overlay]
                y = [self[i].xyz[1] for i in overlay]
                ax.scatter(x=x, y=y, c='blue', s=3)                 # Draw the points

        # Legend and color bar
        if mapType == "LOS-NLOS":
            ax.legend(handles=patches, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
        else:
            cbar = plt.colorbar(im)

            # We need to draw an overlay masking the blocked areas to white color. This overlay is
            # transparent in non-blocked areas.
            blockedColor = colorConverter.to_rgba('white')
            cmap2 = LinearSegmentedColormap.from_list('blockedOverlay',[blockedColor,blockedColor],2)
            cmap2._init()          # create the _lut array, with rgba values
            cmap2._lut[0,-1] = 0   # Set everywhere transparent except where the value is 1 (Blocked)
            ax.imshow(heatMap==1, cmap=cmap2, interpolation='nearest', origin='lower',
                      extent=[xyMin[0],xyMax[0],xyMin[1],xyMax[1]]) # Draw the overlay


        return ax.get_figure(), ax

    # ******************************************************************************************************************
    def animateTrajectory(self, trajectory, numGraphs=0, graphCallback=None, mapType="LOS-NLOS",
                          pointsPerFrame=10, fileName=None):
        r"""
        This visualization function generates a scenario map and animates the movement of a UE device along the 
        specified trajectory within the map. Furthermore, it can animate up to three graphs below the scenario map. A
        complete example of using this function is available at
        :doc:`../Playground/Notebooks/RayTracing/TrajChannelAnim` in the playground.

        Parameters
        ----------
        trajectory: :py:class:`~neoradium.trjchan.Trajectory`
            The trajectory to animate. It must be an instance of :py:class:`~neoradium.trjchan.Trajectory`.
        
        numGraphs: int
            The number of graphs drawn below the scenario map. The default is 0, which animates only the trajectory
            on the map.
        
        graphCallback: function or None
            If ``numGraphs`` is not zero, a callback function should be provided, which will be automatically called 
            by this method. The function is initially called to configure the graphs, and then it is invoked for 
            every frame of the animation to update or draw the graphs. For more information, please refer to the 
            :ref:`callback function section <CallBack>` below.
            
        mapType: str
            This specifies the type of map that will be used as the background of the animation. For more information 
            about the available values, please refer to the :py:meth:`~neoradium.deepmimo.DeepMimoData.drawMap` method.
            
        pointsPerFrame: int
            This function creates a frame of animation every ``pointsPerFrame`` trajectory points. The default is 
            10 which means create an animation frame every other 10 trajectory points. A value of 1, creates a frame 
            for every trajectory point which takes 10 times more memory than the default value of 10.
            
            This function generates an animation frame every ``pointsPerFrame`` trajectory points. By default, it 
            creates an animation frame every other 10 trajectory points. If you set the value to 1, it creates a 
            frame for every trajectory point, which requires 10 times more memory compared to the default value of 10.
            
            .. Note:: For long trajectories, the animation can consume a significant amount of memory, and the 
                matplotlib library may truncate it. To mitigate this issue, you can increase the value of 
                ``pointsPerFrame`` to reduce the memory required for the animation. Alternatively, you can configure 
                the maximum memory usage for the animation by adding a line similar to the following somewhere near 
                the beginning of your code:
                
                .. code-block:: python
                
                    import matplotlib
                    matplotlib.rcParams['animation.embed_limit'] = 100000000  # Set the limit to 100 MB

        fileName: str or None
            If specified, it must be a path to a GIF file, and the animation will be saved to that file.


        Returns
        -------
        `matplotlib.animation.FuncAnimation <https://matplotlib.org/stable/api/_as_gen/matplotlib.animation.FuncAnimation.html>`_
            A ``FuncAnimation`` object containing the information about the animation. In a Jupyter Notebook, you can 
            utilize the ``to_jshtml()`` method of the ``FuncAnimation`` class to display the animation.
            

        .. _CallBack:
            
        **Animation Callback Function**
        
        This function is automatically invoked to configure and display additional graphs below the scenario map 
        during the animation. It accepts the following parameters:
        
            :request: This string can be either ``"Config"`` or ``"Draw"``, indicating the purpose of the call.
            
            :ax: This is an array of 
                `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
                objects. Each element of this array is used to draw a graph below the animated scenario map.
            
            :trajectory: The :py:class:`~neoradium.trjchan.Trajectory` object used for the animation.
            
            :points: A tuple of two integer values, representing the indices of points in the trajectory for the 
                previous and current frames, respectively. This parameter is utilized only when the ``request`` 
                parameter is set to ``"Draw"``.
        
        The following is an example of a callback function that draws two graphs below the animated trajectory on 
        the scenario map.:
        
        .. code-block:: python
        
            def handleGraph(request, ax, trajectory, points=None):
                if request=="Config":
                    # Configure 1st graph for delay of first path
                    ax[0].set_xlim(0,trajectory.numPoints)
                    ax[0].set_ylim(900,1300)
                    ax[0].set_title("Delay of first path (ns)")

                    # Configure 2nd graph for Power of first path
                    ax[1].set_xlim(0,trajectory.numPoints)
                    ax[1].set_ylim(-130,-80)
                    ax[1].set_title("Power of first path (dB)")

                elif request=="Draw":
                    p0, p1 = points
                    ax[0].plot([p0,p1], [trajectory.points[p0].delays[0], trajectory.points[p1].delays[0]], 'blue', markersize=1)
                    ax[1].plot([p0,p1], [trajectory.points[p0].powers[0], trajectory.points[p1].powers[0]], 'red', markersize=1)
        """
        if numGraphs>3: raise("Too many graphs! (This function supports up to 3 graphs)")
        figSize = (6, 4+4*numGraphs/3)                                  # numGraphs→Height: 0→4, 1→5.33, 2→6.66, 3→8
        if numGraphs>0:
            fig, ax = plt.subplots(1+numGraphs,1, figsize=figSize, gridspec_kw={'height_ratios': [4] + numGraphs*[1]})
        else:
            fig, ax = plt.subplots(figsize=figSize)                     # Just animating the map (no graphs)
        axMap = ax if numGraphs==0 else ax[0]
        self.drawMap(mapType, ax=axMap)
        point, = axMap.plot([], [], 'bo', markersize=5)                 # Starting Point

        graphCallback("Config", ax[1:], trajectory)                     # Configure the graphs

        def animate(p):
            p0, p1 = (p-1)*pointsPerFrame, p*pointsPerFrame
            x, y = trajectory.points[p1].xyz[:2]
            point.set_data([x], [y])
            if p>0:
                axMap.plot([trajectory.points[p0].xyz[0],x],            # Update the map for this frame
                           [trajectory.points[p0].xyz[1],y],'black', linewidth=1)
                if numGraphs>0:
                    graphCallback("Draw", ax[1:], trajectory, (p0, p1)) # Draw the graphs for this frame
            return point,

        plt.tight_layout()
        
        frameDuration = 1000.0*pointsPerFrame*trajectory.time/trajectory.numPoints  # Frame duration in milliseconds
        anim = animation.FuncAnimation(fig, animate, frames=trajectory.numPoints//pointsPerFrame,
                                       interval=int(np.round(frameDuration)), blit=True, repeat=False)

        plt.close()
        if fileName is not None:
            # Save to a gif file
            fps = int(min(np.round(1/(frameDuration/1000)), 30))                # Frames per second
            anim.save(fileName, writer=animation.PillowWriter(fps=fps))

        return anim

    # ******************************************************************************************************************
    def interactiveTrajPoints(self, mapType="LOS-NLOS", backEnd="MacOSX", figSize=6):
        r"""
        This function enables you to create a trajectory by selecting points on the map. It opens a separate window 
        displaying the scenario map. You can then click on the map points to create the trajectory. After each click, 
        the map updates to show current trajectory. To end the trajectory, simply close the window. This function 
        returns the selected points after closing the map window. The function :py:meth:`trajectoryFromPoints` can then
        be used to create a :py:class:`~neoradium.trjchan.Trajectory` object based on the captured trajectory points.
        The notebook file :doc:`../Playground/Notebooks/RayTracing/DeepMimo` contains an example of using this
        function.
        
        Parameters
        ----------
        mapType: str
            This specifies the type of map to be drawn by this function:
            
                :LOS-NLOS: The color used for each point depends on whether it has a Line-of-Sight path, or if there
                    is a total blockage at that point.
                
                :1stPathDelays: The color used for each point depends the amount of delay for the strongest path at
                    that point.
                 
                :1stPathPowers: The color used for each point depends the path power for the strongest path at that
                    point.
                        
        backEnd: str
            The name of the interactive backend to be used by the matplotlib library. For more information, please 
            refer to `matplotlib backends <https://matplotlib.org/stable/users/explain/figure/backends.html>`_. The 
            default backend is “MacOSX”.

        figSize: float
            This value determines the approximate size of the drawn map. If the maximum of the map’s width and height 
            is less than the specified value, the map is scaled to match the specified size. The default value is set 
            to ``6``.

        Returns
        -------
        NumPy array
            An array of 2-D points on the current scenario map specifying the trajectory.


        **Example:**
        
        The code below can be utilized to generate a trajectory of points for the “asu_campus1” scenario. The image 
        below illustrates the current trajectory, depicted by blue lines, and the starting point, marked by a small 
        blue circle.
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("asu_campus1", baseStationId=1, gridId=0)
            points = deepMimoData.interactiveTrajPoints(mapType="LOS-NLOS")
        
        .. figure:: ../Images/DeepMimoInteractiveMap.png
            :align: center
            :figwidth: 600px
        """
        import subprocess
        print("Running the interactive map for '%s'..."%(self.scenario))

        if mapType == "LOS-NLOS":           titleStr = "Map of LOS/NLOS paths"
        elif mapType == "1stPathDelays":    titleStr = "Delay of first path (ns)"
        elif mapType == "1stPathPowers":    titleStr = "Power of first path (dB)"
        else:                               raise ValueError("'%s' is an invalid 'mapType'!"%(mapType))
        titleStr += "\\nClick on the map to add trajectory points"
        
        bsId = f"\"{self.baseStationId}\""  if type(self.baseStationId)==str else self.baseStationId
        grId = f"\"{self.gridId}\""         if type(self.gridId)==str        else self.gridId

        pyFileStr = f"""
# This file was auto-generated by the NeoRadium's DeepMimoData class. Please do not edit manually.
from neoradium import DeepMimoData
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("{backEnd}")

DeepMimoData.setScenariosPath("{self.pathToScenarios}")
deepMimoData = DeepMimoData("{self.scenario}", {bsId}, {grId})
fig, ax = deepMimoData.drawMap("{mapType}", figSize={figSize})

points = []
def onClick(event):
    x, y = event.xdata, event.ydata
    if x is not None and y is not None:
        if event.button == 1:
            if len(points)==0:  ax.plot(x, y, 'bo')
            else:               ax.plot([points[-1][0],x],[points[-1][1],y],'blue')
            points.append((x, y))
        elif event.button == 3:
            if len(points)==0:  return
            ax.plot([points[-2][0],points[-1][0]],[points[-2][1],points[-1][1]],'grey')
            points.pop()
        plt.draw()

plt.title("{titleStr}")
cid = fig.canvas.mpl_connect('button_press_event', onClick)
plt.show()
print("Clicked points:", points)
"""
        fileName = "InteractiveTrjPoints.py"
        pyFile = open(fileName, "w")
        pyFile.write(pyFileStr)
        pyFile.close()

        result = subprocess.run(["python", fileName], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "Clicked points: [" not in result.stdout:
            raise ValueError("Something went wrong!\nOutput:\n%s\nError:\n%s\n"%(result.stdout,result.stderr))
        start = result.stdout.find("Clicked points: [") + len("Clicked points: ")
        end = start + result.stdout[start:].find("]") + 1
        x = np.float64(eval(result.stdout[start:end]))
        print(f"Done. {len(x)} points selected.")
        return x

    # ******************************************************************************************************************
    def trajectoryFromPoints(self, points, bwp, speedMps=None):
        r"""
        Creates and returns a :py:class:`~neoradium.trjchan.Trajectory` object based on the given trajectory points
        and parameters. Please refer to the notebook :doc:`../Playground/Notebooks/RayTracing/DeepMimo` for an example
        of using this function.
        
        Parameters
        ----------
        points: NumPy array 
            This array of 2-D points on the current scenario map specifies the trajectory. The 
            :py:meth:`interactiveTrajPoints` function can be used to obtain these points interactively.

        bwp: The bandwidth part used to determine the timing of the interpolated trajectory points. This implementation
            generates one interpolated trajectory point for each communication slot.
        
        speedMps: float or None
            If provided, it specifies the trajectory speed in meters per second. If not provided, the speed is
            automatically determined based on the scenario type (indoor vs outdoor). The current implementation uses
            an average walking speed of 1.2 m/s for indoor scenarios and 14 m/s for outdoor scenarios (which 
            corresponds to a car moving at 31.32 miles per hour). Note that the actual linear speed on the trajectory
            may not be precisely equal to this value due to the approximations in the calculations.
            
        Returns
        -------
        :py:class:`~neoradium.trjchan.Trajectory`
            A :py:class:`~neoradium.trjchan.Trajectory` object containing all the information about the created 
            trajectory
        """
        traj = []
        for i in range(len(points)-1):
            p1,p2 = self.xyToGridXy(points[i:i+2])
            def lineFunc(x=None, y=None):
                if x is not None:   a,i,o = x,0,1      # Input is x, output is y
                elif y is not None: a,i,o = y,1,0      # Input is y, output is x
                else: assert False, "Both x and y cannot be None!"
                return (p2[o]-p1[o])*(a-p1[i])/(p2[i]-p1[i]) + p1[o]
            dx, dy = p2-p1
            xInc,yInc = np.sign([dx,dy])
            if np.abs(dx)>np.abs(dy):
                for x in range(p1[0],p2[0],xInc):  traj += [ [x, np.round(lineFunc(x=x))] ]
            else:
                for y in range(p1[1],p2[1],yInc):  traj += [ [np.round(lineFunc(y=y)), y] ]
        
        trjIndexes = self.gridXyToIndex(np.int32(traj + [p2]))
        
        # If the speed is not specified, set the speed based on the scenario (Indoor vs outdoor).
        inDoorScenarios = [ "I1_2p4", "I1_2p4", "I1_2p4", "I1_2p4" ]
        if speedMps is None:
            if self.scenario in inDoorScenarios:    speedMps = 1.2      # Walking speed
            else:                                   speedMps = 14       # A car at 31.32 miles per hour

        # Now interpolate to create intermediate points on each segment
        return self.interpolateTrajectory(trjIndexes, speedMps, bwp)    # A Trajectory object

    # ******************************************************************************************************************
    def getChanGen(self, numChannels, bwp,
                   los=None, minDist=0, maxDist=np.inf,
                   minX=-np.inf, minY=-np.inf, maxX=np.inf, maxY=np.inf, **kwargs):
        r"""
        Samples random points from the current scenario based on the specified criteria and returns a generator object
        that can generate channel matrices corresponding to those random points.

        The indices of the random points can be retrieved using the ``pointIdx`` property of the returned generator 
        object. These point indices can then be passed to the :py:meth:`drawMap` method as an "overlay" to be 
        displayed on the map
        
        Refer to the notebook :doc:`../Playground/Notebooks/RayTracing/ChannelGeneration` for an example of 
        using this method.
        
        Parameters
        ----------
        numChannels: int 
            This is the number of channel matrices generated by the returned generator which is equal to the number of 
            points sampled from all the points on the grid of the current scenario. However, it disregards points 
            with total blockage (i.e., points with no paths to the base station). If the given filter criteria result
            in insufficient points being available in the current scenario, the number of points sampled (and 
            consequently, the number of channels generated) may be less than ``numChannels``.
            
        bwp : :py:class:`~neoradium.carrier.BandwidthPart` 
            The bandwidth part object used by the returned generator to construct channel matrices.

        los: Boolean or None        
            It can be set to `None`, `True`, or `False`.

                * If set to `None`, the sampled points are not filtered based on their Line-of-Sight communication 
                  path (default).
            
                * If set to `True`, only the points with a Line-of-Sight communication path to the base station are
                  considered.
            
                * If set to `False`, only the points without a Line-of-Sight communication path to the base station 
                  are considered. 
        
        minDist: float
            If specified, this parameter determines the minimum distance between the points and the base station. 
            Points closer than this specified value will not be considered. The default is ``0`` which effectively 
            disables this filter.
            
        maxDist: float
            If specified, this parameter determines the maximum distance between the points and the base station. 
            Points farther than this specified value will not be considered. By default, this parameter is set to 
            ``np.inf``, which effectively disables this filter.

        minX: float
            If specified, parameter determines a lower bound for the ``x`` coordinate of the points to consider. It 
            can be used with other filters to limit the points to a specific region. By default, this parameter is
            set to ``-np.inf``, which effectively disables this filter.
            
        minY: float
            If specified, parameter determines a lower bound for the ``y`` coordinate of the points to consider. It 
            can be used with other filters to limit the points to a specific region. By default, this parameter is
            set to ``-np.inf``, which effectively disables this filter.

        maxX: float
            If specified, parameter determines an upper bound for the ``x`` coordinate of the points to consider. It 
            can be used with other filters to limit the points to a specific region. By default, this parameter is
            set to ``np.inf``, which effectively disables this filter.

        maxY: float
            If specified, parameter determines an upper bound for the ``y`` coordinate of the points to consider. It 
            can be used with other filters to limit the points to a specific region. By default, this parameter is
            set to ``np.inf``, which effectively disables this filter.
                    
        kwargs : dict
            Here is a list of additional optional parameters that can be used to further customize the calculation 
            of the channel matrices:
            
                :normalizeGains: If the default value of `True` is used, the path gains are normalized.
                    
                :normalizeOutput: If the default value of `True` is used, the gains are normalized based on the 
                    number of receive antenna.

                :normalizeDelays: If the default value of `True` is used, the delays are normalized as specified in 
                    “Step 3” of **3GPP TR 38.901 section 8.4**. Otherwise, the original delays obtained from 
                    ray-tracing are used.

                :filterLen: The length of the channel filter. The default is 16 sample.
                
                :delayQuantSize: The size of the delay fraction quantization for the channel filter. The default is 64.
                
                :stopBandAtten: The stop-band attenuation value (in dB) used by the channel filter. The default is 80 dB.
                
                :txAntenna: The transmitter antenna, which is an instance of either the 
                    :py:class:`neoradium.antenna.AntennaPanel` or :py:class:`neoradium.antenna.AntennaArray` class. 
                    By default, it is a single antenna in a 1x1 antenna panel with vertical polarization.
                
                :rxAntenna: The receiver antenna which is an instance of either the 
                    :py:class:`neoradium.antenna.AntennaPanel` or :py:class:`neoradium.antenna.AntennaArray` class. 
                    By default, it is a single antenna in a 1x1 antenna panel with vertical polarization.
                    
                :txOrientation: The orientation of transmitter antenna. This is a list of 3 angle values in degrees for
                    the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901 Section 7.1.3** for more
                    information.

                :rxOrientation: The orientation of receiver antenna. This is a list of 3 angle values in degrees for 
                    the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901 Section 7.1.3** for more
                    information.

                :xPolPower: The cross-polarization power in dB. The default is 10 dB. It is defined as 
                    :math:`X=10 log_{10} \kappa^{RT}` where :math:`\kappa^{RT}` is the cross-polarization ratio (XPR).
                    In the current implementation, this value is used for all paths.
            
                :ueSpeed: Specifies the speed of the UE. It can be one of the following:
                
                        * If it is a tuple of the form ``(speedMin, speedMax)``, at each point a random speed is sampled
                          uniformly between ``speedMin`` and ``speedMax``.
                          
                        * If it is a list of the form [:math:`s_1`, :math:`s_2`, ..., :math:`s_n`], at each point a 
                          random speed is picked from those specified in the list.
                          
                        * If it is a single number, then the same UE speed is used at all points.
                    
                    The default is ``(0,20)``.
                      
                :ueDir: Specifies the direction of UE movement in the X-Y plane as an angle in degrees. It can be one 
                    of the following:
                    
                        * If it is a tuple of the form ``(dirMin, dirMax)``, at each point a random angle is sampled
                          uniformly between ``dirMin`` and ``dirMax``.
                          
                        * If it is a list of the form [:math:`a_1`, :math:`a_2`, ..., :math:`a_n`], at each point a 
                          random angle is picked from those specified in the list.
                          
                        * If it is a single number, then the same UE direction is used at all points. 
                    
                    The default is ``(0, 360)``.
                                         
        Returns
        -------
        ``ChanGen``, a generator object that is used to generate channel matrices.
        

        **Example:**
                       
        .. code-block:: python
        
            deepMimoData = DeepMimoData("asu_campus_3p5", baseStationId=1, gridId=0)
            carrier = Carrier(startRb=0, numRbs=25, spacing=15) # Carrier with 25 PRBs, 15 kHz subcarrier spacing
            
            # Create 100 channel matrices
            chanGen = deepMimoData.getChanGen(100, 
                                              carrier.curBwp,   # Bandwidth Part  
                                              los=False,        # Include only Non-Line-of-Sight channels
                                              minDist=200,      # With distances to the base station between 200
                                              maxDist=250,      # and 250 meters
                                              maxX=100,         # With maximum x coordinate of 100 meters
                                              seed=123)         # Reproducible results
            allChannels = np.stack([chan for chan in chanGen])  # Create the channel matrices
            print(allChannels.shape)                            # Prints (100, 14, 300, 1, 1)
        """
        seed = kwargs.pop("seed", None)
        totalPoints = len(self.allTrjPoints)
        
        # A local function called at each reset (including first instantiation)
        def getPointsAndChannel(seed):
            rangen = random if seed is None else random.getGenerator(seed)          # The random number generator
            allPointIdxs = rangen.choice(totalPoints, totalPoints, replace=False)   # Random shuffling of all indices
            pointIdx = []                                                           # The indices of the random points
            i = -1
            while len(pointIdx)<numChannels:
                i += 1
                if i>=totalPoints:          break       # All points in the whole grid have been processed.
                point = self[allPointIdxs[i]]
                if point.hasLos == -1:      continue    # Ignore the points with total blockage
                if point.xyz[0]<minX:       continue    # Ignore points where x < minX
                if point.xyz[0]>maxX:       continue    # Ignore points where x > maxX
                if point.xyz[1]<minY:       continue    # Ignore points where y < minY
                if point.xyz[1]>maxY:       continue    # Ignore points where y > maxY
                if point.bsDist<minDist:    continue    # Ignore where distance to the base station is less than minDist
                if point.bsDist>maxDist:    continue    # Ignore where distance to the base station is more than maxDist
                if los is not None:
                    if point.hasLos != (1 if los else 0):
                        continue                        # Ignore the points with los flag mismatch with the given "los"
                pointIdx += [ allPointIdxs[i] ]         # Passed all the filters

            points = [self[i] for i in pointIdx]        # A list of "TrjPoint" objects
            points[-1].sampleNo = 1                     # This indicates that we want a "pointSet", not a trajectory
            
            # Set random speeds at each point:
            numPoints = len(points)
            ueSpeed = kwargs.pop("ueSpeed", (0, 20))
            if type(ueSpeed)==tuple:  speeds = rangen.uniform(*ueSpeed, size=numPoints)
            elif type(ueSpeed)==list: speeds = rangen.choice(np.float32(ueSpeed), size=numPoints)
            else:                     speeds = np.float32(numPoints*[ueSpeed])
            ueDir = kwargs.pop("ueDir", (0, 360))
            if type(ueDir)==tuple:    dirs = rangen.uniform(*ueDir, size=numPoints)*np.pi/180
            elif type(ueDir)==list:   dirs = rangen.choice(np.float32(ueDir), size=numPoints)*np.pi/180
            else:                     dirs = np.float32(numPoints*[ueDir])*np.pi/180
            for i, point in enumerate(points):
                point.speed = np.float64([speeds[i]*np.cos(dirs[i]), speeds[i]*np.sin(dirs[i]), 0])
            
            # Create a "Trajectory" object representing the point set.
            pointSet = Trajectory(points, self.carrierFreq)
            channel = TrjChannel(bwp, pointSet, **kwargs)   # Create the trajectory-based channel model.
            return np.int32(pointIdx), channel

        class ChanGen:
            def __init__(self): self.reset()
            def __iter__(self): return self

            def __next__(self):
                if self.cur >= numChannels: raise StopIteration
                chanMat = self.channel.getChannelMatrix()
                self.channel.goNext()
                self.cur += 1
                return chanMat
                    
            def reset(self):
                self.cur = 0
                # Get random point indices and corresponding TrjChannel object
                self.pointIdx, self.channel = getPointsAndChannel(seed)

        return ChanGen()
