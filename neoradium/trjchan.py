# Copyright (c) 2024 InterDigital AI Lab
"""
This module implements the :py:class:`~neoradium.trjchan.TrjChannel` class, which encapsulates the information and 
functionality for creating trajectory-based, spatially and temporally consistent sequences of channels. They can be 
used in simulations to study the evolution of the channel over time and its effect on communication. These channel 
models can also be used to generate datasets for sequential deep learning models, such as LSTMs and Transformers.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 10/02/2024    Shahab Hamidi-Rad       First version of the file.
# 04/01/2025    Shahab                  Restructured the file to work with the new ChannelModel class
# 05/07/2025    Shahab                  Reviewed and updated the documentation
# 06/20/2025    Shahab                  * The Trajectory class now can be used in the "PointSet" mode where there is
#                                         no temporal/spatial correlation between points. It is used to create random
#                                         channel matrices. (See isPointSet in this file and the "getChanGen" method in
#                                         the "deepmimo.py" file.
#                                       * Updated the "restart" and "goNext" with the new parameter "applyToBwp".
#                                       * Added the new method "getChanSeqGen" that can be used to return a generator
#                                         object that can generate sequences of channel matrices based on the given
#                                         parameters.
# 12/18/2025    Shahab                  * Some bug fixes.
#                                       * Added the method "printDiscontinuities" to the "Trajectory" class
#                                       * Added multipath interpolation inside each slot. See "interpolateSlots"
#                                         parameter and "interpolateSlot" method of the "TrjChannel" class.
#                                       * Added the "totalBlockage" readonly property.
#                                       * Re-organised the code related to the "getPathGains" function. "getGains"
#                                         method replaced the old "getLOSgains" and "getNLOSgains" methods. The
#                                         "getDopplerFactor" method has also been updated accordingly.
#                                       * Updated the channel sequence generation in the "getChanSeqGen" function and
#                                         added 2 different callbacks for channels and sequences.
# **********************************************************************************************************************
import numpy as np
import scipy.io
from scipy.signal import lfilter
import matplotlib.pyplot as plt

from .antenna import AntennaElement
from .channelmodel import ChannelModel
from .utils import getMultiLineStr, freqStr, toRadian, toDegrees, toLinear, toDb
from .random import random
from .carrier import SAMPLE_RATE

# This file is based on 3GPP TR 38.901 V17.1.0 (2023-12)
# **********************************************************************************************************************
class TrjPoint:
    r"""
    This class encapsulates the spatial and temporal information at each point along a trajectory. A 
    :py:class:`~neoradium.trjchan.Trajectory` object comprises an ordered set of ``TrjPoint`` objects. This class is
    also used by the :py:class:`~neoradium.deepmimo.DeepMimoData` class to store the multipath information obtained 
    through ray-tracing at each point on the grid in a specified `DeepMIMO <https://www.deepmimo.net>`_ scenario.
    
    .. Note:: You typically don’t create instances of this class directly. However, you can obtain ``TrjPoint`` 
        instances from a :py:class:`~neoradium.trjchan.Trajectory` or a :py:class:`~neoradium.deepmimo.DeepMimoData` 
        object. This documentation may be helpful if you want to implement additional sources of trajectories beyond
        the `DeepMIMO <https://www.deepmimo.net>`_ framework currently implemented in the :py:class:`~neoradium.deepmimo.DeepMimoData` class. 
    """
    # This is a list of parameters that are also available from the Trajectory and TrjChannel classes
    pathParamNames = ["phases",     "delays",     "powers",     "aoas",     "zoas",     "aods",     "zods",  "bounces",
                      "losPhase",   "losDelay",   "losPower",   "losAoa",   "losZoa",   "losAod",   "losZod",
                      "nlosPhases", "nlosDelays", "nlosPowers", "nlosAoas", "nlosZoas", "nlosAods", "nlosZods",
                      "hasLos", "numPaths", "numNlosPaths"]
                      
    def __init__(self, xyz=[0,0,0], hasLos=-1, pathInfo=np.empty((0,8)), bsDist=0,
                 pathLoss=0, speed=np.zeros(3), sampleNo=0):
        r"""
        Parameters
        ----------
        xyz: list
            A list of three floating-point numbers giving the 3-D coordinates of this 
            :py:class:`~neoradium.trjchan.TrjPoint` in meters. The default is ``[0, 0, 0]``.

        hasLos: int
            An integer that can take on any of the following values:
            
                :1: Means there is a line-of-sight (LOS) path between the UE at this 
                    :py:class:`~neoradium.trjchan.TrjPoint` and the base station.
                
                :0: Means there is no line-of-sight (LOS) path between the UE at this
                    :py:class:`~neoradium.trjchan.TrjPoint` and the base station.
                
                :-1: Means there is no path between the UE at this :py:class:`~neoradium.trjchan.TrjPoint` and the 
                    base station. (Total Blockage)
                
        pathInfo: NumPy array
            An ``n × 8`` matrix, where ``n`` is the number of paths. It contains the multipath information at this
            :py:class:`~neoradium.trjchan.TrjPoint`. The default is an empty NumPy array. The eight parameters for 
            each path are stored in this matrix as follows:
            
                :0: Path *phase* angle in degrees.
                
                :1: Path *delay* in nanoseconds.
                
                :2: Path *gain* in dB.
                
                :3: Azimuth angle Of Arrival (AOA) in degrees.

                :4: Zenith angle Of Arrival (ZOA) in degrees.
                
                :5: Azimuth angle Of Departure (AOD) in degrees.

                :6: Zenith angle Of Departure (ZOD) in degrees.
                
                :7: The path interactions (See the ``bounces`` parameter below). For each path, it is an integer that
                    can take values of 0, -1, or a positive number. A value of 0 indicates a line-of-sight path. A 
                    value of -1 typically indicates the path interaction information is not available, for example 
                    when older `DeepMIMO <https://www.deepmimo.net>`_ scenario files are used. If the value is 
                    positive, the number of digits specifies the number of interactions the path had on its journey 
                    from the transmitter to the receiver. Each digit represents a specific type of interaction:
                    
                    :1: Reflection
                    :2: Diffraction 
                    :3: Scattering 
                    :4: Transmission
                
        bsDist: float
            The distance between this :py:class:`~neoradium.trjchan.TrjPoint` and the base station in meters. The 
            default is zero.
            
        speed: NumPy array
            The 3D vector of linear speed at this :py:class:`~neoradium.trjchan.TrjPoint` in meters per second (m/s).
            
        sampleNo: int
            The time when the UE arrives at this :py:class:`~neoradium.trjchan.TrjPoint` as a number of samples from
            the start of :py:class:`~neoradium.trjchan.Trajectory`. This is based on **3GPP** sample rate of 30,720,000 
            samples per second.


        **Other Properties:**
                    
            :numPaths: The number of paths between the UE at this :py:class:`~neoradium.trjchan.TrjPoint` and the 
                base station.
                
            :numNlosPaths: The number of Non-Line-Of-Sight (NLOS) paths between the UE at this 
                :py:class:`~neoradium.trjchan.TrjPoint` and the base station.
                
            :time: The time (in seconds) when the user arrives at this :py:class:`~neoradium.trjchan.TrjPoint`. This 
                read-only value is zero for the first :py:class:`~neoradium.trjchan.TrjPoint` of the 
                :py:class:`~neoradium.trjchan.Trajectory` and increases based on the speed and the distance between 
                the points along the trajectory.
                
            :linearSpeed: UE's linear speed at this :py:class:`~neoradium.trjchan.TrjPoint` (Read-only).
            
            :phases: The *phase* angles in degrees for all paths or `None` if there is are paths.
            :delays: The *delay* in nanoseconds for all paths or `None` if there are no paths.
            :powers: The *gain* in dB for all NLOS paths or `None` if there are no paths.
            :aoas: The Azimuth angle Of Arrival (AOA) in degrees for all paths or `None` if there are no paths.
            :zoas: The Zenith angle Of Arrival (ZOA) in degrees for all paths or `None` if there are no paths.
            :aods: The Azimuth angle Of Departure (AOD) in degrees for all paths or `None` if there are no paths.
            :zods: The Zenith angle Of Departure (ZOD) in degrees for all paths or `None` if there are no paths.
            :bounces: The path interaction information for all paths or `None` if there are no paths. See the 
                explanation of the path interactions for the ``pathInfo`` parameter above.
            :losPhase: The LOS path's *phase* angle in degrees or `None` if there is no LOS path.
            :losDelay: The LOS path's *delay* in nanoseconds or `None` if there is no LOS path.
            :losPower: The LOS path's *gain* in dB or `None` if there is no LOS path.
            :losAoa: The LOS path's Azimuth angle Of Arrival (AOA) in degrees or `None` if there is no LOS path.
            :losZoa: The LOS path's Zenith angle Of Arrival (ZOA) in degrees or `None` if there is no LOS path.
            :losAod: The LOS path's Azimuth angle Of Departure (AOD) in degrees or `None` if there is no LOS path.
            :losZod: The LOS path's Zenith angle Of Departure (ZOD) in degrees or `None` if there is no LOS path.
            :nlosPhases: The *phase* angles in degrees for all NLOS paths or `None` if there are no NLOS paths.
            :nlosDelays: The *delay* in nanoseconds for all NLOS paths or `None` if there are no NLOS paths.
            :nlosPowers: The *gain* in dB for all NLOS paths or `None` if there are no NLOS paths.
            :nlosAoas: The Azimuth angle Of Arrival (AOA) in degrees for all NLOS paths or `None` if there are no NLOS paths.
            :nlosZoas: The Zenith angle Of Arrival (ZOA) in degrees for all NLOS paths or `None` if there are no NLOS paths.
            :nlosAods: The Azimuth angle Of Departure (AOD) in degrees for all NLOS paths or `None` if there are no NLOS paths.
            :nlosZods: The Zenith angle Of Departure (ZOD) in degrees for all NLOS paths or `None` if there are no NLOS paths.
        """
        self.xyz = np.float64(xyz)
        self.hasLos = int(hasLos)      # 1: LOS path present; 0: Only NLOS paths; -1: No path present
        assert self.hasLos in [-1,0,1]
        self.numPaths = len(pathInfo)
        assert (self.numPaths==0)==(self.hasLos==-1), "numPaths=%d, self.hasLos==%d"%(self.numPaths,self.hasLos)
        self.numNlosPaths = 0 if self.hasLos==-1 else (self.numPaths - self.hasLos)

        # Note: We always assume if there is a LOS path, it is the one with the lowest delay. So, we do not keep the
        # LOS flag per path. Path Values: 0:phase, 1:delay, 2:power, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:bounce
        self.pathInfo = np.float64(pathInfo)    # n x 7  or n x 8
        if self.pathInfo.shape[1]==7:
            if self.numPaths>0: self.pathInfo = np.append(self.pathInfo, self.numPaths*[[-1]], axis=1)
            else:               self.pathInfo = np.empty((0,8))
        
        self.bsDist = np.float64(bsDist)
        self.pathLoss = np.float64(pathLoss)
        self.speed = speed                      # linear speed at this point (A 3d vector)
        self.sampleNo = sampleNo                # The sample number of this point from the start point of trajectory
        
        # Initialize path information and make sure pathInfo is sorted by the delay:
        self.phases, self.delays, self.powers, self.aoas, self.zoas, self.aods, self.zods, self.bounces = 8*[None]
        self.losPhase, self.losDelay, self.losPower, self.losAoa, self.losZoa, self.losAod, self.losZod = 7*[None]
        self.nlosPhases, self.nlosDelays, self.nlosPowers, self.nlosAoas, self.nlosZoas, self.nlosAods, \
            self.nlosZods = 7*[None]
        if self.numPaths>0:
            self.pathInfo = self.pathInfo[np.argsort(self.pathInfo[:,1])] # Path with the smallest delay comes first
            
            self.phases, self.delays, self.powers, self.aoas, self.zoas, self.aods, self.zods, self.bounces = \
                self.pathInfo.T
            self.bounces = np.int32(self.bounces)
            if self.hasLos==1:
                self.losPhase, self.losDelay, self.losPower, self.losAoa, self.losZoa, self.losAod, self.losZod = \
                    self.pathInfo.T[:7,0]
                if self.numPaths>1:
                    self.nlosPhases, self.nlosDelays, self.nlosPowers, self.nlosAoas, self.nlosZoas, self.nlosAods, \
                    self.nlosZods = self.pathInfo.T[:7,1:]
            else:
                self.nlosPhases, self.nlosDelays, self.nlosPowers, self.nlosAoas, self.nlosZoas, self.nlosAods, \
                    self.nlosZods = self.pathInfo.T[:7,:]

    # ******************************************************************************************************************
    @property
    def time(self):         return self.sampleNo/SAMPLE_RATE            # Documented above in the __init__ function.

    # ******************************************************************************************************************
    @property
    def linearSpeed(self):  return np.sqrt(np.square(self.speed).sum()) # Documented above in the __init__ function.

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="TrjPoint Properties:", getStr=False):
        r"""
        Prints the properties of this class.
        
        Parameters
        ----------
        indent: int
            Used internally to adjust the indentation of the printed info.
            
        title: str
            The title used for the information. By default the text
            "TrjPoint Properties:" is used.

        getStr: boolean
            If `True`, returns a text string instead of printing it.
                    
        Returns
        -------
        str or None
            If "getStr" is true, this function returns a text string containing the information about the properties 
            of this class. Otherwise, nothing is returned (default).
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"

        repStr += indent*' ' + f"  location:       {'  '.join('%6.2f'%(x) for x in self.xyz)} m\n"
        repStr += indent*' ' + f"  Distance to BS: {self.bsDist:6.2f} m\n"
        if self.pathLoss>0:
            repStr += indent*' ' + f"  pathLoss:       {self.pathLoss}\n"
        repStr += indent*' ' + f"  LOS/NLOS:       {['No Paths', 'All NLOS', 'Has LOS path'][self.hasLos+1]}\n"
        repStr += indent*' ' + f"  numPaths:       {self.numPaths}\n"
        repStr += indent*' ' + f"  sampleNo:       {self.sampleNo}\n"
        repStr += indent*' ' + f"  time:           {self.time:.6f} sec\n"
        repStr += indent*' ' + f"  speed vector:   ({self.speed[0]:.3f},{self.speed[1]:.3f},{self.speed[2]:.3f}) m/s.\n"

        if self.hasLos==1:
            repStr += indent*' ' + "  Line-Of-Sight Path:\n"
            repStr += indent*' ' + f"    Delay (ns):   {self.losDelay:.5f}\n"
            repStr += indent*' ' + f"    Power (dB):   {self.losPower:.5f}\n"
            repStr += indent*' ' + f"    Phases (Deg): {int(self.losPhase)}\n"
            repStr += indent*' ' + f"    AOA (Deg):    {int(self.losAoa)}\n"
            repStr += indent*' ' + f"    ZOA (Deg):    {int(self.losZoa)}\n"
            repStr += indent*' ' + f"    AOD (Deg):    {int(self.losAod)}\n"
            repStr += indent*' ' + f"    ZOD (Deg):    {int(self.losZod)}\n"
    
        if self.numNlosPaths>0:
            repStr += indent*' ' + f"  Non-Line-Of-Sight Paths ({self.numNlosPaths}):\n"
            repStr += getMultiLineStr("  Delays (ns) ", self.nlosDelays, indent, "%-5f", 5, numPerLine=12)
            repStr += getMultiLineStr("  Powers (dB) ", self.nlosPowers, indent, "%-5f", 5, numPerLine=12)
            repStr += getMultiLineStr("  Phases (Deg)", np.round(self.nlosPhases), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  AOAs (Deg)  ", np.round(self.nlosAoas), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  ZOAs (Deg)  ", np.round(self.nlosZoas), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  AODs (Deg)  ", np.round(self.nlosAods), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  ZODs (Deg)  ", np.round(self.nlosZods), indent, "%-5d", 5, numPerLine=12)
            if self.bounces[0]!=-1:
                repStr += getMultiLineStr("  Bounces     ", self.bounces, indent, "%-5d", 5, numPerLine=12)

        if getStr: return repStr
        print(repStr)
        
    # ******************************************************************************************************************
    def matchPathInfo(self, nextPoint, maxDiff=1):         # Not documented
        # This function matches the path information between this point and 'nextPoint'.
        # This function pairs the best matches in the path information and returns
        # a list of indices specifying the correct order of the path information
        # applied to 'nextPoint'. If c'th path at this point (self) matches n'th path in 'nextPoint',
        # then curToNext[c]=n where 'curToNext' is the returned value of this function.
        
        # Path Values: 0:phase, 1:delay, 2:power, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:bounces
        assert self.numPaths>0
        matchParams = [1,2,3,4,5,6,7]   # Use these to match paths (delay, power, aoa, zoa, aod, zod, bounces)
        p5d0 = self.pathInfo[:,None,matchParams]
        p5d1 = nextPoint.pathInfo[None,:,matchParams]
        absDiff = np.abs(p5d0 - p5d1)                                           # numPaths0 x numPaths1 x 7
        absDiff[:,:,6] *= 100                                                   # Boost the difference in bounces

        # Handling azimuth angles correctly (Params 2 and 4 in the 7d vectors)
        correction = np.zeros_like(absDiff)
        correction[:,:,(2,4)] = 360*(absDiff[:,:,(2,4)]>180)                    # Corrections for aod and aoa
        absDiff = np.abs(absDiff-correction)                                    # numPaths0 x numPaths1 x 7
        diff = absDiff.sum(2)                                                   # numPaths0 x numPaths1
        order = np.argsort(diff.flatten()).reshape(-1,1)                        # numPaths0*numPaths1 x 1
        nCur, nNext = self.numPaths, nextPoint.numPaths
        # Each tuple in the list below is (index in Cur, index in next, distance) sorted by distance
        pairs = np.concatenate((order//nNext, order%nNext, diff.flatten()[order]), axis=1)
        curIdx = nCur*[-1]
        nextIdx = nNext*[-1]
        remainingPairs = min(nCur, nNext)

        p = 0
        while remainingPairs and p<len(pairs):
            c, n, diff = int(pairs[p,0]), int(pairs[p,1]), pairs[p,2]
            if (curIdx[ c ] == -1) and (nextIdx[ n ] == -1) and (diff<maxDiff):
                curIdx[ c ] = n
                nextIdx[ n ] = c
                remainingPairs -= 1
            p += 1

        return np.int32(curIdx)

# **********************************************************************************************************************
class Trajectory:
    r"""
    This class maintains a collection of :py:class:`~neoradium.trjchan.TrjPoint` objects that collectively form a UE 
    trajectory. Additionally, it offers the capability to update its internal state as the UE moves along the
    trajectory.
    
    The trajectory objects are typically generated using a Ray-Tracing framework, such as 
    :py:mod:`DeepMIMO <neoradium.deepmimo>`. For instance, the methods 
    :py:meth:`~neoradium.deepmimo.DeepMimoData.interactiveTrajPoints` and 
    :py:meth:`~neoradium.deepmimo.DeepMimoData.trajectoryFromPoints` can be used to interactively define a trajectory.
    You can also use the :py:meth:`~neoradium.deepmimo.DeepMimoData.getRandomTrajectory` method to create a random 
    trajectory based on the provided Ray-Tracing Scenario, along with other spatial and temporal information about 
    the desired trajectory configuration. Please refer to the notebooks 
    :doc:`../Playground/Notebooks/RayTracing/DeepMimo` and :doc:`../Playground/Notebooks/RayTracing/TrajChannelAnim`
    for examples of using this class.
    """
    def __init__(self, points, carrierFreq):
        r"""
        Parameters
        ----------
        points: list
            A list of :py:class:`~neoradium.trjchan.TrjPoint` objects defining this trajectory.
            
        carrierFreq: float
            The carrier frequency (in Hz) of the communication between the UE and the base station while traveling 
            along this trajectory. This is typically specified by the ray-tracing scenario used to create this 
            trajectory.

            
        **Other Properties**:

            :curIdx: The current index in the list of :py:class:`~neoradium.trjchan.TrjPoint` objects. This starts at 
                zero and is incremented as the UE travels along the trajectory.
            
            :minPaths, avgPaths, maxPaths: Statistics representing the minimum, average, and maximum number of paths 
                between the UE and the base station for all points along this trajectory.

            :numPoints: A read-only property that returns the total number of points on this trajectory.
                
            :numLOS: A read-only property that returns the total number of points along this trajectory with a 
                Line-Of-Sight (LOS) path between the UE and the base station.
                
            :numBlockage: A read-only property that returns the total number of points along this trajectory with total 
                blockage (i.e., no paths between the UE and the base station).
                
            :cur: A read-only property returning current :py:class:`~neoradium.trjchan.TrjPoint` on the trajectory. 
                This is the :py:class:`~neoradium.trjchan.TrjPoint` object at index ``curIdx``.
                
            :remainingPoints: A read-only property returning the remaining points from ``cur`` to the end of 
                trajectory (including ``cur``).
                
            :time: A read-only property returning the total travel time (in seconds) along this trajectory. This is 
                the time of the last point on the trajectory (``trajectory[-1].time``).
                
            :totalDist: A read-only property returning the total distance (in meters) traveled along this trajectory.


        **Indexing**:
        
        This class supports direct indexing with :py:class:`~neoradium.trjchan.TrjPoint` objects. For example:
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("O1_3p5B", baseStationId=3)
            carrier = Carrier(startRb=0, numRbs=25, spacing=15)  # Carrier with 25 Resource Blocks, 15 kHz spacing
            xyBounds = np.array([[250, 300], [270, 400]])        # [[minX, minY], [maxX, maxY]]
            trajLen = 100
            segLen = 2
            traj = deepMimoData.getRandomTrajectory(xyBounds, segLen, carrier.curBwp, trajLen, trajDir="-X")
            for point in traj[:5]:   # print the times of the first 5 points on the trajectory
                print(point.time) 
            print(traj[-1].time)     # print the time of the last point on the trajectory

            
        **Iterating through points**:

        This class has a generator function (``__iter__``) which makes it easier to use it in a loop. For example, the 
        following code finds the point on the trajectory that is farthest from the base station.
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("O1_3p5B", baseStationId=3)
            carrier = Carrier(startRb=0, numRbs=25, spacing=15)  # Carrier with 25 Resource Blocks, 15 kHz spacing
            xyBounds = np.array([[250, 300], [270, 400]])        # [[minX, minY], [maxX, maxY]]
            trajLen = 100
            segLen = 2
            traj = deepMimoData.getRandomTrajectory(xyBounds, segLen, carrier.curBwp, trajLen, trajDir="-X")

            farPoint = None
            maxDist = 0
            for point in traj:
                if point.bsDist>maxDist:
                    farPoint = point
                    maxDist = point.bsDist
            print(farPoint) 
            

        **TrjPoint Redirection**:
            
        The properties **phases**, **delays**, **powers**, **aoas**, **zoas**, **aods**, **zods**, **bounces**,
        **losPhase**, **losDelay**, **losPower**, **losAoa**, **losZoa**, **losAod**, **losZod**, 
        **nlosPhases**, **nlosDelays**, **nlosPowers**, **nlosAoas**, **nlosZoas**, **nlosAods**, **nlosZods**,
        **hasLos**, **numPaths***, and **numNlosPaths** are redirected to the corresponding property in the 
        :py:class:`~neoradium.trjchan.TrjPoint` class for the current point (``cur``). For example ``traj.losZod`` is 
        equivalent to ``traj.cur.losZod``.
        """
        self.points = points                    # Points on this trajectory
        self.dist = 0                           # Total distance traveled. See the "totalDist" property below.
        self.carrierFreq = carrierFreq          # Carrier frequency
        self.curIdx = 0                         # Current position in the trajectory
        
        self.avgPaths, self.maxPaths, self.minPaths = 0, 0, 100000
        self.numLOS, self.numBlockage = 0, 0
        self.maxSpeed = 0
        for p in self.points:
            if p.numPaths > self.maxPaths:  self.maxPaths = p.numPaths
            if p.numPaths < self.minPaths:  self.minPaths = p.numPaths
            if p.hasLos==1:     self.numLOS += 1
            if p.numPaths==0:   self.numBlockage += 1
            self.avgPaths += p.numPaths
            if p.linearSpeed > self.maxSpeed:  self.maxSpeed = p.linearSpeed
        self.avgPaths /= len(self.points)
        self.restart()

    # ******************************************************************************************************************
    def restart(self):
        r"""
        This function resets the current trajectory point to the starting point.
        """
        self.curIdx = 0
        
    # ******************************************************************************************************************
    def goNext(self):
        r"""
        This function advances current point in this trajectory.
        """
        self.curIdx += 1
        
    # ******************************************************************************************************************
    def draw(self):
        r"""
        This visualization function draws this trajectory object using the `matplotlib` library. The starting point 
        of the trajectory is shown with a small red circle.
                
        **Example:**
        
        .. code-block:: python
        
            deepMimoData = DeepMimoData("O1_3p5B", baseStationId=3)
            carrier = Carrier(startRb=0, numRbs=25, spacing=15)  # Carrier with 25 Resource Blocks, 15 kHz spacing
            xyBounds = np.array([[250, 300], [270, 400]])        # [[minX, minY], [maxX, maxY]]
            trajLen = 100
            segLen = 2
            traj = deepMimoData.getRandomTrajectory(xyBounds, segLen, carrier.curBwp, trajLen, trajDir="-X")
            traj.draw()

        .. figure:: ../Images/Trajectory.png
            :align: center
            :figwidth: 600px
        """
        x = [tp.xyz[0] for tp in self.points]
        y = [tp.xyz[1] for tp in self.points]
        fig, ax = plt.subplots()
        ax.grid()
        ax.scatter(x=x[0], y=y[0], c='r')      # The starting point
        ax.plot(x, y)
        ax.set_aspect('equal')

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title="Trajectory Properties:", getStr=False):
        r"""
        Prints the properties of this class.
        
        Parameters
        ----------
        indent: int
            Used internally to adjust the indentation of the printed info.
            
        title: str
            The title used for the information. By default the text "Trajectory Properties:" is used.

        getStr: boolean
            If this is `True`, the function returns a text string instead of printing the info. Otherwise when this
            is `False` (default) the function prints the information.
            
        Returns
        -------
        str or None
            If "getStr" is true, this function returns a text string containing
            the information about the properties of this class. Otherwise, nothing
            is returned (default).
        """
        repStr = "\n" if indent==0 else ""
        if self.isPointSet:
            repStr += indent*' ' + "PointSet Properties\n"
            repStr += indent*' ' + f"  No. of points:         {self.numPoints}\n"
            repStr += indent*' ' + f"  Carrier Frequency:     {freqStr(self.carrierFreq)}\n"
            repStr += indent*' ' + f"  Paths (Min, Avg, Max): {self.minPaths}, {self.avgPaths:.2f}, {self.maxPaths}\n"
            repStr += indent*' ' + f"  Totally blocked:       {self.numBlockage}\n"
            repStr += indent*' ' + f"  LOS percentage:        {self.numLOS*100/self.numPoints:.2f}%\n"
        else:
            repStr += indent*' ' + title + "\n"
            repStr += indent*' ' + f"  start (x,y,z):         ({', '.join('%.2f'%(x) for x in self.points[0].xyz)})\n"
            repStr += indent*' ' + f"  No. of points:         {self.numPoints}\n"
            repStr += indent*' ' + f"  curIdx:                {self.curIdx} ({self.curIdx*100/self.numPoints:.2f}%)\n"
            repStr += indent*' ' + f"  curSpeed:              {np.round(self.cur.speed,2)}\n"
            repStr += indent*' ' + f"  Total distance:        {self.totalDist:.2f} meters\n"
            repStr += indent*' ' + f"  Total time:            {self.time:.3f} seconds\n"
            repStr += indent*' ' + f"  Average Speed:         {self.totalDist/self.time:.3f} m/s\n"
            repStr += indent*' ' + f"  Carrier Frequency:     {freqStr(self.carrierFreq)}\n"
            repStr += indent*' ' + f"  Paths (Min, Avg, Max): {self.minPaths}, {self.avgPaths:.2f}, {self.maxPaths}\n"
            repStr += indent*' ' + f"  Totally blocked:       {self.numBlockage}\n"
            repStr += indent*' ' + f"  LOS percentage:        {self.numLOS*100/self.numPoints:.2f}%\n"

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    def isPointSet(self):                                               # Not Documented
        # True means that this is a set of points used solely to create a dataset of channel matrices
        # with no temporal correlation. (See the DeepMimoData.chanMatAtPoints method)
        return self.points[-1].sampleNo==1          # Not Documented.

    # ******************************************************************************************************************
    @property
    def numPoints(self):        return len(self.points)                 # Documented above in the __init__ function.
    
    # ******************************************************************************************************************
    @property
    def remainingPoints(self):  return self.numPoints - self.curIdx     # Documented above in the __init__ function.

    # ******************************************************************************************************************
    @property
    def cur(self):              return self.points[self.curIdx]         # Documented above in the __init__ function.

    # ******************************************************************************************************************
    @property
    def time(self):             return self.points[-1].time             # Documented above in the __init__ function.

    # ******************************************************************************************************************
    @property                   # Not documented (used only internally)
    def losAngles(self):        return toRadian([self.losAoa, self.losZoa, self.losAod, self.losZod])[:,None,None]

    # ******************************************************************************************************************
    @property                   # Not documented (used only internally)
    def nlosAngles(self):       return toRadian([self.nlosAoas, self.nlosZoas, self.nlosAods, self.nlosZods])[:,:,None]

    # ******************************************************************************************************************
    @property
    def totalDist(self):                                                # Documented above in the __init__ function.
        # Calculating this lazily
        if self.dist == 0:
            for i in range(1,self.numPoints):
               self.dist += np.sqrt(np.square(self.points[i-1].xyz-self.points[i].xyz).sum())
        return self.dist
                
    # ******************************************************************************************************************
    def __getattr__(self, property):                                    # Documented above in the __init__ function.
        # Get these properties from the 'cur' object
        if property not in TrjPoint.pathParamNames:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.cur, property)

    # ******************************************************************************************************************
    def __iter__(self):                                                 # Documented above in the __init__ function.
        for point in self.points:
            yield point

    # ******************************************************************************************************************
    def __getitem__(self, idx):                                         # Documented above in the __init__ function.
        return self.points[idx]

    # ******************************************************************************************************************
    def printDiscontinuities(self):
        r"""
        This method prints all the `discontinuities` in this trajectory. A `discontinuity` is a situation where two 
        consecutive points on the trajectory have different sets of paths.
        """
        for i,p in enumerate(self.points):
            if i==0: continue # skip the first one
            prev = self.points[i-1]
            curToNext = prev.matchPathInfo(p)           # Match paths between prev and p
            commonIdxPrev = np.where(curToNext>-1)[0]   # Indexes of common paths in p0
            commonIdxP = curToNext[curToNext!=-1]       # Indexes of common paths in p1
            assert len(commonIdxPrev)==len(commonIdxP)
            numCommon = len(commonIdxPrev)              # Number of common paths

            if (numCommon==prev.numPaths) and (numCommon==p.numPaths): continue
            print(f"Point {i-1:,}: {prev.numPaths} paths, Point {i:,}: {p.numPaths} paths, {numCommon} common paths.")

# **********************************************************************************************************************
class TrjChannel(ChannelModel):
    r"""
    This class implements a trajectory-based channel model that can generate spatially and temporally consistent 
    sequences of channel information. It creates channel instances based on the movement of users along a 
    :py:class:`~neoradium.trjchan.Trajectory`. The :py:class:`~neoradium.trjchan.TrjChannel` class has been 
    implemented based on **3GPP TR 38.901 Section 8** with the following additional assumptions:
    
    * We assume only one *frequency bin* (:math:`K_B=1`) and derive the path power :math:`P_{l_{RT}}^{RT,real}` from
      the multipath information for each path. As explained in "Step 2" of **3GPP TR 38.901 Section 8.4**, 
      this implies that the bandwidth :math:`B` must be lower than :math:`c/D` Hz, where :math:`c` is the speed of 
      light and :math:`D` is the maximum antenna aperture in either azimuth or elevation.   

    * We skip Steps 4 to 10 in **3GPP TR 38.901 Section 8.4**. In essence, we refrain from generating random 
      clusters and rays and only use the deterministic paths calculated through ray-tracing. Consequently, the number 
      of paths is equivalent to the number of clusters, and each cluster contains a single ray.
      
    * We use the same Cross-Polarization Ratio (XPR) for all paths. This value (:math:`\kappa^{RT}`) is derived from 
      the ``xPolPower`` parameter (:math:`X`) using:
      
    .. math::

        \kappa^{RT} = 10^{\frac X {10}}
            
    * For the initial phase values used to calculate the cross-polarization matrix, we opt for the phase values 
      obtained from ray-tracing instead of generating random values (as outlined in "Step 12" of **3GPP TR 38.901 
      Section 8.4**). Additionally, we employ the same phase value for all four initial phase angles: 
      :math:`\Phi^{\theta \theta}`, :math:`\Phi^{\theta \phi}`, :math:`\Phi^{\phi \theta}`, and 
      :math:`\Phi^{\phi \phi}`. This approach is expected to enhance spatial consistency when generating sequences 
      of channels, which is the main goal of this channel model.
    
    All API functions used in most typical use cases are explained in the documentation of 
    :py:class:`~neoradium.channelmodel.ChannelModel` class.
    
    The typical use case involves creating a trajectory (using 
    :py:meth:`~neoradium.deepmimo.DeepMimoData.interactiveTrajPoints` and 
    :py:meth:`~neoradium.deepmimo.DeepMimoData.trajectoryFromPoints`, or 
    :py:meth:`~neoradium.deepmimo.DeepMimoData.getRandomTrajectory`), using it to obtain a :py:class:`TrjChannel`
    object, and then calling functions such as :py:meth:`~neoradium.channelmodel.ChannelModel.getChannelMatrix`,
    :py:meth:`~neoradium.channelmodel.ChannelModel.applyToSignal`, 
    :py:meth:`~neoradium.channelmodel.ChannelModel.applyToGrid`.
    
    Please refer to the notebook :doc:`../Playground/Notebooks/RayTracing/TrajChannel` for an example of using 
    this class.
    """
    # ******************************************************************************************************************
    def __init__(self, bwp, trajectory, **kwargs):
        r"""
        Parameters
        ----------
        bwp : :py:class:`~neoradium.carrier.BandwidthPart` 
            The bandwidth part object used by the channel model to construct channel matrices.
        
        trajectory : :py:class:`~neoradium.trjchan.Trajectory`
            The trajectory along which the channels are created by this channel model. This object contains multipath
            information that is used to create sequences of channels.
        
        kwargs : dict
            Here’s a list of additional optional parameters that can be used to further customize this channel model:

                :normalizeGains: If the default value of `True` is used, the path gains are normalized before 
                    being applied to the signals.
                    
                :normalizeOutput: If the default value of `True` is used, the gains are normalized based on the 
                    number of receive antennas.

                :normalizeDelays: If the default value of `True` is used, the delays are normalized as specified in 
                    “Step 3” of **3GPP TR 38.901 section 8.4**. Otherwise, the original delays obtained from 
                    ray-tracing are used.

                :interpolateSlots: If the default value of `True` is used, the multipath information for each slot is
                    interpolated to derive multipath information per time symbol. Otherwise, the same multipath 
                    information is used for all symbols in the slot, which is faster but less accurate.  

                :filterLen: The length of the channel filter. The default is 16 sample.
                
                :delayQuantSize: The size of delay fraction quantization for the channel filter. The default is 64.
                
                :stopBandAtten: The Stop-band attenuation value (in dB) used by the channel filter. The default is 80dB.
                
                :txAntenna: The transmitter antenna, which is an instance of 
                    :py:class:`~neoradium.antenna.AntennaElement`, :py:class:`~neoradium.antenna.AntennaPanel`,
                    or :py:class:`~neoradium.antenna.AntennaArray` class. If not specified, a single antenna element is
                    automatically created by default.
                
                :rxAntenna: The receiver antenna, which is an instance of 
                    :py:class:`~neoradium.antenna.AntennaElement`, :py:class:`~neoradium.antenna.AntennaPanel`,
                    or :py:class:`~neoradium.antenna.AntennaArray` class. If not specified, a single antenna element is
                    automatically created by default.
                    
                :txOrientation: The orientation of transmitter antenna. This is a list of 3 angle values in degrees for
                    the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901 Section 7.1.3** for more
                    information.

                :rxOrientation: The orientation of receiver antenna. This is a list of 3 angle values in degrees for 
                    the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901 Section 7.1.3** for more
                    information.

                :xPolPower: The Cross-Polarization Power in dB. The default is 10db. It is defined as 
                    :math:`X=10 log_{10} \kappa^{RT}` where :math:`\kappa^{RT}` is the Cross-Polarization Ratio (XPR). 
                    In current implementation this value is used for all paths.

                                        
        **Other Properties:**
        
            :nrNt: A tuple of the form ``(nr,nt)``, where ``nr`` and ``nt`` are the number receiver and transmitter 
                antenna elements, respectively.
                
            :pathPowers: This read-only property returns the path powers for all paths between the base station and 
                the UE at current position on the current trajectory. 

            :pathDelays: This read-only property returns the path delays for all paths between the base 
                station and the UE at current position on the trajectory. If the parameter ``normalizeDelays``
                is `True`, the delays from the ray-tracing are normalized according to "Step 3" in **3GPP TR 38.901 
                Section 8.4**. Note that the delay normalization (if enabled) is only applied to this property; the
                redirected properties "delays", "losDelay", and "nlosDelays" always keep the original delay values from
                the ray-tracing.

            :totalBlockage: This read-only property returns `True` if the channel is currently at a total blockage 
                point, indicating that there is no communication path between the transmitter and receiver.

        **TrjPoint Redirection**:
            
            The properties **phases**, **delays**, **powers**, **aoas**, **zoas**, **aods**, **zods**, **bounces**,
            **losPhase**, **losDelay**, **losPower**, **losAoa**, **losZoa**, **losAod**, **losZod**, 
            **nlosPhases**, **nlosDelays**, **nlosPowers**, **nlosAoas**, **nlosZoas**, **nlosAods**, **nlosZods**,
            **hasLos**, **numPaths***, and **numNlosPaths** are redirected to the corresponding property in the 
            :py:class:`~neoradium.trjchan.TrjPoint` class for the current point (``cur``) in the trajectory. For 
            example ``channel.losZod`` is equivalent to ``channel.trajectory.cur.losZod``.

        """
        super().__init__(bwp, **kwargs)
        self.trajectory = trajectory
        
        self.interpolateSlots = kwargs.get('interpolateSlots', True)
                        
        self.carrierFreq = trajectory.carrierFreq   # Set carrierFreq based on the value from ray-tracing scenario
        
        self.dopplerShift = trajectory.maxSpeed * self.carrierFreq/299792458 # Calculate based on max speed (Not used)
        self.kFactor = None                         # K-Factor and K-Factor scaling apply to stochastic models (CDL/TDL)

        self.txAntenna = kwargs.get('txAntenna', AntennaElement())  # Transmitter AntennaArray/AntennaPanel object
        self.rxAntenna = kwargs.get('rxAntenna', AntennaElement())  # Receiver AntennaArray/AntennaPanel object

        # Orientation of TX and RX antenna arrays.
        # NOTE: To point an RX/TX antena to LOS angles of arrival/departure use: 𝛼=aoa[0]/aod[0], 𝛃=zoa[0]/zod[0]-90° 𝛄=0
        self.txOrientation = toRadian(kwargs.get('txOrientation', [0,0,0])) # Orientation of TX Antenna
        self.rxOrientation = toRadian(kwargs.get('rxOrientation', [0,0,0])) # Orientation of RX Antenna

        self.xPolPower = kwargs.get('xPolPower', 10.0)  # CrossPolarization Power in dB.

        self.normalizeDelays = kwargs.get('normalizeDelays', True )     # Normalize delays so that the 1st one is zero

        self.restart()

    # ******************************************************************************************************************
    @property                   # This property is already documented above in the __init__ function.
    def nrNt(self):             return (self.rxAntenna.getNumElements(), self.txAntenna.getNumElements())

    # ******************************************************************************************************************
    @property                   # This property is already documented above in the __init__ function.
    def totalBlockage(self):    return (self.trajectory.numPaths == 0)

    # ******************************************************************************************************************
    @property                   # This property is already documented above in the __init__ function.
    def pathPowers(self):
        if self.totalBlockage:      return None
        if self.interpolateSlots:
            _, nc, n = self.intPathInfo.shape
            return self.intPathInfo[2,nc//2]    # Return the interpolated powers in the mid point of this slot
        return self.trajectory.powers

    # ******************************************************************************************************************
    @property                   # This property is already documented above in the __init__ function.
    def pathDelays(self):
        if self.totalBlockage:      return None
        delays = self.trajectory.delays
        if self.interpolateSlots:
            _, nc, n = self.intPathInfo.shape
            delays = self.intPathInfo[1,nc//2]    # Return the interpolated delays in the mid point of this slot
        
        # Normalizing the delays. See "Step 3" in 3GPP 3GPP TR 38.901 section 8.4 for more details.
        if self.normalizeDelays:    return delays - delays[0]
        return delays

    # ******************************************************************************************************************
    def __getattr__(self, property):    # Documented above in the __init__ function.
        # Get these properties from the 'cur' object
        if property not in TrjPoint.pathParamNames:
            raise ValueError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, property))
        return getattr(self.trajectory, property)

    # ******************************************************************************************************************
    def restart(self, restartRanGen=False, applyToBwp=True):
        r"""
        This method first resets the current trajectory to its starting position and then calls the base class 
        :py:meth:`~neoradium.channelmodel.ChannelModel.restart`.

        Parameters
        ----------
        restartRanGen : Boolean
            Ignored for the :py:class:`~neoradium.trjchan.TrjChannel` class as it uses deterministic information only.

        applyToBwp : Boolean
            If set to `True` (the default), this function restarts the Bandwidth Part associated with this channel 
            model. Otherwise, the Bandwidth Part’s state remains unchanged.
        """
        self.trajectory.restart()
        super().restart(restartRanGen, applyToBwp)

    # ******************************************************************************************************************
    def goNext(self, applyToBwp=True):
        r"""
        This method is called after each application of the channel to a signal. It updates the timing information of
        the trajectory and the channel model preparing it for the next application to the input signal. It is assumed 
        that the channel is applied to a single slot of the signal at each application (either in the time or 
        frequency domain).
        
        Parameters
        ----------
        applyToBwp : Boolean
            If set to `True` (the default), this function advances the timing state of the Bandwidth Part associated
            with this channel model. Otherwise, the Bandwidth Part’s state remains unchanged.

        """
        self.trajectory.goNext()
        super().goNext(applyToBwp)

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this channel model object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as a title for the printed information. If `None` (default), the text
            "TrjChannel Properties:" is used for the title.

        getStr : Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns
            the information in a text string. Otherwise, nothing is returned.
        """
        # Do not call the parent class's print, because some of parameters do not apply to this subclass
        if title is None:   title = "TrjChannel Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  carrierFreq:          {freqStr(self.carrierFreq)}\n"
        repStr += indent*' ' + f"  normalizeGains:       {str(self.normalizeGains)}\n"
        repStr += indent*' ' + f"  normalizeOutput:      {str(self.normalizeOutput)}\n"
        repStr += indent*' ' + f"  normalizeDelays:      {str(self.normalizeDelays)}\n"
        repStr += indent*' ' + f"  interpolateSlots:     {str(self.interpolateSlots)}\n"
        repStr += indent*' ' + f"  xPolPower:            {self.xPolPower:.2f} (dB)\n"
        repStr += indent*' ' + f"  filterLen:            {self.filterLen} samples\n"
        repStr += indent*' ' + f"  delayQuantSize:       {self.delayQuantSize}\n"
        repStr += indent*' ' + f"  stopBandAtten:        {self.stopBandAtten} dB\n"
        repStr += indent*' ' + f"  dopplerShift:         {freqStr(self.dopplerShift)}\n"
        repStr += indent*' ' + f"  coherenceTime:        {self.coherenceTime} sec\n"

        repStr += self.txAntenna.print(indent+2, "TX Antenna:", True)
        if np.any(self.txOrientation):
            repStr += indent*' ' + "    Orientation (𝛼,𝛃,𝛄): %s°\n"%("° ".join(str(int(np.round(toDegrees(a))))
                                                                                 for a in self.txOrientation ))
        
        repStr += self.rxAntenna.print(indent+2, "RX Antenna:", True)
        if np.any(self.rxOrientation):
            repStr += indent*' ' + "    Orientation (𝛼,𝛃,𝛄): %s°\n"%("° ".join(str(int(np.round(toDegrees(a))))
                                                                                 for a in self.rxOrientation ))

        repStr += self.trajectory.print(indent+2, "Trajectory:", True)

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def prepareForNextSlot(self):
        # Raise exception if at the end of trajectory. Otherwise just call base class's function.
        if self.trajectory.remainingPoints<=0:      raise ValueError("Reached end of trajectory!")
        if self.nextSlotStart > self.curSlotStart:  return  # The parameters for this slot have already been initialized
        if self.interpolateSlots: self.interpolateSlot()
        super().prepareForNextSlot()

    # ******************************************************************************************************************
    def interpolateSlot(self):
        # This function returns `nc` sets of point information (nc = numSym + 1). The first set corresponds to the
        # information at the current trajectory point, the last set corresponds to the information at the next
        # trajectory point, and the intermediate sets contain the interpolated information for each symbol.
        if self.totalBlockage:  return              # No interpolation is needed at total-blockage locations

        symLens = self.bwp.getSymLens()                         # lengths of the next (numSyms+1) symbols
        symStarts = np.cumsum(np.append([0], symLens[:-1]))     # Symbol start sample indices for the next nc symbols
        nc = len(symLens)                                       # nc = numSyms+1 = len(symLens) = len(symStarts)
        p0 = self.trajectory.cur                                # Current trajectory point

        # Initialize interpolated values using the current point by repeating its information for all symbols.
        # If anything fails in this function, the interpolation defaults to using p0 only.
        self.intXyzs = np.array(nc*[p0.xyz])                                            # Shape: nc x 3
        self.intPathInfo = np.transpose( np.array(nc*[p0.pathInfo]), (2,0,1))           # 8 x nc x numPaths
        if self.trajectory.remainingPoints<=1:      return

        
        p1 = self.trajectory.points[self.trajectory.curIdx+1]   # Next trajectory point
        if p0.hasLos != p1.hasLos:                  return      # Ensure that both points are either LOS or NLOS
        
        curToNext = p0.matchPathInfo(p1)                        # Match path indices between p0 and p1
        commonIdxCur = np.where(curToNext>-1)[0]                # Indexes of common paths in p0
        commonIdxNext = curToNext[curToNext!=-1]                # Indexes of common paths in p1
        assert len(commonIdxCur)==len(commonIdxNext)
        n = len(commonIdxCur)                                   # Number of common paths
        if n==0:                                    return      # No common paths. Use p0-based interpolation only
       
        # Endpoints of interpolation (p0 and p1), including only the common paths
        endPointsInfo = np.concatenate(([p0.pathInfo[commonIdxCur]], [p1.pathInfo[commonIdxNext]]))
        # Unwrapping azimuth/phase angles: (0:Phase, 1:delay, 2:RxPower, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:bounces)
        endPointsInfo[:,:,(0,3,5)] = np.unwrap(endPointsInfo[:,:,(0,3,5)],.5, axis=0, period=360)
        endPointsXyz = np.concatenate(([p0.xyz],[p1.xyz]))              # End points 3D coordinates. Shape: 2 x 3
        # All endpoint information including positions:
        endPointsInfo = np.concatenate((endPointsInfo.reshape(2,-1), endPointsXyz), axis=1) # Shape: 2 x n*8+3

        # Compute the interpolated point information
        startRatios = symStarts.reshape(-1,1)/symStarts[-1]                 # nc increasing values between 0 and 1
        intPointInfo = endPointsInfo[0] + (endPointsInfo[1]-endPointsInfo[0])*startRatios       # nc x (n*8+3)

        # Validate start and end values
        assert np.abs(intPointInfo[-1]-endPointsInfo[1]).max()<1e-6, \
               f"End Difference: {np.abs(intPointInfo[-1]-endPointsInfo[1]).max()}"
        assert np.abs(intPointInfo[0]-endPointsInfo[0]).max()<1e-6, \
               f"Start Difference: {np.abs(intPointInfo[0]-endPointsInfo[0]).max()}"
    
        # Store the interpolated information
        self.intXyzs = intPointInfo[:,-3:]                                                      # Shape: nc x 3
        self.intPathInfo = np.transpose(intPointInfo[:,:-3].reshape(nc,n,8), (2,0,1))           # 8 x nc x n

    # ******************************************************************************************************************
    def getPathGains(self):
        r"""
        Calculates the gains for Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) paths separately and combines the 
        results before returning the gains between every RX/TX antenna pair, for every path, at every time symbol. 
        
        Returns
        -------
        NumPy array or None
            This function returns a 4-D complex tensor of shape ``Ns x Nr x Nt x Np``, where ``Ns`` represents the 
            number of time symbols, ``Nr`` and ``Nt`` indicate the number of receiver and transmitter antennas
            respectively, and ``Np`` denotes the number of paths between the base station and the UE at its current
            location along its trajectory. If the UE is at a total blockage point, this function returns `None`.
        """
        # This function creates gain information for the next 'nc' time symbols, which is the number of time symbols
        # per slot plus 1. We always create gains for one more symbol (The first symbol of the next slot)
        if self.numPaths == 0:      return None                         # Total Blockage
        
        if self.interpolateSlots:
            numIntPaths = self.intPathInfo.shape[-1]
            if self.hasLos:
                if numIntPaths==1:  return self.getGains(los=True)       # Only LOS, Shape: nc x nr x nt x 1
                return np.concatenate((self.getGains(los=True),
                                       self.getGains(los=False)),axis=3) # LOS & NLOS, Shape: nc x nr x nt x n
            return self.getGains(los=False)                              # Only NLOS, Shape: nc x nr x nt x n
        
        # Not interpolating
        if self.numNlosPaths>0:
            if self.hasLos==0:      return self.getGains(los=False)      # Only NLOS, Shape: nc x nr x nt x n
            return np.concatenate((self.getGains(los=True),
                                   self.getGains(los=False)),axis=3)     # LOS & NLOS, Shape: nc x nr x nt x n
        return self.getGains(los=True)                                   # Only LOS, Shape: nc x nr x nt x 1

    # ******************************************************************************************************************
    def getGains(self, los):
        # This function calculates the gains matrix for LOS and NLOS paths (based on the `los` parameter).
        # It is called by the `getPathGains` function and returns an nc x nr x nt x n complex NumPy array.
        if self.interpolateSlots:
            # Shape of self.intPathInfo: 8 x nc x n
            # Path information indexes: 0:Phase, 1:delay, 2:RxPower, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:bounces
            if los:     # nc = number of time symbols+1, n = 1
                phiA, thetaA, phiD, thetaD = toRadian(self.intPathInfo[3:7,:,0:1])                  # 4 x nc x n
                pN = toLinear(self.intPathInfo[2,:,0:1])                    # Convert from dB to linear   nc x n
                phases = toRadian(self.intPathInfo[0,:,0:1])                                            # nc x n
            else:       # nc = number of time symbols+1, n = number of NLOS paths
                phiA, thetaA, phiD, thetaD = toRadian(self.intPathInfo[3:7,:,self.hasLos:])         # 4 x nc x n
                pN = toLinear(self.intPathInfo[2,:,self.hasLos:])           # Convert from dB to linear   nc x n
                phases = toRadian(self.intPathInfo[0,:,self.hasLos:])                                   # nc x n
        elif los:       # nc = 1, n = 1
            phiA, thetaA, phiD, thetaD = toRadian([self.losAoa, self.losZoa,
                                                   self.losAod, self.losZod])[:, None, None]        # 4 x nc x n
            pN = toLinear(np.array([[self.losPower]]))                      # Convert from dB to linear   nc x n
            phases = toRadian(self.losPhase)[None,None]                                                 # nc x n
        else:           # nc = 1, n = number of NLOS paths
            phiA, thetaA, phiD, thetaD = toRadian([self.nlosAoas, self.nlosZoas,
                                                       self.nlosAods, self.nlosZods])[:,None,:]     # 4 x nc x n
            pN = toLinear(np.array([self.nlosPowers]))                      # Convert from dB to linear   nc x n
            phases = toRadian(self.nlosPhases)[None,:]                                                  # nc x n

        nr, nt = self.nrNt

        # Get the RX field component and RX location component in:
        #       NLOS: TR 38.901 – Eq. 7.5-28 (the 1st and 4th terms)
        #       LOS:  TR 38.901 – Eq. 7.5-29 (the 1st and 5th terms)
        fieldRx, locRx = self.rxAntenna.getElementsFields(thetaA, phiA, self.rxOrientation) # nrx2xncxn & nrxncxn
        fieldRx, locRx = np.transpose(fieldRx,(2,0,1,3)), np.transpose(locRx,(1,0,2))   # nc x nr x 2 x n & nc x nr x n

        # Get the TX field component and TX location component in:
        #       NLOS: TR38.901 - Eq. 7.5-28 (The 3rd and 5th terms)
        #       LOS:  TR38.901 - Eq. 7.5-29 (The 3rd and 6th terms)
        fieldTx, locTx = self.txAntenna.getElementsFields(thetaD, phiD, self.txOrientation) # ntx2xncxn & ntxncxn
        fieldTx, locTx = np.transpose(fieldTx,(2,0,1,3)), np.transpose(locTx,(1,0,2))   # nc x nt x 2 x n & nc x nt x n

        if los:
            # Get the polarization matrix term in TR 38.901 – Eq. 7.5-29 (the 2nd and 4th terms combined).
            # We need to apply the LOS phase to the polarization matrix. The LOS phase is: -2π d/λ (where d is
            # the 3D distance and λ is the wavelength).
            polMat = np.exp(1j*phases[:,None,None,:])*np.float64([[1,0],[0,-1]])[None,:,:,None]      # nc x 2 x 2 x n
        else:
            # STEP-3 (See Step-3 in TR38.901 - 7.7.1) ------------------------------------------------------------------
            # Get the cross-polarization power ratio:
            k = toLinear(self.xPolPower)    # Cross-Polarization Ratio (XPR)
            
            # Get the polarization matrix term in TR38.901 - Eq. 7.5-28 (The 2nd term)
            polMat = np.exp(1j*phases[:,None,None,:])*(np.sqrt([[1, 1/k], [1/k, 1]]))[None,:,:,None] # nc x 2 x 2 x n

        doppler = self.getDopplerFactor(thetaA, phiA)                                                           # nc x n

        # First fieldRx x polMat x fieldTx
        #   fieldRx      .      polMat    ->  fieldRx.PolMat .     fieldTx     ->       h
        # (nc, nr, 2, n) . (nc, 2, 2, n)  ->  (nc, nr, 2, n) . (nc, nt, 2, n)  ->  (nc, nr, nt, n)
        #      1   2            1  2      ->       1   2            2   1      ->       1   2
        h = np.matmul(np.matmul(fieldRx, polMat, axes=[(1,2),(1,2),(1,2)]),
                      fieldTx, axes=[(1,2),(2,1),(1,2)])                                            # nc x nr x nt x n
        h *= locRx[:,:,None,:] * locTx[:,None,:,:]                                                  # nc x nr x nt x n
        h =  h * doppler[:, None,None,:]                                                            # nc x nr x nt x n
        h *= np.sqrt(pN)[:,None,None,:]                                                             # nc x nr x nt x n
        return h

    # ******************************************************************************************************************
    def getDopplerFactorOld(self, theta, phi):           # Not documented
        # This function calculates the doppler term (the last term) in TR38.901 - Eq. 7.5-28 and 7.5-29
        # 'r' is the unit vector which points to the arrival direction of each path.
        r = np.array([ np.sin(theta)*np.cos(phi),
                       np.sin(theta)*np.sin(phi),
                       np.cos(theta) ])                     # Shape: 3, 1, 1 (LOS) or 3, n, 1 (NLOS)
        v = self.trajectory.cur.speed.reshape(3,1,1)        # 3D Speed vector, Shape: 3, 1, 1
        waveLen = 299792458/self.carrierFreq                # Speed of light: 299792458 m/s
        dopplerShift = (r*v).sum(0)/waveLen                 # Doppler Shift (Hz), Shape: 1,1 (LOS) or n,1 (NLOS)

        chanTimes = self.chanGainSamples/self.sampleRate
        return np.exp(2j * np.pi * chanTimes[:,None,None] * dopplerShift[None,:,:]) # Shape: nc, n, 1 or nc, 1, 1

    # ******************************************************************************************************************
    def getDopplerFactor(self, theta, phi):
        # This function calculates the doppler term (the last term) in TR38.901 - Eq. 7.5-28 and 7.5-29
        # 'r' is the unit vector which points to the arrival direction of each path.
        r = np.array([ np.sin(theta)*np.cos(phi),
                       np.sin(theta)*np.sin(phi),
                       np.cos(theta) ])                     # Shape: 3 x nc x n
        v = self.trajectory.cur.speed.reshape(3,1,1)        # 3D Speed vector, Shape: 3, 1, 1
        waveLen = 299792458/self.carrierFreq                # Speed of light: 299792458 m/s
        dopplerShift = ((r*v).sum(0)/waveLen)               # Doppler Shift (Hz), Shape: nc x n

        chanTimes = self.chanGainSamples/self.sampleRate
        return np.exp(2j * np.pi * chanTimes[:, None] * dopplerShift) # Shape: nc x n

    # ******************************************************************************************************************
    def getChanSeqGen(self, seqPeriod=1, seqLen=10, maxNumSeq=np.inf,
                      chanCallback=None, seqCallback=None):
        r"""
        This function returns an iterable object that generates sequences of channel matrices based on the given parameters.

        Refer to the notebook :doc:`../Playground/Notebooks/RayTracing/ChannelSequences` for an example of
        using this method.
        
        Parameters
        ----------
        seqPeriod: int            
            The sampling period of channel matrices along the trajectory. The default value of ``1`` means all 
            channel matrices are included in the sequence. For instance, if this is set to ``3``, every third 
            channel matrix (i.e., one out of every three slots) is included in the sequence.
            
        seqLen : int
            The length of the returned sequences. The default is ``10``.

        maxNumSeq: int or float           
            The maximum number of sequences to generate. By default, it is set to ``np.inf``, indicating no additional
            limit on the number of sequences. In this case, channel sequences are generated until the end of 
            the trajectory.
    
        chanCallback: function
            If provided, this function is called when a new channel matrix is generated and before it is added to 
            the sequence. The function may modify the channel and return the modified channel matrix. The function may 
            also return `None`, which means the current sequence is not valid and should be dropped. In this case, 
            the generator discards the channels collected in the current sequence and starts a new sequence beginning
            with the next channel matrix according to ``seqPeriod``. The default is `None` which disables this channel 
            preprocessing feature. The user-defined function receives the following parameters:

                :seqNo: The sequence number, starting at zero and incremented for each sequence.
                :elementNo: The element index within the current sequence.
                :channelMatrix: The current channel matrix. An ``L x K x Nr x Nt`` NumPy array where ``L`` represents 
                    the number of OFDM symbols, ``K`` denotes the number of subcarriers, ``Nr`` is the number of 
                    receive antennas, and ``Nt`` indicates the number of transmit antennas.
                :channel: This :py:class:`~neoradium.trjchan.TrjChannel` object.
                            
        seqCallback: function
            If provided, this function is called when a new sequence of channels is generated and before it is used by
            the sequence generator. The function may modify the channel sequence and return the modified sequence. The 
            function may also return `None`, which means the current sequence is not valid and should be dropped. In 
            this case, the generator discards the sequence and initiates a new one, starting with the next
            channel matrix according to ``seqPeriod``. The default is `None` which disables this sequence preprocessing 
            feature. The user-defined function receives the following parameters:
            
                :seqNo: The sequence number, starting at zero and incremented for each sequence.
                :channelSequence: The sequence of channel matrices. An ``S x L x K x Nr x Nt`` NumPy array where
                    ``S`` is the sequence length (the ``seqLen`` parameter above), ``L`` represents the number of 
                    OFDM symbols, ``K`` denotes the number of subcarriers, ``Nr`` is the number of receive antennas, 
                    and ``Nt`` indicates the number of transmit antennas.
                :channel: This :py:class:`~neoradium.trjchan.TrjChannel` object.
        
        Returns
        -------
        ChanSeqGen
            An iterable object that yields sequences of channel matrices.

    
        The following example creates sequences of effective channel matrixes (which includes the precoding effect)
        that do not contain any point with total blockage.
        
        .. code-block:: python
        
            from neoradium import Carrier, PDSCH

            carrier = Carrier(numRbs=25, spacing=30) # Carrier with 25 RBs, 30KHz subcarrier spacing
            channel = TrjChannel(carrier.curBwp, trajectory,
                                 txAntenna = AntennaPanel([2,4], polarization="x"),       # 8 TX antenna
                                 rxAntenna = AntennaPanel([1,2], polarization="x"))       # 2 RX antenna
            pdsch = PDSCH(carrier.curBwp, numLayers=2, nID=carrier.cellId, modulation="16QAM")

            def chanCallback(seqNo, elementNo, channelMatrix, channel):
                # Drop the channels with total blockage
                return None if channel.totalBlockage else channelMatrix
                
            def seqCallback(seqNo, channelSeq, channel):
                # Get the precoder matrix based on the first channel in the sequence
                precoder = pdsch.getPrecodingMatrix(channelSeq[0])
                # Now create and return a new sequence containing the effective channels by applying 
                # the precoder to all channels in the sequence
                newSeq = np.stack([channel.getEffChannel(chanMat, precoder) for chanMat in channelSeq])  
                return newSeq        

            chanSeqGen = channel.getChanSeqGen(seqPeriod=1, seqLen=5, maxNumSeq=20, 
                                               chanCallback=chanCallback, seqCallback=seqCallback)
            channelSequences = np.stack([chanSeq for chanSeq in chanSeqGen])
            print(channelSequences.shape)   # Prints: (20, 5, 14, 300, 4, 2) for (numSeq, seqLen, L, K, Nr, Nt)
        """
        if seqPeriod <= 0:
            raise ValueError("'seqPeriod' must be a positive integer.")
    
        self.restart()
        class ChanSeqGen:
            def __init__(self, channel):
                self.channel = channel
                self.numSeq = 0
                
            def __iter__(self): return self

            def __next__(self):
                if self.numSeq >= maxNumSeq: raise StopIteration
                
                sequence = None
                while sequence is None:
                    sequence = []
                    while len(sequence) < seqLen:
                        while (self.channel.trajectory.curIdx % seqPeriod) > 0:
                            self.channel.goNext()
                            if self.channel.trajectory.remainingPoints <=0: raise StopIteration
                        
                        channelMatrix = self.channel.getChannelMatrix()
                        if chanCallback is not None:    channelMatrix = chanCallback(self.numSeq, len(sequence),
                                                                                     channelMatrix, self.channel)
                        if channelMatrix is None:       sequence = []   # The channel is bad, restart the sequence
                        else:                           sequence += [ channelMatrix ]
                        self.channel.goNext()
                        if self.channel.trajectory.remainingPoints <=0:     raise StopIteration

                    sequence = np.stack(sequence)
                    if seqCallback is not None:         sequence = seqCallback(self.numSeq, sequence, self.channel)
                    
                self.numSeq += 1
                return sequence

            def reset(self):
                self.numSeq = 0
                self.channel.restart()

        return ChanSeqGen(self)
