# Copyright (c) 2024-2026, InterDigital AI Lab
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
# 04/01/2025    Shahab                  Restructured the file to work with the new ChannelModel class.
# 05/07/2025    Shahab                  Reviewed and updated the documentation.
# 06/20/2025    Shahab                  * The Trajectory class now can be used in the "PointSet" mode where there is
#                                         no temporal/spatial correlation between points. It is used to create random
#                                         channel matrices (see isPointSet in this file and the "getChanGen" method in
#                                         the "deepmimo.py" file).
#                                       * Updated the "restart" and "goNext" with the new parameter "applyToBwp".
#                                       * Added the new method "getChanSeqGen" that can be used to return a generator
#                                         object that can generate sequences of channel matrices based on the given
#                                         parameters.
# 12/18/2025    Shahab                  * Some bug fixes.
#                                       * Added the method "printDiscontinuities" to the "Trajectory" class.
#                                       * Added multipath interpolation inside each slot. See "interpolateSlots"
#                                         parameter and "interpolateSlot" method of the "TrjChannel" class.
#                                       * Added the "totalBlockage" readonly property.
#                                       * Re-organised the code related to the "getPathGains" function. "getGains"
#                                         method replaced the old "getLOSgains" and "getNLOSgains" methods. The
#                                         "getDopplerFactor" method has also been updated accordingly.
#                                       * Updated the channel sequence generation in the "getChanSeqGen" function and
#                                         added 2 different callbacks for channels and sequences.
# 03/12/2026    Shahab                  Changes in NeoRadium version 0.5.0:
#                                       * Added pathIds and polPhases to the trajectory points.
#                                       * The new random polarization phases are used when calculating channel gains.
#                                       * Removed the 'interpolateSlots' and interpolation feature. The discontinuity
#                                         problem was related to using "phases" for polarization matrix. Now
#                                         we create random polarization phases, and do not use "phases".
# **********************************************************************************************************************
import numpy as np
from scipy.signal import lfilter

from .antenna import AntennaElement
from .channelmodel import ChannelModel
from .utils import getMultiLineStr, freqStr, toRadian, toDegrees, toLinear, toDb
from .random import random
from .carrier import SAMPLE_RATE

# This file is based on 3GPP TR 38.901
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
                      "hasLos", "numPaths", "numNlosPaths", "pathIds", "polPhases"]
                      
    def __init__(self, xyz=[0,0,0], hasLos=-1, pathInfo=np.empty((0,8)), bsDist=0,
                 pathLoss=0, speed=np.zeros(3), sampleNo=0, pathIds=None, polPhases=None):
        r"""
        Parameters
        ----------
        xyz : list
            A list of three floating-point numbers giving the 3-D coordinates of this 
            :py:class:`~neoradium.trjchan.TrjPoint` in meters. The default is ``[0, 0, 0]``.

        hasLos : int
            An integer that can take on any of the following values:
            
                :1: Means there is a line-of-sight (LOS) path between the UE at this 
                    :py:class:`~neoradium.trjchan.TrjPoint` and the base station.
                
                :0: Means there is no line-of-sight (LOS) path between the UE at this
                    :py:class:`~neoradium.trjchan.TrjPoint` and the base station.
                
                :-1: Means there is no path between the UE at this :py:class:`~neoradium.trjchan.TrjPoint` and the 
                    base station. (Total Blockage)
                
        pathInfo : NumPy array
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
                    value of -1 typically indicates the path interaction information is not available, for example, 
                    when older `DeepMIMO <https://www.deepmimo.net>`_ scenario files are used. If the value is 
                    positive, the number of digits specifies the number of interactions the path had on its journey 
                    from the transmitter to the receiver. Each digit represents a specific type of interaction:
                    
                    :1: Reflection
                    :2: Diffraction 
                    :3: Scattering 
                    :4: Transmission
                
        bsDist : float
            The distance between this :py:class:`~neoradium.trjchan.TrjPoint` and the base station in meters. The 
            default is zero.
            
        speed : NumPy array
            The 3D vector of linear speed at this :py:class:`~neoradium.trjchan.TrjPoint` in meters per second (m/s).
            
        sampleNo : int
            The time when the UE arrives at this :py:class:`~neoradium.trjchan.TrjPoint` as a number of samples from
            the start of :py:class:`~neoradium.trjchan.Trajectory`. This is based on the **3GPP** sample rate of 
            30,720,000 samples per second.

        pathIds : NumPy array
            A numpy array containing the path's unique identifiers for the paths at the current trajectory point. A
            unique ID is assigned to each new path when it is detected first. 
            
        polPhases : NumPy array or None
            An ``n x 2 x 2`` NumPy array containing polarization phases :math:`\Phi^{\theta\theta}`, 
            :math:`\Phi^{\theta\phi}`, :math:`\Phi^{\phi\theta}`, and :math:`\Phi^{\phi\phi}` for all ``n`` paths. This
            is based on **3GPP TR 38.901 Section 7.5** (Step 10). These are initialized randomly (between :math:`-\pi` 
            and :math:`\pi`) when a new path is "born". They stay the same on a trajectory during the lifetime of the 
            path. The default is `None``, which results in the same phase for all four values in the cross polarization
            matrix.


        **Other Properties:**
                    
            :numPaths: The number of paths between the UE at this :py:class:`~neoradium.trjchan.TrjPoint` and the 
                base station.
                
            :numNlosPaths: The number of Non-Line-Of-Sight (NLOS) paths between the UE at this 
                :py:class:`~neoradium.trjchan.TrjPoint` and the base station.
                
            :time: The time (in seconds) when the user arrives at this :py:class:`~neoradium.trjchan.TrjPoint`. This 
                read-only value is zero for the first :py:class:`~neoradium.trjchan.TrjPoint` of the 
                :py:class:`~neoradium.trjchan.Trajectory` and increases based on the speed and the distance between 
                the points along the trajectory.
                
            :linearSpeed: UE's linear speed at this :py:class:`~neoradium.trjchan.TrjPoint` (read-only).
            
            :phases: The *phase* angles in degrees for all paths or `None` if there are no paths.
            :delays: The *delay* in nanoseconds for all paths or `None` if there are no paths.
            :powers: The *gain* in dB for all NLOS paths or `None` if there are no paths.
            :aoas: The Azimuth angle Of Arrival (AOA) in degrees for all paths or `None` if there are no paths.
            :zoas: The Zenith angle Of Arrival (ZOA) in degrees for all paths or `None` if there are no paths.
            :aods: The Azimuth angle Of Departure (AOD) in degrees for all paths or `None` if there are no paths.
            :zods: The Zenith angle Of Departure (ZOD) in degrees for all paths or `None` if there are no paths.
            :bounces: The path interaction information for all paths or `None` if there are no paths. See the 
                explanation of the path interactions for the ``pathInfo`` parameter above.
            :polPhases: The polarization initial phases. This is an ``n x 2 x 2`` matrix.
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

        self.pathIds = pathIds          # Unique identifier for each path at this point.
        self.polPhases = polPhases      # Polarization phases used by the channel model for NLOS paths.

        # Note: We always assume if there is a LOS path, it is the one with the lowest delay. So, we do not keep a
        # LOS flag per path.
        # Path Values: 0:phase, 1:delay, 2:power, 3:aoa, 4:zoa, 5:aod, 6:zod, 7:bounce
        self.pathInfo = np.float64(pathInfo)    # n x 7  or n x 8
        if self.pathInfo.shape[1]==7:
            if self.numPaths>0: self.pathInfo = np.append(self.pathInfo, self.numPaths*[[-1]], axis=1)
            else:               self.pathInfo = np.empty((0,8))
        
        self.bsDist = np.float64(bsDist)
        self.pathLoss = np.float64(pathLoss)
        self.speed = speed                  # linear speed at this point (A 3D vector)
        self.sampleNo = sampleNo            # The sample number of this point from the start point of the trajectory
        
        # Initialize path information and make sure pathInfo is sorted by the delay:
        self.phases, self.delays, self.powers, self.aoas, self.zoas, self.aods, self.zods, self.bounces = 8*[None]
        self.losPhase, self.losDelay, self.losPower, self.losAoa, self.losZoa, self.losAod, self.losZod = 7*[None]
        self.nlosPhases, self.nlosDelays, self.nlosPowers, self.nlosAoas, self.nlosZoas, self.nlosAods, \
            self.nlosZods = 7*[None]
        if self.numPaths>0:
            sortIdx = np.argsort(self.pathInfo[:,1])
            self.pathInfo = self.pathInfo[sortIdx]      # Path with the smallest delay comes first
            if self.pathIds is not None:    self.pathIds = self.pathIds[sortIdx]    # Reorder path IDs
            if self.polPhases is not None:
                self.polPhases = self.polPhases[sortIdx]                    # Reorder Polarization Phases
                if self.hasLos==1:  self.polPhases = self.polPhases[1:]     # Polarization Phases are for NLOS paths

            self.phases, self.delays, self.powers, self.aoas, self.zoas, self.aods, self.zods, self.bounces = \
                self.pathInfo.T
            self.bounces = np.int32(self.bounces)
            if self.hasLos==1:
                self.losPhase, self.losDelay, self.losPower, self.losAoa, self.losZoa, self.losAod, self.losZod = \
                    self.pathInfo.T[:7,0]
                assert self.bounces[0] in [0,-1], "LOS bounce must be 0 if available!"
                if self.numPaths>1:
                    self.nlosPhases, self.nlosDelays, self.nlosPowers, self.nlosAoas, self.nlosZoas, self.nlosAods, \
                        self.nlosZods, self.nlosBounces = self.pathInfo.T[:,1:]
            else:
                # Every path is NLOS
                self.nlosPhases, self.nlosDelays, self.nlosPowers, self.nlosAoas, self.nlosZoas, self.nlosAods, \
                    self.nlosZods, self.nlosBounces = self.pathInfo.T

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
        indent : int
            Used internally to adjust the indentation of the printed info.
            
        title : str
            The title used for the information. By default, the text "TrjPoint Properties:" is used.

        getStr : bool
            If `True`, returns a string instead of printing it.
                    
        Returns
        -------
        str or None
            If ``getStr`` is `True`, this function returns a string containing information about the properties of 
            this class. Otherwise, nothing is returned (default).
        """
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"

        repStr += indent*' ' + f"  location:           {'  '.join('%6.2f'%(x) for x in self.xyz)} m\n"
        repStr += indent*' ' + f"  Distance to BS:     {self.bsDist:6.2f} m\n"
        if self.pathLoss>0:
            repStr += indent*' ' + f"  pathLoss:           {self.pathLoss}\n"
        repStr += indent*' ' + f"  LOS/NLOS:           {['No Paths', 'All NLOS', 'Has LOS path'][self.hasLos+1]}\n"
        repStr += indent*' ' + f"  numPaths:           {self.numPaths}\n"
        repStr += indent*' ' + f"  sampleNo:           {self.sampleNo}\n"
        repStr += indent*' ' + f"  time:               {self.time:.6f} sec\n"
        repStr += indent*' ' + f"  speed vector:       ({self.speed[0]:.3f},{self.speed[1]:.3f},{self.speed[2]:.3f}) m/s.\n"

        if self.hasLos==1:
            repStr += indent*' ' + "  Line-Of-Sight Path:\n"
            if self.pathIds is not None:
                repStr += indent*' ' + f"    Path ID:          {self.pathIds[0]}\n"
            repStr += indent*' ' + f"    Delay (ns):       {self.losDelay:.5f}\n"
            repStr += indent*' ' + f"    Power (dB):       {self.losPower:.5f}\n"
            repStr += indent*' ' + f"    Phases (Deg):     {int(self.losPhase)}\n"
            repStr += indent*' ' + f"    AOA (Deg):        {int(self.losAoa)}\n"
            repStr += indent*' ' + f"    ZOA (Deg):        {int(self.losZoa)}\n"
            repStr += indent*' ' + f"    AOD (Deg):        {int(self.losAod)}\n"
            repStr += indent*' ' + f"    ZOD (Deg):        {int(self.losZod)}\n"
    
        if self.numNlosPaths>0:
            repStr += indent*' ' + f"  Non-Line-Of-Sight Paths ({self.numNlosPaths}):\n"
            if self.pathIds is not None:
                pathIds = self.pathIds[1:] if self.hasLos == 1 else self.pathIds
                repStr += getMultiLineStr("  Path IDs        ", pathIds, indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  Delays (ns)     ", self.nlosDelays, indent, "%-5f", 5, numPerLine=12)
            repStr += getMultiLineStr("  Powers (dB)     ", self.nlosPowers, indent, "%-5f", 5, numPerLine=12)
            repStr += getMultiLineStr("  Phases (Deg)    ", np.round(self.nlosPhases), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  AOAs (Deg)      ", np.round(self.nlosAoas), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  ZOAs (Deg)      ", np.round(self.nlosZoas), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  AODs (Deg)      ", np.round(self.nlosAods), indent, "%-5d", 5, numPerLine=12)
            repStr += getMultiLineStr("  ZODs (Deg)      ", np.round(self.nlosZods), indent, "%-5d", 5, numPerLine=12)
            if self.bounces[0]!=-1:
                repStr += getMultiLineStr("  Bounces         ", self.nlosBounces, indent, "%-5d", 5, numPerLine=12)
                repStr += indent*' ' + "        1: Reflection, 2: Diffraction, 3: Scattering, 4: Transmission\n"

        if getStr: return repStr
        print(repStr)
        
    # ******************************************************************************************************************
    def matchPathInfo(self, nextPoint, maxDiff=1):         # Undocumented
        # This function matches the path information between this point and 'nextPoint'.
        # This function pairs the best matches in the path information and returns
        # a list of indices specifying the correct order of the path information
        # applied to 'nextPoint'. If c-th path at this point (self) matches n-th path in 'nextPoint',
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
        points : list
            A list of :py:class:`~neoradium.trjchan.TrjPoint` objects defining this trajectory.
            
        carrierFreq : float
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
                
            :cur: A read-only property returning the current :py:class:`~neoradium.trjchan.TrjPoint` on the trajectory. 
                This is the :py:class:`~neoradium.trjchan.TrjPoint` object at index ``curIdx``.
                
            :remainingPoints: A read-only property returning the remaining points from ``cur`` to the end of 
                the trajectory (including ``cur``).
                
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

        This class has a generator function (``__iter__``) which makes it easier to use in a loop. For example, the 
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
        **hasLos**, **numPaths**, **numNlosPaths**, **pathIds**, and **polPhases** are redirected to the corresponding
        property in the :py:class:`~neoradium.trjchan.TrjPoint` class for the current point (``cur``). For example, 
        ``traj.losZod`` is equivalent to ``traj.cur.losZod``.
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
        This function advances the current point in this trajectory.
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
        import matplotlib.pyplot as plt
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
        indent : int
            Used internally to adjust the indentation of the printed info.
            
        title : str
            The title used for the information. By default, the text "Trajectory Properties:" is used.

        getStr : bool
            If this is `True`, the function returns a string instead of printing the info. Otherwise, when this
            is `False` (default) the function prints the information.
            
        Returns
        -------
        str or None
            If ``getStr`` is `True`, this function returns a string containing information about the properties of 
            this class. Otherwise, nothing is returned (default).
        """
        repStr = "\n" if indent==0 else ""
        if self.isPointSet:
            repStr += indent*' ' + "PointSet Properties\n"
            repStr += indent*' ' + f"  No. of points:          {self.numPoints:,}\n"
            repStr += indent*' ' + f"  Carrier Frequency:      {freqStr(self.carrierFreq)}\n"
            repStr += indent*' ' + f"  Paths (Min, Avg, Max):  {self.minPaths}, {self.avgPaths:.2f}, {self.maxPaths}\n"
            repStr += indent*' ' + f"  Totally blocked:        {self.numBlockage:,}\n"
            repStr += indent*' ' + f"  LOS percentage:         {self.numLOS*100/self.numPoints:.2f}%\n"
        else:
            repStr += indent*' ' + title + "\n"
            repStr += indent*' ' + f"  start (x,y,z):          ({', '.join('%.2f'%(x) for x in self.points[0].xyz)})\n"
            repStr += indent*' ' + f"  No. of points:          {self.numPoints:,}\n"
            repStr += indent*' ' + f"  curIdx:                 {self.curIdx} ({self.curIdx*100/self.numPoints:.2f}%)\n"
            repStr += indent*' ' + f"  curSpeed:               {np.round(self.cur.speed,2)}\n"
            repStr += indent*' ' + f"  Total distance:         {self.totalDist:.2f} meters\n"
            repStr += indent*' ' + f"  Total time:             {self.time:.3f} seconds\n"
            repStr += indent*' ' + f"  Average Speed:          {self.totalDist/self.time:.3f} m/s\n"
            repStr += indent*' ' + f"  Carrier Frequency:      {freqStr(self.carrierFreq)}\n"
            repStr += indent*' ' + f"  Paths (Min, Avg, Max):  {self.minPaths}, {self.avgPaths:.2f}, {self.maxPaths}\n"
            repStr += indent*' ' + f"  Totally blocked:        {self.numBlockage:,}\n"
            repStr += indent*' ' + f"  LOS percentage:         {self.numLOS*100/self.numPoints:.2f}%\n"

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    def isPointSet(self):                                               # Undocumented
        # True means that this is a set of points used solely to create a dataset of channel matrices
        # with no temporal correlation. (See the DeepMimoData.chanMatAtPoints method)
        return self.points[-1].sampleNo==1          # Undocumented.

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
    @property
    def totalDist(self):                                                # Documented above in the __init__ function.
        # Calculating this lazily
        if self.dist == 0:
            for i in range(1,self.numPoints):
               self.dist += np.sqrt(np.square(self.points[i-1].xyz-self.points[i].xyz).sum())
        return self.dist
                
    # ******************************************************************************************************************
    def __getattr__(self, attrName):                                    # Documented above in the __init__ function.
        # Get these properties from the 'cur' object
        if attrName not in TrjPoint.pathParamNames:
            raise AttributeError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, attrName))
        return getattr(self.cur, attrName)

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
            
    * For the initial phase values used to calculate the cross-polarization matrix, we create reproducible random 
      phases for each path. These phase values remain unchanged as long as the path exists along the trajectory. See
      the `polPhases` parameter of the :py:class:`~neoradium.trjchan.TrjPoint` class for more information
    
    All API functions used in most typical use cases are explained in the documentation of 
    the :py:class:`~neoradium.channelmodel.ChannelModel` class.
    
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

                :filterLen: The length of the channel filter. The default is 16 samples.
                
                :delayQuantSize: The size of delay fraction quantization for the channel filter. The default is 64.
                
                :stopBandAtten: The stop-band attenuation value (in dB) used by the channel filter. The default is 80 dB.
                
                :txAntenna: The transmitter antenna, which is an instance of 
                    :py:class:`~neoradium.antenna.AntennaElement`, :py:class:`~neoradium.antenna.AntennaPanel`,
                    or :py:class:`~neoradium.antenna.AntennaArray` class. If not specified, a single antenna element is
                    automatically created by default.
                
                :rxAntenna: The receiver antenna, which is an instance of 
                    :py:class:`~neoradium.antenna.AntennaElement`, :py:class:`~neoradium.antenna.AntennaPanel`,
                    or :py:class:`~neoradium.antenna.AntennaArray` class. If not specified, a single antenna element is
                    automatically created by default.
                    
                :txOrientation: The orientation of the transmitter antenna. This is a list of 3 angle values in degrees 
                    for the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901 Section 7.1.3** for more
                    information.

                :rxOrientation: The orientation of the receiver antenna. This is a list of 3 angle values in degrees for 
                    the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901 Section 7.1.3** for more
                    information.

                :xPolPower: The cross-polarization power in dB. The default is 10 dB. It is defined as 
                    :math:`X=10 log_{10} \kappa^{RT}` where :math:`\kappa^{RT}` is the Cross-Polarization Ratio (XPR). 
                    In the current implementation, this value is used for all paths.

                                        
        **Other Properties:**
        
            :nrNt: A tuple of the form ``(nr,nt)``, where ``nr`` and ``nt`` are the number of receiver and transmitter 
                antenna elements, respectively.
                
            :pathPowers: This read-only property returns the path powers for all paths between the base station and 
                the UE at the current position on the current trajectory. 

            :pathDelays: This read-only property returns the path delays for all paths between the base 
                station and the UE at the current position on the trajectory. If the parameter ``normalizeDelays``
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
            **hasLos**, **numPaths**, **numNlosPaths**, **pathIds**, and **polPhases** are redirected to the 
            corresponding property in the :py:class:`~neoradium.trjchan.Trajectory` class (``trajectory``). For 
            example ``channel.pathIds`` is equivalent to ``channel.trajectory.pathIds`` which is equivalent to
            ``channel.trajectory.cur.pathIds``.
        """
        super().__init__(bwp, **kwargs)
        self.trajectory = trajectory

        self.carrierFreq = trajectory.carrierFreq   # Set carrierFreq based on the value from ray-tracing scenario
        
        self.dopplerShift = trajectory.maxSpeed * self.carrierFreq/299792458 # Calculate based on max speed (Not used)
        self.kFactor = None                         # K-Factor and K-Factor scaling apply to stochastic models (CDL/TDL)

        self.txAntenna = kwargs.get('txAntenna', AntennaElement())  # Transmitter AntennaArray/AntennaPanel object
        self.rxAntenna = kwargs.get('rxAntenna', AntennaElement())  # Receiver AntennaArray/AntennaPanel object

        # Orientation of TX and RX antenna arrays.
        # NOTE:
        #   To point an RX/TX antenna to LOS angles of arrival/departure use: 𝛼=aoa[0]/aod[0], 𝛃=zoa[0]/zod[0]-90° 𝛄=0
        self.txOrientation = kwargs.get('txOrientation', [0,0,0])   # Orientation of TX Antenna (Degrees)
        self.rxOrientation = kwargs.get('rxOrientation', [0,0,0])   # Orientation of RX Antenna (Degrees)
        self.xPolPower = kwargs.get('xPolPower', 10.0)              # CrossPolarization Power in dB.
        self.normalizeDelays = kwargs.get('normalizeDelays', True ) # Normalize delays so that the 1st one is zero
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
        return self.trajectory.powers

    # ******************************************************************************************************************
    @property                   # This property is already documented above in the __init__ function.
    def pathDelays(self):
        if self.totalBlockage:      return None
        delays = self.trajectory.delays

        # Normalizing the delays. See "Step 3" in 3GPP TR 38.901 section 8.4 for more details.
        if self.normalizeDelays:    return delays - delays[0]
        return delays

    # ******************************************************************************************************************
    def __getattr__(self, attrName):    # Documented above in the __init__ function.
        # Get these properties from the 'cur' object
        if attrName not in (TrjPoint.pathParamNames +
                            ['isPointSet', 'numPoints', 'remainingPoints', 'cur', 'time']):
            raise AttributeError("Class '%s' does not have any property named '%s'!"%(self.__class__.__name__, attrName))
        return getattr(self.trajectory, attrName)

    # ******************************************************************************************************************
    def restart(self, restartRanGen=False, applyToBwp=True):
        r"""
        This method first resets the current trajectory to its starting position and then calls the base class 
        :py:meth:`~neoradium.channelmodel.ChannelModel.restart`.

        Parameters
        ----------
        restartRanGen : bool
            Ignored for the :py:class:`~neoradium.trjchan.TrjChannel` class as it uses deterministic information only.

        applyToBwp : bool
            If set to `True` (the default), this function restarts the Bandwidth Part associated with this channel 
            model. Otherwise, the Bandwidth Part’s state remains unchanged.
        """
        self.trajectory.restart()
        super().restart(restartRanGen, applyToBwp)

    # ******************************************************************************************************************
    def goNext(self, applyToBwp=True):
        r"""
        This method is called after each application of the channel to a signal. It updates the timing information of
        the trajectory and the channel model, preparing it for the next application to the input signal. It is assumed 
        that the channel is applied to a single slot of the signal at each application (either in the time or 
        frequency domain).
        
        Parameters
        ----------
        applyToBwp : bool
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
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "TrjChannel Properties:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns
            the information in a string. Otherwise, nothing is returned.
        """
        # Do not call the parent class's print, because some of the parameters do not apply to this subclass
        if title is None:   title = "TrjChannel Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  carrierFreq:              {freqStr(self.carrierFreq)}\n"
        repStr += indent*' ' + f"  normalizeGains:           {str(self.normalizeGains)}\n"
        repStr += indent*' ' + f"  normalizeOutput:          {str(self.normalizeOutput)}\n"
        repStr += indent*' ' + f"  normalizeDelays:          {str(self.normalizeDelays)}\n"
        repStr += indent*' ' + f"  xPolPower:                {self.xPolPower:.2f} (dB)\n"
        repStr += indent*' ' + f"  filterLen:                {self.filterLen} samples\n"
        repStr += indent*' ' + f"  delayQuantSize:           {self.delayQuantSize}\n"
        repStr += indent*' ' + f"  stopBandAtten:            {self.stopBandAtten} dB\n"
        repStr += indent*' ' + f"  dopplerShift:             {freqStr(self.dopplerShift)}\n"
        repStr += indent*' ' + f"  coherenceTime:            {self.coherenceTime*1000:.3f} milliseconds\n"

        repStr += self.txAntenna.print(indent+2, "TX Antenna:", True)
        if np.any(self.txOrientation):
            repStr += indent*' ' + "    Orientation (𝛼,𝛃,𝛄):     %s°\n"%("° ".join(str(a) for a in self.txOrientation))
        repStr += self.rxAntenna.print(indent+2, "RX Antenna:", True)
        if np.any(self.rxOrientation):
            repStr += indent*' ' + "    Orientation (𝛼,𝛃,𝛄):     %s°\n"%("° ".join(str(a) for a in self.rxOrientation))
        repStr += self.trajectory.print(indent+2, "Trajectory:", True)

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def prepareForNextSlot(self):
        # Raise an exception if at the end of the trajectory. Otherwise, just call base class's function.
        if self.trajectory.remainingPoints<=0:      raise ValueError("Reached end of trajectory!")
        if self.nextSlotStart > self.curSlotStart:  return  # The parameters for this slot have already been initialized
        super().prepareForNextSlot()

    # ******************************************************************************************************************
    def getPathGains(self):
        r"""
        Calculates the gains for Line-Of-Sight (LOS) and Non-Line-Of-Sight (NLOS) paths separately and combines the 
        results before returning the gains between every RX/TX antenna pair, for every path, at every time symbol. 
        
        Returns
        -------
        NumPy array or None
            This function returns a 4-D complex tensor of shape ``Ns x Nr x Nt x Np``, where ``Ns`` represents the 
            number of time symbols, ``Nr`` and ``Nt`` indicate the number of receiver and transmitter antennas
            respectively, and ``Np`` denotes the number of paths between the transmitter and the receiver at its 
            current location along its trajectory. If the UE is at a `total blockage` point, this function returns 
            `None`.
        """
        # This function creates gain information for the next 'ns' time symbols, which is the number of time symbols
        # per slot plus 1. We always create gains for one more symbol (The first symbol of the next slot)
        if self.numPaths == 0:      return None                             # Total Blockage
        if self.numNlosPaths>0:
            if self.hasLos==0:      return self.getGains(los=False)         # Only NLOS, Shape:  ns x nr x nt x n
            return np.concatenate((self.getGains(los=True),
                                   self.getGains(los=False)),axis=3)        # LOS & NLOS, Shape: ns x nr x nt x n
        return self.getGains(los=True)                                      # Only LOS, Shape:   ns x nr x nt x 1

    # ******************************************************************************************************************
    def getGains(self, los):
        # This function calculates the gains matrix for LOS and NLOS paths (based on the `los` parameter).
        # It is called by the `getPathGains` function and returns an n x ns x nr x nt complex NumPy array.
        if los:       # ns = 1, n = 1
            phiA, thetaA = np.array([self.losAoa, self.losZoa])[:, None, None]      # Both in degrees.    ns x n
            phiD, thetaD = np.array([self.losAod, self.losZod])[:, None, None]      # Both in degrees.    ns x n
            pN = toLinear(np.array([[self.losPower]]))                      # Convert from dB to linear   ns x n
        else:           # ns = 1, n = number of NLOS paths
            phiA, thetaA = np.array([self.nlosAoas, self.nlosZoas])[:,None,:]       # Both in degrees.    ns x n
            phiD, thetaD = np.array([self.nlosAods, self.nlosZods])[:,None,:]       # Both in degrees.    ns x n
            pN = toLinear(np.array([self.nlosPowers]))                      # Convert from dB to linear   ns x n
            polPhases = 0 if self.polPhases is None else self.polPhases

        nr, nt = self.nrNt
        # Get the RX field component and RX location component in:
        #       NLOS: TR 38.901 – Eq. 7.5-28 (the 1st and 4th terms)
        #       LOS:  TR 38.901 – Eq. 7.5-29 (the 1st and 5th terms)
        fieldRx, locRx = self.rxAntenna.getElementsFields(thetaA, phiA, self.rxOrientation) # (ns,n,nr,2), (ns,n,nr)

        # Get the TX field component and TX location component in:
        #       NLOS: TR38.901 - Eq. 7.5-28 (The 3rd and 5th terms)
        #       LOS:  TR38.901 - Eq. 7.5-29 (The 3rd and 6th terms)
        fieldTx, locTx = self.txAntenna.getElementsFields(thetaD, phiD, self.txOrientation) # (ns,n,nt,2), (ns,n,nt)

        if los:
            # Get the polarization matrix term in TR 38.901 – Eq. 7.5-29 (the 2nd and 4th terms combined).
            polMat = np.float64([[1,0],[0,-1]])[None,None,:,:]                                      # (1,1,2,2)
        else:
            # STEP-3 (See Step-3 in TR38.901 - 7.7.1) ------------------------------------------------------------------
            # Get the cross-polarization power ratio:
            kappa = toLinear(self.xPolPower)    # Cross-Polarization Ratio (XPR)
            # Get the polarization matrix term in TR38.901 - Eq. 7.5-28 (The 2nd term)
            polPhases = 0 if self.polPhases is None else self.polPhases
            polMat = np.exp(1j*polPhases) * np.sqrt([[1, 1/kappa], [1/kappa, 1]])[None,:,:]         # n x 2 x 2
            polMat = polMat[None,...]                                                               # 1 x n x 2 x 2

        doppler = self.getDopplerFactor(thetaA, phiA)                                               # ns x n

        # First fieldRx x polMat x fieldTx
        h = fieldRx @ polMat @ np.swapaxes(fieldTx,-2,-1)                           # Shape: ns x n x nr x nt
        h = h * locRx[:,:,:,None] * locTx[:,:,None,:]           # Location factors,   Shape: ns x n x nr x nt
        h = h * doppler[:,:,None,None]                          # Apply the doppler,  Shape: ns x n x nr x nt
        h = h * np.sqrt(pN)[:,:,None,None]                      # Apply the scaling,  Shape: ns x n x nr x nt
        return np.transpose(h,(0,2,3,1))                                            # Shape: ns x nr x nt x n

    # ******************************************************************************************************************
    def getDopplerFactor(self, theta, phi):                 # Undocumented
        # This function calculates the doppler term (the last term) in TR38.901 - Eq. 7.5-28 and 7.5-29
        # 'r' is the unit vector which points to the arrival direction of each path.
        𝜃, 𝜑 = np.deg2rad(theta), np.deg2rad(phi)
        r = np.array([ np.sin(𝜃)*np.cos(𝜑), np.sin(𝜃)*np.sin(𝜑), np.cos(𝜃) ])    # Shape: 3 x ns x n
        v = self.trajectory.cur.speed.reshape(3,1,1)                            # 3D Speed vector, Shape: 3, 1, 1
        waveLen = 299792458/self.carrierFreq                                    # Speed of light: 299792458 m/s
        dopplerShift = ((r*v).sum(0)/waveLen)                                   # Doppler Shift (Hz), Shape: ns x n

        chanTimes = self.chanGainSamples/self.sampleRate
        return np.exp(2j * np.pi * chanTimes[:, None] * dopplerShift)           # Shape: ns x n

    # ******************************************************************************************************************
    def getChanSeqGen(self, seqPeriod=1, seqLen=10, maxNumSeq=np.inf,
                      chanCallback=None, seqCallback=None):
        r"""
        This function returns an iterable object that generates sequences of channel matrices based on the given parameters.

        Refer to the notebook :doc:`../Playground/Notebooks/RayTracing/ChannelSequences` for an example of
        using this method.
        
        Parameters
        ----------
        seqPeriod : int            
            The sampling period of channel matrices along the trajectory. The default value of ``1`` means all 
            channel matrices are included in the sequence. For instance, if this is set to ``3``, every third 
            channel matrix (i.e., one out of every three slots) is included in the sequence.
            
        seqLen : int
            The length of the returned sequences. The default is ``10``.

        maxNumSeq : int or float           
            The maximum number of sequences to generate. By default, it is set to ``np.inf``, indicating no additional
            limit on the number of sequences. In this case, channel sequences are generated until the end of 
            the trajectory.
    
        chanCallback : function
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
                            
        seqCallback : function
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

    
        The following example creates sequences of effective channel matrices (which include the precoding effect)
        that do not contain any point with total blockage.
        
        .. code-block:: python
        
            from neoradium import Carrier, PDSCH

            carrier = Carrier(numRbs=25, spacing=30) # Carrier with 25 RBs, 30 kHz subcarrier spacing
            channel = TrjChannel(carrier.curBwp, trajectory,
                                 txAntenna = AntennaPanel([2,4], polarization="x"),       # 8 TX antennas
                                 rxAntenna = AntennaPanel([1,2], polarization="x"))       # 2 RX antennas
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
