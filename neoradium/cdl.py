# Copyright (c) 2024 InterDigital AI Lab
"""
This module implements the :py:class:`~neoradium.cdl.CdlChannel` class which
encapsulates the Clustered Delay Line (CDL) channel model functionality.
"""
# ****************************************************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 11/30/2023    Shahab Hamidi-Rad       Completed the documentation
# ****************************************************************************************************************************************************

import numpy as np
import scipy.io
from scipy.signal import lfilter

from .antenna import AntennaPanel
from .channel import ChannelBase, ChannelFilter
from .utils import getMultiLineStr
from .random import random

# This file is based on 3GPP TR 38.901 V17.0.0 (2022-03) available at:
#   https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173

# ****************************************************************************************************************************************************
# Note: All angles are received in degrees but used and kept internally as radians
def toRadian(angle):    return (None if angle is None else np.float64(angle)*np.pi/180.0)
def toDegrees(angle):   return (None if angle is None else np.float64(angle)*180.0/np.pi)

# ****************************************************************************************************************************************************
clusterInfo = {
                'A':   # TR38.901 - Table 7.7.1-1 CDL-A
                   ## Delay     Power      AOD       AOA      ZOD     ZOA         Cluster #
                   [[ 0.0000,   -13.4,     -178.1,   51.3,    50.2,   125.4 ],    #  1
                    [ 0.3819,   0,         -4.2,     -152.7,  93.2,   91.3  ],    #  2
                    [ 0.4025,   -2.2,      -4.2,     -152.7,  93.2,   91.3  ],    #  3
                    [ 0.5868,   -4,        -4.2,     -152.7,  93.2,   91.3  ],    #  4
                    [ 0.4610,   -6,        90.2,     76.6,    122,    94    ],    #  5
                    [ 0.5375,   -8.2,      90.2,     76.6,    122,    94    ],    #  6
                    [ 0.6708,   -9.9,      90.2,     76.6,    122,    94    ],    #  7
                    [ 0.5750,   -10.5,     121.5,    -1.8,    150.2,  47.1  ],    #  8
                    [ 0.7618,   -7.5,      -81.7,    -41.9,   55.2,   56    ],    #  9
                    [ 1.5375,   -15.9,     158.4,    94.2,    26.4,   30.1  ],    #  10
                    [ 1.8978,   -6.6,      -83,      51.9,    126.4,  58.8  ],    #  11
                    [ 2.2242,   -16.7,     134.8,    -115.9,  171.6,  26    ],    #  12
                    [ 2.1718,   -12.4,     -153,     26.6,    151.4,  49.2  ],    #  13
                    [ 2.4942,   -15.2,     -172,     76.6,    157.2,  143.1 ],    #  14
                    [ 2.5119,   -10.8,     -129.9,   -7,      47.2,   117.4 ],    #  15
                    [ 3.0582,   -11.3,     -136,     -23,     40.4,   122.7 ],    #  16
                    [ 4.0810,   -12.7,     165.4,    -47.2,   43.3,   123.2 ],    #  17
                    [ 4.4579,   -16.2,     148.4,    110.4,   161.8,  32.6  ],    #  18
                    [ 4.5695,   -18.3,     132.7,    144.5,   10.8,   27.2  ],    #  19
                    [ 4.7966,   -18.9,     -118.6,   155.3,   16.7,   15.2  ],    #  20
                    [ 5.0066,   -16.6,     -154.1,   102,     171.7,  146   ],    #  21
                    [ 5.3043,   -19.9,     126.5,    -151.8,  22.7,   150.7 ],    #  22
                    [ 9.6586,   -29.7,     -56.2,    55.2,    144.9,  156.1 ]],   #  23

                'B':   # TR38.901 - Table 7.7.1-2 CDL-B
                   ## Delay     Power      AOD       AOA      ZOD     ZOA         Cluster #
                   [[ 0.0000,   0,         9.3,      -173.3,  105.8,  78.9 ],     #  1
                    [ 0.1072,   -2.2,      9.3,      -173.3,  105.8,  78.9 ],     #  2
                    [ 0.2155,   -4,        9.3,      -173.3,  105.8,  78.9 ],     #  3
                    [ 0.2095,   -3.2,      -34.1,    125.5,   115.3,  63.3 ],     #  4
                    [ 0.2870,   -9.8,      -65.4,    -88.0,   119.3,  59.9 ],     #  5
                    [ 0.2986,   -1.2,      -11.4,    155.1,   103.2,  67.5 ],     #  6
                    [ 0.3752,   -3.4,      -11.4,    155.1,   103.2,  67.5 ],     #  7
                    [ 0.5055,   -5.2,      -11.4,    155.1,   103.2,  67.5 ],     #  8
                    [ 0.3681,   -7.6,      -67.2,    -89.8,   118.2,  82.6 ],     #  9
                    [ 0.3697,   -3,        52.5,     132.1,   102.0,  66.3 ],     #  10
                    [ 0.5700,   -8.9,      -72,      -83.6,   100.4,  61.6 ],     #  11
                    [ 0.5283,   -9,        74.3,     95.3,    98.3,   58.0 ],     #  12
                    [ 1.1021,   -4.8,      -52.2,    103.7,   103.4,  78.2 ],     #  13
                    [ 1.2756,   -5.7,      -50.5,    -87.8,   102.5,  82.0 ],     #  14
                    [ 1.5474,   -7.5,      61.4,     -92.5,   101.4,  62.4 ],     #  15
                    [ 1.7842,   -1.9,      30.6,     -139.1,  103.0,  78.0 ],     #  16
                    [ 2.0169,   -7.6,      -72.5,    -90.6,   100.0,  60.9 ],     #  17
                    [ 2.8294,   -12.2,     -90.6,    58.6,    115.2,  82.9 ],     #  18
                    [ 3.0219,   -9.8,      -77.6,    -79.0,   100.5,  60.8 ],     #  19
                    [ 3.6187,   -11.4,     -82.6,    65.8,    119.6,  57.3 ],     #  20
                    [ 4.1067,   -14.9,     -103.6,   52.7,    118.7,  59.9 ],     #  21
                    [ 4.2790,   -9.2,      75.6,     88.7,    117.8,  60.1 ],     #  22
                    [ 4.7834,   -11.3,     -77.6,    -60.4,   115.7,  62.3 ]],    #  23
  
                'C':   # TR38.901 - Table 7.7.1-3 CDL-C
                   ## Delay     Power      AOD       AOA      ZOD     ZOA         Cluster #
                   [[ 0,        -4.4,      -46.6,    -101,    97.2,   87.6  ],    # 1
                    [ 0.2099,   -1.2,      -22.8,    120,     98.6,   72.1  ],    # 2
                    [ 0.2219,   -3.5,      -22.8,    120,     98.6,   72.1  ],    # 3
                    [ 0.2329,   -5.2,      -22.8,    120,     98.6,   72.1  ],    # 4
                    [ 0.2176,   -2.5,      -40.7,    -127.5,  100.6,  70.1  ],    # 5
                    [ 0.6366,   0,         0.3,      170.4,   99.2,   75.3  ],    # 6
                    [ 0.6448,   -2.2,      0.3,      170.4,   99.2,   75.3  ],    # 7
                    [ 0.6560,   -3.9,      0.3,      170.4,   99.2,   75.3  ],    # 8
                    [ 0.6584,   -7.4,      73.1,     55.4,    105.2,  67.4  ],    # 9
                    [ 0.7935,   -7.1,      -64.5,    66.5,    95.3,   63.8  ],    # 10
                    [ 0.8213,   -10.7,     80.2,     -48.1,   106.1,  71.4  ],    # 11
                    [ 0.9336,   -11.1,     -97.1,    46.9,    93.5,   60.5  ],    # 12
                    [ 1.2285,   -5.1,      -55.3,    68.1,    103.7,  90.6  ],    # 13
                    [ 1.3083,   -6.8,      -64.3,    -68.7,   104.2,  60.1  ],    # 14
                    [ 2.1704,   -8.7,      -78.5,    81.5,    93.0,   61.0  ],    # 15
                    [ 2.7105,   -13.2,     102.7,    30.7,    104.2,  100.7 ],    # 16
                    [ 4.2589,   -13.9,     99.2,     -16.4,   94.9,   62.3  ],    # 17
                    [ 4.6003,   -13.9,     88.8,     3.8,     93.1,   66.7  ],    # 18
                    [ 5.4902,   -15.8,     -101.9,   -13.7,   92.2,   52.9  ],    # 19
                    [ 5.6077,   -17.1,     92.2,     9.7,     106.7,  61.8  ],    # 20
                    [ 6.3065,   -16,       93.3,     5.6,     93.0,   51.9  ],    # 21
                    [ 6.6374,   -15.7,     106.6,    0.7,     92.9,   61.7  ],    # 22
                    [ 7.0427,   -21.6,     119.5,    -21.9,   105.2,  58    ],    # 23
                    [ 8.6523,   -22.8,     -123.8,   33.6,    107.8,  57    ]],   # 24

                'D':   # TR38.901 - Table 7.7.1-4 CDL-D
                   ## Delay     Power      AOD       AOA      ZOD     ZOA         Cluster #
                   [[ 0,        -0.2,      0,        -180,    98.5,   81.5 ],     # 1    Specular(LOS path)
                    [ 0,        -13.5,     0,        -180,    98.5,   81.5 ],     # 1    Laplacian
                    [ 0.035,    -18.8,     89.2,     89.2,    85.5,   86.9 ],     # 2
                    [ 0.612,    -21,       89.2,     89.2,    85.5,   86.9 ],     # 3
                    [ 1.363,    -22.8,     89.2,     89.2,    85.5,   86.9 ],     # 4
                    [ 1.405,    -17.9,     13,       163,     97.5,   79.4 ],     # 5
                    [ 1.804,    -20.1,     13,       163,     97.5,   79.4 ],     # 6
                    [ 2.596,    -21.9,     13,       163,     97.5,   79.4 ],     # 7
                    [ 1.775,    -22.9,     34.6,     -137,    98.5,   78.2 ],     # 8
                    [ 4.042,    -27.8,     -64.5,    74.5,    88.4,   73.6 ],     # 9
                    [ 7.937,    -23.6,     -32.9,    127.7,   91.3,   78.3 ],     # 10
                    [ 9.424,    -24.8,     52.6,     -119.6,  103.8,  87   ],     # 11
                    [ 9.708,    -30.0,     -132.1,   -9.1,    80.3,   70.6 ],     # 12
                    [ 12.525,   -27.7,     77.2,     -83.8,   86.5,   72.9 ]],    # 13

                'E':   # TR38.901 - Table 7.7.1-5 CDL-E
                   ## Delay     Power      AOD       AOA      ZOD     ZOA         Cluster #
                   [[ 0.000,    -0.03,     0,        -180,    99.6,   80.4 ],    # 1    Specular(LOS path)
                    [ 0.000,    -22.03,    0,        -180,    99.6,   80.4 ],    # 1    Laplacian
                    [ 0.5133,   -15.8,     57.5,     18.2,    104.2,  80.4 ],    # 2
                    [ 0.5440,   -18.1,     57.5,     18.2,    104.2,  80.4 ],    # 3
                    [ 0.5630,   -19.8,     57.5,     18.2,    104.2,  80.4 ],    # 4
                    [ 0.5440,   -22.9,     -20.1,    101.8,   99.4,   80.8 ],    # 5
                    [ 0.7112,   -22.4,     16.2,     112.9,   100.8,  86.3 ],    # 6
                    [ 1.9092,   -18.6,     9.3,      -155.5,  98.8,   82.7 ],    # 7
                    [ 1.9293,   -20.8,     9.3,      -155.5,  98.8,   82.7 ],    # 8
                    [ 1.9589,   -22.6,     9.3,      -155.5,  98.8,   82.7 ],    # 9
                    [ 2.6426,   -22.3,     19,       -143.3,  100.8,  82.9 ],    # 10
                    [ 3.7136,   -25.6,     32.7,     -94.7,   96.4,   88   ],    # 11
                    [ 5.4524,   -20.2,     0.5,      147,     98.9,   81   ],    # 12
                    [ 12.0034,  -29.8,     55.9,     -36.2,   95.6,   88.6 ],    # 13
                    [ 20.6419,  -29.2,     57.6,     -26,     104.6,  78.3 ]],   # 14
               }

# ****************************************************************************************************************************************************
perClusterParams = {#           C_ASD  C_ASA  C_ZSD  C_ZSA   XPR
                        'A':  ([5,     11,    3,     3],     10),     # TR38.901 - Table 7.7.1-1 CDL-A
                        'B':  ([10,    22,    3,     7],     8),      # TR38.901 - Table 7.7.1-2 CDL-B
                        'C':  ([2,     15,    3,     7],     7),      # TR38.901 - Table 7.7.1-3 CDL-C
                        'D':  ([5,     8,     3,     3],     11),     # TR38.901 - Table 7.7.1-4 CDL-D
                        'E':  ([5,     11,    3,     7],     8),      # TR38.901 - Table 7.7.1-5 CDL-E
                   }

# ****************************************************************************************************************************************************
# TR38.901 - Table 7.5-3: Ray offset angles within a cluster, given for rms angle spread normalized to 1
rayOffsets = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
              0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551]

# ****************************************************************************************************************************************************
class CdlChannel(ChannelBase):
    r"""
    This class implements the Clustered Delay Line (CDL) channel model based
    on **3GPP TR 38.901**. It is derived from the
    :py:class:`~neoradium.channel.ChannelBase` class.
    
    All of API functions used in most typical use cases are explained in the
    documentation of :py:class:`~neoradium.channel.ChannelBase` class.
    
    The typical use case involves instantiating a :py:class:`CdlChannel` object
    and then calling functions such as
    :py:meth:`~neoradium.channel.ChannelBase.getChannelMatrix`,
    :py:meth:`~neoradium.channel.ChannelBase.applyToSignal`,
    :py:meth:`~neoradium.channel.ChannelBase.applyToGrid`, etc.
    """
    # ************************************************************************************************************************************************
    def __init__(self, profile='A', **kwargs):
        r"""
        Parameters
        ----------
        profile : str or None (default: 'A')
            The CDL profile. It can be one of 'A', 'B', 'C', 'D', 'E',
            or ``None``. See **3GPP TR 38.90 section 7.7.1** for more
            information. Use ``None`` to indicate a customized version of
            CDL channel (See :ref:`Customizing CDL Model <CustomizingCDL>`).

        kwargs : dict
            A set of optional arguments. Please refer to
            :py:class:`~neoradium.channel.ChannelBase` for a list of inherited
            parameters. Here is a list of additional parameters specific to
            ``CdlChannel``.

                :ueDirAZ: This is a list of 2 angles for the Azimuth and Zenith
                    of the UE's direction of movement in degrees. The default
                    is [0, 90] which indicates moving along the x-axis. In this
                    version, the base station is assumed to be stationary.
                    
                :txAntenna: The transmitter antenna. This must be an instance of
                    :py:class:`~neoradium.antenna.AntennaPanel` or
                    :py:class:`~neoradium.antenna.AntennaArray` class.
                
                :rxAntenna: The receiver antenna. This must be an instance of
                    :py:class:`~neoradium.antenna.AntennaPanel` or
                    :py:class:`~neoradium.antenna.AntennaArray` class.
                    
                :txOrientation: The orientation of transmitter antenna. This
                    is a list of 3 angle values in degrees for the *bearing* angle
                    :math:`\alpha`, *down-tilt* angle :math:`\beta`, and *slant* angle
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP
                    TR 38.901 Section 7.1.3** for more information.

                :rxOrientation: The orientation of receiver antenna. This
                    is a list of 3 angle values in degrees for the *bearing* angle
                    :math:`\alpha`, *down-tilt* angle :math:`\beta`, and *slant* angle
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP
                    TR 38.901 Section 7.1.3** for more information.

                :xPolPower: The Cross Polarization Power in db. The default is 10db.
                    For more details please refer to **Step 3 in 3GPP TR 38.901 Section
                    7.7.1**.

                :angleScaling: The :ref:`Angle Scaling <AngleScaling>` parameters. If
                    specified, it must be a tuple of 2 numpy arrays.
                    
                    The first item specifies the angle scaling mean values. It is a
                    1-D numpy array of 4 values corresponding to the *Azimuth of
                    Departure*, *Azimuth of Arrival*, *Zenith of Departure*, and
                    *Zenith of Arrival* angles.
                    
                    The second item specifies the RMS angle spread values. It is a
                    1-D numpy array of 4 values corresponding to the *Azimuth of
                    Departure*, *Azimuth of Arrival*, *Zenith of Departure*, and
                    *Zenith of Arrival* angles. If this is set to ``None`` (the
                    default), the *Angle Scaling* is disabled. For more information
                    please see :ref:`Angle Scaling <AngleScaling>` below.
                    
                    If this is set to ``None`` (the default), the *Angle Scaling*
                    is disabled.
                    
                :pathDelays: Use this to customize or override the path delays which
                    by default are set based on the CDL channel model as defined in
                    **3GPP TR 38.901**. You don't need to specify this parameter for
                    most use cases. See :ref:`Customizing CDL Model <CustomizingCDL>`
                    below for more information.
                
                :pathPowers: Use this to customize or override the path powers which
                    by default are set based on the CDL channel model as defined in
                    **3GPP TR 38.901**. You don't need to specify this parameter for
                    most use cases. See :ref:`Customizing CDL Model <CustomizingCDL>`
                    below for more information.

                :aods: Use this to customize or override the Azimuth of Departure angles
                    which by default are set based on the CDL channel model as defined
                    in **3GPP TR 38.901**. You don't need to specify this parameter for
                    most use cases. See :ref:`Customizing CDL Model <CustomizingCDL>`
                    below for more information.

                :aoas: Use this to customize or override the Azimuth of Arrival angles
                    which by default are set based on the CDL channel model as defined
                    in **3GPP TR 38.901**. You don't need to specify this parameter for
                    most use cases. See :ref:`Customizing CDL Model <CustomizingCDL>`
                    below for more information.

                :zods: Use this to customize or override the Zenith of Departure angles
                    which by default are set based on the CDL channel model as defined
                    in **3GPP TR 38.901**. You don't need to specify this parameter for
                    most use cases. See :ref:`Customizing CDL Model <CustomizingCDL>`
                    below for more information.

                :zoas: Use this to customize or override the Zenith of Arrival angles
                    which by default are set based on the CDL channel model as defined
                    in **3GPP TR 38.901**. You don't need to specify this parameter for
                    most use cases. See :ref:`Customizing CDL Model <CustomizingCDL>`
                    below for more information.

                :angleSpreads: Use this to customize or override the RMS Angle spread
                    (in degrees) which is used to normalized angles. This a list of 4
                    values corresponding to the *Azimuth of Departure*, *Azimuth of
                    Arrival*, *Zenith of Departure*, and *Zenith of Arrival* angles.
                    These values by default are set based on the CDL channel model as
                    defined in **3GPP TR 38.901**. You don't need to specify this
                    parameter for most use cases. See
                    :ref:`Customizing CDL Model <CustomizingCDL>` below for more
                    information.
                    
                    Please note that this should not be confused with angle spread
                    values used for angle scaling.
        
                :hasLos: Use this to customize or override the ``hasLos`` property of
                    this channel model which by default is set based on the CDL channel
                    model as defined in **3GPP TR 38.901**. You don't need to specify
                    this parameter for most use cases. See
                    :ref:`Customizing CDL Model <CustomizingCDL>` below for more
                    information.
                    
                :kFactorLos: Use this when customizing the CDL model. This is the K-Factor
                    ratio (in dB) for the LOS cluster (1st cluster). You don't need to specify
                    this parameter for most use cases. See
                    :ref:`Customizing CDL Model <CustomizingCDL>` below for more
                    information. When customizing the CDL model, this value by default is set
                    to the difference of path powers (in dB) for the first and second clusters.


        **Other Properties:**
        
        All of the parameters mentioned above are directly available. Please also refer to
        the :py:class:`~neoradium.channel.ChannelBase` class for a list of inherited
        properties. Here is a list of additional properties specific to this class.
        
            :nrNt: A tuple of the form ``(nr,nt)``, where ``nr`` and ``nt`` are the
                number receiver and transmitter antenna elements correspondingly.
            
            :rayCoupling: This property is used internally for ray coupling. See **Step 2
                in 3GPP TR 38.901 Section 7.7.1** for more details.
                
            :initialPhases: The random initial phases used when creating channel gains.
                See **Step 10 in 3GPP TR 38.901 Section 7.5**  for more details.

        Note
        ----
        All of the angle values are provided to this class in degrees. Internally, this
        class uses radian values for all the calculations. So, when you access any of the
        angle values such as ``aods``, ``aoas``, ``zods``, and ``zoas``, remember that
        they are in radians.


        .. _AngleScaling:
        
        **Angle Scaling:**

        If ``angleScaling`` is set to ``None``, the angle scaling is disabled.
        Otherwise, angle scaling is applied to all angles of arrival and departure
        for all clusters according to **3GPP TR 38.901 Section 7.7.5.1 and Annex A**.


        .. _CustomizingCDL:
        
        **Customizing CDL Model:**

        There are two different ways to customize the CDL model:
        
        a) You can choose one of the predefined CDL profiles (A, B, C, D, or E) and then
           modify the parameters of the model by passing in additional information. For
           example you can choose CDL-B model and then pass your own path delays to
           override the path delays specified in the standard.
           
        b) You can also create your own model completely from scratch. You first pass
           ``None`` for the ``profile`` parameter and then specify all channel model
           parameters. Please note that you **must** specify at least the following
           parameters in this case:
           
                * pathDelays
                * pathPowers
                * aods
                * aoas
                * zods
                * zoas
                * hasLos
            
           You can also optionally specify the following values:
           
                * angleSpreads (defaults to [4.0, 10.0, 2.0, 2.0] if not specified)
                * kFactorLos (defaults to ``pathPowers[0]-pathPowers[1]``)
           
           Also note that if your channel model contains a LOS cluster, it **must** be
           the first cluster in the lists and the ``hasLos`` parameter should be set
           to ``True``.
        """
        super().__init__(**kwargs)
        self.profile = profile                                # Can be 'A', 'B', 'C', 'D', or 'E'. None -> Custom model
        if self.profile is not None:
            if self.profile not in "ABCDE":    raise ValueError("Unsupported CDL profile '%s'!"%(self.profile))
                                
        self.ueDirAZ = toRadian(kwargs.get('ueDirAZ', [0,90]))  # Direction of UE. [Azimuth, Zenith] in degrees (Use radians internally)

        self.txAntenna = kwargs.get('txAntenna', None)          # Transmitter AntennaArray object (AntennaPanel or AntennaArray object)
        self.rxAntenna = kwargs.get('rxAntenna', None)          # Receiver AntennaArray object (AntennaPanel or AntennaArray object)

        # Orientation of TX and RX antenna arrays.
        # NOTE: To point an RX/TX antena to LOS angles of arrival/departure use: 𝛼=aoa[0]/aod[0], 𝛃=zoa[0]/zod[0]-90° 𝛄=0
        self.txOrientation = toRadian(kwargs.get('txOrientation', [0,0,0])) # Orientation of TX antenna array (alpha, beta, gamma) - degrees
        self.rxOrientation = toRadian(kwargs.get('rxOrientation', [0,0,0])) # Orientation of RX antenna array (alpha, beta, gamma) - degrees

        # Angle Scaling according to TR38.901 - 7.5.1 and TR38.901 - Annex A
        #  - To disable, set both 'angleScaling' to None (This is the default)
        #  - To enable, provide 2 lists of 4 desired values for aods, aoas, zods, zoas respectively for the mean and spread used for scaling.
        self.angleScaling = kwargs.get('angleScaling', None)
        if self.angleScaling is not None:
            assertMsg = "'angleScaling' must be a tuple of two lists of length 4!"
            if type(self.angleScaling) != tuple:                        raise ValueError(asserMsg)
            if type(self.angleScaling[0]) not in (list,np.ndarray):     raise ValueError(asserMsg)
            if type(self.angleScaling[1]) not in (list,np.ndarray):     raise ValueError(asserMsg)
            if len(self.angleScaling[0]) != 4:                          raise ValueError(asserMsg)
            if len(self.angleScaling[1]) != 4:                          raise ValueError(asserMsg)
            self.scalingAngleMeans = toRadian(self.angleScaling[0])
            self.scalingAngleSpreads = toRadian(self.angleScaling[1])
            
        # Set the default values based on the CDL profile. They can be overridden to create customized channels.
        def getCdlValue(x):
            return None if self.profile is None else np.float64(clusterInfo[self.profile])[:,x]
        
        self.pathDelays = kwargs.get('pathDelays', getCdlValue(0))  # Normalized Path Delays. The actual delays are set in the "scaleDelays"
        self.pathPowers = kwargs.get('pathPowers', getCdlValue(1))  # Path Powers in db

        # Note: We use radians internally
        self.aods = toRadian(kwargs.get('aods', getCdlValue(2)))    # Azimuth Angles of departure in degrees
        self.aoas = toRadian(kwargs.get('aoas', getCdlValue(3)))    # Azimuth Angles of arrival in degrees
        self.zods = toRadian(kwargs.get('zods', getCdlValue(4)))    # Zenith Angles of departure in degrees
        self.zoas = toRadian(kwargs.get('zoas', getCdlValue(5)))    # Zenith Angles of arrival in degrees

        self.hasLos = kwargs.get('hasLos', False if self.profile is None else (self.profile in "DE")) # True if cluster is a line of sight cluster
        self.xPolPower = kwargs.get('xPolPower', 10.0 if self.profile is None else perClusterParams[self.profile][1]) # CrossPolarization Power in db
        
        if self.pathDelays is None: raise ValueError("'pathDelays' is not specified for the custom CDL model!")
        if self.pathPowers is None: raise ValueError("'pathPowers' is not specified for the custom CDL model!")
        if self.aods is None:       raise ValueError("'aods' is not specified for the custom CDL model!")
        if self.aoas is None:       raise ValueError("'aoas' is not specified for the custom CDL model!")
        if self.zods is None:       raise ValueError("'zods' is not specified for the custom CDL model!")
        if self.zoas is None:       raise ValueError("'zoas' is not specified for the custom CDL model!")
        if ( (len(self.pathDelays)!=len(self.pathPowers)) or
             (len(self.pathDelays)!=len(self.aods)) or
             (len(self.pathDelays)!=len(self.aoas)) or
             (len(self.pathDelays)!=len(self.zods)) or
             (len(self.pathDelays)!=len(self.zoas)) ): raise ValueError("Cluster information must have the same size!")

        # Note that there is at most one LOS cluster and that is assumed to be the first cluster.
        self.kFactorLos = kwargs.get('kFactorLos', (self.pathPowers[0]-self.pathPowers[1]) if self.hasLos else None)    # K-Factor of the first cluster in db
        if self.profile is not None:
            self.scaleDelays()
            if self.kFactor is not None:    self.applyKFactorScaling()
        elif self.hasLos:
            # For custom models with LOS path, we split the first path into LOS and NLOS
            # Also note that it is assumed the custom values for powers and delays do not need angle and K-Factor scaling
            k1st = 10**(self.kFactorLos/10.0)
            p1st = 10**(self.pathPowers[0]/10.0)
            pathPowers1st = -10*np.log10(p1st + p1st/k1st)
            self.pathPowers = np.concatenate( ([pathPowers1st, pathPowers1st-self.kFactorLos], self.pathPowers[1:]))
            self.pathDelays = np.concatenate(([self.pathDelays[0]], self.pathDelays))
            self.aods = np.concatenate(([self.aods[0]], self.aods))
            self.aoas = np.concatenate(([self.aoas[0]], self.aoas))
            self.zods = np.concatenate(([self.zods[0]], self.zods))
            self.zoas = np.concatenate(([self.zoas[0]], self.zoas))

        # RMS angle spreads for aods, aoas, zods, zoas respectively.
        # This is the main angle spread not to be confused with the 'angleSpread' used for angle scaling.
        angleSpreadsDefault = [4.0, 10.0, 2.0, 2.0] if self.profile is None else perClusterParams[self.profile][0]
        self.angleSpreads = toRadian(kwargs.get('angleSpreads', angleSpreadsDefault))     # The angle spreads applied to normalized angles in degrees
        
        n, m = len(self.aods) - (1 if self.hasLos else 0), 20
        
        # Note:
        # The rayCoupling and initialPhases do not need to be specified. These are set randomly. The capability to specify
        # them is not documented and may be removed later.
        self.rayCoupling = kwargs.get('rayCoupling', None)                  # Ray Coupling values - Not documented
        self.randomRayCoupling = True
        if self.rayCoupling is not None:
            self.randomRayCoupling = False
            self.rayCoupling = np.int32(self.rayCoupling)
            if self.rayCoupling.shape != (3,n,m):
                raise ValueError("Invalide \"rayCoupling\" shape! Must be %s but it is %s"%(str((3,n,m)), str(self.rayCoupling.shape)))
            if np.any(self.rayCoupling>=m) or np.any(self.rayCoupling<0):
                raise ValueError("\"rayCoupling\" values must be between 0 and %d (inclusive)!"%(m))

        self.initialPhases = toRadian(kwargs.get('initialPhases', None))    # Initial Phases in degrees (Using radians internally) - Not documented
        self.randomInitialPhases = True
        if self.initialPhases is not None:
            self.randomInitialPhases = False
            self.initialPhases = np.float64(self.initialPhases)
            if self.initialPhases.shape != (2,2,n,m):
                raise ValueError("Invalide \"initialPhases\" shape! Must be %s but it is %s"%(str((2,2,n,m)), str(self.initialPhases.shape)))
            if np.any(self.initialPhases<-np.pi) or np.any(self.initialPhases>np.pi):
                raise ValueError("\"initialPhases\" values must be between -𝛑 and 𝛑!")

        # Channel Filter Info:
        self.channelFilter = self.makeFilter()  # makeFilter is defined in the base class.
        self.restart()
        
    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this channel model object. It first calls
        the base class :py:meth:`~neoradium.channel.ChannelBase.print` and
        then adds the information specific to :py:class:`~neoradium.cdl.CdlChannel`
        class.

        Parameters
        ----------
        indent : int (default: 0)
            The number of indentation characters.
            
        title : str or None (default: None)
            If specified, it is used as a title for the printed information.

        getStr : Boolean (default: False)
            If ``True``, it returns the information in a text string instead
            of printing the information.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is ``True``, then this function returns
            the information in a text string. Otherwise, nothing is returned.
        """
        if title is None:   title = "Customized CDL Channel Properties:" if self.profile is None else "CDL-%s Channel Properties:"%(self.profile)
        repStr = super().print(indent, title, True)
        repStr += indent*' ' + "  ueDirAZ: %s°, %s°\n"%(str(int(np.round(toDegrees(self.ueDirAZ[0])))), str(int(np.round(toDegrees(self.ueDirAZ[1])))))
        
        if self.angleScaling is not None:
            repStr += indent*' ' + "  Angle Scaling:\n"
            repStr += indent*' ' + "    Means: %s°\n"%("° ".join(str(int(np.round(toDegrees(angle)))) for angle in self.scalingAngleMeans ))
            repStr += indent*' ' + "    RMS Spreads: %s°\n"%("° ".join(str(int(np.round(toDegrees(angle)))) for angle in self.scalingAngleSpreads ))

        repStr += getMultiLineStr("pathDelays (ns)", self.pathDelays, indent, "%6f", 6, numPerLine=10)
        repStr += getMultiLineStr("pathPowers (db)", self.pathPowers, indent, "%6f", 6, numPerLine=10)
        repStr += getMultiLineStr("AODs (Degree)", np.round(toDegrees(self.aods)), indent, "%4d", 4, numPerLine=15)
        repStr += getMultiLineStr("AOAs (Degree)", np.round(toDegrees(self.aoas)), indent, "%4d", 4, numPerLine=15)
        repStr += getMultiLineStr("ZODs (Degree)", np.round(toDegrees(self.zods)), indent, "%4d", 4, numPerLine=15)
        repStr += getMultiLineStr("ZOAs (Degree)", np.round(toDegrees(self.zoas)), indent, "%4d", 4, numPerLine=15)
        repStr += indent*' ' + "  hasLOS: %s\n"%(str(self.hasLos))
        repStr += indent*' ' + "  Cross Pol. Power: %s db\n"%(str(self.xPolPower))
        if self.profile is None and self.hasLos:
            repStr += indent*' ' + "  K-Factor for LOS path: %s db\n"%(str(self.kFactorLos))
        repStr += indent*' ' + "  angleSpreads: %s°\n"%("° ".join(str(int(np.round(toDegrees(angle)))) for angle in self.angleSpreads ))
        
        repStr += self.txAntenna.print(indent+2, "TX Antenna:", True)
        if np.prod(self.txOrientation)!=0:
            repStr += indent*' ' + "  TX Antenna Orientation (𝛼,𝛃,𝛄): %s°\n"%("° ".join(str(int(np.round(toDegrees(angle)))) for angle in self.txOrientation ))
        
        repStr += self.rxAntenna.print(indent+2, "RX Antenna:", True)
        if np.prod(self.rxOrientation)!=0:
            repStr += indent*' ' + "  RX Antenna Orientation (𝛼,𝛃,𝛄): %s°\n"%("° ".join(str(int(np.round(toDegrees(angle)))) for angle in self.rxOrientation ))

        repStr += self.channelFilter.print(indent+2, "Channel Filter:", True)

        if getStr: return repStr
        print(repStr)

    # ************************************************************************************************************************************************
    def restart(self, restartRanGen=False):
        r"""
        This method first calls the base class :py:meth:`~neoradium.channel.ChannelBase.restart`
        and then re-initializes the ray coupling and initial phases randomly.

        Parameters
        ----------
        restartRanGen : Boolean (default: False)
            If a ``seed`` was not provided to this channel model, this parameter
            is ignored. Otherwise, if ``restartRanGen`` is set to ``True``, this
            channel model's random generator is reset and if ``restartRanGen`` is
            ``False`` (default), the random generator is not reset. This means
            if ``restartRanGen`` is ``False``, calling this function starts a new
            sequence of channel instances which are different from the sequence when
            the channel was instantiated.
        """
        super().restart(restartRanGen)
        if self.randomRayCoupling:      self.rayCoupling = self.getRandomRayCoupling()
        if self.randomInitialPhases:    self.initialPhases = self.getRandomInitialPhases()
        
    # ************************************************************************************************************************************************
    @property           # This property is already documented above in the __init__ function.
    def nrNt(self):     return (self.rxAntenna.getNumElements(), self.txAntenna.getNumElements())

    # ************************************************************************************************************************************************
    def scaleDelays(self):                      # Not documented
        self.pathDelays *= self.delaySpread     # Path delays in nanoseconds (See TR38.901 - Sec. 7.7.3 Scaling of delays)
            
    # ************************************************************************************************************************************************
    def getChannelGains(self, channelTimes):        # Not documented (See the documentation of "getPathGains" in the base class)
        gains = self.getNLOSgains(channelTimes)                                                 # Shape: nc x nr x nt x numNLOS
        if self.hasLos: gains = np.concatenate((self.getLOSgains(channelTimes), gains), axis=3) # Shape: nc x nr x nt x (numNLOS+1)
        return gains                                                                            # Shape: nc x nr x nt x np

    # ************************************************************************************************************************************************
    def wrapAngles(self, angles, how):              # Not documented
        # This function is used to handle the correct wrapping of the angles. Four types
        # of wrapping is handled as follows:
        if how in ["-180,180", "-𝛑,𝛑"]:
            # Wrap the angles to be between -𝛑, 𝛑
            return (angles + np.pi)%(2*np.pi) - np.pi
        
        if how in ["0,180", "0,𝛑"]:
            # Wrap the angles to be between 0 and 𝛑 (Note that angles between 180 and 360 are wrapped in reverse order)
            angles %= 2*np.pi
            angles[angles>np.pi] = 2*np.pi - angles[angles>np.pi]
            return angles

        if how in ["0,360", "0,2𝛑"]:
            # Wrap the angles to be between 0, 2𝛑
            return angles % (2*np.pi)
            
        if how in ["Clip-0,180", "Clip-0,𝛑"]:
            # Clip the angles to the range 0..𝛑 (Different from wrapping between 0 and 𝛑 above)
            return np.clip(angles,0,np.pi)
            
        assert False, "Don't know how to wrap with \"%s\"!"%(how)

    # ************************************************************************************************************************************************
    def getLOSgains(self, channelTimes):        # Not documented
        # This function calculates the gain for the LOS cluster. It must be called only if
        # current channel model contains LOS clusters.
        assert self.hasLos, "'getLOS' function was called for a profile that does not contain LOS information!"
        
        # STEP-1 (See Step-1 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Generate departure and arrival angles. These are all 1x1 matrixes (one cluster, one ray).
        # Note: No need to use "rayOffsets" for LOS case.
        phiD = self.aods[0:1].reshape(1,1)      # Shape: 1 x 1
        phiA = self.aoas[0:1].reshape(1,1)      # Shape: 1 x 1
        thetaD = self.zods[0:1].reshape(1,1)    # Shape: 1 x 1
        thetaA = self.zoas[0:1].reshape(1,1)    # Shape: 1 x 1
        pN = 10.0**(self.pathPowers[0]/10.0)    # Shape: Scaler     (Also converted from db to linear)

        if self.angleScaling is not None:
            # Need to do angle scaling:
            phiD, phiA, thetaD, thetaA = self.applyAngleScaling(phiD, phiA, thetaD, thetaA, pN)

        phiD = self.wrapAngles(phiD, "-180,180")
        phiA = self.wrapAngles(phiA, "-180,180")
        thetaD = self.wrapAngles(thetaD, "0,180")
        thetaA = self.wrapAngles(thetaA, "0,180")

        nr, nt = self.nrNt
        # STEP-2 (See Step-2 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # No random coupling is needed for LOS case because there is only one cluster and one ray
        
        # STEP-3 (See Step-3 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Generate the cross polarization power ratios:
        kappa = 10.0**(self.xPolPower/10.0)

        # STEP-4 (See Step-4 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Draw initial random phases (See Step-10 in TR38.901 - 7.5)
        # No Initial phases needed for LOS case. The polarization matrix is fixed
        
        # Get the TX field part and TX location part in TR38.901 - Eq. 7.5-29
        fieldTx, locTx = self.txAntenna.getElementsFields(thetaD, phiD, self.txOrientation)  # Shapes: "nt x 2 x 1 x 1" and "nt x 1 x 1"

        # Get the RX field part and RX location part in TR38.901 - Eq. 7.5-29
        fieldRx, locRx = self.rxAntenna.getElementsFields(thetaA, phiA, self.rxOrientation)  # Shapes: "nr x 2 x n x m" and "nr x n x m"

        # Get the polarization matrix part in TR38.901 - Eq. 7.5-29
        polMat = np.float64([[1,0],[0,-1]])                                                         # Shape:  2 x 2
        
        # Get the doppler term in TR38.901 - Eq. 7.5-29
        doppler = self.getDopplerFactor(channelTimes, thetaA, phiA)                                 # Shape:  nc

        # Now that we have built all parts of TR38.901 - Eq. 7.5-29, we need to combine all of them together. Here are shapes of different parts
        # of TR38.901 - Eq. 7.5-29  complex tensor. (Squeezing out the 1 x 1 parts)
        #       fieldRx: nr x 2
        #       polMat:  2 x 2
        #       fieldTx: nt x 2
        #       locRx:   nr
        #       locTx:   nt
        #       doppler: t
        # The output will be a "nr x nt x t x 1"
        
        # First fieldRx x polMat x fieldTx
        hLOS = ((fieldRx.reshape(-1,1,2,1) * polMat.reshape(1,1,2,2)).sum(2).reshape(-1,1,2)*fieldTx.reshape(1, -1, 2)).sum(2)  # Shape: nr x nt
        # Now apply location factors
        hLOS = hLOS * locRx.reshape(-1, 1) * locTx.reshape(1, -1)                                                               # Shape: nr x nt
        # Applying the doppler:
        hLOS = hLOS.reshape(1,nr,nt) * doppler.reshape(-1,1,1)                                                                  # Shape: nc x nr x nt
        # Apply the scaling
        hLOS *= np.sqrt(pN)                                                                                                     # Shape: nc x nr x nt
        return hLOS.reshape(-1, nr, nt, 1)                                                                                      # Shape: nc x nr x nt x 1
        
    # ************************************************************************************************************************************************
    def getNLOSgains(self, channelTimes):       # Not documented
        # This function calculates the gains for all NLOS clusters.
        
        # STEP-1 (See Step-1 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Generate departure and arrival angles
        offset = 1 if self.hasLos else 0
        cASD, cASA, cZSD, cZSA = self.angleSpreads
        phiD   = self.aods[offset:].reshape(-1,1) + cASD*np.float64(rayOffsets)     # Shape: n x m
        phiA   = self.aoas[offset:].reshape(-1,1) + cASA*np.float64(rayOffsets)     # Shape: n x m
        thetaD = self.zods[offset:].reshape(-1,1) + cZSD*np.float64(rayOffsets)     # Shape: n x m
        thetaA = self.zoas[offset:].reshape(-1,1) + cZSA*np.float64(rayOffsets)     # Shape: n x m
        pN = 10.0**(self.pathPowers[offset:]/10.0)                                  # Shape: n     (linear power values)
        
        if self.angleScaling is not None:
            # Need to do angle scaling:
            phiD, phiA, thetaD, thetaA = self.applyAngleScaling(phiD, phiA, thetaD, thetaA, pN)

        phiD = self.wrapAngles(phiD, "-180,180")
        phiA = self.wrapAngles(phiA, "-180,180")
        thetaD = self.wrapAngles(thetaD, "0,180")
        thetaA = self.wrapAngles(thetaA, "0,180")

        n, m = phiD.shape
        nr, nt = self.nrNt

        # STEP-2 (See Step-2 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Random coupling of rays within clusters
        phiD, phiA, thetaD, thetaA = self.shuffleRays(phiD, phiA, thetaD, thetaA)

        # STEP-3 (See Step-3 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Generate the cross polarization power ratios:
        kappa = 10.0**(self.xPolPower/10.0)

        # STEP-4 (See Step-4 in TR38.901 - 7.7.1) ----------------------------------------------------------------------------------------------------
        # Draw initial random phases (See Step-10 in TR38.901 - 7.5)
        phiInit = self.initialPhases        # Uniform between -𝛑,𝛑. Shape: 2 x 2 x n x m

        # Get the TX field part and TX location part in TR38.901 - Eq. 7.5-22
        fieldTx, locTx = self.txAntenna.getElementsFields(thetaD, phiD, self.txOrientation)  # Shapes: "nt x 2 x n x m" and "nt x n x m"

        # Get the RX field part and RX location part in TR38.901 - Eq. 7.5-22
        fieldRx, locRx = self.rxAntenna.getElementsFields(thetaA, phiA, self.rxOrientation)  # Shapes: "nr x 2 x n x m" and "nr x n x m"

        # Get the polarization matrix part in TR38.901 - Eq. 7.5-22
        polMat = np.exp(1j*phiInit) * np.sqrt([[1, 1/kappa], [1/kappa, 1]]).reshape(2,2,1,1)        # Shape:  2 x 2 x n x m

        # Get the doppler term in TR38.901 - Eq. 7.5-22
        doppler = self.getDopplerFactor(channelTimes, thetaA, phiA)                                 # Shape:  nc x n x m

        # Now that we have built all parts of TR38.901 - Eq. 7.5-22, we need to combine all of them together. Here are shapes of different parts
        # of TR38.901 - Eq. 7.5-22  complex tensor.
        #       fieldRx: nr x 2 x n x m
        #       polMat:  2 x 2 x n x m
        #       fieldTx: nt x 2 x n x m
        #       locRx:   nr x n x m
        #       locTx:   nt x n x m
        #       doppler: t x n x m
        # The output will be a "nr x nt x t x n"
        
        # First fieldRx x polMat x fieldTx
        hNLOS = ((fieldRx.reshape(-1,1,2,1,n,m) * polMat.reshape(1,1,2,2,n,m)).sum(2).reshape(-1,1,2,n,m)*fieldTx).sum(2)   # Shape: nr x nt x n x m
        # Now apply location factors
        hNLOS = hNLOS * locRx.reshape(-1, 1, n, m) * locTx.reshape(1, -1, n, m)                                             # Shape: nr x nt x n x m
        # Applying the doppler:
        hNLOS = hNLOS.reshape(1,nr,nt,n,m) * doppler.reshape(-1,1,1,n, m)                                                   # Shape: nc x nr x nt x n x m
        # Now sum over m (Combining rays in each cluster)
        hNLOS = hNLOS.sum(4)                                                                                                # Shape: nc x nr x nt x n
        # Apply the scaling
        hNLOS *= np.sqrt(pN/m).reshape(1,1,1,-1)
        return hNLOS

    # ************************************************************************************************************************************************
    def getRandomRayCoupling(self):             # Not documented
        # This function randomly creates the rayCoupling values (See Step-2 in TR38.901 - section 7.7.1)
        n, m = len(self.aods) - (1 if self.hasLos else 0), 20
        return np.int32([ [self.rangen.choice(range(m), size=m, replace=False) for _ in range(n)] for _ in range(3) ])  # Shape: 3 x n x m

    # ************************************************************************************************************************************************
    def getRandomInitialPhases(self):
        # Draw initial random phases (See Step-10 in TR38.901 - 7.5)
        n, m = len(self.aods) - (1 if self.hasLos else 0), 20
        return 2*np.pi * self.rangen.random(size=(2,2,n,m)) - np.pi     # Uniform between -𝛑,𝛑. Shape: 2 x 2 x n x m

    # ************************************************************************************************************************************************
    @classmethod
    def getMatlabRandomInit(cls, profile, seed):       # Not documented
        # This is a helper class method that can be used to create random ray coupling and initial phases to
        # match Matlab values. It can be used when comparing results with Matlab.
        from neoradium.cdl import clusterInfo
        tempGen = random.getGenerator(np.random.RandomState(seed))    # Create a random generator matching Matlab's
        hasLos = 1 if (profile in "DE") else 0

        n, m = len(clusterInfo[profile]), 20
        phi = tempGen.random(size=(4, m, n))
        phi = np.transpose(phi,(0,2,1))[:,hasLos:,:]    # Skip the LOS value (Not used)
        phiInit = (360*phi - 180).reshape(2,2,n-hasLos,m)

        cp = tempGen.random(size=(3, m, n))
        cpIdx = np.argsort(cp,axis=1)
        coupling = np.zeros((3, m, n))
        coupling[[0,2],:,:] = cpIdx[[0,2],:,:]
        for i in range(n):
            idx = np.argsort(cpIdx[2,:,i])
            coupling[1,:,i] = cpIdx[1,idx,i]

        coupling = np.int32(coupling.transpose((0,2,1))[:,hasLos:,:])
        coupling.shape
        # Make sure the dimensions match
        assert coupling.shape==(3,n-hasLos,m)

        # Matlab shuffles 'thetaA' twice. The following fixes this problem:
        rows = np.int32([m*[rr] for rr in range(n-hasLos)]) # Shape: n x m
        coupling[1] = coupling[1][(rows, coupling[2])]
        return phiInit, coupling

    # ************************************************************************************************************************************************
    def shuffleRays(self, phiD, phiA, thetaD, thetaA):              # Not documented
        # This function shuffles the rays in the clusters randomly using the rayCoupling parameter.
        n, m = phiD.shape
        rowIndexes = np.int32([m*[rr] for rr in range(n)])          # Shape: n x m

        # Shuffle rays for each path
        phiA    = phiA  [ (rowIndexes, self.rayCoupling[0]) ]
        thetaA  = thetaA[ (rowIndexes, self.rayCoupling[1]) ]
        thetaD  = thetaD[ (rowIndexes, self.rayCoupling[2]) ]
        return phiD, phiA, thetaD, thetaA

    # ************************************************************************************************************************************************
    def getDopplerFactor(self, channelTimes, theta, phi):           # Not documented
        # This function calculates the doppler term in TR38.901 - Eq. 7.5-22
        vPhi, vTheta = self.ueDirAZ         # Direction (angles) of UE movement in phi and theta in radians
        # Simplifying : d = speed/wavelen. Instead of using vBar and v, we use dBar and doppler and remove the lambda in the denuminator.
        # The following is the adapted version of TR38.901 - Eq. 7.5-25
        dBar = self.dopplerShift * np.array([ np.sin(vTheta) * np.cos(vPhi),
                                              np.sin(vTheta) * np.sin(vPhi),
                                              np.cos(vTheta) ])
        
        sinTheta = np.sin(theta)
        rHatRx = np.array([ sinTheta * np.cos(phi),
                            sinTheta * np.sin(phi),
                            np.cos(theta) ])

        return np.exp(2j * np.pi * channelTimes.reshape(-1,1,1) * (rHatRx*dBar.reshape(3,1,1)).sum(0))  # Shape:  nc x n x m

    # ************************************************************************************************************************************************
    def applyAngleScaling(self, phiD, phiA, thetaD, thetaA, p):     # Not documented
        # This function applies the Angle Scaling according to 3GPP TR 38.901 Section 7.7.5.1 and Annex A.
        assert self.angleScaling is not None
        n,m = phiA.shape

        # Desired mean and spread as provided
        asPhiD, asPhiA, asThetaD, asThetaA = self.scalingAngleSpreads   # Angle Spread for phiD, phiA, thetaD, thetaA
        maPhiD, maPhiA, maThetaD, maThetaA = self.scalingAngleMeans     # Mean Angle for phiD, phiA, thetaD, thetaA
        
        # Calculate Model mean and spread: (See TR38.901 - Annex A)
        def getModelMeanAndSpread(angles):
            weightedSum = (np.exp(1j*angles)*p.reshape(-1,1)).sum()/m     # This is the numinator of the fraction in TR38.901 - Eq. A-1
            angularSpread = np.sqrt(-2*np.log(np.abs(weightedSum/(p.sum()))))       # This is the 'AS' as defined in TR38.901 - Eq. A-1
            meanAngle = np.angle(weightedSum)
            return meanAngle, angularSpread
            
        # In the following, 'ma' is for "Mean Angle", and 'as' is for "Angle Spread"
        maPhiDmodel, asPhiDmodel = getModelMeanAndSpread(phiD)
        maPhiAmodel, asPhiAmodel = getModelMeanAndSpread(phiA)
        maThetaDmodel, asThetaDmodel = getModelMeanAndSpread(thetaD)
        maThetaAmodel, asThetaAmodel = getModelMeanAndSpread(thetaA)
        
        # Scale the angles: (See TR38.901 - Section 7.7.5.1)
        def transformAngles(angles, asD, maD, asM, maM):    # See TR38.901 - Eq. 7.7-5
            # Names format:
            #   'D' is for "Desired", 'M' is for "Model"
            #   'ma' is for "Mean Angle", and 'as' is for "Angle Spread"
            if asM==0:  return angles - maM + maD
            return asD * (angles - maM)/asM + maD
        scaledPhiD = transformAngles(phiD, asPhiD, maPhiD, asPhiDmodel, maPhiDmodel)
        scaledPhiA = transformAngles(phiA, asPhiA, maPhiA, asPhiAmodel, maPhiAmodel)
        scaledThetaD = transformAngles(thetaD, asThetaD, maThetaD, asThetaDmodel, maThetaDmodel)
        scaledThetaA = transformAngles(thetaA, asThetaA, maThetaA, asThetaAmodel, maThetaAmodel)

        # Wrapping the angles. See the note near the end of TR38.901 - Section 7.7.5.1
        scaledPhiD = self.wrapAngles(scaledPhiD, "0,360")               # Wrap azimuth angles around to be within [0, 360] degrees
        scaledPhiA = self.wrapAngles(scaledPhiA, "0,360")               # Wrap azimuth angles around to be within [0, 360] degrees
        scaledThetaD = self.wrapAngles(scaledThetaD, "Clip-0,180")      # Clip zenith angles to be within [0, 180] degrees
        scaledThetaA = self.wrapAngles(scaledThetaA, "Clip-0,180")      # Clip zenith angles to be within [0, 180] degrees

        return scaledPhiD, scaledPhiA, scaledThetaD, scaledThetaA
