# Copyright (c) 2024 InterDigital AI Lab
"""
This module implements the :py:class:`~neoradium.cdl.CdlChannel` class which encapsulates the functionality of the 
Clustered Delay Line (CDL) channel model.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 06/05/2023    Shahab Hamidi-Rad       First version of the file.
# 11/30/2023    Shahab                  Completed the documentation
# 04/01/2025    Shahab                  Restructured the file to work with the new ChannelModel class
# 05/07/2025    Shahab                  * Changed the default orientation of receiver antenna to [180,0,0] instead
#                                         of [0,0,0].
#                                       * Reviewed and updated the documentation.
# 06/20/2025    Shahab                  * The default antenna for the CDL channel model is now a 1x1 antenna panel.
#                                       * The new "getChanGen" class method can be used to return a generator object
#                                         that can generate CDL channel matrices according to the given parameters and
#                                         criteria.
#                                       * Updated the "restart" method with the new parameter "applyToBwp".
# **********************************************************************************************************************
import numpy as np
import scipy.io
from scipy.signal import lfilter

from .antenna import AntennaElement
from .channelmodel import ChannelModel
from .utils import getMultiLineStr, toRadian, toDegrees, toLinear, toDb, freqStr
from .random import random

# This file is based on 3GPP TR 38.901 V17.1.0
# **********************************************************************************************************************
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

# **********************************************************************************************************************
perClusterParams = {#           C_ASD  C_ASA  C_ZSD  C_ZSA   XPR
                        'A':  ([5,     11,    3,     3],     10),     # TR38.901 - Table 7.7.1-1 CDL-A
                        'B':  ([10,    22,    3,     7],     8),      # TR38.901 - Table 7.7.1-2 CDL-B
                        'C':  ([2,     15,    3,     7],     7),      # TR38.901 - Table 7.7.1-3 CDL-C
                        'D':  ([5,     8,     3,     3],     11),     # TR38.901 - Table 7.7.1-4 CDL-D
                        'E':  ([5,     11,    3,     7],     8),      # TR38.901 - Table 7.7.1-5 CDL-E
                   }

# **********************************************************************************************************************
# TR38.901 - Table 7.5-3: Ray offset angles within a cluster, given for rms angle spread normalized to 1
# These are the 𝛼m in Eq. 7.7-0a (20 values for 20 rays)
rayOffsets = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,
              0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551]

# **********************************************************************************************************************
class CdlChannel(ChannelModel):
    r"""
    This class implements the Clustered Delay Line (CDL) channel model based on **3GPP TR 38.901**. It is derived 
    from the :py:class:`~neoradium.channelmodel.ChannelModel` class.
    
    All of the API functions used in typical use cases are explained in the documentation of the
    :py:class:`~neoradium.channelmodel.ChannelModel` class.
    
    The typical use case involves instantiating a :py:class:`CdlChannel` object and then calling functions such as
    :py:meth:`~neoradium.channelmodel.ChannelModel.getChannelMatrix`,
    :py:meth:`~neoradium.channelmodel.ChannelModel.applyToSignal`,
    :py:meth:`~neoradium.channelmodel.ChannelModel.applyToGrid`, etc. Please refer to the notebook 
    :doc:`../Playground/Notebooks/Channels/ChannelMatrix` for an example of using this class.
    """
    # ******************************************************************************************************************
    def __init__(self, bwp, profile='A', **kwargs):
        r"""
        Parameters
        ----------
        bwp : :py:class:`~neoradium.carrier.BandwidthPart` 
            The bandwidth part object used by the channel model to create channel matrices.
            
        profile : str or None
            The CDL profile. It can be one of 'A', 'B', 'C', 'D', 'E', or `None`. See **3GPP TR 38.90, Section 
            7.7.1** for more information. Use `None` to indicate a customized version of CDL channel (See 
            :ref:`Customizing CDL Model <CustomizingCDL>`).

        kwargs : dict
            Here’s a list of additional optional parameters that can be used to further customize this channel model:

                :normalizeGains: A boolean flag. The default value is `True`, indicating that the path gains 
                    are normalized before they are applied to the signals.
                    
                :normalizeOutput: A boolean flag. The default value is `True`, indicating that the gains are 
                    normalized based on the number of receive antennas.
                    
                :txDir: A string that represents the transmission direction, which can be either “Downlink” or 
                    “Uplink”. By default, it is set to “Downlink”.
                    
                :filterLen: The length of the channel filter. The default is 16 samples.
                
                :delayQuantSize: The size of delay fraction quantization for the channel filter. The default is 64.
                
                :stopBandAtten: The stop-band attenuation (in dB) used by the channel filter. The default is 80 dB.
                
                :seed: The seed used by the random functions in the channel model. Setting this to a fixed value ensures
                    that the channel model generates repeatable results. The default value is `None`, indicating 
                    that this channel model uses the **NeoRadium**’s :doc:`global random generator <./Random>`.
                    
                :dopplerShift: The maximum Doppler shift in Hertz. The default value is 40 Hertz, which corresponds to
                    a speed of approximately 10 kilometers per hour. A value of zero makes the channel model static. 
                    For trajectory-based channel models, this value is automatically assigned based on the maximum 
                    trajectory speed.
                    
                :carrierFreq: The carrier frequency of the channel model in Hz. The default is 3.5 GHz.
                
                :delaySpread: The delay spread in nanoseconds. The default is 30 ns. It can also be a string 
                    containing one of the values in following table (See **3GPP TR 38.901, table 7.7.3-1**)
                    
                    ======================  ==============
                    Delay Spread str        Delay spread
                    ======================  ==============
                    'VeryShort'             10 ns
                    'Short'                 30 ns
                    'Nominal'               100 ns
                    'Long'                  300 ns
                    'VeryLong'              1000 ns
                    ======================  ==============

                :ueDirAZ: This is a list of two angles for the Azimuth and Zenith of the UE’s direction of movement 
                    in degrees. The default value is [0, 90], which indicates movement along the x-axis. In the current
                    version, the base station is assumed to be stationary.
                    
                :txAntenna: The transmitter antenna, which is an instance of 
                    :py:class:`~neoradium.antenna.AntennaElement`, :py:class:`~neoradium.antenna.AntennaPanel`,
                    or :py:class:`~neoradium.antenna.AntennaArray` class. If not specified, a single antenna element is
                    automatically created by default.
                
                :rxAntenna: The receiver antenna, which is an instance of 
                    :py:class:`~neoradium.antenna.AntennaElement`, :py:class:`~neoradium.antenna.AntennaPanel`,
                    or :py:class:`~neoradium.antenna.AntennaArray` class. If not specified, a single antenna element is
                    automatically created by default.
                    
                :txOrientation: The orientation of the transmitter antenna. This is a list of three angle values in 
                    degrees: bearing angle (math:`\alpha`), downtilt angle (math:`\beta`), and slant angle 
                    (math:`\gamma`). The default orientation is [0,0,0]. For more information, please refer to 
                    **3GPP TR 38.901, Section 7.1.3**.

                :rxOrientation: The orientation of receiver antenna. This is a list of three angle values in 
                    degrees: bearing angle (math:`\alpha`), downtilt angle (math:`\beta`), and slant angle 
                    (math:`\gamma`). The default orientation is [180,0,0]. For more information, please refer to 
                    **3GPP TR 38.901, Section 7.1.3**.

                :kFactor: The K-Factor (in dB) used for scaling. The default is `None`. If not specified 
                    (``kFactor=None``), K-factor scaling is disabled.

                :xPolPower: The cross-polarization Power in dB. The default is 10 dB. For more details please refer 
                    to "Step 3" in **3GPP TR 38.901, Section 7.7.1**.

                :angleScaling: The :ref:`Angle Scaling <AngleScaling>` parameters. If specified, it must be a tuple of
                    2 NumPy arrays.
                    
                    The first item specifies the mean values for angle scaling. It’s a 1-D NumPy array containing 
                    four values for: the *Azimuth angle of Departure*, *Azimuth angle of Arrival*, *Zenith angle of 
                    Departure*, and *Zenith angle of Arrival*.
                    
                    The second item specifies the RMS angle spread values. It is a 1-D NumPy array containing four RMS 
                    values for: the *Azimuth angle of Departure*, *Azimuth angle of Arrival*, *Zenith angle of 
                    Departure*, and *Zenith angle of Arrival*. For more information, please refer to 
                    :ref:`Angle Scaling <AngleScaling>` below.
                    
                    If this value is set to `None` (the default), the *Angle Scaling* is disabled.
                    
                :pathDelays: Use this parameter to customize or override the default path delays, which are set 
                    based on the CDL channel model as defined in **3GPP TR 38.901**. In most use cases, you don’t 
                    need to specify this parameter. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.
                
                :pathPowers: Use this parameter to customize or override the path power settings, which are set by 
                    default based on the CDL channel model as defined in **3GPP TR 38.901**. You don’t need to specify 
                    this parameter for most use cases. For more information, refer to 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.

                :aods: Use this parameter to customize or override the Azimuth angles of Departure, which are set 
                    by default based on the CDL channel model as defined in **3GPP TR 38.901**. You don’t need to 
                    specify this parameter for most use cases. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.

                :aoas: Use this parameter to customize or override the Azimuth angles of Arrival, which are set by 
                    default based on the CDL channel model as defined in **3GPP TR 38.901**. You don’t need to specify
                    this parameter for most use cases. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.
                    
                :zods: Use this parameter to customize or override the Zenith angles of Departure, which are set 
                    by default based on the CDL channel model as defined in **3GPP TR 38.901**. You don’t need to 
                    specify this parameter for most use cases. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.
                    
                :zoas: Use this parameter to customize or override the Zenith angles of Arrival, which are set by 
                    default based on the CDL channel model as defined in **3GPP TR 38.901**. You don’t need to specify
                    this parameter for most use cases. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.
                    
                :angleSpreads: Use this parameter to customize or override the RMS Angle spread (in degrees) used to 
                    normalize angles. This parameter specifies four values corresponding to the *Azimuth angle of 
                    Departure*, *Azimuth angle of Arrival*, *Zenith angle of Departure*, and *Zenith angle of Arrival*.
                    By default, these values are set based on the CDL channel model as defined in **3GPP TR 38.901**. 
                    You don’t need to specify this parameter for most use cases. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.
                    
                    Please note that this should not be confused with angle spread values used for angle scaling (See
                    ``angleScaling`` above).
        
                :hasLos: Use this parameter to customize or override the ``hasLos`` property of this channel model. 
                    By default, this property is set based on the CDL channel model as defined in **3GPP TR 38.901**. 
                    You don’t need to specify this parameter for most use cases. For more information, see 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below.
                    
                :kFactorLos: Use this parameter when customizing the CDL model. It represents the K-Factor ratio 
                    (in dB) for the LOS (Line of Sight) cluster (the first cluster). You don’t need to specify this 
                    parameter for most use cases. For more information, refer to 
                    :ref:`Customizing CDL Model <CustomizingCDL>` below. By default, when customizing the CDL model, 
                    this value is set to the difference in path powers (in dB) between the first and second clusters.


        .. Note:: All angle values provided to this class are in degrees. However, internally, the class uses 
            radian values for all calculations. Therefore, when you access any of the angle values, such as ``aods``,
            ``aoas``, ``zods``, and ``zoas``, remember that they are in radians.


        **Other Properties:**
        
        All of the parameters mentioned above are directly available. Here is a list of additional properties:
        
            :coherenceTime: The `Coherence time <https://en.wikipedia.org/wiki/Coherence_time_(communications_systems)>`_
                of the channel model in seconds. This is calculated based on the ``dopplerShift`` parameter.
            :sampleRate: The sample rate used by this channel model. For 3GPP standard, this is set to 30,720,000 
                samples per second.
            :nrNt: A tuple of the form ``(nr,nt)``, where ``nr`` and ``nt`` are the number of receiver and transmitter
                antennas elements correspondingly.
            :rayCoupling: This property is used internally for ray coupling. See **Step 2 in 3GPP TR 38.901, Section
                7.7.1** for more details.
            :initialPhases: The random initial phases used when creating channel gains. See **Step 10 in 3GPP 
                TR 38.901, Section 7.5**  for more details.


        .. _AngleScaling:
        
        **Angle Scaling:**

        If ``angleScaling`` is set to `None`, angle scaling is disabled. Otherwise, it is applied to all angles of 
        arrival and departure for all clusters, as per **3GPP TR 38.901, Section 7.7.5.1 and Annex A**.


        .. _CustomizingCDL:
        
        **Customizing CDL Model:**

        There are two different ways to customize the CDL model:
        
        a) You can select one of the predefined CDL profiles (A, B, C, D, or E) and then modify the model’s parameters 
           by providing additional information. For instance, you can choose the CDL-B model and override the standard
           path delays by specifying your own path delays.
           
        b) You can also create your own model entirely from scratch. Initially, pass `None` for the ``profile`` 
           parameter and then specify all the channel model parameters. Please note that in this case, you 
           **must** specify at least the following parameters:
           
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
           
           Also note that if your channel model contains a LOS cluster, it **must** be the first cluster in the lists, 
           and the ``hasLos`` parameter should be set to `True`.
        """
        super().__init__(bwp, **kwargs)
        self.profile = profile                              # Can be 'A', 'B', 'C', 'D', or 'E'. None -> Custom model
        if self.profile is not None:
            if self.profile not in "ABCDE":    raise ValueError(f"Unsupported CDL profile '{self.profile}'!")

        self.delaySpread = kwargs.get('delaySpread', 30)    # Default: 30ns
        if type(self.delaySpread)==str:
            # See TR38.901 - Table 7.7.3-1
            strToDelaySpread = {"VeryShort": 10, "Short": 30, "Nominal": 100, "Long": 300, "VeryLong": 1000}
            if self.delaySpread not in strToDelaySpread:
                raise ValueError("'delaySpread' must be a number or one of 'VeryShort', 'Short', 'Nominal', 'Long', "+
                                 "or 'VeryLong'")
            self.delaySpread = strToDelaySpread[self.delaySpread]

        self.ueDirAZ = toRadian(kwargs.get('ueDirAZ', [0,90]))      # Direction of UE. [Azimuth, Zenith] in degrees

        self.txAntenna = kwargs.get('txAntenna', AntennaElement())  # Transmitter AntennaArray/AntennaPanel object
        self.rxAntenna = kwargs.get('rxAntenna', AntennaElement())  # Receiver AntennaArray/AntennaPanel object

        # Orientation of TX and RX antenna arrays (alpha, beta, gamma) - degrees
        # NOTE1: To point an RX/TX antenna to LOS angles of arrival/departure use:
        #           𝛼=aoa[0]/aod[0], 𝛃=zoa[0]/zod[0]-90°, 𝛄=0
        # Note2: Based on some experiments, it was decided to change the default orientation of RX antenna to
        #        [180,0,0] instead of the original [0,0,0]. See the notebook "CdlBearingAngles.ipynb" in
        #        the "OtherExperiments" folder for more information.
        self.txOrientation = toRadian(kwargs.get('txOrientation', [0,0,0]))     # Orientation of TX antenna array
        self.rxOrientation = toRadian(kwargs.get('rxOrientation', [180,0,0]))   # Orientation of RX antenna array
        
        # K-factor scaling: (See the function "applyKFactorScaling" in the base class)
        self.kFactor = kwargs.get('kFactor', None)  # The K-factor for scaling in dB. 'None' disables K-factor scaling
        
        # Angle Scaling according to TR38.901 - 7.7.5.1 and TR38.901 - Annex A
        #  - To disable, set 'angleScaling' to None (This is the default)
        #  - To enable, provide 2 lists of 4 desired values for aods, aoas, zods, zoas respectively for the
        #    mean and spread used for scaling.
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
        
        self.pathDelays = kwargs.get('pathDelays', getCdlValue(0))  # Normalized Path Delays. See "scaleDelays"
        self.pathPowers = kwargs.get('pathPowers', getCdlValue(1))  # Path Powers in db

        # Note: We use radians internally
        self.aods = toRadian(kwargs.get('aods', getCdlValue(2)))    # Azimuth angles of departure (in degrees)
        self.aoas = toRadian(kwargs.get('aoas', getCdlValue(3)))    # Azimuth angles of arrival (in degrees)
        self.zods = toRadian(kwargs.get('zods', getCdlValue(4)))    # Zenith angles of departure (in degrees)
        self.zoas = toRadian(kwargs.get('zoas', getCdlValue(5)))    # Zenith angles of arrival (in degrees)

        self.hasLos = kwargs.get('hasLos', False if self.profile is None else (self.profile in "DE"))
        
        # Cross-polarization Power in dB
        self.xPolPower = kwargs.get('xPolPower', 10.0 if self.profile is None else perClusterParams[self.profile][1])
        
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

        # Note that there is at most one LOS cluster and that is assumed to be the first cluster. This is the K-Factor
        # of the first cluster in dB
        self.kFactorLos = kwargs.get('kFactorLos', (self.pathPowers[0]-self.pathPowers[1]) if self.hasLos else None)
        if self.profile is not None:
            self.scaleDelays()
            if self.kFactor is not None:    self.applyKFactorScaling()
        elif self.hasLos:
            # For custom models with LOS path, we split the first path into LOS and NLOS
            # Also note that it is assumed the custom values for powers and delays do not need angle and K-Factor scaling
            k1st = toLinear(self.kFactorLos)
            p1st = toLinear(self.pathPowers[0])
            pathPowers1st = -toDb(p1st + p1st/k1st)
            self.pathPowers = np.concatenate( ([pathPowers1st, pathPowers1st-self.kFactorLos], self.pathPowers[1:]))
            self.pathDelays = np.concatenate(([self.pathDelays[0]], self.pathDelays))
            self.aods = np.concatenate(([self.aods[0]], self.aods))
            self.aoas = np.concatenate(([self.aoas[0]], self.aoas))
            self.zods = np.concatenate(([self.zods[0]], self.zods))
            self.zoas = np.concatenate(([self.zoas[0]], self.zoas))

        # RMS angle spreads for aods, aoas, zods, zoas respectively.
        # This is the main angle spread not to be confused with the one used for angle scaling.
        angleSpreadsDefault = [4.0, 10.0, 2.0, 2.0] if self.profile is None else perClusterParams[self.profile][0]
        
        # The angle spreads applied to normalized angles in degrees
        self.angleSpreads = toRadian(kwargs.get('angleSpreads', angleSpreadsDefault))
        
        n, m = len(self.aods) - (1 if self.hasLos else 0), 20   # n clusters, m rays per cluster
        
        # Note:
        # The rayCoupling and initialPhases do not need to be specified. These are set randomly. The capability to
        # specify them is not documented and may be removed later.
        self.rayCoupling = kwargs.get('rayCoupling', None)                  # Ray Coupling values - Not documented
        self.randomRayCoupling = True
        if self.rayCoupling is not None:
            self.randomRayCoupling = False
            self.rayCoupling = np.int32(self.rayCoupling)
            if self.rayCoupling.shape != (3,n,m):
                raise ValueError(f"Invalid 'rayCoupling' shape! Must be {(3,n,m)} but it is {self.rayCoupling.shape}")
            if np.any(self.rayCoupling>=m) or np.any(self.rayCoupling<0):
                raise ValueError(f"'rayCoupling' values must be between 0 and {m} (inclusive)!")

        self.initialPhases = toRadian(kwargs.get('initialPhases', None))    # Initial phases in degrees - Not documented
        self.randomInitialPhases = True
        if self.initialPhases is not None:
            self.randomInitialPhases = False
            self.initialPhases = np.float64(self.initialPhases)
            if self.initialPhases.shape != (2,2,n,m):
                raise ValueError(f"Invalid 'initialPhases' shape! Must be {(2,2,n,m)} but it is {self.initialPhases.shape}")
            if np.any(self.initialPhases<-np.pi) or np.any(self.initialPhases>np.pi):
                raise ValueError("'initialPhases' values must be between -𝛑 and 𝛑!")

        self.restart()
        
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this CDL channel model object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it serves as the title for the printed information. If `None` (the default), an 
            automatic title is generated based on the channel model parameters.

        getStr : Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string.
            Otherwise, nothing is returned.
        """
        if title is None:
            if self.profile is None:    title = "Customized CDL Channel Properties:"
            else:                       title = f"CDL-{self.profile} Channel Properties:"

        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  carrierFreq:          {freqStr(self.carrierFreq)}\n"
        repStr += indent*' ' + f"  normalizeGains:       {str(self.normalizeGains)}\n"
        repStr += indent*' ' + f"  normalizeOutput:      {str(self.normalizeOutput)}\n"
        repStr += indent*' ' + f"  txDir:                {self.txDir}\n"
        repStr += indent*' ' + f"  filterLen:            {self.filterLen} samples\n"
        repStr += indent*' ' + f"  delayQuantSize:       {self.delayQuantSize}\n"
        repStr += indent*' ' + f"  stopBandAtten:        {self.stopBandAtten} dB\n"
        repStr += indent*' ' + f"  dopplerShift:         {freqStr(self.dopplerShift)}\n"
        repStr += indent*' ' + f"  coherenceTime:        {self.coherenceTime*1000:.3f} milliseconds\n"

        repStr += indent*' ' + f"  delaySpread:          {self.delaySpread} ns\n"
        repStr += indent*' ' + f"  ueDirAZ:              {np.round(toDegrees(self.ueDirAZ[0]))}°, " + \
                                                       f"{np.round(toDegrees(self.ueDirAZ[1]))}°\n"

        if self.angleScaling is not None:
            repStr += indent*' ' + "  Angle Scaling:\n"
            repStr += indent*' ' + "    Means:               %s°\n"%("° ".join(str(int(np.round(toDegrees(angle))))
                                                                             for angle in self.scalingAngleMeans ))
            repStr += indent*' ' + "    RMS Spreads:         %s°\n"%("° ".join(str(int(np.round(toDegrees(angle))))
                                                                           for angle in self.scalingAngleSpreads ))

        repStr += indent*' ' + f"  Cross Pol. Power:     {self.xPolPower} dB\n"
        if self.profile is None and self.hasLos:
            repStr += indent*' ' + f"  K-Factor:             {self.kFactorLos} dB\n"
        repStr += indent*' ' + "  angleSpreads:         %s°\n"%("° ".join(str(int(np.round(toDegrees(angle))))
                                                                             for angle in self.angleSpreads ))
        
        repStr += self.txAntenna.print(indent+2, "TX Antenna:", True)
        if np.any(self.txOrientation):
            repStr += indent*' ' + "    Orientation (𝛼,𝛃,𝛄): %s°\n"%("° ".join(str(int(np.round(toDegrees(a))))
                                                                                 for a in self.txOrientation ))
        
        repStr += self.rxAntenna.print(indent+2, "RX Antenna:", True)
        if np.any(self.rxOrientation):
            repStr += indent*' ' + "    Orientation (𝛼,𝛃,𝛄): %s°\n"%("° ".join(str(int(np.round(toDegrees(a))))
                                                                                 for a in self.rxOrientation ))

        repStr += indent*' ' + f"  hasLOS:               {self.hasLos}\n"
        if self.hasLos:
            repStr += indent*' ' + "  LOS Path:\n"
            repStr += indent*' ' + f"    Delay (ns):         {self.pathDelays[0]:.5f}\n"
            repStr += indent*' ' + f"    Power (dB):         {self.pathPowers[0]:.5f}\n"
            repStr += indent*' ' + f"    AOD (Deg):          {int(self.aods[0])}\n"
            repStr += indent*' ' + f"    AOA (Deg):          {int(self.aoas[0])}\n"
            repStr += indent*' ' + f"    ZOD (Deg):          {int(self.zods[0])}\n"
            repStr += indent*' ' + f"    ZOA (Deg):          {int(self.zoas[0])}\n"
        o = 1 if self.hasLos else 0
        repStr += indent*' ' + f"  NLOS Paths ({len(self.pathDelays)-o}):\n"
        repStr += getMultiLineStr("  Delays (ns)       ", self.pathDelays[o:], indent, "%-5f", 5, numPerLine=12)
        repStr += getMultiLineStr("  Powers (dB)       ", self.pathPowers[o:], indent, "%-5f", 5, numPerLine=12)
        repStr += getMultiLineStr("  AODs (Deg)        ", np.round(toDegrees(self.aods[o:])), indent, "%-4d", 4, numPerLine=12)
        repStr += getMultiLineStr("  AOAs (Deg)        ", np.round(toDegrees(self.aoas[o:])), indent, "%-4d", 4, numPerLine=12)
        repStr += getMultiLineStr("  ZODs (Deg)        ", np.round(toDegrees(self.zods[o:])), indent, "%-4d", 4, numPerLine=12)
        repStr += getMultiLineStr("  ZOAs (Deg)        ", np.round(toDegrees(self.zoas[o:])), indent, "%-4d", 4, numPerLine=12)

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def restart(self, restartRanGen=False, applyToBwp=True):
        r"""
        This method first re-initializes the random object if a ``seed`` was provided to this channel model and the 
        ``restartRanGen`` parameter is set to `True`. It then randomly re-initializes the ray coupling and initial 
        phases and calls the base class :py:meth:`~neoradium.channelmodel.ChannelModel.restart`.

        Parameters
        ----------
        restartRanGen : Boolean
            If a ``seed`` was not provided to this channel model, this parameter is ignored. Otherwise, if 
            ``restartRanGen`` is set to `True`, this channel model's random generator is reset and if 
            ``restartRanGen`` is `False` (default), the random generator is not reset. This means if 
            ``restartRanGen`` is `False`, calling this function starts a new sequence of channel instances, 
            which differs from the sequence when the channel was instantiated.

        applyToBwp : Boolean
            If set to `True` (the default), this function restarts the :py:class:`~neoradium.carrier.BandwidthPart` 
            associated with this channel model. Otherwise, the :py:class:`~neoradium.carrier.BandwidthPart` state 
            remains unchanged.
        """
        if (self.seed is not None) and restartRanGen: self.rangen = random.getGenerator(self.seed)
        if self.randomRayCoupling:      self.rayCoupling = self.getRandomRayCoupling()
        if self.randomInitialPhases:    self.initialPhases = self.getRandomInitialPhases()
        super().restart(restartRanGen, applyToBwp)

    # ******************************************************************************************************************
    @property           # This property is already documented above in the __init__ function.
    def nrNt(self):     return (self.rxAntenna.getNumElements(), self.txAntenna.getNumElements())

    # ******************************************************************************************************************
    def scaleDelays(self):                  # Not documented
        self.pathDelays *= self.delaySpread # Path delays in nanoseconds (See TR38.901 - Sec. 7.7.3, Scaling of delays)
            
    # ******************************************************************************************************************
    def getPathGains(self):                 # Not documented (See "getPathGains" in the base class)
        gains = self.getNLOSgains()                                         # Shape: nc x nr x nt x numNLOS
        if self.hasLos:
            gains = np.concatenate((self.getLOSgains(), gains), axis=3)     # Shape: nc x nr x nt x (numNLOS+1)
        return gains                                                        # Shape: nc x nr x nt x np

    # ******************************************************************************************************************
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

    # ******************************************************************************************************************
    def getLOSgains(self):                          # Not documented
        # This function calculates the gain for the LOS cluster. It must be called only if current channel model
        # contains LOS clusters.
        assert self.hasLos, "'getLOS' function was called for a profile that does not contain LOS information!"
        
        # STEP-1 (See Step-1 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Generate departure and arrival angles. These are all 1x1 matrices (one cluster, one ray).
        # Note: No need to use "rayOffsets" for LOS case.
        phiD = self.aods[0:1].reshape(1,1)      # Shape: 1 x 1
        phiA = self.aoas[0:1].reshape(1,1)      # Shape: 1 x 1
        thetaD = self.zods[0:1].reshape(1,1)    # Shape: 1 x 1
        thetaA = self.zoas[0:1].reshape(1,1)    # Shape: 1 x 1
        pN = toLinear(self.pathPowers[0])       # Shape: Scalar

        if self.angleScaling is not None:
            # Need to do angle scaling:
            phiD, phiA, thetaD, thetaA = self.applyAngleScaling(phiD, phiA, thetaD, thetaA, pN)

        phiD = self.wrapAngles(phiD, "-180,180")
        phiA = self.wrapAngles(phiA, "-180,180")
        thetaD = self.wrapAngles(thetaD, "0,180")
        thetaA = self.wrapAngles(thetaA, "0,180")

        nr, nt = self.nrNt
        # STEP-2 (See Step-2 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # No random coupling is needed for LOS case because there is only one cluster and one ray
        
        # STEP-3 (See Step-3 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Generate the cross-polarization power ratios:
        # "Kappa" is not needed for LOS case. The polarization matrix is fixed.

        # STEP-4 (See Step-4 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Draw initial random phases (See Step-10 in TR38.901 - 7.5)
        # No initial phases are needed for LOS case. The polarization matrix is fixed
        
        # Get the TX field part and TX location part in TR38.901 - Eq. 7.5-29
        fieldTx, locTx = self.txAntenna.getElementsFields(thetaD, phiD, self.txOrientation)# nt x 2 x 1 x 1 & nt x 1 x 1

        # Get the RX field part and RX location part in TR38.901 - Eq. 7.5-29
        fieldRx, locRx = self.rxAntenna.getElementsFields(thetaA, phiA, self.rxOrientation)# nr x 2 x 1 x 1 & nr x 1 x 1

        # Get the polarization matrix part in TR38.901 - Eq. 7.5-29
        polMat = np.float64([[1,0],[0,-1]])                                                 # Shape:  2 x 2
        
        # Get the doppler term in TR38.901 - Eq. 7.5-29
        doppler = self.getDopplerFactor(thetaA, phiA)                                       # Shape:  nc

        # Now that we have built all parts of TR38.901 - Eq. 7.5-29, we need to combine all of them together. Here are
        # shapes of different parts of TR38.901 - Eq. 7.5-29  complex tensor. (Squeezing out the 1 x 1 parts)
        #       fieldRx: nr x 2
        #       polMat:  2 x 2
        #       fieldTx: nt x 2
        #       locRx:   nr
        #       locTx:   nt
        #       doppler: t
        # The output will be a "nr x nt x t x 1"
        
        # First fieldRx x polMat x fieldTx
        hLOS = ((fieldRx.reshape(-1,1,2,1) * polMat.reshape(1,1,2,2)).sum(2).reshape(-1,1,2) * \
                 fieldTx.reshape(1, -1, 2)).sum(2)                                          # Shape: nr x nt
        # Now apply location factors
        hLOS = hLOS * locRx.reshape(-1, 1) * locTx.reshape(1, -1)                           # Shape: nr x nt
        # Applying the doppler
        hLOS = hLOS.reshape(1,nr,nt) * doppler.reshape(-1,1,1)                              # Shape: nc x nr x nt
        # Apply the scaling
        hLOS *= np.sqrt(pN)                                                                 # Shape: nc x nr x nt
        return hLOS.reshape(-1, nr, nt, 1)                                                  # Shape: nc x nr x nt x 1
        
    # ******************************************************************************************************************
    def getNLOSgains(self):                         # Not documented
        # This function calculates the gains for all NLOS clusters.
        
        # STEP-1 (See Step-1 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Generate departure and arrival angles
        offset = 1 if self.hasLos else 0
        cASD, cASA, cZSD, cZSA = self.angleSpreads
        phiD   = self.aods[offset:].reshape(-1,1) + cASD*np.float64(rayOffsets)     # Shape: n x m
        phiA   = self.aoas[offset:].reshape(-1,1) + cASA*np.float64(rayOffsets)     # Shape: n x m
        thetaD = self.zods[offset:].reshape(-1,1) + cZSD*np.float64(rayOffsets)     # Shape: n x m
        thetaA = self.zoas[offset:].reshape(-1,1) + cZSA*np.float64(rayOffsets)     # Shape: n x m
        pN = toLinear(self.pathPowers[offset:])                                     # Shape: n
        
        if self.angleScaling is not None:
            # Need to do angle scaling:
            phiD, phiA, thetaD, thetaA = self.applyAngleScaling(phiD, phiA, thetaD, thetaA, pN)

        phiD = self.wrapAngles(phiD, "-180,180")
        phiA = self.wrapAngles(phiA, "-180,180")
        thetaD = self.wrapAngles(thetaD, "0,180")
        thetaA = self.wrapAngles(thetaA, "0,180")

        n, m = phiD.shape
        nr, nt = self.nrNt

        # STEP-2 (See Step-2 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Random coupling of rays within clusters
        phiD, phiA, thetaD, thetaA = self.shuffleRays(phiD, phiA, thetaD, thetaA)

        # STEP-3 (See Step-3 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Generate the cross-polarization power ratios:
        kappa = toLinear(self.xPolPower)    # See Eq. 7.7-0b in Step-3 of TR38.901 - 7.7.1

        # STEP-4 (See Step-4 in TR38.901 - 7.7.1) ----------------------------------------------------------------------
        # Draw initial random phases (See Step-10 in TR38.901 - 7.5)
        phiInit = self.initialPhases        # Uniform between -𝛑,𝛑. Shape: 2 x 2 x n x m

        # Get the TX field part and TX location part in TR38.901 - Eq. 7.5-22
        fieldTx, locTx = self.txAntenna.getElementsFields(thetaD, phiD, self.txOrientation)# nt x 2 x n x m & nt x n x m

        # Get the RX field part and RX location part in TR38.901 - Eq. 7.5-22
        fieldRx, locRx = self.rxAntenna.getElementsFields(thetaA, phiA, self.rxOrientation)# nr x 2 x n x m & nr x n x m

        # Get the polarization matrix part in TR38.901 - Eq. 7.5-22
        polMat = np.exp(1j*phiInit) * np.sqrt([[1, 1/kappa], [1/kappa, 1]]).reshape(2,2,1,1)    # Shape:  2 x 2 x n x m

        # Get the doppler term in TR38.901 - Eq. 7.5-22
        doppler = self.getDopplerFactor(thetaA, phiA)                                           # Shape:  nc x n x m

        # Now that we have built all parts of TR38.901 - Eq. 7.5-22, we need to combine all of them together. Here
        # are shapes of different parts of TR38.901 - Eq. 7.5-22  complex tensor.
        #       fieldRx: nr x 2 x n x m
        #       polMat:  2 x 2 x n x m
        #       fieldTx: nt x 2 x n x m
        #       locRx:   nr x n x m
        #       locTx:   nt x n x m
        #       doppler: t x n x m
        # The output will be an "nr x nt x t x n"
        
        # First fieldRx x polMat x fieldTx
        hNLOS = ((fieldRx.reshape(-1,1,2,1,n,m) * polMat.reshape(1,1,2,2,n,m)).sum(2).reshape(-1,1,2,n,m) * \
                 fieldTx).sum(2)                                                    # Shape: nr x nt x n x m
        # Now apply location factors
        hNLOS = hNLOS * locRx.reshape(-1, 1, n, m) * locTx.reshape(1, -1, n, m)     # Shape: nr x nt x n x m
        # Applying the doppler:
        hNLOS = hNLOS.reshape(1,nr,nt,n,m) * doppler.reshape(-1,1,1,n, m)           # Shape: nc x nr x nt x n x m
        # Now sum over m (Combining rays in each cluster)
        hNLOS = hNLOS.sum(4)                                                        # Shape: nc x nr x nt x n
        # Apply the scaling
        hNLOS *= np.sqrt(pN/m).reshape(1,1,1,-1)
        return hNLOS                                                                # Shape: nc x nr x nt x n

    # ******************************************************************************************************************
    def getRandomRayCoupling(self):             # Not documented
        # This function randomly creates the ray-coupling values (See Step-2 in TR38.901, Section 7.7.1)
        n, m = len(self.aods) - (1 if self.hasLos else 0), 20
        return np.int32([ [self.rangen.choice(range(m), size=m, replace=False)
                                for _ in range(n)] for _ in range(3) ])             # Shape: 3 x n x m

    # ******************************************************************************************************************
    def getRandomInitialPhases(self):
        # Draw initial random phases (See Step-10 in TR38.901 - 7.5)
        n, m = len(self.aods) - (1 if self.hasLos else 0), 20
        return 2*np.pi * self.rangen.random(size=(2,2,n,m)) - np.pi     # Uniform between -𝛑,𝛑. Shape: 2 x 2 x n x m

    # ******************************************************************************************************************
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

    # ******************************************************************************************************************
    def shuffleRays(self, phiD, phiA, thetaD, thetaA):              # Not documented
        # This function shuffles the rays in the clusters randomly using the rayCoupling parameter.
        n, m = phiD.shape
        rowIndexes = np.int32([m*[rr] for rr in range(n)])          # Shape: n x m

        # Shuffle rays for each path
        phiA    = phiA  [ (rowIndexes, self.rayCoupling[0]) ]
        thetaA  = thetaA[ (rowIndexes, self.rayCoupling[1]) ]
        thetaD  = thetaD[ (rowIndexes, self.rayCoupling[2]) ]
        return phiD, phiA, thetaD, thetaA

    # ******************************************************************************************************************
    def getDopplerFactor(self, theta, phi):           # Not documented
        # This function calculates the doppler term in TR38.901 - Eq. 7.5-22
        vPhi, vTheta = self.ueDirAZ         # Direction (angles) of UE movement in phi and theta in radians
        # Simplifying : d = speed/wavelen. Instead of using vBar and v, we use dBar and doppler and remove the lambda
        # in the denuminator.
        # The following is the adapted version of TR38.901 - Eq. 7.5-25
        dBar = self.dopplerShift * np.array([ np.sin(vTheta) * np.cos(vPhi),
                                              np.sin(vTheta) * np.sin(vPhi),
                                              np.cos(vTheta) ])
        
        sinTheta = np.sin(theta)
        rHatRx = np.array([ sinTheta * np.cos(phi),
                            sinTheta * np.sin(phi),
                            np.cos(theta) ])

        chanTimes = self.chanGainSamples/self.sampleRate
        return np.exp(2j * np.pi * chanTimes.reshape(-1,1,1) * (rHatRx*dBar.reshape(3,1,1)).sum(0)) # Shape: nc x n x m

    # ******************************************************************************************************************
    def applyAngleScaling(self, phiD, phiA, thetaD, thetaA, p):     # Not documented
        # This function applies the Angle Scaling according to 3GPP TR 38.901, Section 7.7.5.1 and Annex A.
        assert self.angleScaling is not None
        n,m = phiA.shape

        # Desired mean and spread as provided
        asPhiD, asPhiA, asThetaD, asThetaA = self.scalingAngleSpreads   # Angle Spread for phiD, phiA, thetaD, thetaA
        maPhiD, maPhiA, maThetaD, maThetaA = self.scalingAngleMeans     # Mean Angle for phiD, phiA, thetaD, thetaA
        
        # Calculate Model mean and spread: (See TR38.901 - Annex A)
        def getModelMeanAndSpread(angles):
            weightedSum = (np.exp(1j*angles)*p.reshape(-1,1)).sum()/m  # Numinator of the fraction in TR38.901 - Eq. A-1
            angularSpread = np.sqrt(-2*np.log(np.abs(weightedSum/(p.sum()))))   # The 'AS' defined in TR38.901 - Eq. A-1
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
        scaledPhiD = self.wrapAngles(scaledPhiD, "0,360")   # Wrap azimuth angles around to be within [0, 360] degrees
        scaledPhiA = self.wrapAngles(scaledPhiA, "0,360")   # Wrap azimuth angles around to be within [0, 360] degrees
        scaledThetaD = self.wrapAngles(scaledThetaD, "Clip-0,180")  # Clip zenith angles to be within [0, 180] degrees
        scaledThetaA = self.wrapAngles(scaledThetaA, "Clip-0,180")  # Clip zenith angles to be within [0, 180] degrees

        return scaledPhiD, scaledPhiA, scaledThetaD, scaledThetaA

    # ******************************************************************************************************************
    @classmethod
    def getChanGen(cls, numChannels, bwp, profiles="ABCDE", delaySpread=(10,500),
                   ueSpeed=(10,70), ueDir=(0,360), **kwargs):
        r"""
        Returns a generator object that can generate CDL channel matrices based on the given parameters and criteria.

        Refer to the notebook :doc:`../Playground/Notebooks/Channels/CdlChannelDataset` for an example of 
        using this method.
        
        Parameters
        ----------
        numChannels: int 
            The number of channel matrices generated by the returned generator.
            
        bwp : :py:class:`~neoradium.carrier.BandwidthPart` 
            The bandwidth part object used by the returned generator to construct channel matrices.

        profiles: str        
            A string containing a combination of upper case letters 'A', 'B', 'C', 'D', and 'E'. For example the
            sting "ACE", means the CDL profiles 'A', 'C', and 'E' are considered when creating the channel matrices.
            The default is "ABCDE", which means all CDL profiles are included.

        delaySpread: float, tuple, or list
            Specifies the delay spread in nanoseconds. It can be one of the following:
            
                * If it is a tuple of the form ``(dsMin, dsMax)``, a random value is uniformly sampled between 
                  ``dsMin`` and ``dsMax`` for each channel matrix.
                  
                * If it is a list of the form [:math:`d_1`, :math:`d_2`, ..., :math:`d_n`], for each channel matrix a 
                  random delay spread is picked from those specified in the list.
                  
                * If it is a single number, then the same delay spread is used for all channel matrices. 
            
            The default is ``(10,500)``.

        ueSpeed: float, tuple, or list
            Specifies the speed of the UE in meters per second. It can be one of the following:
            
                * If it is a tuple of the form ``(speedMin, speedMax)``, a random value is uniformly sampled between 
                  ``speedMin`` and ``speedMax`` for each channel matrix.
                  
                * If it is a list of the form [:math:`s_1`, :math:`s_2`, ..., :math:`s_n`], for each channel matrix a 
                  random speed is picked from those specified in the list.
                  
                * If it is a single number, then the same UE speed is used for all channel matrices.
            
            The default is ``(0,20)``.
              
        ueDir: float, tuple, or list
            Specifies the direction of UE movement in the X-Y plane as an angle in degrees. It can be one 
            of the following:
            
                * If it is a tuple of the form ``(dirMin, dirMax)``, a random value is uniformly sampled between 
                  ``dirMin`` and ``dirMax`` for each channel matrix.
                  
                * If it is a list of the form [:math:`a_1`, :math:`a_2`, ..., :math:`a_n`], for each channel matrix a 
                  random UE direction is picked from those specified in the list.
                  
                * If it is a single number, then the same UE direction is used for all channel matrices. 
            
            The default is ``(0, 360)``.
                   
        kwargs : dict
            Here is a list of additional optional parameters that can be used to further customize the calculation 
            of the channel matrices:
            
                :normalizeGains: If the default value of `True` is used, the path gains are normalized.
                    
                :normalizeOutput: If the default value of `True` is used, the gains are normalized based on the 
                    number of receive antennas.

                :filterLen: The length of the channel filter. The default is 16 sample.
                
                :delayQuantSize: The size of the delay fraction quantization for the channel filter. The default is 64.
                
                :stopBandAtten: The stop-band attenuation value (in dB) used by the channel filter. The default 
                    is 80 dB.
                
                :txAntenna: The transmitter antenna, which is an instance of either the 
                    :py:class:`neoradium.antenna.AntennaPanel` or :py:class:`neoradium.antenna.AntennaArray` class. 
                    By default, it is a single antenna in a 1x1 antenna panel with vertical polarization.
                
                :rxAntenna: The receiver antenna which is an instance of either the 
                    :py:class:`neoradium.antenna.AntennaPanel` or :py:class:`neoradium.antenna.AntennaArray` class. 
                    By default, it is a single antenna in a 1x1 antenna panel with vertical polarization.
                    
                :txOrientation: The orientation of the transmitter antenna. This is a list of 3 angle values in degrees
                    for the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901, Section 7.1.3** for more
                    information.

                :rxOrientation: The orientation of the receiver antenna. This is a list of 3 angle values in degrees 
                    for the *bearing* angle :math:`\alpha`, *downtilt* angle :math:`\beta`, and *slant* angle 
                    :math:`\gamma`. The default is [0,0,0]. Please refer to **3GPP TR 38.901, Section 7.1.3** for more
                    information.

                :seed: The seed used to generate CDL channel matrices. The default value is `None`, indicating 
                    that this channel model uses the **NeoRadium**’s :doc:`global random generator <./Random>`. In
                    this case the results are not reproducible.
                    
                :carrierFreq: The carrier frequency of the CDL channel model in Hz. The default is 3.5 GHz.
                
                :kFactor: The K-Factor (in dB) used for scaling. The default is `None`. If not specified 
                    (``kFactor=None``), K-factor scaling is disabled.

                :xPolPower: The cross-polarization Power in dB. The default is 10db. For more details please refer 
                    to "Step 3" in **3GPP TR 38.901, Section 7.7.1**.

                :angleScaling: The :ref:`Angle Scaling <AngleScaling>` parameters. If specified, it must be a tuple of
                    2 NumPy arrays.
                    
                    The first item specifies the mean values for angle scaling. It’s a 1-D NumPy array containing 
                    four values for: the *Azimuth angle of Departure*, *Azimuth angle of Arrival*, *Zenith angle of 
                    Departure*, and *Zenith angle of Arrival*.
                    
                    The second item specifies the RMS angle spread values. It is a 1-D NumPy array containing four RMS 
                    values for: the *Azimuth angle of Departure*, *Azimuth angle of Arrival*, *Zenith angle of 
                    Departure*, and *Zenith angle of Arrival*. For more information, please refer to 
                    :ref:`Angle Scaling <AngleScaling>`.
                    
                    If this value is set to `None` (the default), the *Angle Scaling* is disabled.
                    
        Returns
        -------
        ``ChanGen``, a generator object that is used to generate channel matrices.
        
        
        **Example:**
                       
        .. code-block:: python
        
            # First create a carrier object with 25 PRBs and 15 kHz subcarrier spacing
            carrier = Carrier(startRb=0, numRbs=25, spacing=15)

            # Now create the generator
            chanGen = CdlChannel.getChanGen(1000, carrier.curBwp,       # Number of channels and bandwidth part
                                            profiles="ABCDE",           # Randomly pick a CDL profile
                                            delaySpread=(10,500),       # Uniformly sample between 10 and 500 ns
                                            ueSpeed=(5,20),             # Uniformly sample between 5 and 20 m/s
                                            ueDir=[45, 135, 225, 315],  # Randomly pick one of these UE directions
                                            carrierFreq=4e9,            # Carrier frequency
                                            txAntenna=AntennaPanel([2,4], polarization="x"),  # 16 TX antennas
                                            rxAntenna=AntennaPanel([1,2], polarization="x"),  # 4 RX antennas
                                            seed=123)

            # Create the channel matrices
            allChannels = np.stack([chan for chan in chanGen])  
            print(f"Shape of 'allChannels': {allChannels.shape}")       # Prints (1000, 14, 300, 4, 16)       
        """
        seed = kwargs.pop("seed", None)
        carrierFreq = kwargs.get("carrierFreq", 3.5e9)
        class ChanGen:
            def __init__(self): self.reset()
            def __iter__(self): return self
            def __next__(self):
                if self.cur >= numChannels: raise StopIteration
                # Create a CDL channel and get the channel matrix
                self.curChan = CdlChannel(bwp, str(self.profiles[self.cur]),
                                          delaySpread=self.delaySpreads[self.cur],
                                          dopplerShift=self.dopplerShifts[self.cur],
                                          ueDirAZ=[self.ueDirs[self.cur], 90],
                                          seed=self.chanSeeds[self.cur],
                                          **kwargs)
                self.cur += 1
                return self.curChan.getChannelMatrix()
                   
            def __len__(self):
                return len(self.chanSeeds)

            def reset(self):
                rangen = random if seed is None else random.getGenerator(seed)  # The random number generator
                self.cur = 0
                self.curChan = None

                self.profiles = rangen.choice(list(profiles), size=numChannels) # Pick a profile for each channel matrix

                # Get a speed (in m/s) for each channel matrix
                if type(ueSpeed)==tuple:        speeds = rangen.uniform(*ueSpeed, numChannels)
                elif type(ueSpeed)==list:       speeds = rangen.choice(np.float32(ueSpeed), numChannels)
                else:                           speeds = np.float32(numChannels*[ueSpeed])
                # Calculate the doppler shift based on the speed and carrier frequency
                self.dopplerShifts = speeds * carrierFreq/299792458

                # Get a UE direction of movement for each channel matrix
                if type(ueDir)==tuple:          self.ueDirs = rangen.uniform(*ueDir, numChannels)*np.pi/180
                elif type(ueDir)==list:         self.ueDirs = rangen.choice(np.float32(ueDir), numChannels)*np.pi/180
                else:                           self.ueDirs = np.float32(numChannels*[ueDir])*np.pi/180
            
                # Get a delay spread for each channel matrix
                if type(delaySpread)==tuple:    self.delaySpreads = rangen.uniform(*delaySpread, numChannels)
                elif type(delaySpread)==list:   self.delaySpreads = rangen.choice(np.float32(delaySpread), numChannels)
                else:                           self.delaySpreads = np.float32(numChannels*[delaySpread])

                # Get a seed for each channel matrix
                self.chanSeeds = rangen.integers(10,1000, size=numChannels)
                    
        return ChanGen()
