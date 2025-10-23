# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``polar.py`` contains the API used for 
`Polar coding <https://en.wikipedia.org/wiki/Polar_code_(coding_theory)>`_. It implements the class
:py:class:`PolarBase`, which is the base class for other polar coding classes and is derived from the
:py:class:`~neoradium.chancodebase.ChanCodeBase` class. It also implements the classes :py:class:`PolarEncoder` and 
:py:class:`PolarDecoder` both of which are derived from :py:class:`PolarBase`.

This implementation is based on **3GPP TS 38.212**.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 01/09/2024    Shahab Hamidi-Rad       Completed the documentation
# **********************************************************************************************************************
import numpy as np
from .chancodebase import ChanCodeBase

# Suggested Course:
# LDPC and Polar Codes in 5G Standard: https://www.youtube.com/playlist?list=PLyqSpQzTE6M81HJ26ZaNv0V3ROBrcv-Kc
# This file is based on 3GPP TS 38.212 V17.0.0 (2021-12)
# The decoder class uses a recursive implementation of Successive Cancellation List decoder written from
# scratch. (See ``sclDecode``)

# **********************************************************************************************************************
# This is based on TS 38.212 Table 5.3.1.2-1: Polar sequence
reliabilitySeq = np.int16([
    0,    1,    2,    4,    8,    16,   32,   3,    5,    64,   9,    6,    17,   10,   18,   128,
    12,   33,   65,   20,   256,  34,   24,   36,   7,    129,  66,   512,  11,   40,   68,   130,
    19,   13,   48,   14,   72,   257,  21,   132,  35,   258,  26,   513,  80,   37,   25,   22,
    136,  260,  264,  38,   514,  96,   67,   41,   144,  28,   69,   42,   516,  49,   74,   272,
    160,  520,  288,  528,  192,  544,  70,   44,   131,  81,   50,   73,   15,   320,  133,  52,
    23,   134,  384,  76,   137,  82,   56,   27,   97,   39,   259,  84,   138,  145,  261,  29,
    43,   98,   515,  88,   140,  30,   146,  71,   262,  265,  161,  576,  45,   100,  640,  51,
    148,  46,   75,   266,  273,  517,  104,  162,  53,   193,  152,  77,   164,  768,  268,  274,
    518,  54,   83,   57,   521,  112,  135,  78,   289,  194,  85,   276,  522,  58,   168,  139,
    99,   86,   60,   280,  89,   290,  529,  524,  196,  141,  101,  147,  176,  142,  530,  321,
    31,   200,  90,   545,  292,  322,  532,  263,  149,  102,  105,  304,  296,  163,  92,   47,
    267,  385,  546,  324,  208,  386,  150,  153,  165,  106,  55,   328,  536,  577,  548,  113,
    154,  79,   269,  108,  578,  224,  166,  519,  552,  195,  270,  641,  523,  275,  580,  291,
    59,   169,  560,  114,  277,  156,  87,   197,  116,  170,  61,   531,  525,  642,  281,  278,
    526,  177,  293,  388,  91,   584,  769,  198,  172,  120,  201,  336,  62,   282,  143,  103,
    178,  294,  93,   644,  202,  592,  323,  392,  297,  770,  107,  180,  151,  209,  284,  648,
    94,   204,  298,  400,  608,  352,  325,  533,  155,  210,  305,  547,  300,  109,  184,  534,
    537,  115,  167,  225,  326,  306,  772,  157,  656,  329,  110,  117,  212,  171,  776,  330,
    226,  549,  538,  387,  308,  216,  416,  271,  279,  158,  337,  550,  672,  118,  332,  579,
    540,  389,  173,  121,  553,  199,  784,  179,  228,  338,  312,  704,  390,  174,  554,  581,
    393,  283,  122,  448,  353,  561,  203,  63,   340,  394,  527,  582,  556,  181,  295,  285,
    232,  124,  205,  182,  643,  562,  286,  585,  299,  354,  211,  401,  185,  396,  344,  586,
    645,  593,  535,  240,  206,  95,   327,  564,  800,  402,  356,  307,  301,  417,  213,  568,
    832,  588,  186,  646,  404,  227,  896,  594,  418,  302,  649,  771,  360,  539,  111,  331,
    214,  309,  188,  449,  217,  408,  609,  596,  551,  650,  229,  159,  420,  310,  541,  773,
    610,  657,  333,  119,  600,  339,  218,  368,  652,  230,  391,  313,  450,  542,  334,  233,
    555,  774,  175,  123,  658,  612,  341,  777,  220,  314,  424,  395,  673,  583,  355,  287,
    183,  234,  125,  557,  660,  616,  342,  316,  241,  778,  563,  345,  452,  397,  403,  207,
    674,  558,  785,  432,  357,  187,  236,  664,  624,  587,  780,  705,  126,  242,  565,  398,
    346,  456,  358,  405,  303,  569,  244,  595,  189,  566,  676,  361,  706,  589,  215,  786,
    647,  348,  419,  406,  464,  680,  801,  362,  590,  409,  570,  788,  597,  572,  219,  311,
    708,  598,  601,  651,  421,  792,  802,  611,  602,  410,  231,  688,  653,  248,  369,  190,
    364,  654,  659,  335,  480,  315,  221,  370,  613,  422,  425,  451,  614,  543,  235,  412,
    343,  372,  775,  317,  222,  426,  453,  237,  559,  833,  804,  712,  834,  661,  808,  779,
    617,  604,  433,  720,  816,  836,  347,  897,  243,  662,  454,  318,  675,  618,  898,  781,
    376,  428,  665,  736,  567,  840,  625,  238,  359,  457,  399,  787,  591,  678,  434,  677,
    349,  245,  458,  666,  620,  363,  127,  191,  782,  407,  436,  626,  571,  465,  681,  246,
    707,  350,  599,  668,  790,  460,  249,  682,  573,  411,  803,  789,  709,  365,  440,  628,
    689,  374,  423,  466,  793,  250,  371,  481,  574,  413,  603,  366,  468,  655,  900,  805,
    615,  684,  710,  429,  794,  252,  373,  605,  848,  690,  713,  632,  482,  806,  427,  904,
    414,  223,  663,  692,  835,  619,  472,  455,  796,  809,  714,  721,  837,  716,  864,  810,
    606,  912,  722,  696,  377,  435,  817,  319,  621,  812,  484,  430,  838,  667,  488,  239,
    378,  459,  622,  627,  437,  380,  818,  461,  496,  669,  679,  724,  841,  629,  351,  467,
    438,  737,  251,  462,  442,  441,  469,  247,  683,  842,  738,  899,  670,  783,  849,  820,
    728,  928,  791,  367,  901,  630,  685,  844,  633,  711,  253,  691,  824,  902,  686,  740,
    850,  375,  444,  470,  483,  415,  485,  905,  795,  473,  634,  744,  852,  960,  865,  693,
    797,  906,  715,  807,  474,  636,  694,  254,  717,  575,  913,  798,  811,  379,  697,  431,
    607,  489,  866,  723,  486,  908,  718,  813,  476,  856,  839,  725,  698,  914,  752,  868,
    819,  814,  439,  929,  490,  623,  671,  739,  916,  463,  843,  381,  497,  930,  821,  726,
    961,  872,  492,  631,  729,  700,  443,  741,  845,  920,  382,  822,  851,  730,  498,  880,
    742,  445,  471,  635,  932,  687,  903,  825,  500,  846,  745,  826,  732,  446,  962,  936,
    475,  853,  867,  637,  907,  487,  695,  746,  828,  753,  854,  857,  504,  799,  255,  964,
    909,  719,  477,  915,  638,  748,  944,  869,  491,  699,  754,  858,  478,  968,  383,  910,
    815,  976,  870,  917,  727,  493,  873,  701,  931,  756,  860,  499,  731,  823,  922,  874,
    918,  502,  933,  743,  760,  881,  494,  702,  921,  501,  876,  847,  992,  447,  733,  827,
    934,  882,  937,  963,  747,  505,  855,  924,  734,  829,  965,  938,  884,  506,  749,  945,
    966,  755,  859,  940,  830,  911,  871,  639,  888,  479,  946,  750,  969,  508,  861,  757,
    970,  919,  875,  862,  758,  948,  977,  923,  972,  761,  877,  952,  495,  703,  935,  978,
    883,  762,  503,  925,  878,  735,  993,  885,  939,  994,  980,  926,  764,  941,  967,  886,
    831,  947,  507,  889,  984,  751,  942,  996,  971,  890,  509,  949,  973,  1000, 892,  950,
    863,  759,  1008, 510,  979,  953,  763,  974,  954,  879,  981,  982,  927,  995,  765,  956,
    887,  985,  997,  986,  943,  891,  998,  766,  511,  988,  1001, 951,  1002, 893,  975,  894,
    1009, 955,  1004, 1010, 957,  983,  958,  987,  1012, 999,  1016, 767,  989,  1003, 990,  1005,
    959,  1011, 1013, 895,  1006, 1014, 1017, 1018, 991,  1020, 1007, 1015, 1019, 1021, 1022, 1023])

# **********************************************************************************************************************
# Input interleaving Pattern according to TS 38.212 V17.0.0 (2021-12) Table 5.3.1.1-1
inputInterleaver = [  0,   2,   4,   7,   9,  14,  19,  20,  24,  25,  26,  28,  31,  34,
                     42,  45,  49,  50,  51,  53,  54,  56,  58,  59,  61,  62,  65,  66,
                     67,  69,  70,  71,  72,  76,  77,  81,  82,  83,  87,  88,  89,  91,
                     93,  95,  98, 101, 104, 106, 108, 110, 111, 113, 115, 118, 119, 120,
                    122, 123, 126, 127, 129, 132, 134, 138, 139, 140,   1,   3,   5,   8,
                     10,  15,  21,  27,  29,  32,  35,  43,  46,  52,  55,  57,  60,  63,
                     68,  73,  78,  84,  90,  92,  94,  96,  99, 102, 105, 107, 109, 112,
                    114, 116, 121, 124, 128, 130, 133, 135, 141,   6,  11,  16,  22,  30,
                     33,  36,  44,  47,  64,  74,  79,  85,  97, 100, 103, 117, 125, 131,
                    136, 142,  12,  17,  23,  37,  48,  75,  80,  86, 137, 143,  13,  18,
                     38, 144,  39, 145,  40, 146,  41, 147, 148, 149, 150, 151, 152, 153,
                    154, 155, 156, 157, 158, 159, 160, 161, 162, 163 ]

# **********************************************************************************************************************
# Sub-block interleaving pattern based on TS 38.212 V17.0.0 (2021-12) Table 5.4.1.1-1
subBlockInterleaver = np.uint16([ 0,  1,  2,  4,  3,  5,  6,  7,  8, 16,  9, 17, 10, 18, 11, 19,
                                 12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27, 29, 30, 31])

# **********************************************************************************************************************
# The base class for polar coding
class PolarBase(ChanCodeBase):
    r"""
    This is the base class for all polar coding classes. Both :py:class:`PolarEncoder` and :py:class:`PolarDecoder`
    classes are derived from it. In 5G NR, polar coding is used for the following cases:
    
        - Downlink Control Information (DCI)
        - Uplink Control Information (UCI)
        - Physical Broadcast Channel (PBCH)
    """
    # ******************************************************************************************************************
    def __init__(self, payloadSize=0, rateMatchedLen=0, dataType=None, **kwargs):
        r"""
        Parameters
        ----------
        payloadSize: int
            The size of input bitstream not including the CRC bits. This is the value :math:`A` in **3GPP TS 38.212,
            Section 5.2.1**.
            
        rateMatchedLen: int
            The total length of rate-matched output bitstream. This is the value :math:`E` in **3GPP TS 38.212,
            Sections 5.3.1 and 5.4.1**.
            
        dataType: str or None
            The type of data using this Polar encoder/decoder. It can be one of the
            following:
            
            :"DCI": Downlink Control Information
            :"UCI": Uplink Control Information
            :"PBCH": Physical Broadcast Channel
            :None: Customized Polar Coding.

        kwargs : dict
            A set of optional arguments depending on the ``dataType``:

                :iBIL: Coded bits Interleaving flag. This is a boolean value that indicates whether coded bits
                    interleaving is enabled (`True`) or disabled (`False`). By default ``iBIL=False``. This
                    is the value :math:`I_{BIL}` in **3GPP TS 38.212, Section 5.4.1.3**. This parameter is ignored
                    if the ``dataType`` is not `None`. In this case, ``iBIL`` is set to `True` for 
                    ``dataType="UCI"``, and `False` for ``dataType="DCI"`` and ``dataType="PBCH"`` cases.

                :nMax: Max value of :math:`n` where :math:`N=2^n` is the length of the polar code. By default this
                    is set to 10 (which means :math:`N=1024`. This is the value :math:`N_{max}` in **3GPP TS 38.212,
                    Section 5.3.1.2**. This parameter is ignored if the ``dataType`` is not `None`. In this case,
                    ``nMax=10`` when ``dataType="UCI"``, and ``nMax=9`` for ``dataType="DCI"`` and ``dataType="PBCH"``
                    cases.

                :iIL: Input Interleaving flag. This is a boolean value that indicates whether input interleaving
                    is enabled (`True`) or disabled (`False`). By default ``iIL=False``. This is the value
                    :math:`I_{IL}` in **3GPP TS 38.212, Section 5.3.1.1**. This parameter is ignored if the 
                    ``dataType`` is not `None`. In this case, ``iIL`` is set to `False` for ``dataType="UCI"``,
                    and `True` for ``dataType="DCI"`` and ``dataType="PBCH"`` cases.

                :nPC: Total number of parity-check bits. By default this is set to 0. This is the value :math:`N_{PC}`
                    in **3GPP TS 38.212, Section 5.3.1**. This parameter is ignored if the ``dataType`` is not 
                    `None`. In this case, ``nPC=0`` when ``dataType`` is set to ``"DCI"`` or ``"PBCH"``. For the
                    ``"UCI"`` case, this value may be set to 0 or 3 which is determined based on the procedure
                    explained in **3GPP TS 38.212, Section 5.3.1.2**.
                    
                :nPCwm: The number of *Low-weight*, *High-Reliability* parity-check bits out of the total parity-check
                    bits ``nPC``. By default this is set to 0. This is the value :math:`n_{PC}^{wm}` in **3GPP TS
                    38.212, Sections 5.3.1.2, 6.3.1.3.1, and 6.3.2.3.1**. This parameter is ignored if the 
                    ``dataType`` is not `None`. In this case, ``nPCwm=0`` when ``dataType`` is set to ``"DCI"``
                    or ``"PBCH"``. For the ``"UCI"`` case, this value may be set to 0 or 1 which is determined based
                    on the procedure explained in **3GPP TS 38.212, Sections 6.3.1.3.1 and 6.3.2.3.1**.

                :iSeg: Segmentation flag. This is a boolean value that indicates whether segmentation is enabled 
                    (`True`) or disabled (`False`). By default ``iSeg=False``. This is the value :math:`I_{seg}`
                    in **3GPP TS 38.212, Section 5.2.1**. This parameter is ignored if the ``dataType`` is not
                    `None`. In this case, ``iSeg=False`` when ``dataType="DCI"`` or ``dataType="PBCH"``. When
                    ``dataType="UCI"``, ``iSeg`` is set based on the value of ``payloadSize``.

                :crcPoly: The CRC polynomial. This is a string specifying the CRC polynomial or `None`. If
                    specified, it must be one of the values specified in 
                    :py:meth:`~neoradium.chancodebase.ChanCodeBase.getCrc` for the ``poly`` parameter. The default 
                    value is ``"11"``. This parameter is ignored if the ``dataType`` is not `None`. In this case
                    ``crcPoly`` is set to ``"6"`` or ``"11"`` depending on ``payloadSize`` for ``dataType="UCI"``,
                    and ``"24C"`` for ``dataType="DCI"`` and ``dataType="PBCH"`` cases.
                    

        **Other Properties:**
        
            :rateMatchedBlockLen: The number of rate-matched bits transmitted for each code block when segmented. This
                is the same as ``rateMatchedLen`` if segmentation is disabled. This is the value :math:`E_r` in
                **3GPP TS 38.212, Section 5.5**.
                
            :codeBlockSize: The code block size. This is the value :math:`K` in **3GPP TS 38.212, Section 5.3.1**
                which includes the CRC bits (if any).

            :polarCodeSize: The polar code size :math:`N`. This is always a power of 2.
                    
            :msgBits: A list of indices of the message bits in the coded bitstream.
            
            :frozenBits: A list of indices of the *frozen bits* in the coded bitstream.
            
            :pcBits: A list of indices of the parity-check bits in the coded bitstream. This can be empty depending 
                on ``nPC``.
    
            :generator: The polar coding *generator* matrix as a 2-D NumPy array.
        """
        super().__init__()
        
        self.payloadSize = int(payloadSize)             # Payload size not including the CRC bits (A)
        self.rateMatchedLen = int(rateMatchedLen)       # Total transmitted bits (E)
        self.rateMatchedBlockLen = int(rateMatchedLen)  # The bits transmitted for each code block when segmented (Er)
        self.codeBlockSize = 0                          # Code Block Size including CRC bits (K)
        self.polarCodeSize = 0                          # The polar code size (N = 2^n)
        self.inInterleaveIndexes = None                 # Used for Input interleaving only if enabled (iIL=True)
        self.cbInterleaveIndexes = None                 # Used only if coded bit interleaving is enabled (iBIL=True)
        self.sbInterleaveIndexes = None                 # Sub-block interleaving indices. See in "initialize" function
        
        if dataType is None:                # If the dataType is not given all parameters should be specified
            self.iBIL =  kwargs.get('iBIL', False)      # Coded bits Interleaving flag
            self.nMax =  kwargs.get('nMax', 10)         # Max value of n. The total number of Polar-Coded bits is 2^n
            self.iIL =   kwargs.get('iIL', False)       # Input Interleaving flag
            self.nPC =   kwargs.get('nPC', 0)           # Total number of parity-check bits
            self.nPCwm = kwargs.get('nPCwm', 0)     # "Low-weight, High-Reliability" parity-check bits (included in nPC)
            self.iSeg =  kwargs.get('iSeg', False)      # Segmentation flag
            self.crcPoly = kwargs.get('crcPoly', "11")  # CRC polynomial Identifier Str
            crcLen = self.getCrcLen(self.crcPoly)
            self.codeBlockSize = ((payloadSize+1)//2 + crcLen) if self.iSeg else (payloadSize+crcLen)   # Kr
        else:
            self.dataType = dataType.lower()
            if self.dataType == 'uci':      # Uplink control information
                # See TS 38.212 V17.0.0 (2021-12), section 6.3.1.4.1
                self.iBIL = True
                
                # See TS 38.212 V17.0.0 (2021-12), section 6.3.1.3.1 for PUCCH and section 6.3.2.3.1 for PUSCH
                self.nMax = 10
                self.iIL = False

            elif self.dataType == 'pbch':       # Broadcast Channel
                # See TS 38.212 V17.0.0 (2021-12), sections 7.1.4 and 7.1.5
                self.nMax = 9
                self.iIL = True
                self.nPC = 0
                self.nPCwm = 0
                self.iBIL = False
                self.iSeg = False           # Segmentation flag
                self.crcPoly = '24C'        # CRC bits (See TS 38.212 V17.0.0 (2021-12), section 7.1.3)

            elif self.dataType == 'dci':        # Downlink control information
                # See TS 38.212 V17.0.0 (2021-12), sections 7.3.3 and 7.3.4
                self.nMax = 9
                self.iIL = True
                self.nPC = 0
                self.nPCwm = 0
                self.iBIL = False
                self.iSeg = False           # Segmentation flag
                self.crcPoly = '24C'        # CRC bits (See TS 38.212 V17.0.0 (2021-12), section 7.3.2)

            else:
                raise ValueError("'dataType' value must be one of 'UCI', 'DCI', or 'PBCH'.")

        if (dataType is not None) and (payloadSize>0) and (rateMatchedLen>0):
            self.initialize(payloadSize, rateMatchedLen)
            
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)      # Not documented - Not called directly by the user
    def print(self, indent, title, getStr):                     # Not documented - Not called directly by the user
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        if self.dataType is not None:
            repStr += indent*' ' + "  dataType ......................: %s\n"%(self.dataType.upper())
        repStr += indent*' ' + "  payloadSize (A) ...............: %d\n"%(self.payloadSize)
        if self.iSeg:
            repStr += indent*' ' + "  rateMatchedLen (Etot) .........: %d\n"%(self.rateMatchedLen)
            repStr += indent*' ' + "  rateMatchedBlockLen (Er) ......: %d\n"%(self.rateMatchedBlockLen)
        else:
            repStr += indent*' ' + "  rateMatchedLen (E) ............: %d\n"%(self.rateMatchedLen)
        repStr += indent*' ' + "  codeBlockSize (K) .............: %d\n"%(self.codeBlockSize)
        repStr += indent*' ' + "  polarCodeSize (N) .............: %d\n"%(self.polarCodeSize)
        repStr += indent*' ' + "  Max Log2(N) (nMax) ............: %d\n"%(self.nMax)
        repStr += indent*' ' + "  Segmentation (iSeg) ...........: %s\n"%("Enabled" if self.iSeg else "Disabled")
        repStr += indent*' ' + "  Code Block CRC (crcPoly) ......: %s\n"%(self.crcPoly)
        repStr += indent*' ' + "  Input Interleaving (iIL) ......: %s\n"%("Enabled" if self.iIL else "Disabled")
        repStr += indent*' ' + "  Coded bit Interleaving (iBIL)..: %s\n"%("Enabled" if self.iBIL else "Disabled")
        repStr += indent*' ' + "  Parity-check bits (nPC, nPCwm) : %d,%d\n"%(self.nPC, self.nPCwm)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def initialize(self, payloadSize, rateMatchedLen):          # Not documented - Not called directly by the user
        self.payloadSize = int(payloadSize)
        self.rateMatchedLen = int(rateMatchedLen)
        a, eTot = self.payloadSize, self.rateMatchedLen
        
        if self.dataType == 'uci':      # Uplink control information
            # Note: UCI can be sent in PUCCH or PUSCH. The parameters are the same for both cases.
            # See TS 38.212 V17.0.0 (2021-12), section 6.3.1.2.1 for PUCCH and section 6.3.2.2.1 for PUSCH
            if a<12:    raise ValueError("Polar coding is not supported for UCI with payload size smaller than 12!")

            self.iSeg = ((a>=360 and eTot>=1088) or a>=1013)
            self.crcPoly = '6' if a<20 else '11'
            l = int(self.crcPoly)
            k = ((a+1)//2 + l) if self.iSeg else (a+l)
            
            # See TS 38.212 V17.0.0 (2021-12), section 6.3.1.4.1
            # Note that in this case eTot is "Euci" and rate-matched len 'eR' for each code block depends on iSeg
            eR = self.rateMatchedBlockLen = eTot//(self.iSeg+1)

            self.nPC = 0
            self.nPCwm = 0
            if (k > 17) and (k < 26):     self.nPC = 3      # a=12..19 => k=18..25
            elif k > 30:                  self.nPC = 0      # a=20...  => k=31...

            # See TS 38.212 V17.0.0 (2021-12), section 6.3.1.3.1 for PUCCH and section 6.3.2.3.1 for PUSCH
            # nPCwm is 1 only if number of rate-matched output (e) is more than 189+k (i.e., e>189+k)
            if (k > 17) and (k < 26):   self.nPCwm = 1 if (eR-k+3) > 192 else 0
            elif k > 30:                self.nPCwm = 0
        else:
            l = 24      # Using '24C' CRC for the DCI and PBCH cases
            k = a + l
            eR = self.rateMatchedBlockLen = eTot
            
                
        self.codeBlockSize = k
        n1 = self.ceilLog2(eR)-1
        if (k/eR >= 9/16.0):         n1 += 1
        elif (eR > (9/8)*(1<<n1)):   n1 += 1

        rMin = 1/8
        n2 = self.ceilLog2(k/rMin)
        nMin = 5
        n = max( min(n1, n2, self.nMax), nMin)
        nn = self.polarCodeSize = 1<<n              # N

        # Input interleaving indices:
        if self.iIL:
            # See TS 38.212 V17.0.0 (2021-12), Section 5.3.1.1
            kIlMax = 164
            kILminusk = kIlMax-k
            self.inInterleaveIndexes = [ piIlMax - kILminusk for piIlMax in inputInterleaver if piIlMax >= kILminusk ]
        
        reliabilitySeqN = reliabilitySeq[reliabilitySeq<nn]

        # Sub-block interleaving indices. See TS 38.212 V17.0.0 (2021-12), Section 5.4.1.1, Shape: (nn,)
        jj = self.sbInterleaveIndexes = [ subBlockInterleaver[(i<<5)//nn]*(nn>>5) + i%(nn>>5) for i in range(nn) ]

        # Getting frozen and message bit indices:
        fTemp = set()
        if eR<nn:
            if k/eR <= 7.0/16:       # Puncturing
                fTemp.update(jj[:nn-eR-1])
                if eR >= 3.0*nn/4:  fTemp.update(range((3*nn-2*eR+3)//4-1))
                else:               fTemp.update(range((9*nn-4*eR+15)//16-1))
            else:                   # Shortening
                fTemp.update(jj[eR:])
        
        # Using the last K+nPC bits (most reliable bits)
        self.msgBits = sorted([x for x in reliabilitySeqN if x not in fTemp][-(k+self.nPC):])
        self.frozenBits = sorted(x for x in reliabilitySeqN if x not in self.msgBits)

        # Making the generator matrix for polar coding
        g = [1]
        for _ in range(n): g = np.kron([[1, 0], [1, 1]], g)
        self.generator = g

        # Getting parity-check bits. See TS 38.212 V17.0.0 (2021-12), Section 5.3.1.2
        self.pcBits = []
        if self.nPC>0:
            self.pcBits = self.msgBits[:(self.nPC - self.nPCwm)]                    # (nPC - nPCwm) Least reliable bits
            if self.nPCwm>0:
                # The remaining message bits reversed so that most reliable bits come first
                mostReliableMsgBits = self.msgBits[(self.nPC - self.nPCwm):][::-1]
                
                # The argsort indices of weights based on 1st row-weights and then highest reliability
                idx = np.argsort(g[ mostReliableMsgBits ].sum(1), kind='stable')
                pcWmBits = mostReliableMsgBits[idx][::-1][:nPCwm]   # The nPCwm lowest row-weight bits in message bits.
                self.pcBits += pcWmBits.tolist()                                # Add to the parity-check bits
            self.msgBits = [b for b in self.msgBits if b not in self.pcBits]    # Remove the parity-check bits

        if self.iBIL:
            # Pre-calculate coded bit interleaving indices
            # See TS 38.212 V17.0.0 (2021-12), Section 5.4.1.3

            # Find smallest t such that t*(t+1)/2 >= eR
            t = int(np.floor(np.sqrt(2*eR)))
            if t*(t+1)<2*eR: t += 1
            # Now make the 'v' matrix
            idx = np.arange(eR)
            v = []
            k = 0
            for i in range(t):
                v += [t*[-1]]
                for j in range(t-i):
                    if k<eR: v[i][j] = idx[k]
                    k += 1
                if k>=eR: break
            self.cbInterleaveIndexes = np.transpose(v[:i+1]).flatten()
            
            # This is the indices for the encoder. For decoder we need the inverse of this.
            self.cbInterleaveIndexes = self.cbInterleaveIndexes[ self.cbInterleaveIndexes>=0 ]

    # ******************************************************************************************************************
    def setIoSizes(self, payloadSize, rateMatchedLen):
        r"""
        This function can be called to re-initialize the class properties. When the ``payloadSize`` or
        ``rateMatchedLen`` parameters change but other properties remain the same, you can either create a new polar
        encoder/decoder object or reuse the existing objects and re-initialize them using this method.
        
        Note that if there is no change in the values of ``payloadSize`` and ``rateMatchedLen``, this function
        returns without doing anything.
        
        Parameters
        ----------
        payloadSize: int
            The new size of input bitstream not including the CRC bits. This is the value :math:`A` in **3GPP 
            TS 38.212, Section 5.2.1**.
            
        rateMatchedLen: int
            The new total length of rate-matched output bitstream. This is the value :math:`E` in **3GPP TS 38.212,
            Sections 5.3.1 and 5.4.1**.
        """
        if (self.payloadSize != payloadSize) or (self.rateMatchedLen != rateMatchedLen):
            self.initialize(payloadSize, rateMatchedLen)

    # ******************************************************************************************************************
    @classmethod
    def intLog2(cls, num):                                      # Not documented - Not called directly by the user
        n,i = int(num), 0
        while n>1: n >>= 1; i+=1
        return i

    # ******************************************************************************************************************
    @classmethod
    def ceilLog2(cls, num):                                     # Not documented - Not called directly by the user
        n,i = int(num)-1, 1
        while n>1: n >>= 1; i+=1
        return i

# **********************************************************************************************************************
class PolarEncoder(PolarBase):
    r"""
    This class is used to encode a bitstream using 
    `Polar coding <https://en.wikipedia.org/wiki/Polar_code_(coding_theory)>`_. It is derived from the
    :py:class:`PolarBase` class and performs the following tasks:
    
    - Segmentation of the transport block based on **3GPP TS 38.212, Section 5.2.1**
    - Polar encoding based on **3GPP TS 38.212, Section 5.3.1**
    - Rate Matching with Sub-block interleaving, bit selection, and interleaving of coded bits based on **3GPP 
      TS 38.212, Section 5.4.1**
    """
    # ******************************************************************************************************************
    def __init__(self, payloadSize=0, rateMatchedLen=0, dataType=None, **kwargs):
        # NOTE: Documentation is inherited from PolarBase
        super().__init__(payloadSize, rateMatchedLen, dataType, **kwargs)

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`PolarEncoder` object.

        Parameters
        ----------
        indent: int
            The number of indentation characters.
            
        title: str
            If specified, it is used as a title for the printed information.

        getStr: Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string.
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Polar Encoder Properties:"
        repStr = super().print(indent, title, True)
        if getStr: return repStr
        print(repStr)
    
    # ******************************************************************************************************************
    def doSegmentation(self, txBlock):
        r"""
        If segmentation is enabled, the first step in Polar encoding process is breaking down the transport block
        into smaller code blocks. This function receives a transport block ``txBlock``, performs segmentation
        depending on the value of ``iSeg`` property based on **3GPP TS 38.212, Section 5.2.1**, and outputs a 2D
        ``C x K`` NumPy array containing ``C`` code blocks of length ``K``. Note that ``C`` can only be 1 or 2 and
        if ``iSeg=False``, then ``C=1``.

        Parameters
        ----------
        txBlock: NumPy array
            A NumPy array of bits containing the transport block information.
            
        Returns
        -------
        NumPy array
            A 2D ``C x K`` NumPy array containing ``C`` code blocks of length ``K``.
        """
        a = len(txBlock)
            
        # For polar code segmentation, c can be 1 or 2 only. See TS 38.212 V17.0.0 (2021-12), Section 5.2.1
        if self.iSeg:
            a = len(txBlock)
            if a%2:  codeBlocks = np.int8( [[0]+txBlock[:a//2].tolist(), txBlock[a//2:]] )
            else:    codeBlocks = txBlock.reshape(2,-1)
            c = 2
        else:
            codeBlocks = txBlock[None,:]
            c = 1
        
        if self.crcPoly is None:    return codeBlocks                                           # Shape: (c,k)
        return np.int8([ self.appendCrc(codeBlocks[i], self.crcPoly) for i in range(c) ])       # Shape: (c,k)

    # ******************************************************************************************************************
    def encode(self, codeBlocks):
        r"""
        This function encodes a set of code blocks and returns a set of Polar-coded code blocks based on the procedure
        explained in **3GPP TS 38.212, Section 5.3.1**.

        Parameters
        ----------
        codeBlocks: NumPy array
            A ``C x K`` NumPy array containing ``C`` code blocks of length ``K`` being Polar-encoded by this function.

        Returns
        -------
        NumPy array
            A ``C x N`` NumPy array containing the ``C`` encoded code blocks.
        """
        c, k = codeBlocks.shape
        nn, e = self.polarCodeSize, self.rateMatchedBlockLen
        
        # Do the input interleaving if it is enabled (iIL=True)
        if self.iIL:
            codeBlocks = codeBlocks[:,self.inInterleaveIndexes]

        # Applying the generator matrix to get the code block
        encodedBlocks = []
        for codeBlock in codeBlocks:
            u = np.zeros(nn, dtype=np.uint8)
            u[self.msgBits] = codeBlock                 # Populate the message bits
            if self.nPC>0:
                # Now Populate the parity-check bits
                y = [0,0,0,0,0]   # y0 ... y4
                for n in range(nn):
                    y = np.roll(y,-1)
                    if n in self.pcBits:    u[n] = y[0]
                    else:                   y[0] ^= u[n]
                    
            encodedBlocks += [u.dot(self.generator)%2]  # Apply the generator matrix and add to the encoded code blocks
                    
        return np.int8(encodedBlocks)

    # ******************************************************************************************************************
    def rateMatch(self, codeBlocks):
        r"""
        This function receives a set of encoded code blocks and returns the rate-matched code blocks. It first performs
        Sub-block interleaving based on **3GPP TS 38.212, Section 5.4.1.1**, then bit selection is done based on
        **3GPP TS 38.212, Section 5.4.1.2**. Finally, if *Coded bits Interleaving* is enabled (``iBIL=True``), this
        function applies the procedure in **3GPP TS 38.212, Section 5.4.1.3** for *Coded bits Interleaving*.

        Parameters
        ----------
        codeBlocks: NumPy array
            A ``C x N`` NumPy array containing ``C`` encoded code blocks of length ``N`` being rate-matched by
            this function.

        Returns
        -------
        NumPy array
            A ``C x E`` NumPy array containing the ``C`` rate-matched code blocks of length ``E`` where
            ``E=rateMatchedBlockLen``.
        """
        # Sub-block Interleaving indices. See TS 38.212 V17.0.0 (2021-12), Section 5.4.1.1
        jj = self.sbInterleaveIndexes
        codeBlocks = codeBlocks[:,jj]
        nn, k, e = self.polarCodeSize, self.codeBlockSize, self.rateMatchedBlockLen
        
            
        # Bit Selection. TS 38.212, Section 5.4.1.2
        if e>=nn:                   rateMatchedCWs = codeBlocks[:, [k%nn for k in range(e)] ]    # Repetition
        elif (k/e) <= (7.0/16):     rateMatchedCWs = codeBlocks[:, nn-e:]                        # Puncturing
        else:                       rateMatchedCWs = codeBlocks[:, :e]                           # Shortening

        if self.iBIL:
            # Interleaving of coded bits.
            if e>8192:
                raise ValueError("The rate-matched output length (%d) should not be larger than 8192!"%(e))
            rateMatchedCWs = rateMatchedCWs[:,self.cbInterleaveIndexes]   # Shape: c, e
           
        return rateMatchedCWs       # Shape: c, e

# **********************************************************************************************************************
class SclDecoder:                                               # Not documented - Not used directly by the user
    # ******************************************************************************************************************
    def __init__(self, frozenBits, maxCount=8, useMinSum=True):
        self.frozenBits = frozenBits        # The indices of the frozen and punctured bits
        self.maxCount = maxCount            # Max number of candidates to keep
        self.useMinSum = useMinSum          # Use Min-Sum approximation
        self.reset()                        # Reset all candidate info
        
    # ******************************************************************************************************************
    def reset(self):
        self.pathCosts = np.float64([0])    # Path Costs (Lower value means more probable results)
        self.uHats = np.int8([[]])          # uHats. The 'U' candidates (Including frozen bits)
        self.xHats = np.int8([[]])          # xHat. The code block candidates
        self.llrIdxs = np.int16([0])        # The indices mapping candidates to LLRs
        self.count = 1                      # Current number of candidates
        
    # ******************************************************************************************************************
    def updateFrozen(self, llrs):
        zeroBits = np.int8(self.count*[[0]])
        self.pathCosts -= np.minimum(0,llrs)                        # 0 cost if llr>0 else cost: abs(llr)=-llr
        self.uHats = np.concatenate([self.uHats,zeroBits],axis=1)   # Add the new bit(zero) to the U for all candidates
        self.xHats = zeroBits                               # The new bit(zero) for all xHat candidates
        self.llrIdxs = np.arange(self.count)                # The LLR indices just map to the same candidate numbers
 
    # ******************************************************************************************************************
    def updateMessage(self, llrs):
        zeroBits = np.int8(self.count*[[0]])
        oneBits = np.int8(self.count*[[1]])
        # New candidates assuming 0:
        pathCosts0 = self.pathCosts - np.minimum(0,llrs)
        uHats0 = np.concatenate([self.uHats,zeroBits],axis=1)
        
        # New candidates assuming 1:
        pathCosts1 = self.pathCosts + np.maximum(0,llrs)
        uHats1 = np.concatenate([self.uHats,oneBits],axis=1)

        # All new candidates:
        pathCosts = np.concatenate([pathCosts0,pathCosts1], axis=0)
        uHats = np.concatenate([uHats0,uHats1], axis=0)
        xHats = np.concatenate([zeroBits,oneBits], axis=0)
        llrIdxs = np.concatenate([np.arange(self.count),np.arange(self.count)], axis=0)
        
        # Get best 'maxCount' candidate indices
        bestKPaths = np.argsort(pathCosts)[:self.maxCount]
        
        # Keep the best candidates and discard the others
        self.pathCosts = pathCosts[bestKPaths]
        self.uHats = uHats[bestKPaths]
        self.xHats = xHats[bestKPaths]
        self.llrIdxs = llrIdxs[bestKPaths]
        self.count = len(self.pathCosts)

    # ******************************************************************************************************************
    def getRightLLRs(self, llr2s):
        c, n = self.xHats.shape
        rightLLRs =llr2s[self.llrIdxs,:,:]                                                              # shape: (c,2,n)
        rightLLRs *= np.concatenate([ (1-2*self.xHats)[:,None,:], np.ones((c, 1, n),np.int8) ], axis=1) # shape: (c,2,n)
        return rightLLRs.sum(1)                                                                         # shape: (c, n)
    
    # ******************************************************************************************************************
    def updateEnd(self, xHatsLeft, leftToOriginal):
        self.xHats = np.concatenate([ xHatsLeft[self.llrIdxs]^self.xHats, self.xHats ], axis=1)         # shape: (c, n)
        self.llrIdxs = leftToOriginal[ self.llrIdxs ]                                                   # shape: (c,)

    # ******************************************************************************************************************
    def uHatsSorted(self):
        # Sort based on path costs (Lowest cost first), then return the uHats
        pathOrders = np.argsort(self.pathCosts)
        self.pathCosts = self.pathCosts[pathOrders]
        self.xHats = self.xHats[pathOrders]
        self.uHats = self.uHats[pathOrders]
        self.llrIdxs = self.llrIdxs[pathOrders]
        return self.uHats
    
    # ******************************************************************************************************************
    def decode(self, rxLlrs, idx=-1):
        firstCall = (idx==-1)
        if firstCall:
            # This is the first time this is called. Initialize all candidates.
            self.reset()
            rxLlrs = rxLlrs[None,:]     # Shape: (1,n)
            idx = 0

        c,n = rxLlrs.shape
        assert c==self.count
        if n==1:
            # We are at a leaf node
            if idx in self.frozenBits: self.updateFrozen(rxLlrs[:,0])
            else:                      self.updateMessage(rxLlrs[:,0])
            return

        llr2s = rxLlrs.reshape(c, 2, n//2)                              # shape: (c, 2, n/2)
        sign = np.sign(llr2s).prod(axis=1)                              # shape: (c, n/2)

        def f(x): return np.abs(np.log(np.tanh(np.abs(x/2.0))+1e-12))   # SPC extrinsic likelihood function
        if self.useMinSum:  lext = np.abs(llr2s).min(axis=1)            # Min-sum approximation (Fast)
        else:               lext = f(f(llr2s).sum(axis=1))              # The extrinsic likelihood (Slow, precise)

        leftLLRs = sign*lext                                            # The LLRs for the left child, shape: (c, n/2)
        self.decode(leftLLRs, idx)                                      # Process the left child

        # The candidate information can change after returning from left child. Save a lookup table of
        # indices mapping from each new candidate to the corresponding LLR.
        leftToOriginal = self.llrIdxs.copy()
        xHatsLeft = self.xHats.copy()               # Save left xHats before going to the right child, shape: (c, n/2)

        rightLLRs = self.getRightLLRs(llr2s)        # Calculate the LLRs for the right child for all candidates
        self.decode(rightLLRs, idx+n//2)            # Process the right child
        
        self.updateEnd(xHatsLeft, leftToOriginal)   # Update xHat and the candidate ↔︎ LLR mappings before returning

        if firstCall:
            # If this is the first call, then we are done. Sort the candidates (Based on path costs) and
            # return the uHats
            return self.uHatsSorted()
               
# **********************************************************************************************************************
class PolarDecoder(PolarBase):
    r"""
    This class is used to decode a set of Log-Likelihood-Ratios (LLRs) to a transport block using the Successive 
    Cancellation List (SCL) [2]_ algorithm. It is derived from the :py:class:`PolarBase` class and performs rate
    recovery and Polar decoding which are basically the opposite of the encoding tasks performed in reverse order.
    
    The following example shows a typical use case for decoding the received Polar-coded information into transport
    blocks:
    
    .. code-block:: python
        :caption: An example of Polar decoding

        payloadLen = 30             # A
        rateMatchedLen = 120        # E

        # Creating a polar decoder object for "DCI" data
        polarDecoder = PolarDecoder(payloadLen, rateMatchedLen, 'dci', sclListSize=8, useMinsum=True)
        
        # Rate recovery (Assuming "llrs" contains the LLR values from demodulation process)
        rateRecoveredRxBlocks = polarDecoder.recoverRate(llrs)

        # Polar Decoding using SCL algorithm
        decTxBlock, numCrcErrors = polarDecoder.decode(rateRecoveredRxBlocks)
    """
    # ******************************************************************************************************************
    def __init__(self, payloadSize=0, rateMatchedLen=0, dataType=None, **kwargs):
        r"""
        Parameters
        ----------
        payloadSize: int
            The size of input bitstream not including the CRC bits. This is the value :math:`A` in **3GPP TS 38.212,
            Section 5.2.1**.
            
        rateMatchedLen: int
            The total length of rate-matched output bitstream. This is the value :math:`E` in **3GPP TS 38.212,
            Sections 5.3.1 and 5.4.1**.
            
        dataType: str or None
            The type of data using this Polar decoder. It can be one of the following:
            
            :"DCI": Downlink Control Information
            :"UCI": Uplink Control Information
            :"PBCH": Physical broadcast channel
            :None: Customized Polar Coding.

        kwargs : dict
            A set of optional arguments depending on the ``dataType``:

                :iBIL: Coded bits Interleaving flag. This is a boolean value that indicates whether coded bits
                    interleaving is enabled (`True`) or disabled (`False`). By default ``iBIL=False``. This
                    is the value :math:`I_{BIL}` in **3GPP TS 38.212, Section 5.4.1.3**. This parameter is ignored
                    if the ``dataType`` is not `None`. In this case, ``iBIL`` is set to `True` for 
                    ``dataType="UCI"``, and `False` for ``dataType="DCI"`` and ``dataType="PBCH"`` cases.

                :nMax: Max value of :math:`n` where :math:`N=2^n` is the length of the polar code. By default this
                    is set to 10 (which means :math:`N=1024`. This is the value :math:`N_{max}` in **3GPP TS 38.212,
                    Section 5.3.1.2**. This parameter is ignored if the ``dataType`` is not `None`. In this case,
                    ``nMax=10`` when ``dataType="UCI"``, and ``nMax=9`` for ``dataType="DCI"`` and ``dataType="PBCH"``
                    cases.

                :iIL: Input Interleaving flag. This is a boolean value that indicates whether input interleaving
                    is enabled (`True`) or disabled (`False`). By default ``iIL=False``. This is the value
                    :math:`I_{IL}` in **3GPP TS 38.212, Section 5.3.1.1**. This parameter is ignored if the 
                    ``dataType`` is not `None`. In this case, ``iIL`` is set to `False` for ``dataType="UCI"``,
                    and `True` for ``dataType="DCI"`` and ``dataType="PBCH"`` cases.

                :nPC: Total number of parity-check bits. By default this is set to 0. This is the value :math:`N_{PC}`
                    in **3GPP TS 38.212, Section 5.3.1**. This parameter is ignored if the ``dataType`` is not 
                    `None`. In this case, ``nPC=0`` when ``dataType`` is set to ``"DCI"`` or ``"PBCH"``. For the
                    ``"UCI"`` case, this value may be set to 0 or 3 which is determined based on the procedure
                    explained in **3GPP TS 38.212, Section 5.3.1.2**.

                :nPCwm: The number of *Low-weight*, *High-Reliability* parity-check bits out of the total parity-check
                    bits ``nPC``. By default this is set to 0. This is the value :math:`n_{PC}^{wm}` in **3GPP TS
                    38.212, Sections 5.3.1.2, 6.3.1.3.1, and 6.3.2.3.1**. This parameter is ignored if the 
                    ``dataType`` is not `None`. In this case, ``nPCwm=0`` when ``dataType`` is set to ``"DCI"``
                    or ``"PBCH"``. For the ``"UCI"`` case, this value may be set to 0 or 1 which is determined based
                    on the procedure explained in **3GPP TS 38.212, Sections 6.3.1.3.1 and 6.3.2.3.1**.

                :iSeg: Segmentation flag. This is a boolean value that indicates whether segmentation is enabled 
                    (`True`) or disabled (`False`). By default ``iSeg=False``. This is the value :math:`I_{seg}`
                    in **3GPP TS 38.212, Section 5.2.1**. This parameter is ignored if the ``dataType`` is not
                    `None`. In this case, ``iSeg=False`` when ``dataType="DCI"`` or ``dataType="PBCH"``. When
                    ``dataType="UCI"``, ``iSeg`` is set based on the value of ``payloadSize``.

                :crcPoly: The CRC polynomial. This is a string specifying the CRC polynomial or `None`. If
                    specified, it must be one of the values specified in 
                    :py:meth:`~neoradium.chancodebase.ChanCodeBase.getCrc` for the ``poly`` parameter. The default 
                    value is ``"11"``. This parameter is ignored if the ``dataType`` is not `None`. In this case
                    ``crcPoly`` is set to ``"6"`` or ``"11"`` depending on ``payloadSize`` for ``dataType="UCI"``,
                    and ``"24C"`` for ``dataType="DCI"`` and ``dataType="PBCH"`` cases.
                    
                :sclListSize: The list size of the *Successive Cancellation List (SCL)* algorithm used for decoding. 
                    The default is 8.
                    
                :useMinsum: A Boolean value indicating whether the *Min-Sum* approximation should be used in the SCL 
                    algorithm. `True` (default) means the "Min-Sum" approximation is used resulting in faster
                    decoding with slightly less precise results. `False` means the actual extrinsic likelihood
                    function based on hyperbolic tangent function is used.


        .. Note:: For a pair of :py:class:`PolarEncoder`/:py:class:`PolarDecoder` objects to work properly, the 
                  above parameters used to configure them should match.

        Please refer to :py:class:`PolarBase` class for a list of properties inherited from the base class.
        """
        super().__init__(payloadSize, rateMatchedLen, dataType, **kwargs)
        self.sclListSize = kwargs.get('sclListSize', 8)     # The list size for the SCL decoding algorithm.
        self.useMinsum = kwargs.get('useMinsum', True)      # If True, use "min-sum" approximation in the SCL algorithm
        
        self.candidates = None
        
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`PolarDecoder` object.

        Parameters
        ----------
        indent: int
            The number of indentation characters.
            
        title: str
            If specified, it is used as a title for the printed information.

        getStr: Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string.
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Polar Decoder Properties:"
        repStr = super().print(indent, title, True)
        repStr += indent*' ' + "  SCL List Size .................: %s\n"%(self.sclListSize)
        repStr += indent*' ' + "  Min-sum Approximation .........: %s\n"%("Enabled" if self.useMinsum else "Disabled")
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def initialize(self, payloadSize, rateMatchedLen):          # Not documented - Not called directly by the user
        super().initialize(payloadSize, rateMatchedLen)
        
        # For the decoder we need the inverse of the three interleaving indices.
        # Input Interleaver:
        if self.inInterleaveIndexes is not None:
            self.inInterleaveIndexes = np.argsort(self.inInterleaveIndexes)

        # Sub-block interleaver:
        self.sbInterleaveIndexes = np.argsort(self.sbInterleaveIndexes)
        
        # Coded block interleaver:
        if self.cbInterleaveIndexes is not None:
            self.cbInterleaveIndexes = np.argsort(self.cbInterleaveIndexes)

    # ******************************************************************************************************************
    def recoverRate(self, rxBlock):
        r"""
        This function receives an array of Log-Likelihood Ratios (LLRs) in ``rxBlock`` and returns a set of
        rate-recovered LLRs for each code block which are ready for Polar decoding. This function does the exact opposite
        of the :py:class:`PolarEncoder`'s :py:meth:`rateMatch` method. Note that while the :py:meth:`rateMatch`
        works with bits, this method works on LLRs which are usually obtained by performing demodulation process.

        Parameters
        ----------
        rxBlock: NumPy array
            A NumPy array of Log-Likelihood Ratios (LLRs) obtained as a result of demodulation process. Each element
            is a real LLR value corresponding to a each received bit. The larger the LLR value, the more likely it is
            for that bit to be a ``0``.

        Returns
        -------
        NumPy array
            A ``C x N`` NumPy array of ``C`` received coded blocks of length ``N`` containing the LLR values for
            each coded block ready to be polar-decoded.
        """
        c, e = rxBlock.shape
        assert e == self.rateMatchedBlockLen
        nn, k = self.polarCodeSize, self.codeBlockSize

        if self.iBIL:
            # We need to undo the coded bit interleaving
            rxBlock = rxBlock[ :, self.cbInterleaveIndexes ]
            
        # Bit Selection. TS 38.212 Section 5.4.1.2
        if e>=nn:
            # Repetition
            # Need to add the LLRs of the repeated bits (I think Matlab implementation may be wrong here!)
            rateRecoveredCWs = np.zeros((c,nn))
            for i in range(e): rateRecoveredCWs[i%nn] += rxBlock[i]
        elif (k/e) <= (7.0/16):
            # Puncturing
            # Need to add 0 LLRs (at the beginning) for the punctured bits.
            rateRecoveredCWs = np.concatenate([np.zeros((c,nn-e)), rxBlock], axis=1)
        else:
            # Shortening
            # Need to add Large LLR values (at the end) for the Shortened bits.
            rateRecoveredCWs = np.concatenate([rxBlock, self.LARGE_LLR*np.ones((c,nn-e))], axis=1)

        # Sub-block Interleaving indices. See TS 38.212 V17.0.0 (2021-12), Section 5.4.1.1
        rateRecoveredCWs = rateRecoveredCWs[:,self.sbInterleaveIndexes]
        
        return rateRecoveredCWs         # Shape: c,nn

    # ******************************************************************************************************************
    def decode(self, rxLlrBlocks):
        r"""
        This function implements the *Successive Cancellation List (SCL)* algorithm for Polar-decoding of LLRs into
        decoded transport blocks. This implementation was inspired mostly by `LDPC and Polar Codes in 5G Standard
        <https://www.youtube.com/playlist?list=PLyqSpQzTE6M81HJ26ZaNv0V3ROBrcv-Kc>`_ set of videos and was written
        from scratch using a recursive algorithm to efficiently perform the SCL decoding process.

        Parameters
        ----------
        rxLlrBlocks: NumPy array
            A ``C x N`` NumPy array of ``C`` received coded blocks of length ``N`` containing the LLR values for each
            coded block.

        Returns
        -------
        txBlock: NumPy array of bits
            A 1D NumPy array of length :math:`A` containing the decoded transport block bits where :math:`A` is
            equal to the parameter ``payloadSize``.
            
        numCrcErrors: int
            The total number of CRC errors if ``crcPoly`` is not `None`, otherwise, zero.
        """
        c, nn = rxLlrBlocks.shape
        if nn != self.polarCodeSize:
            raise ValueError("The rxLLRs's second dimension(%d) must match the configured Polar Code Size(%d)"%
                             (nn, self.polarCodeSize))

        rxLlrBlocks = np.clip(rxLlrBlocks, -20, 20)
        txpBlock = []
        crcErrors = 0
        for rxLlrBlock in rxLlrBlocks:
            # uHats contains up to 'sclListSize' candidates
            uHats = SclDecoder(self.frozenBits, self.sclListSize).decode(rxLlrBlock)
            
            messages = uHats[:,self.msgBits]
                
            if self.iIL:                                # Do the input De-interleaving if it is enabled (iIL=True)
                messages = messages[:,self.inInterleaveIndexes]

            if self.crcPoly is None:                    # If there is no CRC ...
                message = messages[0]                   #    ... just use the first candidate
            else:
                crcResults = self.checkCrc(messages, self.crcPoly)                  # All CRC results
                goodIndexes = np.where(crcResults)[0]                               # The ones that passed CRC check
                message = messages[goodIndexes[0]] if len(goodIndexes)>0 else messages[0]   # The one with lowest cost
                if len(goodIndexes)==0: crcErrors += 1                              # If CRC failed for all candidates
                message = message[:-self.getCrcLen(self.crcPoly)]                   # Remove the CRC block
                                
            txpBlock += message.tolist()
        
        # Return the last 'payloadSize' bits. (If payloadSize is Odd, the first bit is zero and must be skipped)
        return np.int8(txpBlock)[-self.payloadSize:], crcErrors

