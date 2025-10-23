# Copyright (c) 2025 InterDigital AI Lab
"""
The module ``harq.py`` contains the API for Hybrid Automatic Repeat reQuest (HARQ), a fundamental mechanism in 5G NR,
designed for error correction and reliable data transfer over radio links. The class structure follows the 3GPP 
HARQ hierarchy:

- :py:class:`~neoradium.harq.HarqEntity`: The main HARQ management object. It implements the “HARQ Entity” as specified
  in **3GPP TS 38.321, Section 5.3.2.1**.
- :py:class:`~neoradium.harq.HarqProcess`: Each HARQ entity contains one or more HARQ processes. This class implements 
  the “HARQ Process” object as specified in **3GPP TS 38.321, Section 5.3.2.2**.
- :py:class:`~neoradium.harq.HarqCW`: Each HARQ process can handle up to two codewords. This class implements HARQ 
  processing for a single codeword. A :py:class:`~neoradium.harq.HarqProcess` can have one or two 
  :py:class:`~neoradium.harq.HarqCW` objects, depending on the number of codewords. 

This implementation adheres to the procedures specified in **3GPP TS 38.321, TS 38.212, and TS 38.214**.

**A typical HARQ workflow:**

Here’s a typical workflow example for using HARQ in your simulations:

1) Create an LDPC encoder, for instance:

    .. code-block:: python
            
        ldpcEncoder = LdpcEncoder(baseGraphNo=1, modulation='QPSK', txLayers=1, targetRate=490/1024)

2) Create a HARQ entity object, passing in the LDPC encoder:

    .. code-block:: python
            
        harq = HarqEntity(ldpcEncoder, harqType="IR", numProc=16)  # Using "Incremental Redundancy" with 16 HARQ processes

3) For each transmission and each codeword, check the ``needNewData`` property. If it is `True` for a codeword, create
   the transport block bits for the corresponding codeword. Otherwise, set the transport block to `None` (for 
   retransmission). Once the transport blocks are ready for each codeword, call the 
   :py:meth:`HarqEntity.getRateMatchedCodeBlocks` function to prepare the rate-matched bitstreams for transmission. 
   Here is an example:

    .. code-block:: python
            
        txBlocks = []                                         # Transport blocks, one per codeword.
        for c in range(numCodewords):
            if harq.needNewData[c]:                           # New transmission.
                txBlocks += [ random.bits(txBlockSizes[c]) ]  # Create random bits for the new transport block
            else:                                             # Retransmission
                txBlocks += [ None ]                          # Set transport block to None indicating retransmission

        rateMatchedCodeWords = harq.getRateMatchedCodeBlocks(txBlocks)  # Prepare the bitstream for transmission    

4) At the receiving end, after demodulating the LLRs, the :py:meth:`HarqEntity.decodeLLRs` method is invoked to obtain
   a list of decoded blocks for each codeword. For instance:

    .. code-block:: python

        decodedTxBlocks, blockErrors = harq.decodeLLRs(llrs, txBlockSize)  # Decode received LLRs

5) Near the end of the transmission loop, the HARQ entity’s :py:meth:`~HarqEntity.goNext` method is called to 
   transition to the next HARQ process for the subsequent transmission.

    .. code-block:: python

        harq.goNext()
        
Please refer to the notebook :doc:`../Playground/Notebooks/HARQ/Harq` for a complete example of the above workflow.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/31/2025    Shahab Hamidi-Rad       First version of the file.
# 08/01/2025    Shahab                  Added documentation.
# **********************************************************************************************************************
import numpy as np
from .ldpc import LdpcEncoder

# **********************************************************************************************************************
class HarqCW:
    r"""
    This class implements the HARQ Processing for a single codeword. The :py:class:`neoradium.harq.HarqProcess` class 
    can have either one or two :py:class:`HarqCW` objects.
    """
    # ******************************************************************************************************************
    def __init__(self, process, cwIdx):
        r"""
        Parameters
        ----------
        process: :py:class:`~neoradium.harq.HarqProcess`
            The :py:class:`~neoradium.harq.HarqProcess` object that holds this :py:class:`~neoradium.harq.HarqCW` 
            instance.
            
        cwIdx: int
            The codeword index, which is 0 for the first codeword, or 1 for the second codeword.
             
                    
        **Other Read-Only Properties:**
        
            :curTry: This indicates the current number of retransmissions. It starts at zero for the first transmission
                for this codeword. This value increments after each transmission failure.
            :rv: This represents the redundancy version used for retransmissions. For “Chase Combining” retransmission,
                this value is always zero. For “Incremental Redundancy” retransmission, it is updated based on the 
                ``rvSequence`` parameter of the :py:class:`~neoradium.harq.HarqEntity` object.
            :txBlockNo: This indicates the number of unique transport blocks (not including retransmissions) 
                transmitted so far since the start of communication before the one currently being processed by this 
                :py:class:`~neoradium.harq.HarqCW` instance.
            :needNewData: This indicates whether the :py:class:`~neoradium.harq.HarqCW` object is ready to receive a 
                new transport block for transmission. If it is `True`, it means it is ready to transmit the new block. 
                If it is `False`, it means it is still busy retransmitting the previous block. This corresponds to the 
                **New Data Indicator (NDI)** in **3GPP TS 38.212**
        """
        self.process = process  # HARQ process object "owning" this HarqCW
        self.cwIdx = cwIdx      # The codeword Index associated with this object 0=>"Codeword 1", 1=>"Codeword 2"
        self.reset()            # Reset the parameters and statistics of this HarqCW

    # ******************************************************************************************************************
    def reset(self):
        # Called after a successful transmission, a timeout, or when called by the HARQ Entity/Process
        self.curTry = 0         # Current Try number (AKA retransmission number)
        self.txBlockNo = 0      # The transport block number for this HarqCW
        self.rv = 0             # Current revision number of this HarqCW
        self.encBuffer = None   # The encoder buffer for this HarqCW which contains the encoded bits.
        self.decBuffer = None   # The decoder buffer for this HarqCW which contains the circular queue of LDPC decoder

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        # Prints the information about this HarqCW
        if title is None:   title = f"HARQ Codeword Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  curTry:             {self.curTry}\n"
        repStr += indent*' ' + f"  txBlockNo:          {self.txBlockNo}\n"
        repStr += indent*' ' + f"  rv:                 {self.rv}\n"
        if self.encBuffer is not None:
            repStr += indent*' ' + f"  encBuffer Shape:    {self.encBuffer.shape}\n"
        if self.decBuffer is not None:
            repStr += indent*' ' + f"  decBuffer Shape:    {self.decBuffer.shape}\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    def needNewData(self):  return self.curTry==0   # True means the start of new txBlock transmission

    # ******************************************************************************************************************
    def getRateMatchedCodeBlocks(self, txBlock, g=None, concatCBs=True):
        # Uses the LdpcEncoder's methods to encode 'txBlock'
        encoder = self.process.entity.encoder                           # Get LDPC encoder object from the HARQ entity
        if txBlock is None:                                             # Retransmission
            assert self.curTry>0
            assert self.encBuffer is not None
            # In this case we already have the encoded bits in the "encBuffer". Just need to rate-match them
            rateMatchedCodeWords = encoder.rateMatch(self.encBuffer, g, concatCBs, self.rv)
            
        else:                                                           # New transmission
            assert self.curTry==0
            assert self.encBuffer is None
            txBlockWithCrc = encoder.appendCrc(txBlock,'24A')           # Append txBlock CRC
            codeBlocksCrc = encoder.doSegmentation(txBlockWithCrc)      # Perform segmentation
            # Encode to Code Blocks and save to "encBuffer" to be used for possible future retransmission
            self.encBuffer = encoder.encode(codeBlocksCrc)
            rateMatchedCodeWords = encoder.rateMatch(self.encBuffer, g, concatCBs, self.rv) # Apply rate matching
        return rateMatchedCodeWords

    # ******************************************************************************************************************
    def decodeLLRs(self, llrs, txBlockSize, numIter):
        # Uses the LdpcDecoder's methods to decode the 'llrs'
        decoder = self.process.entity.decoder                           # Get LDPC decoder object from the HARQ entity

        rxCodedBlocks = decoder.recoverRate(llrs, txBlockSize, self)              # Recover Rate
        decodedBlocks = decoder.decode(rxCodedBlocks, numIter=numIter)            # LDPC Decoding
        decodedTxBlockWithCRC, crcMatch = decoder.checkCrcAndMerge(decodedBlocks) # Check code block CRCs and Merge them
        decodedTxBlock = decodedTxBlockWithCRC[:-24]                              # remove transport block CRC
        blockErrors = len(crcMatch)-sum(crcMatch)                                 # Number of Block errors
        
        # Note that we update the state of this HarqCW all statistics of HARQ Entity after decoding the received
        # transmission.
        self.update(blockErrors, txBlockSize)
        return decodedTxBlock, blockErrors

    # ******************************************************************************************************************
    def update(self, blockErrors, txBlockSize):
        # Update all stats after decoding the received transmission.
        entity = self.process.entity                                # Get the entity object
        if self.curTry == 0:  self.txBlockNo = entity.txBlocks[0]   # If this was a new transmission, set txBlockNo
        entity.txBits[self.curTry] += txBlockSize                   # Update number of transmitted bits for this try
        entity.txBlocks[self.curTry] += 1                           # Update number of transmitted blocks for this try
        if blockErrors==0:                                          # Successful transmission
            entity.rxBits[self.curTry] += txBlockSize               # Update number of received bits for this try
            entity.rxBlocks[self.curTry] += 1                       # Update number of received blocks for this try

        if blockErrors>0:                                           # Transmission failed
            entity.handleEvent("RXFAILED", self )                   # Create "RXFAILED" event
            self.curTry += 1                                        # Increase the current try count
            if self.curTry == entity.maxTries:                        # Reached Max. try count -> timed out!
                entity.handleEvent("TIMEOUT", self)                 # Create "TIMEOUT" event
                entity.numTimeouts += 1                             # Increase the number of timeouts
                self.reset()                                        # Prepare for a new transmission
            else:                                                   # Try again
                self.rv = entity.getRV(self.curTry)                 # Set the Redundancy Version for the retransmission
        else:                                                       # Successful transmission
            entity.handleEvent("RXSUCCESS", self)                   # Create "RXSUCCESS" event
            self.reset()                                            # Prepare for a new transmission

# **********************************************************************************************************************
class HarqProcess:
    r"""
    This class encapsulates the functionality of a HARQ process as outlined in **3GPP TS 38.321, Section 5.3.2.2**. A 
    HARQ entity manages multiple parallel HARQ processes, each identified by a unique HARQ process identifier. The UE 
    capabilities determine the maximum number of HARQ processes per cell, which can be 16 or 32. A single HARQ process
    can support one or two Transport Blocks, depending on the number of codewords. In this implementation, each 
    codeword is processed by a dedicated :py:class:`HarqCW` object.
    """
    # ******************************************************************************************************************
    def __init__(self, entity, id, numCW):
        r"""
        Parameters
        ----------
        entity: :py:class:`~neoradium.harq.HarqEntity`
            The :py:class:`~neoradium.harq.HarqEntity` object that holds this :py:class:`~neoradium.harq.HarqProcess`
            instance.
            
        id: int
            The unique identifier associated with this HARQ process.
                    
        numCW: int
            The number of codewords processed by this HARQ process. It can be 1 or 2.


        **Other Read-Only Properties:**
        
            :needNewData: A list of one or two boolean values, corresponding to each codeword. For each element in the 
                list, a `True` value indicates that the HARQ process is ready to receive a new transport block for 
                transmission, while a `False` value signifies that it is currently busy retransmitting the previous 
                transport block. This aligns with the **New Data Indicator (NDI)** defined in **3GPP TS 38.212**.
        """
        self.id = id                    # HARQ process identifier
        self.entity = entity            # HARQ entity
        self.cws = [ HarqCW(self,i) for i in range(numCW) ]

    # ******************************************************************************************************************
    def reset(self):
        r"""
        Resets this HARQ process to prepare it for new transmissions. It resets the counters and releases any
        encoder/decoder retransmission buffers by calling the `reset` method of the :py:class:`~neoradium.harq.HarqCW`
        objects.
        """
        for cw in self.cws: cw.reset()
            
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`~neoradium.harq.HarqProcess` object and its 
        :py:class:`~neoradium.harq.HarqCW` objects.

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
        if title is None:   title = f"HARQ Process Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  id:                   {self.id}\n"
        repStr += indent*' ' + f"  numCW:                {len(self.cws)}\n"
        for i,cw in enumerate(self.cws):
             repStr += cw.print(indent+2, f"HARQ CW {i+1}:", True)
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def needNewData(self):  return [cw.curTry==0 for cw in self.cws]

    # ******************************************************************************************************************
    def getRateMatchedCodeBlocks(self, txBlocks, gs=None, concatCBs=True):
        r"""
        This function takes a list of one or two transport blocks (``txBlocks``) based on the number of codewords 
        and returns a list of LDPC-encoded and rate-matched bitstreams for each codeword. If a transport block in 
        the ``txBlocks`` list is set to `None`, it uses the buffered encoded bitstream and only applies rate matching 
        for retransmitting the previously encoded transport block. Otherwise, it assumes new transmission and encodes 
        the transport block, saving the encoded bits for future retransmissions. For more information about the LDPC 
        encoding process, refer to the :py:meth:`~neoradium.ldpc.LdpcEncoder.getRateMatchedCodeBlocks` method of the 
        :py:class:`~neoradium.ldpc.LdpcEncoder` class.

        Parameters
        ----------
        txBlocks: list
            A list of one or two NumPy arrays for each codeword. The presence of a ‘None’ value for each transport 
            block in the list indicates a retransmission of previously buffered encoded transport block. 

        g: list or None
            A list of one or two integer values for each codeword. Each element in the list represents the total 
            number of bits available for transmitting the transport block. This value corresponds to the value 
            :math:`G` in the *bit selection* process explained in **3GPP TS 38.212, Section 5.4.2.1**. If not 
            provided (default), it is calculated using the formula :math:`G=\lceil \frac {B-24} R \rceil`, where 
            :math:`B` is the transport block size and :math:`R` is the code rate.
            
        concatCBs: Boolean
            If `True` (Default), the rate-matched code blocks are concatenated, and a single array of bits is 
            returned for each codeword. Otherwise, for each codeword, a list of NumPy arrays is returned, where 
            each element in the list represents the bit array corresponding to each code block.
                        
        Returns
        -------
        NumPy array or list of NumPy arrays
            If ``concatCBs`` is `True`, a one-dimensional NumPy array is returned, containing the concatenation of 
            all rate-matched coded blocks. Otherwise, a list of NumPy arrays is returned, where each element in the 
            list corresponds to the bit array of a coded block.
        """
        return [cw.getRateMatchedCodeBlocks(txBlocks[i], None if gs is None else gs[i], concatCBs) for i,cw in enumerate(self.cws)]

    # ******************************************************************************************************************
    def decodeLLRs(self, llrs, txBlockSizes, numIter=5):
        r"""
        This function takes a list of one or two NumPy arrays, each containing Log-Likelihood Ratios (LLRs) for the 
        demodulated received signals for each codeword. It then returns a list of one or two decoded transport blocks, 
        with the size of each transport block specified by the ``txBlockSizes`` list, which provides the transport 
        block sizes for each codeword.

        For each codeword, the function recovers the rate, decodes the code blocks using LDPC decoding, checks the 
        code block CRCs, and merges all code blocks to reassemble the transport block. In case of retransmissions, 
        the function combines the LLR values of the retransmission with those from previous transmissions before 
        decoding the code blocks. 

        Parameters
        ----------
        llrs: list
            A list of one or two NumPy arrays, each containing the Log-Likelihood Ratios (LLRs) from the demodulated 
            received signals corresponding to each codeword. 

        txBlockSizes: list
            A list containing one or two integer values, each representing the transport block size for each codeword.
            
        numIter: int            
            The number of iterations in the *Layered Belief Propagation* decoding algorithm. In some cases, larger 
            values may lead to more accurate decoding but can slow down the entire decoding process. For more 
            information, please refer to the :py:meth:`~neoradium.ldpc.LdpcDecoder.decode` method of the 
            :py:class:`~neoradium.ldpc.LdpcDecoder` class. The default value is 5.
                        
        Returns
        -------
        decodedTxBlock: list
            A list of one or two NumPy arrays, each containing the decoded transport blocks for each codeword.
            
        blockErrors: list
            A list containing one or two integer values, each representing the number of code block CRC errors 
            for the corresponding codeword.
        """
        retvals = [cw.decodeLLRs(llrs[i], txBlockSizes[i], numIter) for i,cw in enumerate(self.cws)]
        return tuple(list(x) for x in zip(*retvals))   # returns a tuple of lists for decodedTxBlocks, blockErrors

# **********************************************************************************************************************
class HarqEntity:
    r"""
    This class encapsulates the functionality of a HARQ entity as specified in **3GPP TS 38.321, Section 5.3.2.1**. A
    HARQ entity is configured for each Serving Cell to manage downlink (DL) and uplink (UL) HARQ operations. The 
    primary purpose of a HARQ entity is to maintain multiple parallel HARQ processes and direct HARQ information and 
    associated Transport Blocks to these corresponding processes. This HARQ information includes key parameters 
    such as the New Data Indicator (NDI), Transport Block Size (TBS), Redundancy Version (RV), and the HARQ process
    ID.
    """
    # ******************************************************************************************************************
    def __init__(self, encoder, harqType="CC", numProc=8, rvSequence=[0,2,3,1], maxTries=4, eventCallback=None):
        r"""
        Parameters
        ----------
        encoder: :py:class:`~neoradium.ldpc.LdpcEncoder`
            The :py:class:`neoradium.ldpc.LdpcEncoder` object, used by this HARQ entity for LDPC encoding. It is also
            used to create the corresponding :py:class:`neoradium.ldpc.LdpcDecoder` object, which is used for LDPC
            decoding.
                         
        harqType: str
            The retransmission method used by this HARQ entity. It can be one of ``"CC"``(default) or ``"IR"``:
            
            :"CC": Indicates **Chase Combining** which is a straightforward HARQ method where each retransmission 
                   is an exact copy of the original data. The receiver simply combines the energy (or LLRs) from 
                   these identical transmissions, which increases the signal-to-noise ratio and makes it more likely
                   that the combined signal can be decoded correctly.
                   
            :"IR": Indicates **Incremental Redundancy**, a more efficient and advanced method. Instead of re-sending
                   identical copies, each retransmission contains new and different parity bits, known as redundancy 
                   versions. The receiver combines these unique pieces of information with the original transmission, 
                   progressively building a stronger and more complete coded block. This approach significantly 
                   improves the chances of successful decoding with fewer retransmissions compared to Chase Combining.

            numProc: int
                The number of HARQ processes utilized by this HARQ entity. It varies depending on the capabilities 
                of the UE. A HARQ entity can manage up to 32 HARQ processes, with the default being 8.

            rvSequence: list of integers
                For the Incremental Redundancy HARQ, this specifies the sequence of redundancy versions (RV) values 
                used for each consecutive retransmission. The default sequence is ``[0, 2, 3, 1]``, which is based 
                on **3GPP TS 38.214, Table 5.1.2.1-2**.
                
            maxTries: int
                The maximum number of transmission attempts for a specified transport block. If a transport block's 
                transmission fails after this many attempts, a timeout event occurs. In the case of PDSCH, this value 
                is equivalent to the parameter ``pdsch-AggregationFactor``, as explained in **3GPP TS 38.321, 
                Section 5.3.2.1**.
                
            eventCallback: function or None
                If this callback function is provided, it will be invoked on the following events:
                
                :“RXFAILED”: This event is triggered when a transport block transmission fails, whether it’s the 
                    original transmission or a retransmission.

                :“RXSUCCESS”: This event is triggered when a transport block transmission succeeds, whether it’s the 
                    original transmission or a retransmission.

                :“TIMEOUT”: This event is triggered when a transport block transmission fails after the ``maxTries``
                    transmissions.

                For more information, refer to the :ref:`Event callback function section <EventCallback>` below.



        **Other Read-Only Properties:**
        
            :processes: A list of :py:class:`~neoradium.harq.HarqProcess` objects managed by this HARQ entity.
            :curProcess: The HARQ process that is currently transmitting or retransmitting the transport blocks.
            :curProcIdx: The current HARQ process that is currently transmitting or retransmitting the transport blocks. 
            :numCW: The number of codewords. This is based on the LDPC encoder's ``txLayers`` parameter. 
            :needNewData: A list of one or two boolean values, corresponding to each codeword. For each element in the 
                list, a `True` value indicates that current HARQ process (``curProcess``) is ready to receive a new 
                transport block for transmission, while a `False` value signifies that it is currently busy 
                retransmitting the previous transport block. This aligns with the **New Data Indicator (NDI)** defined 
                in **3GPP TS 38.212**.
            :totalTxBlocks: The total number of transport blocks transmitted, including retransmissions.
            :totalRxBlocks: The total number of transport blocks received and successfully decoded.
            :totalTxBits: The total number of transport block bits transmitted, including retransmissions.
            :totalRxBits: The total number of transport block bits received and successfully decoded.
            :throughput: The communication throughput expressed as a percentage. It’s calculated as
                ``totalRxBits*100/totalTxBits``.
            :bler: The Block Error Rate expressed as a percentage. It’s calculated as 
                ``(totalTxBlocks-totalRxBlocks)*100/totalTxBlocks``.
            :numTimeouts: The total number of timeout events, which occur when a transport block fails after all 
                retransmissions attempts.
            :meanTries: The average number of transmission attempts per transport block. This value ranges from zero
                (indicating no retransmissions) to ``maxTries`` (indicating all transmissions timed out).
                

        .. _EventCallback:
            
        **Event Callback Function**
        
        This function is automatically invoked when a transmission event occurs. It accepts the following parameters:
        
            :eventStr: This string can be one of ``”RXFAILED”``, ``”RXSUCCESS”``, or ``”TIMEOUT”``, as explained above.
            :harqCW: The :py:class:`~neoradium.harq.HarqCW` object that triggered the event. This object can be used 
                to obtain more information about the event.
            
            Here is an example of an event callback function that prints all triggered events:
        
            .. code-block:: python
            
                def myEventHandler(eventStr, harqCW):
                    print(f"HARQ Process {harqCW.process.id:2d} CW{harqCW.cwIdx+1}:{event:10s} curTry:{harqCW.curTry} RV:{harqCW.rv}")
                
            Please refer to the notebook :doc:`../Playground/Notebooks/HARQ/HarqEventCallback` for a complete example 
            of using event callback functions.
        """
        self.encoder = encoder
        self.decoder = encoder.getDecoder()
        self.numCW = 2 if encoder.txLayers>4 else 1
        assert harqType in ["CC", "IR"]
        self.harqType = harqType
        assert numProc>0 and numProc<=32
        self.numProc = numProc
        self.processes = [ HarqProcess(self,i,self.numCW) for i in range(numProc) ]
        self.rvSequence = rvSequence
        self.maxTries = maxTries
        self.eventCallback = eventCallback
        self.reset()

    # ******************************************************************************************************************
    def reset(self):
        r"""
        Resets this HARQ entity to prepare it for a new set of transmissions. It resets the counters and releases 
        any encoder/decoder retransmission buffers by invoking :py:meth:`HarqProcess.reset` for all HARQ
        processes.        
        """
        for p in self.processes:    p.reset()
        self.curProcIdx = 0
        self.rxBits = np.zeros(self.maxTries, dtype=np.int32)   # rxBits[i]: Total bits received at i'th try
        self.txBits = np.zeros(self.maxTries, dtype=np.int32)   # txBits[i]: Total bits transmitted at i'th try
        self.rxBlocks = np.zeros(self.maxTries, dtype=np.int32) # rxBlocks[i]: Total TxBlocks received at i'th try
        self.txBlocks = np.zeros(self.maxTries, dtype=np.int32) # txBlocks[i]: Total TxBlocks transmitted at i'th try
        self.numTimeouts = 0

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`~neoradium.harq.HarqEntity` object.

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
        if title is None:   title = f"HARQ Entity Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  HARQ Type:            {self.harqType}\n"
        repStr += indent*' ' + f"  Num. Processes:       {self.numProc}\n"
        repStr += indent*' ' + f"  Num. Codewords:       {len(self.processes[0].cws)}\n"
        repStr += indent*' ' + f"  RV sequence:          {self.rvSequence}\n"
        repStr += indent*' ' + f"  maxTries:             {self.maxTries}\n"

        repStr += self.encoder.print(indent+2, f"Encoder:", True)
        repStr += self.decoder.print(indent+2, f"Decoder:", True)

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def printStats(self, getStr=False):
        r"""
        Prints the statistics of this :py:class:`~neoradium.harq.HarqEntity` object.

        Parameters
        ----------
        getStr: Boolean
            If `True`, returns a text string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a text string. 
            Otherwise, nothing is returned.
        """
        repStr = "\nHARQ Entity Statistics:\n"
        repStr += f"  txBits (per try):     {self.txBits}\n"
        repStr += f"  rxBits (per try):     {self.rxBits}\n"
        repStr += f"  txBlocks (per try):   {self.txBlocks}\n"
        repStr += f"  rxBlocks (per try):   {self.rxBlocks}\n"
        repStr += f"  numTimeouts:          {self.numTimeouts}\n"
        repStr += f"  totalTxBlocks:        {self.totalTxBlocks}\n"
        repStr += f"  totalRxBlocks:        {self.totalRxBlocks}\n"
        repStr += f"  totalTxBits:          {self.totalTxBits}\n"
        repStr += f"  totalRxBits:          {self.totalRxBits}\n"
        repStr += f"  throughput:           {self.throughput:.2f}%\n"
        repStr += f"  bler:                 {self.bler:.2f}%\n"
        repStr += f"  Average Num. Retries: {self.meanTries:.2f}%\n"
        if getStr: return repStr
        print(repStr)
        
    # ******************************************************************************************************************
    def handleEvent(self, event, process):      # Not documented
        # Called to handle events. Currently it only calls the callback function if one is specified.
        if self.eventCallback is not None:
            self.eventCallback(event, process)
    
    # ******************************************************************************************************************
    def getRV(self, tryNum):                    # Not documented
        # Returns the rv value based on current number of retransmissions (tryNum)
        if self.harqType == "CC":   return 0                                              # CC: Always 0
        else:                       return self.rvSequence[ tryNum%len(self.rvSequence) ] # IR: Based on rvSequence
    
    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def totalTxBlocks(self):        return self.txBlocks.sum().item()

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def totalRxBlocks(self):        return self.rxBlocks.sum().item()

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def totalTxBits(self):          return self.txBits.sum().item()

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def totalRxBits(self):          return self.rxBits.sum().item()

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def throughput(self):           return self.totalRxBits*100/self.totalTxBits

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def bler(self):                 return (self.totalTxBlocks-self.totalRxBlocks)*100/self.totalTxBlocks
        
    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def meanTries(self):    return (((self.rxBlocks*np.arange(self.maxTries)).sum()+self.numTimeouts*self.maxTries)/ \
                                    max(self.totalRxBlocks+self.numTimeouts,1)).item()

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def curProcess(self):           return self.processes[self.curProcIdx]

    # ******************************************************************************************************************
    @property                       # Not documented (Already mentioned in the __init__ documentation)
    def needNewData(self):          return self.curProcess.needNewData

    # ******************************************************************************************************************
    def __getitem__(self, idx):     return self.processes[idx]

    # ******************************************************************************************************************
    def goNext(self):
        r"""
        This function should be called after the transmission of each transport block. It updates the internal 
        pointer ``curProcIdx`` to point to the next HARQ process.
        """
        self.curProcIdx = (self.curProcIdx+1)%self.numProc
    
    # ******************************************************************************************************************
    def getRateMatchedCodeBlocks(self, txBlocks, gs=None, concatCBs=True):
        r"""
        This function takes a list of one or two transport blocks (``txBlocks``) based on the number of codewords 
        and returns a list of LDPC-encoded and rate-matched bitstreams for each codeword. If a transport block in 
        the ``txBlocks`` list is set to `None`, it uses the buffered encoded bitstream and only applies rate matching 
        for retransmitting the previously encoded transport block. Otherwise, it assumes new transmission and encodes 
        the transport block and saves the encoded bits into the HARQ process for future retransmissions. This function 
        internally calls the :py:meth:`~HarqProcess.getRateMatchedCodeBlocks` method of the :py:class:`~HarqProcess` 
        class. For more details, refer to the documentation of :py:meth:`HarqProcess.getRateMatchedCodeBlocks`.
        """
        if not isinstance(txBlocks,list):
            return self.curProcess.getRateMatchedCodeBlocks([txBlocks], [gs], concatCBs)[0]
        return self.curProcess.getRateMatchedCodeBlocks(txBlocks, gs, concatCBs)
        
    # ******************************************************************************************************************
    def decodeLLRs(self, llrs, txBlockSize, numIter=5):
        r"""
        This function takes a list of one or two NumPy arrays, each containing Log-Likelihood Ratios (LLRs) for the 
        demodulated received signals for each codeword. It then returns a list of one or two decoded transport blocks, 
        with the size of each transport block specified by the ``txBlockSizes`` list, which provides the transport 
        block sizes for each codeword.

        For each codeword, the function recovers the rate, decodes the code blocks using LDPC decoding, checks the 
        code block CRCs, and merges all code blocks to reassemble the transport block. In case of retransmissions, 
        the function combines the LLR values of the retransmissions with those from previous transmissions before 
        decoding the code blocks. 
        
        This function internally calls the :py:meth:`~HarqProcess.decodeLLRs` method of the :py:class:`~HarqProcess` 
        class. For more details, refer to the documentation of :py:meth:`HarqProcess.decodeLLRs`.
        """
        if not isinstance(llrs,list):
            retVals = self.curProcess.decodeLLRs([llrs], [txBlockSize], numIter)
            return retVals[0][0], retVals[1][0]
        return self.curProcess.decodeLLRs(llrs, txBlockSize, numIter)

