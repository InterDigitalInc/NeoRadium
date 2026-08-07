# Copyright (c) 2025-2026, InterDigital AI Lab
"""
The module ``harq.py`` contains the API for Hybrid Automatic Repeat reQuest (HARQ), a fundamental mechanism in 5G NR
used for error correction and reliable data transfer over radio links. The class structure follows the 3GPP 
HARQ hierarchy:

- :py:class:`~neoradium.harq.HarqEntity`: The main HARQ management object. It implements the "HARQ Entity" as specified
  in **3GPP TS 38.321, Section 5.3.2.1**.
- :py:class:`~neoradium.harq.HarqProcess`: Each HARQ entity contains one or more HARQ processes. This class implements 
  the "HARQ Process" object as specified in **3GPP TS 38.321, Section 5.3.2.2**.
- :py:class:`~neoradium.harq.HarqCW`: Each HARQ process can handle up to two codewords. This class implements HARQ 
  processing for a single codeword. A :py:class:`~neoradium.harq.HarqProcess` can have one or two 
  :py:class:`~neoradium.harq.HarqCW` objects, depending on the number of codewords. 

This implementation adheres to the procedures specified in **3GPP TS 38.321, TS 38.212, and TS 38.214**.

**A typical HARQ workflow:**

Here’s a typical workflow example for using HARQ in your simulations:

1) Create an :py:class:`~neoradium.ldpccodec.LdpcCodec` object, for instance:

    .. code-block:: python
            
        ldpc = pdsch.getLdpcCodec(coderates=490/1024)

2) Create a HARQ entity object, passing in the :py:class:`~neoradium.ldpccodec.LdpcCodec` object:

    .. code-block:: python
            
        harq = HarqEntity(ldpc, harqType="IR", numProc=16)  # Using "Incremental Redundancy" with 16 HARQ processes

3) For each transmission and each codeword, check the ``needNewData`` property. If it is `True` for a codeword, create
   the transport block bits for the corresponding codeword. Otherwise, set the transport block to `None` (for 
   retransmission). Once the transport blocks are ready for each codeword, call the 
   :py:meth:`HarqEntity.encode` function to prepare the rate-matched bitstreams for transmission. 
   Here is an example:

    .. code-block:: python
            
        txBlocks = []                                         # Transport blocks, one per codeword.
        for c in range(numCodewords):
            if harq.needNewData[c]:                           # New transmission.
                txBlocks += [ random.bits(txBlockSizes[c]) ]  # Create random bits for the new transport block
            else:                                             # Retransmission
                txBlocks += [ None ]                          # Set transport block to None indicating retransmission

        rateMatchedCodeBlocks = harq.encode(txBlocks)         # Prepare the bitstream for transmission    

4) At the receiving end, after demodulating the LLRs, the :py:meth:`HarqEntity.decode` method is invoked to obtain 
   the decoded transport block or blocks and their CRC results. For instance:
    
    .. code-block:: python

        decodedTxBlocks, crcMatches = harq.decode(llrs)    # Decode received LLRs

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
# 04/12/2026    Shahab Hamidi-Rad       Changes in NeoRadium version 0.5.0:
#                                       * Updated the code to use the new 'LdpcCodec' API. See the "Migration Guide"
#                                         in the module documentation in 'ldpccodec.py'.
#                                       * Deprecated the 'decodeLLRs' and 'getRateMatchedCodeBlocks'. Use the new
#                                         functions 'decode' and 'encode' respectively.
#                                       * Removed the 'meanTries' and added the new 'meanFailedTransmissions', and
#                                         'meanRetransmissions' properties to the HarqEntity class.
#                                       * Now keeping statistics (numRxBits, numRxBlocks, numTxBits, numTxBlocks, and
#                                         numTimeouts) for each codeword.
#                                       * Allow changing LDPC codec. See the 'setLdpc' methods.
# **********************************************************************************************************************
import numpy as np
from .ldpccodec import LdpcCodec
from .utils import validateRange, deprecated, DOCS_LOC

docFile = "Harq"            # Used by the 'deprecated' decorators

# **********************************************************************************************************************
class HarqCW:
    r"""
    This class implements HARQ processing for a single codeword. The :py:class:`neoradium.harq.HarqProcess` class 
    can have either one or two :py:class:`HarqCW` objects.
    """
    # ******************************************************************************************************************
    def __init__(self, process, cwIdx):
        r"""
        Parameters
        ----------
        process : :py:class:`~neoradium.harq.HarqProcess`
            The :py:class:`~neoradium.harq.HarqProcess` object that holds this :py:class:`~neoradium.harq.HarqCW` 
            instance.
            
        cwIdx : int
            The codeword index, which is 0 for the first codeword, or 1 for the second codeword.
             
                    
        **Other Read-Only Properties:**
        
            :curTry: This indicates the current number of retransmissions. It starts at zero for the first transmission
                for this codeword. This value increments after each transmission failure.
            :rv: This represents the redundancy version used for retransmissions. For "Chase Combining" retransmission,
                this value is always zero. For "Incremental Redundancy" retransmission, it is updated based on the 
                ``rvSequence`` parameter of the :py:class:`~neoradium.harq.HarqEntity` object.
            :needNewData: This indicates whether the :py:class:`~neoradium.harq.HarqCW` object is ready to receive a 
                new transport block for transmission. If it is `True`, it means it is ready to transmit the new block. 
                If it is `False`, it means it is still busy retransmitting the previous block. This mirrors the role of 
                **New Data Indicator (NDI)** in **3GPP TS 38.212**
        """
        self.process = process  # HARQ process object "owning" this HarqCW
        self.cwIdx = cwIdx      # The codeword Index associated with this object 0=>"Codeword 1", 1=>"Codeword 2"
        self.reset()            # Reset the parameters and statistics of this HarqCW

    # ******************************************************************************************************************
    def reset(self):                                                            # Undocumented
        # Called after a successful transmission, a timeout, or when called by the HARQ Entity/Process
        self.curTry = 0         # Current Try number (AKA retransmission number)
        self.rv = 0             # Current redundancy version of this HarqCW
        self.encBuffer = None   # The encoder buffer for this HarqCW which contains the encoded bits.
        self.decBuffer = None   # The decoder buffer for this HarqCW which contains the decoding circular queue
        self.ldpcCodec = self.process.entity.ldpcCodec.cwCodecs[self.cwIdx]     # LDPC codec for this codeword

    # ******************************************************************************************************************
    def setLdpc(self):
        # If we are in the middle of retransmission, we keep using current LDPC codec until
        # this transmission is complete (e.g. timed out or successfully decoded). Then a call to reset() function
        # updates the ldpcCodec.
        if self.curTry == 0:    # Not in the middle of a retransmission -> Update the LDPC codec immediately
            self.ldpcCodec = self.process.entity.ldpcCodec.cwCodecs[self.cwIdx]
        
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)                      # Undocumented
    def print(self, indent=0, title=None, getStr=False):                        # Undocumented
        # Prints the information about this HarqCW
        if title is None:   title = f"HARQ Codeword Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  curTry:             {self.curTry}\n"
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
    def encode(self, txBlock, g, concatCBs):                                    # Undocumented
        # Uses the LdpcCodecCW methods to encode 'txBlock'
        if txBlock is None:                                                     # Retransmission
            if self.curTry==0: raise ValueError("'txBlock' cannot be 'None' for first transmission!")
            assert self.encBuffer is not None
            # In this case we already have the encoded bits in the "encBuffer". Just need to rate-match them
            rateMatchedCodeBlocks = self.ldpcCodec.rateMatch(self.encBuffer, g, concatCBs, self.rv)
            
        else:                                                                   # New transmission
            if self.curTry>0: raise ValueError("'txBlock' must be 'None' for retransmissions!")
            assert self.encBuffer is None
            # Note that the following call will update the 'self.encBuffer' with the encoded code blocks
            rateMatchedCodeBlocks = self.ldpcCodec.encode(txBlock, g, concatCBs, self)
        return rateMatchedCodeBlocks

    # ******************************************************************************************************************
    def decode(self, llrs):                                                     # Undocumented
        decodedTxBlock, crcMatch = self.ldpcCodec.decode(llrs, self)
       
        # Note that we update the state of this HarqCW and all statistics of the HARQ Entity after decoding the
        # received transmission.
        
        # Note: crcMatch[0] is the CRC match flag for the whole transport block and crcMatch[1,2,...] are the
        # CRC match flags for individual code blocks. Here we only care about the whole transport block, so
        # ``not crcMatch[0]`` means a transport block error.
        self.update(not crcMatch[0], self.ldpcCodec.txBlockSize)
        return decodedTxBlock, crcMatch

    # ******************************************************************************************************************
    def update(self, blockError, txBlockSize):                                  # Undocumented
        # Update all stats after decoding the received transmission.
        entity = self.process.entity                                # Get the entity object
        entity.numTxBits[self.cwIdx,self.curTry] += txBlockSize     # Update number of transmitted bits for this try
        entity.numTxBlocks[self.cwIdx,self.curTry] += 1             # Update number of transmitted blocks for this try

        if blockError:                                          # Transmission failed
            entity.handleEvent("RXFAILED", self )                   # Create "RXFAILED" event
            self.curTry += 1                                        # Increase the current try count
            if self.curTry == entity.maxTries:                      # Reached Max. try count -> timed out!
                entity.handleEvent("TIMEOUT", self)                 # Create "TIMEOUT" event
                entity.numTimeouts[self.cwIdx] += 1                 # Increase the number of timeouts
                self.reset()                                        # Prepare for a new transmission
            else:                                                   # Try again
                self.rv = entity.getRV(self.curTry)                 # Set the Redundancy Version for the retransmission
        else:                                                   # Successful transmission
            entity.numRxBits[self.cwIdx,self.curTry] += txBlockSize # Update number of received bits for this try
            entity.numRxBlocks[self.cwIdx,self.curTry] += 1         # Update number of received blocks for this try
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
        entity : :py:class:`~neoradium.harq.HarqEntity`
            The :py:class:`~neoradium.harq.HarqEntity` object that holds this :py:class:`~neoradium.harq.HarqProcess`
            instance.
            
        id : int
            The unique identifier associated with this HARQ process.
                    
        numCW : int
            The number of codewords processed by this HARQ process. It can be 1 or 2.


        **Other Read-Only Properties:**
        
            :needNewData: A list of one or two boolean values, corresponding to each codeword. For each element in the 
                list, a `True` value indicates that the HARQ process is ready to receive a new transport block for 
                transmission, while a `False` value signifies that it is currently busy retransmitting the previous 
                transport block. This aligns with the **New Data Indicator (NDI)** defined in **3GPP TS 38.212**.
        """
        self.id = id                                            # HARQ process identifier
        self.entity = entity                                    # HARQ entity
        self.cws = [ HarqCW(self,i) for i in range(numCW) ]     # One or two HarqCW objects for each codeword

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
    @property                                                                   # Undocumented
    def needNewData(self):  return [cw.needNewData for cw in self.cws]

    # ******************************************************************************************************************
    def encode(self, txBlocks, g, concatCBs):
        r"""
        This function takes a list of one or two transport blocks (``txBlocks``) based on the number of codewords 
        and returns a list of LDPC-encoded and rate-matched bitstreams for each codeword. If a transport block in 
        the ``txBlocks`` list is set to `None`, it uses the buffered encoded bitstream and only applies rate matching 
        for retransmitting the previously encoded transport block. Otherwise, it assumes new transmission and encodes 
        the transport block, saving the encoded bits for future retransmissions. For more information about the LDPC 
        encoding process, refer to the :py:meth:`~neoradium.ldpccodec.LdpcCodec.encode` method of the 
        :py:class:`~neoradium.ldpccodec.LdpcCodec` class.

        Parameters
        ----------
        txBlocks : list
            A list of one or two NumPy arrays for each codeword. The presence of a ‘None’ value for each transport 
            block in the list indicates a retransmission of previously buffered encoded transport block. 

        g : list or None
            A list of one or two integer values for each codeword. Each element in the list represents the total 
            number of bits available for transmitting the transport block. This value corresponds to the value 
            :math:`G` in the *bit selection* process explained in **3GPP TS 38.212, Section 5.4.2.1**. If not 
            provided (default), it is calculated using the formula :math:`G=\lceil \frac {B-24} R \rceil`, where 
            :math:`B` is the transport block size and :math:`R` is the code rate.
            
        concatCBs : bool
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
        if isinstance(txBlocks,(list,tuple)):
            # If txBlock and gs are lists, then return a list of results for items in the lists
            numCW = self.entity.numCW
            if len(txBlocks)!=numCW:
                raise ValueError(f"'txBlocks' must have exactly {numCW} transport block array{['','s'][numCW-1]}!")
            if not isinstance(g, (list, tuple)):    g = [g] * numCW     # Includes the case where g is None.
            if len(g)!=numCW:
                raise ValueError(f"'g' must have exactly {numCW} value{['','s'][numCW-1]}!")
            return [ self.cws[c].encode(txBlocks[c], g[c], concatCBs) for c in range(numCW) ]
        
        return self.cws[0].encode(txBlocks, g, concatCBs)

    # ******************************************************************************************************************
    @deprecated("encode", docFile)
    def getRateMatchedCodeBlocks(self, txBlocks, gs=None, concatCBs=True):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`encode` method instead.
        """
        return self.encode(txBlocks, gs, concatCBs)

    # ******************************************************************************************************************
    def decode(self, llrs):
        r"""
        This function takes a list of one or two NumPy arrays, each containing Log-Likelihood Ratios (LLRs) for the 
        demodulated received signals for each codeword. It then returns the decoded transport block for a single 
        codeword, or the decoded transport blocks for two codewords.

        For each codeword, the function uses the :py:class:`~neoradium.ldpccodec.LdpcCodecCW` methods to recover the
        rate, decode the code blocks, check the code block CRCs, and merge all code blocks to reassemble the 
        transport block. In case of retransmissions, the function combines the LLR values of the retransmission with 
        those from previous transmissions before decoding the code blocks. 
        
        Parameters
        ----------
        llrs : list
            A list of one or two NumPy arrays, each containing the Log-Likelihood Ratios (LLRs) from the demodulated 
            received signals corresponding to each codeword. 

        Returns
        -------
        tuple
            A tuple containing:

            - decodedTxBlocks : NumPy array or list
                The decoded transport block(s) (without CRC bits).

                - For a single codeword: a 1D NumPy array of length :math:`A`.
                - For two codewords: a list of two NumPy arrays.

            - crcMatches : NumPy array or list
                CRC check results for each codeword.

                Each element is a boolean array of length ``C + 1``:
                
                - The first element corresponds to the transport-block CRC (CRC24A),
                - The remaining elements correspond to code-block CRCs (CRC24B) when segmentation is used.

                For two codewords, a list of two such arrays is returned.
        """
        numCW = self.entity.numCW
        if isinstance(llrs,(list,tuple)):
            if len(llrs)!=numCW:
                raise ValueError(f"'llrs' must have exactly {numCW} LLR array{['','s'][numCW-1]}!")

            crcMatches = []
            decodedTxBlocks = []
            ack = numCW*[-1]        # Used by link adaptation. -1 -> no updates, 1 -> ACK, 0 -> NACK
            for cw in range(numCW):
                decodedTxBlock, crcMatch = self.cws[cw].decode(llrs[cw])
                crcMatches += [ crcMatch ]
                decodedTxBlocks += [ decodedTxBlock ]
                if crcMatch[0]:
                    if self.cws[cw].curTry == 0:    ack[cw] = 1     # First transmission ACK
                elif self.cws[cw].curTry == 1:      ack[cw] = 0     # First transmission NACK

            if (self.entity.la is not None) and (max(ack)>-1):      self.entity.la.update(ack)
            return decodedTxBlocks, crcMatches
#            txBlockCrcMatchPairs = [ self.cws[c].decode(llrs[c]) for c in range(numCW) ]
#            
#            # Return a list of decoded txBlocks and a list of crcMatch info
#            return tuple(list(x) for x in zip(*txBlockCrcMatchPairs))

        if numCW>1:
            raise ValueError(f"'llrs' must be a list or tuple of {numCW} LLR array{['','s'][numCW-1]}!")

        decodedTxBlock, crcMatch = self.cws[0].decode(llrs)
        if self.entity.la is not None:
            if crcMatch[0]:
                if self.cws[0].curTry == 0:     self.entity.la.update(True)     # First transmission ACK
            elif self.cws[0].curTry == 1:       self.entity.la.update(False)    # First transmission NACK
        return decodedTxBlock, crcMatch

    # ******************************************************************************************************************
    @deprecated("decode", docFile)
    def decodeLLRs(self, llrs, txBlockSizes, numIter=5):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`decode` method instead.
        """
        return self.decode(llrs)
        
# **********************************************************************************************************************
class HarqEntity:
    r"""
    This class encapsulates the functionality of a HARQ entity as specified in **3GPP TS 38.321, Section 5.3.2.1**. A
    HARQ entity is configured for each Serving Cell to manage downlink (DL) and uplink (UL) HARQ operations. The 
    primary purpose of a HARQ entity is to maintain multiple parallel HARQ processes and direct HARQ information and 
    associated Transport Blocks to these corresponding processes. This HARQ information includes key parameters 
    such as the New Data Indicator (NDI), Transport Block Size (TBS), Redundancy Version (RV), and the HARQ process
    ID.
    
    .. Note:: In the absence of scheduler/DCI modeling, this implementation advances HARQ processes in round-robin 
        order. This is a simulation simplification that provides deterministic use of parallel HARQ processes, but 
        it does not model scheduler-controlled HARQ process selection used in NR systems.
    """
    # ******************************************************************************************************************
    def __init__(self, ldpcCodec, harqType="CC", numProc=8, rvSequence=[0,2,3,1], maxTries=4, eventCallback=None):
        r"""
        Parameters
        ----------
        ldpcCodec : :py:class:`~neoradium.ldpccodec.LdpcCodec`
            The :py:class:`neoradium.ldpccodec.LdpcCodec` object, used by this HARQ entity for LDPC coding. 
                         
        harqType : str
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

        numProc : int
            The number of HARQ processes utilized by this HARQ entity. It varies depending on the capabilities 
            of the UE. A HARQ entity can manage up to 32 HARQ processes, with the default being 8.

        rvSequence : list of integers
            Defines the order in which standardized redundancy-version identifiers are used across retransmissions. 
            In NR LDPC rate matching, the valid RV identifiers are 0, 1, 2, and 3, each corresponding to a different 
            starting position in the circular buffer defined in **TS 38.212 Table 5.4.2.1-2**. The default sequence 
            ``[0, 2, 3, 1]`` is a common simulation choice for incremental-redundancy HARQ because it cycles through
            the four standardized RVs and exposes different portions of the coded block across retransmissions. It 
            should be treated as a configurable retransmission policy rather than a universally mandated sequence 
            for every scenario.  
            
        maxTries : int
            The maximum number of transmission attempts for a specified transport block, including the initial
            transmission and any retransmissions. If a transport block still fails after this many attempts,
            a timeout event occurs.
            
        eventCallback : function or None
            If this callback function is provided, it will be invoked on the following events:
            
            :"RXFAILED": This event is triggered when a transport block transmission fails, whether it’s the 
                original transmission or a retransmission.

            :"RXSUCCESS": This event is triggered when a transport block transmission succeeds, whether it’s the 
                original transmission or a retransmission.

            :"TIMEOUT": This event is triggered when a transport block transmission fails after the ``maxTries``
                transmissions.

            For more information, refer to the :ref:`Event callback function section <EventCallback>` below.


        **Other Read-Only Properties:**
        
            :processes: A list of :py:class:`~neoradium.harq.HarqProcess` objects managed by this HARQ entity.
            :curProcess: The HARQ process that is currently transmitting or retransmitting the transport blocks.
            :curProcIdx: The current HARQ process that is currently transmitting or retransmitting the transport blocks. 
            :numCW: The number of codewords. This is based on the LDPC encoder's ``txLayers`` parameter. 
            :needNewData: A list of one or two boolean values, corresponding to each codeword. For each element in the 
                list, a `True` value indicates that the current HARQ process (``curProcess``) is ready to receive a new 
                transport block for transmission, while a `False` value signifies that it is currently busy 
                retransmitting the previous transport block. This mirrors the role of the **New Data Indicator (NDI)** 
                defined in **3GPP TS 38.212**.
            :totalTxBlocks: The total number of transport blocks transmitted, including retransmissions.
            :totalRxBlocks: The total number of transport blocks received and successfully decoded.
            :totalTxBits: The total number of transport block bits transmitted, including retransmissions.
            :totalRxBits: The total number of transport block bits received and successfully decoded.
            :throughput: The communication throughput expressed as a percentage. It’s calculated as
                ``totalRxBits*100/totalTxBits``.
            :bler: The block error rate expressed as a percentage. It’s calculated as 
                ``(totalTxBlocks-totalRxBlocks)*100/totalTxBlocks``.
            :bler1st: The first-transmission block error rate expressed as a percentage.
            :numTimeouts: The total number of timeout events for each codeword. A timeout occurs when a transport 
                block fails after all retransmission attempts.
            :meanFailedTransmissions: The average number of failed transmissions. This values ranges from zero 
                (indicating no failed transmissions) up to ``maxTries`` (indicating all timed out transmissions).
            :meanRetransmissions: The average number of retransmissions per transport block. This value ranges from zero
                (indicating no retransmissions) up to ``maxTries-1``.

        .. _EventCallback:
            
        **Event Callback Function**
        
        This function is automatically invoked when a transmission event occurs. It accepts the following parameters:
        
            :eventStr: This string can be one of ``"RXFAILED"``, ``"RXSUCCESS"``, or ``"TIMEOUT"``, as explained above.
            :harqCW: The :py:class:`~neoradium.harq.HarqCW` object that triggered the event. This object can be used 
                to obtain more information about the event.
            
            Here is an example of an event callback function that prints all triggered events:
        
            .. code-block:: python
            
                def myEventHandler(eventStr, harqCW):
                    print(f"HARQ Process {harqCW.process.id:2d} CW{harqCW.cwIdx+1}:{eventStr:10s} curTry:{harqCW.curTry} RV:{harqCW.rv}")
                
            Please refer to the notebook :doc:`../Playground/Notebooks/HARQ/HarqEventCallback` for a complete example 
            of using event callback functions.
        """
        self.ldpcCodec = ldpcCodec
        if not isinstance(self.ldpcCodec, LdpcCodec):
            raise ValueError("'ldpcCodec' must be an instance of the 'LdpcCodec' class. " +
                             "If you are still using 'LdpcEncoder' class, please visit " +
                             DOCS_LOC + "source/API/ChanCode.html#" +
                             "migration-guide-ldpcencoder-ldpcdecoder-ldpccodec for information about how to "+
                             "migrate your code to use the new 'LdpcCodec' class.")

        self.numCW = 2 if self.ldpcCodec.numLayers>4 else 1
        self.harqType = harqType
        validateRange(self.harqType, ["CC", "IR"])
        self.numProc = numProc
        validateRange(self.numProc, (1,32))
        self.processes = [ HarqProcess(self,i,self.numCW) for i in range(numProc) ]
        self.rvSequence = rvSequence        # The default choice is based on 3GPP TS 38.214, Table 5.1.2.1-2
        if len(self.rvSequence)==0: raise ValueError("'rvSequence' must not be empty!")
        for rv in self.rvSequence:
            if rv not in [0,1,2,3]: raise ValueError("The elements in the 'rvSequence' must be one of {0,1,2,3}!")
        self.maxTries = maxTries
        validateRange(self.maxTries, (1, 16))
        self.eventCallback = eventCallback
        self.la = None
        self.reset()

    # ******************************************************************************************************************
    def reset(self):
        r"""
        Resets this HARQ entity to prepare it for a new set of transmissions. It resets the counters and releases 
        any LDPC retransmission buffers by invoking :py:meth:`HarqProcess.reset` for all HARQ processes.        
        """
        for p in self.processes:    p.reset()
        self.curProcIdx = 0

        # In the following the item [c,i] corresponds to codeword 'c' and try number 'i'
        self.numRxBits = np.zeros((self.numCW, self.maxTries), dtype=np.int32)      # Total RX bits
        self.numTxBits = np.zeros((self.numCW, self.maxTries), dtype=np.int32)      # Total TX bits
        self.numRxBlocks = np.zeros((self.numCW, self.maxTries), dtype=np.int32)    # Total RX blocks
        self.numTxBlocks = np.zeros((self.numCW, self.maxTries), dtype=np.int32)    # Total TX blocks
        self.numTimeouts = self.numCW*[0]
        if self.la is not None: self.la.reset()

    # ******************************************************************************************************************
    def setLdpc(self, ldpcCodec):
        r"""
        Update the LDPC codec used for new HARQ transmissions.

        The new codec is applied only to HARQ processes that are idle or starting a
        new transport block. Processes with an active retransmission keep using the
        LDPC codec associated with their original transmission until that HARQ process
        completes. This preserves soft-combining consistency across retransmissions,
        even if the modulation scheme, code rate, or other LDPC-related parameters
        change.

        Parameters
        ----------
        ldpcCodec : :py:class:`~neoradium.ldpccodec.LdpcCodec`
            New LDPC codec to use for subsequent HARQ transmissions.
        """
        self.ldpcCodec = ldpcCodec
        if not isinstance(self.ldpcCodec, LdpcCodec):
            raise ValueError("'ldpcCodec' must be an instance of the 'LdpcCodec' class. " +
                             "If you are still using 'LdpcEncoder' class, please visit " +
                             DOCS_LOC + "source/API/ChanCode.html#" +
                             "migration-guide-ldpcencoder-ldpcdecoder-ldpccodec for information about how to " +
                             "migrate your code to use the new 'LdpcCodec' class.")
        for process in self.processes:
            for cw in process.cws:
                cw.setLdpc()

    # ******************************************************************************************************************
    def setLA(self, la):
        r"""
        Set the link-adaptation object associated with this HARQ entity.

        The link-adaptation object is notified when ACK/NACK feedback becomes available, allowing it 
        to update its adaptation state, such as an OLLA CQI offset. This method is typically called 
        by the link-adaptation object during initialization.

        Parameters
        ----------
        la : :py:class:`~neoradium.csireport.OLLA`
            Link-adaptation object associated with this HARQ entity. The object is expected to provide
            an ``update`` method that accepts ACK/NACK feedback.
        """
        self.la = la
    
    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`~neoradium.harq.HarqEntity` object.

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
        if title is None:   title = f"HARQ Entity Properties:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  HARQ Type:            {self.harqType}\n"
        repStr += indent*' ' + f"  Num. Processes:       {self.numProc}\n"
        repStr += indent*' ' + f"  Num. Codewords:       {len(self.processes[0].cws)}\n"
        repStr += indent*' ' + f"  RV sequence:          {self.rvSequence}\n"
        repStr += indent*' ' + f"  maxTries:             {self.maxTries}\n"
        repStr += self.ldpcCodec.print(indent+2, f"LDPC codec:", True)

        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    def printStats(self, getStr=False):
        r"""
        Prints the statistics of this :py:class:`~neoradium.harq.HarqEntity` object.

        Parameters
        ----------
        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
            Otherwise, nothing is returned.
        """
        repStr = "\nHARQ Entity Statistics:\n"
        if self.numCW == 1:
            repStr += f"  numTxBits (per try):      {self.numTxBits[0]}\n"
            repStr += f"  numRxBits (per try):      {self.numRxBits[0]}\n"
            repStr += f"  numTxBlocks (per try):    {self.numTxBlocks[0]}\n"
            repStr += f"  numRxBlocks (per try):    {self.numRxBlocks[0]}\n"
            repStr += f"  numTimeouts:              {self.numTimeouts[0]}\n"
        else:
            repStr += "\n  First codeword:\n"
            repStr += f"    numTxBits (per try):    {self.numTxBits[0]}\n"
            repStr += f"    numRxBits (per try):    {self.numRxBits[0]}\n"
            repStr += f"    numTxBlocks (per try):  {self.numTxBlocks[0]}\n"
            repStr += f"    numRxBlocks (per try):  {self.numRxBlocks[0]}\n"
            repStr += f"    numTimeouts:            {self.numTimeouts[0]}\n"
            repStr += "\n  Second codeword:\n"
            repStr += f"    numTxBits (per try):    {self.numTxBits[1]}\n"
            repStr += f"    numRxBits (per try):    {self.numRxBits[1]}\n"
            repStr += f"    numTxBlocks (per try):  {self.numTxBlocks[1]}\n"
            repStr += f"    numRxBlocks (per try):  {self.numRxBlocks[1]}\n"
            repStr += f"    numTimeouts:            {self.numTimeouts[1]}\n"

        repStr += f"  totalTxBlocks:            {self.totalTxBlocks}\n"
        repStr += f"  totalRxBlocks:            {self.totalRxBlocks}\n"
        repStr += f"  totalTxBits:              {self.totalTxBits}\n"
        repStr += f"  totalRxBits:              {self.totalRxBits}\n"
        repStr += f"  throughput:               {self.throughput:.2f}%\n"
        repStr += f"  bler:                     {self.bler:.2f}%\n"
        repStr += f"  bler1st:                  {self.bler1st:.2f}%\n"
        repStr += f"  Avg. retransmissions:     {self.meanRetransmissions:.2f}\n"
        repStr += f"  Avg. failed transmissions:{self.meanFailedTransmissions:.2f}\n"
        if getStr: return repStr
        print(repStr)
 
    # ******************************************************************************************************************
    def handleEvent(self, event, harqCW):      # Undocumented
        # Called to handle events. Currently it only calls the callback function if one is specified.
        if self.eventCallback is not None:
            self.eventCallback(event, harqCW)
    
    # ******************************************************************************************************************
    def getRV(self, tryNum):                    # Undocumented
        # Returns the rv value based on current number of retransmissions (tryNum)
        if self.harqType == "CC":   return 0                                              # CC: Always 0
        else:                       return self.rvSequence[ tryNum%len(self.rvSequence) ] # IR: Based on rvSequence
    
    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def totalTxBlocks(self):        return self.numTxBlocks.sum().item()

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def totalRxBlocks(self):        return self.numRxBlocks.sum().item()

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def totalTxBits(self):          return self.numTxBits.sum().item()

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def totalRxBits(self):          return self.numRxBits.sum().item()

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def throughput(self):           return 0 if self.totalTxBits==0 else self.totalRxBits*100/self.totalTxBits

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def bler(self):
        return 0 if self.totalTxBlocks==0 else (self.totalTxBlocks-self.totalRxBlocks)*100/self.totalTxBlocks

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def bler1st(self):
        num1stTx = self.numTxBlocks[:,0].sum()
        return 0 if num1stTx==0 else (num1stTx-self.numRxBlocks[:,0].sum())*100/num1stTx

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def meanFailedTransmissions(self):
        # Returns the mean number of failed transmissions per completed block.
        # The value ranges from 0 (no retransmissions) to maxTries (all timed out).
        
        # With maxTries=4, a returned value of 4 means all blocks timed out.
        numCompletedBlocks = self.totalRxBlocks + sum(self.numTimeouts)
        if numCompletedBlocks == 0: return 0
        numFailedTransmissions = (self.numRxBlocks.sum(0) * np.arange(self.maxTries)).sum() + \
                                 sum(self.numTimeouts) * self.maxTries
        return numFailedTransmissions/numCompletedBlocks

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def meanRetransmissions(self):
        # Returns the mean number of retransmissions per completed block.
        # The value ranges from 0 (no retransmissions) to maxTries-1.
        #
        # With maxTries=4, a timeout means 3 retransmissions occurred (all failed).
        # numRxBlocks[3] only counts successful 4th attempts, so we add numTimeouts
        # to account for failed 4th attempts, which also had 3 retransmissions.
        #
        # Compared to 'meanFailedTransmissions': the only difference is how timeouts
        # are counted. That method counts maxTries failed transmissions per timeout;
        # this one counts maxTries-1 retransmissions per timeout.
        
        # With maxTries=4, a returned value of 3 does not necessarily mean zero communication
        # because some of the 4th attempts may be successful.
        numCompletedBlocks = self.totalRxBlocks + sum(self.numTimeouts)
        if numCompletedBlocks == 0: return 0
        numRetransmissions = (self.numRxBlocks.sum(0) * np.arange(self.maxTries)).sum() + \
                             sum(self.numTimeouts) * (self.maxTries - 1)
        return numRetransmissions/numCompletedBlocks

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
    def curProcess(self):           return self.processes[self.curProcIdx]

    # ******************************************************************************************************************
    @property                       # Undocumented (Already mentioned in the __init__ documentation)
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
    def encode(self, txBlocks, gs=None, concatCBs=True):
        r"""
        This function takes a list of one or two transport blocks (``txBlocks``) based on the number of codewords 
        and returns a list of LDPC-encoded and rate-matched bitstreams for each codeword. If a transport block in 
        the ``txBlocks`` list is set to `None`, it uses the buffered encoded bitstream and only applies rate matching 
        for retransmitting the previously encoded transport block. Otherwise, it assumes new transmission and encodes 
        the transport block and saves the encoded bits into the HARQ process for future retransmissions. This function 
        internally calls the :py:meth:`~HarqProcess.encode` method of the :py:class:`~HarqProcess` 
        class. For more details, refer to the documentation of :py:meth:`HarqProcess.encode`.
        
      
        .. Note:: This function replaces the deprecated function :py:meth:`getRateMatchedCodeBlocks`. The 
            following example shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                rateMatchedCodeBlocks = harq.getRateMatchedCodeBlocks(txBlocks, numBits)

                # New:
                rateMatchedCodeBlocks = harq.encode(txBlocks, numBits)
        """
        return self.curProcess.encode(txBlocks, gs, concatCBs)

    # ******************************************************************************************************************
    @deprecated("encode", docFile)
    def getRateMatchedCodeBlocks(self, txBlocks, gs=None, concatCBs=True):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`encode` method instead.
        """
        return self.encode(txBlocks, gs, concatCBs)
        
    # ******************************************************************************************************************
    def decode(self, llrs):
        r"""
        This function takes a list of one or two NumPy arrays, each containing Log-Likelihood Ratios (LLRs) for the 
        demodulated received signals for each codeword. It then returns one or two decoded transport blocks.

        For each codeword, the function performs rate recovery, decodes the code blocks using LDPC decoding, checks the 
        code block CRCs, and merges all code blocks to reassemble the transport block. In case of retransmissions, 
        the function combines the LLR values of the retransmissions with those from previous transmissions before 
        decoding the code blocks. 
        
        This function internally calls the :py:meth:`~HarqProcess.decode` method of the :py:class:`~HarqProcess` 
        class. For more details, refer to the documentation of :py:meth:`HarqProcess.decode`.
        
        
        .. Note:: This function replaces the deprecated function :py:meth:`decodeLLRs`. The 
            following example shows how to migrate existing code to use this method:

            .. code-block:: python

                # Old:
                decodedTxBlocks, blockErrors = harq.decodeLLRs(llrs, txBlockSizes, numIter=2)

                # New:
                # numIter is set at instantiation
                # crcMatch is returned instead of blockErrors (consistent with LdpcCodec.decode) 
                decodedTxBlock, crcMatch = pdsch.decode(llrs)                       
        """
        return self.curProcess.decode(llrs)
        
    # ******************************************************************************************************************
    @deprecated("decode", docFile)
    def decodeLLRs(self, llrs, txBlockSizes, numIter=5):
        r"""
        :red:`DEPRECATED`: This method is deprecated and will be removed in future releases. Please use the 
        :py:meth:`decode` method instead.
        """
        return self.decode(llrs)

