Migration Guide
===============

This release of **NeoRadium** introduces a significant reorganization of classes and functions to make the API simpler and more consistent to use. All example notebooks in :doc:`Playground/Playground` have been updated to reflect the new API. The documentation has also been revised to clearly mark deprecated classes and functions and to point to their recommended replacements. In addition, the documentation for the new APIs includes examples that show how to update existing code.

Most deprecated functionality remains available for now to ease the transition. However, **NeoRadium** may emit warning messages to remind users to migrate to the new API.

The only deprecated classes are :py:class:`~neoradium.ldpc.LdpcBase`, :py:class:`~neoradium.ldpc.LdpcEncoder`, and :py:class:`~neoradium.ldpc.LdpcDecoder`. These classes have been replaced by :py:class:`~neoradium.ldpccodec.LdpcCodec`, whose documentation includes a dedicated section on :ref:`migrating existing code <ldpccodec-migration-guide>`.

The table below lists deprecated functions and their recommended replacements.

.. list-table:: Deprecated functions and replacements
   :header-rows: 1
   :widths: 45 45

   * - Deprecated
     - Use instead
   * - :py:meth:`Grid.precode <neoradium.grid.Grid.precode>`
     - :py:meth:`PDSCH.precodeTo <neoradium.pdsch.PDSCH.precodeTo>`

   * - :py:meth:`Grid.equalize <neoradium.grid.Grid.equalize>`
     - :py:meth:`PDSCH.equalize <neoradium.pdsch.PDSCH.equalize>`

   * - :py:meth:`Grid.estimateChannelLS <neoradium.grid.Grid.estimateChannelLS>`
     - :py:meth:`PDSCH.estimateChannel <neoradium.pdsch.PDSCH.estimateChannel>`

   * - :py:meth:`Modem.getLLRsFromSymbols <neoradium.modulation.Modem.getLLRsFromSymbols>`
     - :py:meth:`Modem.getLLRs <neoradium.modulation.Modem.getLLRs>`

   * - :py:meth:`PDSCH.getGrid <neoradium.pdsch.PDSCH.getGrid>`
     - :py:meth:`PDSCH.initGrid <neoradium.pdsch.PDSCH.initGrid>`

   * - :py:meth:`PDSCH.getReIndexes <neoradium.pdsch.PDSCH.getReIndexes>`
     - :py:meth:`Grid.getReIndexes <neoradium.grid.Grid.getReIndexes>`

   * - :py:meth:`PDSCH.getBitSizes <neoradium.pdsch.PDSCH.getBitSizes>`
     - :py:meth:`PDSCH.getBitCapacity <neoradium.pdsch.PDSCH.getBitCapacity>`

   * - :py:meth:`PDSCH.populateGrid <neoradium.pdsch.PDSCH.populateGrid>`
     - :py:meth:`PDSCH.setPdschData <neoradium.pdsch.PDSCH.setPdschData>`

   * - :py:meth:`PDSCH.getLLRsFromGrid <neoradium.pdsch.PDSCH.getLLRsFromGrid>`
     - :py:meth:`PDSCH.getLLRs <neoradium.pdsch.PDSCH.getLLRs>`

   * - :py:meth:`PDSCH.getHardBitsFromGrid <neoradium.pdsch.PDSCH.getHardBitsFromGrid>`
     - :py:meth:`PDSCH.getHardBits <neoradium.pdsch.PDSCH.getHardBits>`

   * - :py:meth:`HarqEntity.getRateMatchedCodeBlocks <neoradium.harq.HarqEntity.getRateMatchedCodeBlocks>`
     - :py:meth:`HarqEntity.encode <neoradium.harq.HarqEntity.encode>`

   * - :py:meth:`HarqEntity.decodeLLRs <neoradium.harq.HarqEntity.decodeLLRs>`
     - :py:meth:`HarqEntity.decode <neoradium.harq.HarqEntity.decode>`
     

