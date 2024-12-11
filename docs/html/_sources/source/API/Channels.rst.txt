Channel Models
==============
**NeoRadium** currently supports Clustered Delay Line (CDL) and Tapped Delay
Line (TDL) channel models implemented as :py:class:`~neoradium.cdl.CdlChannel`
and :py:class:`~neoradium.tdl.TdlChannel` classes correspondingly. You
can also derive your customized channel models from the
:py:class:`~neoradium.channel.ChannelBase` class explained below.

.. automodule:: neoradium.channel
   :members: ChannelBase
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __weakref__, __dict__, __getitem__, seqParams

-----------------------------------------------

CDL Channel Model
-----------------
.. automodule:: neoradium.cdl
   :members: CdlChannel
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __weakref__, __dict__, __getitem__, seqParams

-----------------------------------------------

TDL Channel Model
-----------------
.. automodule:: neoradium.tdl
   :members: TdlChannel
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __weakref__, __dict__, __getitem__, seqParams
