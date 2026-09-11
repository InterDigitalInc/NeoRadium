Channel Models
==============
**NeoRadium** currently supports two stochastic channel models: Clustered Delay Line (CDL) and Tapped Delay
Line (TDL). These models are implemented as :py:class:`~neoradium.cdl.CdlChannel` and
:py:class:`~neoradium.tdl.TdlChannel` classes based on **3GPP TR 38.901**. Additionally, **NeoRadium** provides the
:py:class:`~neoradium.trjchan.TrjChannel` class, which is based on UE trajectories created using ray-tracing
scenarios. The deterministic aspects of the :py:class:`~neoradium.trjchan.TrjChannel` class are primarily derived
from **3GPP TR 38.901 Section 8.4**. Furthermore, you can derive your own customized channel models from the
:py:class:`~neoradium.channelmodel.ChannelModel` class, as explained below.


Channel Model Base Class
------------------------
.. automodule:: neoradium.channelmodel
   :members: ChannelModel
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __weakref__, __dict__, __getitem__

-----------------------------------------------

CDL Channel Model
-----------------
.. automodule:: neoradium.cdl
   :members: CdlChannel
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __weakref__, __dict__, __getitem__

-----------------------------------------------

TDL Channel Model
-----------------
.. automodule:: neoradium.tdl
   :members: TdlChannel
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __weakref__, __dict__, __getitem__

-----------------------------------------------

Trajectory Channel Model
------------------------
.. automodule:: neoradium.trjchan
   :members: TrjPoint, Trajectory, TrjChannel
   :member-order: bysource
   :special-members:
   :exclude-members: __init__, __repr__, __dict__, __getitem__, __weakref__
