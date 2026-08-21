# Copyright (c) 2024-2026, InterDigital AI Lab
"""
**NeoRadium** supports the antenna elements, panels, and arrays as defined in the 3-GPP standard **TR 38.901**. Using
this API, you can easily create antenna arrays and study their characteristics.

**Example**

.. code-block:: python
        
    elementTemplate = AntennaElement(beamWidth=[65,65], maxAttenuation=30)
    panelTemplate = AntennaPanel([4,4], elements=elementTemplate, polarization="+")
    antennaArray = AntennaArray([2,2], spacing=[3,3], panels=panelTemplate)
    antennaArray.showElements(zeroTicks=True)


.. figure:: ../Images/AntennaArray.png
   :align: center

.. code-block::
        
    antennaArray.drawRadiation(theta=90, radiationType="Directivity", normalize=False)


.. figure:: ../Images/AntennaArrayRad.png
   :align: center

This file contains the implementation of Antenna Elements, Panels, and Arrays.

.. Note:: **NeoRadium** distinguishes between physical antenna elements and logical CSI-RS ports. When using the
    :py:class:`AntennaPanel` class, physical and logical antennas are treated as identical (i.e., each physical 
    element corresponds directly to one logical transmit dimension and the internal mapping matrix ``B`` is the 
    identity). This is suitable for fully digital MIMO simulations. When using the :py:class:`AntennaArray` class, 
    the array consists of multiple physical panels, and each physical panel/polarization is mapped to one logical 
    CSI-RS port through an internal port-to-element mapping matrix ``B``. This provides a reduced-dimension logical 
    transmit space, similar to subarray or hybrid beamforming architectures. Users may optionally override the default
    mapping and provide a custom ``B`` matrix to model alternative beamforming or hardware implementations.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 05/18/2023    Shahab Hamidi-Rad       First version of the file.
# 05/01/2025    Shahab                  Updated the documentation and fixed some minor bugs.
# 07/11/2025    Shahab                  Added support for omnidirectional antenna.
# 03/12/2026    Shahab                  Changes in NeoRadium version 0.5.0:
#                                       * Added support for beam sweeping and probing.
#                                       * Default polModel for antenna elements is now 1. (A related bug was also fixed)
#                                       * Different methods of mapping antenna ports to physical ports for antenna
#                                         panels and antenna arrays.
#                                       * drawRadiation now receives "weights" for beamforming and ax to allow subplots.
#                                       * 'getElementsFields' and 'getRotationMatrix' now receive angles in degrees.
#                                       * New methods: numEl, __len__, numPanels, local2Global, global2Local,
#                                         applyRotation, getPortSteeringVector, getSweepingBeams, getProbingBeams.
# 06/11/2026    Shahab                  Changed the default 'shape' for AntennaPanel.__init__ from [2,2] to [1,1]
#                                       for consistency with AntennaArray.__init__.
# **********************************************************************************************************************
import numpy as np
import scipy.io

from .utils import freqStr, toLinear, toDb, herm, validateRange

# **********************************************************************************************************************
# This file is based on 3GPP TR 38.901
# Other good reads:
# https://scholar.valpo.edu/engineering_oer/1/  (the last few chapters)
# https://www.antenna-theory.com

# **********************************************************************************************************************
class AntennaBase:
    r"""
    This is the base class for all Antenna objects in **NeoRadium**. The classes :py:class:`AntennaElement`, 
    :py:class:`AntennaArray`, and :py:class:`AntennaPanel` are all derived from this class.
    """
    # ******************************************************************************************************************
    def __init__(self, **kwargs):
        self.isElement = isinstance(self, AntennaElement)

    # ******************************************************************************************************************
    def getMaxDim(self):                                            # Undocumented - Not intended for direct use
        # First get the difference between the two farthest elements. Then return the maximum value among all
        # dimensions. This is also known as "aperture length". (normalized aperture length: normalized by λ)
        if self.isElement:      return 0
        return (self.getElementPosition(-1) - self.getElementPosition(0)).max()

    # ******************************************************************************************************************
    def anglesToNumpy(self, angle, minAngle=None, maxAngle=None):   # Undocumented - Not intended for direct use
        # Converts/creates a NumPy array of angle values based on the given arguments
        if angle is None:               angle = np.arange(minAngle,maxAngle)
        if type(angle) == np.ndarray:   return angle
        if type(angle) == list:         return np.float64(angle)
        if type(angle) == tuple:
            if angle[0]==angle[1]: angle = (angle[0], angle[0]+1)
            return np.float64(range(*angle))
        return np.float64([angle])

    # ******************************************************************************************************************
    @property
    def numEl(self):    return self.getNumElements()
    def __len__(self):  return self.getNumElements()
    def getNumElements(self):
        # This function returns the number of antenna elements for classes derived from AntennaBase.
        # This is overridden in Panel and Array classes.
        return 1

    # ******************************************************************************************************************
    def getElementsDelays(self, theta, phi, frequency):             # Undocumented - Not intended for direct use
        # This function calculates the delays between different elements of a panel or array.
        # Currently, this function is not used.
        if self.isElement:  raise ValueError("'getElementsDelays' should not be called on 'AntennaElement' objects!")
        
        𝜃 = theta.reshape(-1,1) *np.pi/180
        𝜑 = phi.reshape(1,-1)   *np.pi/180

        # This is a 3 x numTheta x numPhi matrix
        xyzFactors = -np.float64([ np.sin(𝝷) * np.cos(𝞅),
                                   np.sin(𝝷) * np.sin(𝞅),
                                   np.cos(𝝷) * np.ones_like(𝞅) ])
        # This is a numElements x 3 matrix
        elementPositions = self.getAllPositions()

        # This is a numElements x numTheta x numPhi matrix giving the delay for each element at each theta/phi
        # combination
        delays = np.tensordot(elementPositions, xyzFactors, axes=1)/frequency
        return delays

    # ******************************************************************************************************************
    def getSteeringVector(self, theta, phi):
        r"""
        This method calculates the steering vector (also known as the Array Response) of an Antenna Array or Antenna 
        Panel for the given Azimuth and Zenith angles. Note that this function can only be called on the 
        :py:class:`AntennaPanel` and :py:class:`AntennaArray` classes. An exception is thrown if it is called on 
        :py:class:`AntennaElement` objects.
        
        .. Note:: This function returns the receiver steering vector. To use it for a transmitter, you need to use the 
            complex conjugate of the returned value.
        
        Parameters
        ----------
        theta : NumPy array
            A 1-D array of zenith angles in degrees. (between 0 and 180)
            
        phi : NumPy array
            A 1-D array of azimuth angles in degrees. (between -180 and 180)
            
        Returns
        -------
        NumPy array
            A 3-D complex NumPy array containing steering vectors for every combination of `theta` and `phi`. The 
            shape of the output is (numElements, numTheta, numPhi).
        """
        if self.isElement:  raise ValueError("'getSteeringVector' should not be called on 'AntennaElement' objects!")

        𝜃 = np.asarray(theta).reshape(-1,1) *np.pi/180
        𝜑 = np.asarray(phi).reshape(1,-1)   *np.pi/180

        xyzPhases = np.float64([ np.sin(𝝷) * np.cos(𝞅),
                                 np.sin(𝝷) * np.sin(𝞅),
                                 np.cos(𝝷) * np.ones_like(𝞅) ])                # Shape: 3 x numTheta x numPhi

        return np.exp(2j * np.pi *
                      np.tensordot(self.getAllPositions(), xyzPhases, axes=1)) # Shape: numElements x numTheta x numPhi

    # ******************************************************************************************************************
    def getFieldPattern(self, theta=None, phi=None):
        r"""
        This method is used to calculate the field patterns around an antenna panel or array in the directions given
        by the arguments ``theta`` and ``phi``.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles 
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles 
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the fields are calculated for all azimuth angles between -180 and 180 degrees.

        Returns
        -------
        NumPy array
            A 3-D complex NumPy array containing steering vectors for each combination of ``theta`` and ``phi``. The 
            shape of the output is (numElements, numTheta, numPhi)
        """
        # Only used to calculate directivity. We are interested in the power pattern, so we ignore polarization here.
        if self.isElement:  raise ValueError("'getFieldPattern' should not be called on 'AntennaElement' objects!")

        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)

        # We assume all elements have the same power pattern. They may have different polarized field patterns.
        elementField = self.getElement(0).getField(theta, phi)  # Field for the first element.  Shape: nTheta x nPhi
        
        steeringVector = self.getSteeringVector(theta, phi)                     # Shape: numElements x nTheta x nPhi
        nEl, nTheta, nPhi = steeringVector.shape

        # Field pattern per element for the whole array
        fieldPattern = (elementField.reshape((1,nTheta,nPhi)) * steeringVector) # Shape: numElements x nTheta x nPhi
        return fieldPattern

    # ******************************************************************************************************************
    def getPolarizedFields(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the polarized fields and outputs 2 matrices of the field values for vertical and 
        horizontal polarizations.
        
        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified zenith angle
            (in degrees)

            If this is None, the fields are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the fields are calculated for all azimuth angles between -180 and 180 degrees.

        weights : NumPy array
            A vector of weights to be applied to the field values. The weights can be used to steer the beams to the
            desired direction. If this is `None`, the field pattern is returned without any beamforming.

        Returns
        -------
        2 NumPy arrays
            * **arrayFieldV**:
                A NumPy array of shape (numTheta x numPhi) containing the field values with vertical 
                polarization at directions specified by ``theta`` and ``phi``.

            * **arrayFieldH**:
                A NumPy array of shape (numTheta x numPhi) containing the field values with horizontal
                polarization at directions specified by ``theta`` and ``phi``.
        """
        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)

        steeringVector = self.getSteeringVector(theta, phi)                         # Shape: numElements x nTheta x nPhi
        nEl, nTheta, nPhi = steeringVector.shape
        
        elementFieldV, elementFieldH = self.getElement(p=0).getPolarizedFields(theta, phi)
        elementFieldV = elementFieldV.reshape(nTheta,nPhi)                          # Shape: nTheta x nPhi
        elementFieldH = elementFieldH.reshape(nTheta,nPhi)                          # Shape: nTheta x nPhi
        if self.polarization in "+x":
            # The panel contains antennas with different polarizations. We need to get samples for both polarizations
            elementFieldVP2, elementFieldHP2 = self.getElement(p=1).getPolarizedFields(theta, phi)
            elementFieldVP2 = elementFieldVP2.reshape(nTheta,nPhi)                  # Shape: nTheta x nPhi
            elementFieldHP2 = elementFieldHP2.reshape(nTheta,nPhi)                  # Shape: nTheta x nPhi
            
            elementFieldV = np.array((nEl//2)*[elementFieldV] + (nEl//2)*[elementFieldVP2]) # Shape: nEl x nTheta x nPhi
            elementFieldH = np.array((nEl//2)*[elementFieldH] + (nEl//2)*[elementFieldHP2]) # Shape: nEl x nTheta x nPhi
        else:
            elementFieldV = np.array(nEl*[elementFieldV])       # Repeat nEl times.   Shape: nEl x nTheta x nPhi
            elementFieldH = np.array(nEl*[elementFieldH])       # Repeat nEl times.   Shape: nEl x nTheta x nPhi

        # Steered vertical and horizontal field patterns per element for the whole array
        elementsFieldV = elementFieldV * steeringVector         # Shape: nEl x nTheta x nPhi
        elementsFieldH = elementFieldH * steeringVector         # Shape: nEl x nTheta x nPhi

        if weights is not None:
            if len(weights)!=nEl:  raise ValueError( "'weights' must be a %d-dimensional vector!"%(nEl) )
            elementsFieldV *= weights[:,None,None]
            elementsFieldH *= weights[:,None,None]

        arrayFieldV = np.squeeze(elementsFieldV.sum(axis=0))    # Sum over the elements. Shape: nTheta x nPhi (squeezed)
        arrayFieldH = np.squeeze(elementsFieldH.sum(axis=0))    # Sum over the elements. Shape: nTheta x nPhi (squeezed)
        return arrayFieldV, arrayFieldH                         # Shapes: nTheta x nPhi (squeezed)

    # ******************************************************************************************************************
    def getField(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the fields in directions specified by ``theta`` and ``phi``. It calls the 
        :py:meth:`getPolarizedFields` method to get the vertical and horizontal polarized fields and combines them
        to get fields at the specified directions.

        .. math::

            F = \sqrt {F_v^2 + F_h^2}

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the fields are calculated for all azimuth angles between -180 and 180 degrees.

        weights : NumPy array
            A vector of weights to be applied to the field values. The weights can be used to steer the beams to the 
            desired direction. If this is `None`, the field pattern is returned without any beamforming.

        Returns
        -------
        NumPy array
            A NumPy array of shape (numTheta x numPhi) containing the field values at the directions specified by
            ``theta`` and ``phi``.
        """
        arrayFieldV, arrayFieldH = self.getPolarizedFields(theta, phi, weights) # Shapes: nTheta x nPhi (squeezed)
        return np.hypot(np.abs(arrayFieldV),np.abs(arrayFieldH))                # Shape: nTheta x nPhi (squeezed)
    
    # ******************************************************************************************************************
    def getPowerPattern(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the field power pattern in the directions specified by ``theta`` and ``phi``. It calls 
        the :py:meth:`getField` method to get the fields then calculates the field powers.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the 
            field powers.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the field power is calculated only for the single specified zenith angle
            (in degrees)

            If this is None, the field powers are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the 
            field powers.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the field power is calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the field powers are calculated for all azimuth angles between -180 and 180 degrees.

        weights : NumPy array
            A vector of weights to be applied to the field values. The weights can be used to steer the beams to the 
            desired direction. If this is `None`, the field pattern is returned without any beamforming.

        Returns
        -------
        NumPy array
            A NumPy array of shape (numElements x numTheta x numPhi) containing the field powers at the directions
            specified by ``theta`` and ``phi``.
        """
        arrayField = self.getField(theta, phi, weights)         # Shape: nTheta x nPhi (squeezed)
        return np.square(np.abs(arrayField))                    # Shape: nTheta x nPhi (squeezed)

    # ******************************************************************************************************************
    def getPowerPatternDb(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the field power pattern (in dB) in the
        directions specified by ``theta`` and ``phi``. It calls the
        :py:meth:`getPowerPattern` method to get the field powers and
        then converts them to dB.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scalar value, the field power is calculated only
            for the single specified zenith angle (in degrees)

            If this is None, the field powers are calculated for all zenith
            angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scalar value, the field power is calculated only
            for the single specified azimuth angle (in degrees)

            If this is None, the field powers are calculated for all azimuth
            angles between -180 and 180 degrees.

        weights : NumPy array
            A vector of weights to be applied to the field values. The weights
            can be used to steer the beams to the desired direction. If this is
            `None`, the field pattern is returned without any beamforming.

        Returns
        -------
        NumPy array
            A NumPy array of shape (numElements x numTheta x numPhi) containing
            the field powers in dB at the directions specified by ``theta`` and ``phi``.
        """
        power = self.getPowerPattern(theta, phi, weights)                   # Shape: nTheta x nPhi (squeezed)
        power = np.maximum(1e-12, power)    # Make sure no zeros in power
        return toDb(power)                  # Return the power in dB          Shape: nTheta x nPhi (squeezed)

    # ******************************************************************************************************************
    def getIntegralAngleStep(self):                                 # Undocumented - Not intended for direct use
        # This function returns the angle step for the integral used to calculate the
        # directivity. See the getDirectivity function below.
        maxSpan = self.getMaxDim()  # Get the farthest distance between antenna elements
        if maxSpan == 0:    return 1
        
        # Using the approximation: beamWidth = 70 * wavelength / D
        # The maxSpan above is: D/wavelength, so:
        beamWidth = 70/maxSpan
        angleStep = beamWidth               # Make sure we have at least three angle steps per beamWidth
        
        # Pick one of 1, 0.5, 0.2, or 0.1 for the step. This makes it easier to handle the range of angles.
        if angleStep>=1:     return 1
        if angleStep>=0.5:   return 0.5
        if angleStep>=0.2:   return 0.2
        return 0.1

    # ******************************************************************************************************************
    def getDirectivity(self, theta=None, phi=None, weights=None):
        r"""
        Directivity at a specific direction is defined as:
        
        .. math::

            D = \frac {P} {P_{avg}}

        where :math:`P` is the power radiated at the specified angle and :math:`P_{avg}` is the average power 
        radiated in all directions. The average power is calculated by integrating the field values at all
        angles (See `this web page <https://www.antenna-theory.com/basics/directivity.php>`_ for more details):
        
        .. math::

            P_{avg} = \frac {1} {4 \pi} \int_0^{2 \pi} \int_0^{\pi} |F(\theta, \phi)|^2 \sin \theta d\theta d\phi

        
        Directivity (without any specific direction) is defined as:
        
        .. math::

            D_{max} = \frac {P_{max}} {P_{avg}}
            
        where :math:`P_{max}` is the maximum power radiated at a direction. Directivity is usually measured in dBi 
        which is the relative directivity in dB with respect to an "isotropic" radiator.
                
        This method calculates the directivity (in dbi) at directions specified by ``theta`` and ``phi``.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the
            directivity.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the directivity is calculated only for the single specified zenith angle
            (in degrees)

            If this is None, the directivity is calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the
            directivity.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the directivity is calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the directivity is calculated for all azimuth angles between -180 and 180 degrees.

        weights : NumPy array
            A vector of weights to be applied to the field values. The weights can be used to steer the beams to the
            desired direction. If this is `None`, the field pattern is returned without any beamforming.

        Returns
        -------
        NumPy array
            A NumPy array of shape (numElements x numTheta x numPhi) containing the directivity in dBi at the 
            directions specified by ``theta`` and ``phi``.
        """
        # Directivity:
        #   AKA Directive Gain
        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)

        elementsField = self.getFieldPattern(theta, phi)   # Fields for each element. Shape: nEl x nTheta x nPhi

        # Now we calculate "Directivity" based on the formula in:
        #       https://www.antenna-theory.com/basics/directivity.php
        # We first need to calculate the average power in all directions which is the denominator integral
        # in the directivity formula.
        angleStep = self.getIntegralAngleStep()
        allTheta = np.arange(0, 180+angleStep, angleStep)
        allPhi = np.arange(-180, 180+angleStep, angleStep)
        
        if (allTheta.shape != theta.shape) or (allPhi.shape != phi.shape):
            elementsFieldAllD = self.getFieldPattern(allTheta, allPhi)
        elif np.any(allTheta!=theta) or np.any(allPhi!=phi):
            elementsFieldAllD = self.getFieldPattern(allTheta, allPhi)
        else:
            elementsFieldAllD = elementsField               # elementsField already calculated

        # Shape of elementsFieldAllD: nEl x nAllTheta x nAllPhi
        n = elementsFieldAllD.shape[0]

        # Now calculating steering vector covariance matrix. Shape: nEl x nEl
        svCov = (elementsFieldAllD * np.sin(allTheta*np.pi/180)[None,:,None]).reshape(n,-1).dot(herm(elementsFieldAllD.reshape(n,-1)))
        
        dTheta = dPhi = angleStep*np.pi/180
        if weights is not None:
            w = weights.reshape(1, n)
            integral = (w.dot(svCov).dot(w.T)*dTheta*dPhi)[0,0].real
            elementsField *= weights.reshape((-1,1,1))
        else:
            integral = svCov.real.sum()*dTheta*dPhi

        arrayField = elementsField.sum(axis=0)                      # Shape: nTheta x nPhi
        arrayPower = np.squeeze(np.square(np.abs(arrayField)))      # Shape: nTheta x nPhi  (squeezed)
        
        # Note that since totalPower is not normalized we have it in the numerator of directivity formula instead of 1
        directivity = 4*np.pi*arrayPower/integral
        directivity = np.maximum(1e-12, directivity)    # Make sure no zeros in directivity so the log below works.
        directivityDbi = toDb(directivity)              # Convert to "dBi" (dB with respect to an isotropic radiator)
        return directivityDbi                           # Shape: nTheta x nPhi  (squeezed)

    # ******************************************************************************************************************
    def drawRadiation(self, theta=None, phi=None, radiationType="Directivity", normalize=True, title=None,
                      viewAngles=(45,20), figSize=6.0, ax=None, weights=None):
        r"""
        This is a multi-purpose visualization function that shows the radiation around antenna elements, panels, and
        arrays in the directions specified by ``theta`` and ``phi``.
        
        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to visualize the 
            radiations.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the radiations are visualized only for the single specified zenith angle
            (in degrees)

            If this is None, the radiations are visualized for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to visualize the
            radiations.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the radiations are visualized only for the single specified azimuth angle
            (in degrees)

            If this is None, the radiations are visualized for all azimuth angles between -180 and 180 degrees.

        radiationType : str
            This parameter specifies the type of radiation to plot. Here is a list of supported values:
                
                * **Directivity** (default)
                * **Power**
                * **PowerDb**
                * **Field**
            
        normalize : bool
            If `True` (default) all the values are normalized before being plotted.

        title  str
            The title to be used for the plot. If not specified, then this function creates a title based on the
            given parameters.
            
        viewAngles : tuple
            For 3-D plots, you can use this parameter to specify your desired viewing angle. For non-3D plots, this
            parameter is ignored. The default is ``(45,20)``.
            
        figSize : float
            The figure size. Use this to control size of the plot. The default is 6.0.
            
        ax : `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_ or None
            If specified, it must be a matplotlib ``Axis`` object on which the radiation pattern is drawn. This can 
            be used to create a group of matplotlib subplots and draw the radiation pattern in one of the subplots. For 
            example:
            
            .. code-block:: python
            
                panel = AntennaPanel([4,4], polarization="x")
                fig, ax = plt.subplots(1,2, layout='constrained', subplot_kw={'projection': 'polar'})
                panel.drawRadiation(theta=90, radiationType="Field", normalize=False, title="Horizontal cut", ax=ax[0])
                panel.drawRadiation(phi=0, radiationType="Field", normalize=False, title="Vertical cut", ax=ax[1])
        
        weights : NumPy array
            A vector of weights to be applied to the radiation pattern. The weights can be used to steer the beams to 
            the desired direction. If this is `None`, beamforming is disabled.

        Returns
        -------
        `matplotlib.axes.Axes <https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.html#matplotlib.axes.Axes>`_
            The matplotlib ``Axis`` object used to draw the radiation pattern.


        **Plot Types:**
        
            :Horizontal Cut at specified elevation: For this case, specify one ``theta`` value and include all 
                azimuth angles (:math:`-\pi < \phi < \pi`). One common use case is the horizontal cut at zero 
                elevation (:math:`\theta = \pi / 2`).
            :Vertical Cut at specified azimuth: For this case, specify one ``phi`` value and include all zenith
                angles (:math:`0 < \theta < \pi`). One common use case is the vertical cut at zero 
                azimuth (:math:`\phi = 0`).
            :3-D pattern: For this case, specify the complete range for both ``theta`` and ``phi`` 
                (:math:`0 < \theta < \pi` and :math:`-\pi < \phi < \pi`). This is the default case if both ``theta``
                and ``phi`` are not specified.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)
                
        if radiationType=="Directivity":
            radValues = self.getDirectivity(theta, phi, weights)
        elif radiationType=="PowerDb":
            radValues = self.getPowerPatternDb(theta, phi, weights)
            if normalize:   radValues -= radValues.max()
        elif radiationType=="Power":
            radValues = self.getPowerPattern(theta, phi, weights)
            if normalize:   radValues /= radValues.max()
        elif radiationType=="Field":
            radValues = self.getField(theta, phi, weights)
            radValues = np.abs(radValues)   # We want to draw the magnitude of field
            if normalize:   radValues /= radValues.max()+1e-12
        else:
            raise ValueError( "Unsupported 'radiationType' value \"%s\"!"%(radiationType) )

        # For logarithmic values, limit the range; otherwise, the plot looks weird! In this case we push all small
        # values to the center for the 2-D polar graphs.
        radRange = radValues.max() - radValues.min()
        if radiationType in ["Directivity", "PowerDb"]:
            plotValues = np.maximum(radValues, radValues.max()-60)
            plotRange = plotValues.max() - plotValues.min()
            plotMin = plotValues.min() if radRange > 60 else (plotValues.min()-plotRange/20)
            plotMax = plotValues.max()
        else:
            plotMin, plotMax = 0, radValues.max()
            plotValues = radValues
            plotRange = radRange

        # Make sure plotMin and plotMax are not the same
        if plotMax==0 and plotMin==0:   plotMin, plotMax = -1, 0.25
        elif plotMax == plotMin:        plotMin, plotMax = plotMin-np.abs(plotMin)/4, plotMax+np.abs(plotMax)/8
        elif radRange==0:               plotMax = plotMax+np.abs(plotMax)/8

        if title is None:
            radTypeStr = {"Directivity":"Directivity", "PowerDb":"Radiation Power (dB)",
                          "Power":"Radiation Power",   "Field": "Electric Field"}[radiationType]
            if normalize and (radiationType!="Directivity"):    radTypeStr = "Normalized "+radTypeStr
            if len(theta)==1:
                if theta[0]==90:    title = f"Horizontal Cut of {radTypeStr} at zero elevation ($\\theta=\\pi/2$)"
                else:               title = f"Horizontal Cut of {radTypeStr} at $\\theta$={int(theta[0])}°"

            elif len(phi)==1:
                if phi[0]==0:       title = f"Vertical Cut of {radTypeStr} at zero azimuth ($\\phi=0$)"
                else:               title = f"Vertical Cut of {radTypeStr} at $\\phi$={int(phi[0])}°"

            else:
                if max(theta)>=179 and min(theta)==0 and max(phi)>=179 and min(phi)==-180: title = radTypeStr
                else: title = f"{radTypeStr} for {theta[0]}°$\\leq\\theta\\leq${theta[-1]}° and {phi[0]}°$\\leq\\phi\\leq${phi[-1]}°"
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(figSize,figSize), layout='constrained',
                                   subplot_kw={'projection': 'polar' if len(theta)==1 or len(phi)==1 else '3d'})

        if len(theta)==1:
            ax.plot(phi*np.pi/180, plotValues)
            ax.set_ylim(plotMin, plotMax)
#            if radiationType in ["Directivity", "PowerDb"]: ax.set_ylim(plotValues.min()-plotRange/20,plotValues.max())
#            else:                                           ax.set_ylim(0, plotValues.max())
            ax.set_title(title, size=16)
            return ax

        if len(phi)==1:
            ax.plot(theta*np.pi/180, plotValues)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            ax.set_ylim(plotMin,plotValues.max())
#            if radiationType in ["Directivity", "PowerDb"]: ax.set_ylim(plotValues.min()-plotRange/20,plotValues.max())
#            else:                                           ax.set_ylim(0, plotValues.max())
            ax.set_title(title, size=16)
            return ax

        # Now doing surface plot
        if type(viewAngles)!=tuple: raise ValueError( "'viewAngles' must be a tuple!" )
        ax.view_init(elev=viewAngles[1], azim=viewAngles[0])

        # For logarithmic values, we shift values so that the radius of minimum value in polar coordinates is 10%
        # of the range of all values.
        if radiationType in ["Directivity", "PowerDb"]:
            plotValues = (plotValues - plotValues.min() + 0.1*(plotValues.max()-plotValues.min()) )
        
        𝞅, 𝝷 = np.pi*phi/180, np.pi*theta/180
        surface = np.float64([plotValues * (np.sin(𝝷).reshape(-1,1) * np.cos(𝞅).reshape(1,-1)),
                              plotValues * (np.sin(𝝷).reshape(-1,1) * np.sin(𝞅).reshape(1,-1)),
                              plotValues * np.cos(𝝷).reshape(-1,1) ])

        r = np.square(surface).sum(0)
        r /= r.max()
        minMins, maxMaxs = surface.min(), (surface.max()+1)
        surface = ax.plot_surface(surface[0,:,:], surface[1,:,:], surface[2,:,:],
                          facecolors=cm.winter(r), alpha=0.8,
                          rstride=1, cstride=1, linewidth=0, antialiased=False)

        # Draw our own axes:
        ax.axis('off')
        qX, qY, qZ = np.zeros((3,3))
        qU, qV, qW = maxMaxs * np.eye(3)
        ax.quiver(qX, qY, qZ, 1.3*qU, qV, qW, arrow_length_ratio=0.1, color='black', linewidth=2)
        ax.text(1.4*maxMaxs, 0, 0, "X", color='black')
        ax.text(0, 1.1*maxMaxs, 0, "Y", color='black')
        ax.text(0, 0, 1.1*maxMaxs, "Z", color='black')

        # Draw the elevation/Azimuth small axis on the X-axis
        qElAzX, qElAzY, qElAzZ = np.float64([[1.1*maxMaxs,1.1*maxMaxs], [0,0], [0,0]])
        qElAzU, qElAzV, qElAzW = np.float64([[0,0], [maxMaxs/5,0], [0,-maxMaxs/5]])
        ax.quiver(qElAzX, qElAzY, qElAzZ, qElAzU, qElAzV, qElAzW, arrow_length_ratio=0.2, color='black')
        ax.text(qElAzX[0], 0, -maxMaxs/5, "$\\theta$", color='black')
        ax.text(qElAzX[0], maxMaxs/5, 0, "$\\phi$", color='black')

        # Force same axis scales for X, Y, and Z:
        ax.set_xlim(minMins,maxMaxs)
        ax.set_ylim(minMins,maxMaxs)
        ax.set_zlim(minMins,maxMaxs)
        ax.set_title(title, size=16)
        return ax

    # ******************************************************************************************************************
    @classmethod
    def getRotationMatrix(cls, orientation):
        r"""
        This class method calculates and returns the forward composite rotation matrix used to convert coordinates from 
        the local to the global system. It is important to note that since the rotation matrix is orthogonal, its 
        inverse matrix is the same as its transpose, which can be used to convert from the global to the local 
        coordinate system. For more information, please refer to **3GPP TR 38.901 equation (7.1-4)**.
                
        Parameters
        ----------
        orientation : list or NumPy array
            A list or NumPy array containing the orientation angles :math:`\alpha` (bearing angle), :math:`\beta` 
            (downtilt angle), and :math:`\gamma` (slant angle) in degrees.

        Returns
        -------
        NumPy array
            A 3x3 rotation matrix that is used to transform the local coordinates to global coordinates.
        """
        # Note: The input units of this function was changed from radians to degrees starting in Neoradium version 0.5
        # Important: This rotation matrix converts from local to global if applied from left.
        if not np.any(orientation): return np.eye(3)            # If all zeros, return Identity
        sinAlpha, sinBeta, sinGamma = np.sin(np.deg2rad(orientation))
        cosAlpha, cosBeta, cosGamma = np.cos(np.deg2rad(orientation))
        # See TR38.901 - Eq. 7.1-4
        return np.float64(
        [[ cosAlpha*cosBeta, cosAlpha*sinBeta*sinGamma-sinAlpha*cosGamma, cosAlpha*sinBeta*cosGamma+sinAlpha*sinGamma ],
         [ sinAlpha*cosBeta, sinAlpha*sinBeta*sinGamma+cosAlpha*cosGamma, sinAlpha*sinBeta*cosGamma-cosAlpha*sinGamma ],
         [ -sinBeta,         cosBeta*sinGamma,                            cosBeta*cosGamma]])
        
    # ******************************************************************************************************************
    @classmethod
    def applyRotation(cls, theta, phi, r):                          # Undocumented - Not intended for direct use
        # Applies the rotation matrix r to the angles theta and phi. It can be used for local->global or vice versa.
        𝜃i = np.deg2rad(theta)                         # Any dimensions
        𝜑i = np.deg2rad(phi)                           # Any dimensions
        ui = np.stack([np.sin(𝜃i) * np.cos(𝜑i),        # ... x 3
                       np.sin(𝜃i) * np.sin(𝜑i),
                       np.cos(𝜃i) ], axis=-1)
        # Note: in 3GPP TR 38.901 equation (7.1-4), the rotation matrix is supposed to be applied from left. Since
        #       we are applying it from right, we transpose the rotation matrix to get the same results.
        uo = ui @ r.T                                  # ... x 3
        𝜃o = np.rad2deg(np.arccos(np.clip(uo[...,2], -1.0, 1.0))) # Same dimensions as theta
        𝜑o = np.rad2deg(np.arctan2(uo[...,1], uo[...,0]))         # Same dimensions as phi
        # Phi does not make sense when theta is 0 or 𝛑
        if 𝜑o.shape != ():
            𝜑o[𝜃o==0] = 0
            𝜑o[𝜃o==180] = 0
        elif 𝜃o==0 or 𝜃o==180:
            𝜑o = 0
        return 𝜃o, 𝜑o

    # ******************************************************************************************************************
    @classmethod
    def local2Global(cls, theta, phi, orientation):
        r"""
        This class method converts a set of local angles to their corresponding global angles.
                
        Parameters
        ----------
        theta : NumPy array
            A NumPy array containing the zenith angles (in degrees) in the local coordinate system.

        phi : NumPy array
            A NumPy array containing the azimuth angles (in degrees) in the local coordinate system.

        orientation : list or NumPy array
            A list or NumPy array containing the orientation angles :math:`\alpha` (bearing angle), :math:`\beta` 
            (downtilt angle), and :math:`\gamma` (slant angle) in degrees.

        Returns
        -------
        NumPy arrays
            A tuple of ``(thetaGlobal, phiGlobal)`` containing zenith and azimuth in the global coordinate system. 
            ``thetaGlobal`` and ``phiGlobal`` are the same shape as ``theta`` and ``phi``.
        """
        return cls.applyRotation(theta, phi, cls.getRotationMatrix(orientation))

    # ******************************************************************************************************************
    @classmethod
    def global2Local(cls, theta, phi, orientation):
        r"""
        This class method converts a set of global angles to their corresponding local angles.
                
        Parameters
        ----------
        theta : NumPy array
            A NumPy array containing the zenith angles (in degrees) in the global coordinate system.

        phi : NumPy array
            A NumPy array containing the azimuth angles (in degrees) in the global coordinate system.

        orientation : list or NumPy array
            A list or NumPy array containing the orientation angles :math:`\alpha` (bearing angle), :math:`\beta` 
            (downtilt angle), and :math:`\gamma` (slant angle) in degrees.

        Returns
        -------
        NumPy arrays
            A tuple of ``(thetaLocal, phiLocal)`` containing zenith and azimuth in the local coordinate system. 
            ``thetaLocal`` and ``phiLocal`` are the same shape as ``theta`` and ``phi``.
        """
        return cls.applyRotation(theta, phi, cls.getRotationMatrix(orientation).T)
        
    # ******************************************************************************************************************
    def getElementsFields(self, theta, phi, orientation=np.float64([0,0,0])):
        r"""
        This method returns the electric fields used to calculate the channel response for different channel models.
        It returns polarized field values in the directions specified by the ``theta`` and ``phi``. This function also
        handles the conversion from local to global coordinates using the rotation angles provided in ``orientation``.
        Please refer to **3GPP TR 38.901 sections 7.1 and 7.5** for more details.

        Parameters
        ----------
        theta : NumPy array
            A 2-D NumPy array containing the zenith angles (in degrees) used to calculate the fields. This is an 
            ``n`` by ``m`` matrix where ``n`` is the number of clusters and ``m`` is the number of rays per cluster.

        phi : NumPy array
            A 2-D NumPy array containing the azimuth angles (in degrees) used to calculate the fields. This is an 
            ``n`` by ``m`` matrix where ``n`` is the number of clusters and ``m`` is the number of rays per cluster.

        orientation : list or NumPy array
            A list or NumPy array containing the orientation angles :math:`\alpha` (bearing angle), :math:`\beta` 
            (downtilt angle), and :math:`\gamma` (slant angle) in degrees.

        Returns
        -------
        2 NumPy arrays
            * **field**:
                A NumPy array of shape (n x m x numAntenna x 2) containing the field information for each antenna 
                element and each one of ``m`` rays in each one of ``n`` clusters. The second dimension (2) is used 
                to separate the vertical and horizontal polarization.

            * **locFactor**:
                A NumPy array of shape (n x m x numAntenna) containing the location factor. For more information 
                please refer to **3GPP TR 38.901 equations 7.5-28 and 7.5-29**.
        """
        # Note: The input units of this function was changed from radians to degrees starting in Neoradium version 0.5
        # This is called by the channel models. theta and phi are n x m matrices of azimuth and zenith angles of
        # arrival (Rx Antenna) or departure (Tx Antenna), where n is the number of clusters and m is the number
        # of rays per cluster.
        𝜃G, 𝜑G = np.deg2rad(theta), np.deg2rad(phi)                                          # Shape: ...
        𝜃L, 𝜑L = np.deg2rad(self.global2Local(theta, phi, orientation))                      # Shape: ...

        # The spherical unit vectors of the GCS. See TR38.901 - Eq. 7.1-13 and 7.1-14
        𝜃HatG = np.stack([ np.cos(𝜃G) * np.cos(𝜑G), np.cos(𝜃G) * np.sin(𝜑G), -np.sin(𝜃G) ], axis=-1)  # ... x 3
        𝜑HatG = np.stack([ -np.sin(𝜑G), np.cos(𝜑G), np.zeros_like(𝜑G) ], axis=-1)                    # ... x 3

        # The spherical unit vectors of the LCS. See TR38.901 - Eq. 7.1-13 (Applied to local phi and theta)
        𝜃HatL = np.stack([ np.cos(𝜃L) * np.cos(𝜑L), np.cos(𝜃L) * np.sin(𝜑L), -np.sin(𝜃L) ], axis=-1)  # ... x 3

        r = self.getRotationMatrix(orientation)
        psi = np.arctan2( ((𝜑HatG @ r)*𝜃HatL).sum(-1), ((𝜃HatG @ r)*𝜃HatL).sum(-1) )    # TR38.901 - Eq. 7.1-12
        
        # Polarized local fields for all elements. fieldPairs is a list of tuples (fTheta, fPhi)
        fieldPairs = [ e.getPolarizedFields(np.rad2deg(𝜃L), np.rad2deg(𝜑L)) for e in self.allElements() ]
        f𝜃L, f𝜑L = (np.stack(x,axis=-1) for x in zip(*fieldPairs))            # Shapes: ..., numElements
        f𝜃G = np.cos(psi)[...,None]*f𝜃L - np.sin(psi)[...,None]*f𝜑L           # Shape: ..., numElements
        f𝜑G = np.sin(psi)[...,None]*f𝜃L + np.cos(psi)[...,None]*f𝜑L           # Shape: ..., numElements
        fields = np.stack([f𝜃G, f𝜑G], axis=-1)                                # Shape: ..., numElements, 2
        
        # The spherical unit vector at theta,phi (See TR38.901 - Eq. 7.5-23 and 7.5-24) - using the local values
        rHatL = np.stack([ np.sin(𝜃L) * np.cos(𝜑L), np.sin(𝜃L) * np.sin(𝜑L), np.cos(𝜃L) ], axis=-1)  # ... x 3
        posL = self.getAllPositions()      # Local antenna positions.                                 numAntenna x 3
        locAngle = 2*np.pi*(rHatL @ posL.T)                                                     # ... x numAntenna
        locFactor = np.exp(1j * locAngle)                                                       # ... x numAntenna

        return fields, locFactor                    # Shapes: (..., numElements, 2) and (..., numElements)
    
    # ******************************************************************************************************************
    def getPortSteeringVector(self, theta, phi):                    # Undocumented - Not intended for direct use
        if isinstance(self, AntennaPanel):
            # Note: getSteeringVector returns the receiver steering vector. For transmitter
            # we need to use the complex conjugate of that.
            return self.getSteeringVector(theta,phi).conj()                     # numElements x numTheta x numPhi
        
        assert isinstance(self, AntennaArray)
        # This is an Antenna array. Each antenna port corresponds to a panel. For the dual polarization case, each
        # CSI-RS port corresponds to the corresponding polarization of a panel. (numPorts = 2*numPanels)
        # Get steering vector based on centers of the panels
        𝜃 = np.asarray(theta).reshape(-1,1) *np.pi/180
        𝜑 = np.asarray(phi).reshape(1,-1)   *np.pi/180
        xyzPhases = np.float64([ np.sin(𝝷) * np.cos(𝞅),
                                 np.sin(𝝷) * np.sin(𝞅),
                                 np.cos(𝝷) * np.ones_like(𝞅) ])                 # Shape: 3 x numTheta x numPhi

        # positions of centers of panels
        panelPositions = np.float64([p.position for p in self.allPanels()])     # Shape: numPanels x 3
        # Note: We are using -2j𝜋 (instead of +2j𝜋) because we want transmitter steering vectors.
        w = np.exp(-2j*np.pi * np.tensordot(panelPositions, xyzPhases, axes=1)) # Shape: numPanels x numTheta x numPhi
        if self.polarization in "|-":   return w                                # Single polarization
        
        # Dual polarization -> Duplicate steering vectors for the 2nd polarization:
        return np.vstack([w,w])                                                 # Shape: numPorts x numTheta x numPhi
        
    # ******************************************************************************************************************
    def getSweepingBeams(self, numTheta, numPhi, thetaSpan=20, phiSpan=120,
                         angleMethod="sincos", polStrategy="equal"):
        r"""
        Generate a beam-sweeping grid of steering vectors.

        This method constructs a grid of steering directions in the antenna’s **local**
        coordinate system and returns one steering vector per direction.
        For dual-polarized antenna panels/arrays, the steering vectors can either excite both 
        polarizations equally ("equal") or generate two sets of beams that probe the two
        polarizations independently ("probe").

        Parameters
        ----------
        numTheta : int
            Number of zenith (theta) directions in the sweep.
            If the panel/array has a single vertical element/panel (nV == 1), this is forced to 1.
        numPhi : int
            Number of azimuth (phi) directions in the sweep.
            If the panel/array has a single horizontal element/panel (nH == 1), this is forced to 1.
        thetaSpan  float, optional
            Total sweep span in zenith (degrees) centered around 90° (broadside).
            Used when numTheta > 1. Default is 20.
        phiSpan : float, optional
            Total sweep span in azimuth (degrees) centered around 0° (broadside).
            Used when numPhi > 1. Default is 120.
        angleMethod : {"sincos", "linear"}, optional
            Angle grid generation method:
            
                :"sincos": uniform sampling in sin(phi) and cos(theta) domains (uniform in spatial-frequency space)
                :"linear": uniform sampling directly in degrees
                
            Default is "sincos".
        polStrategy : {"equal", "probe"}, optional
            Polarization strategy for dual-polarized antenna:

                :"equal": each beam excites both polarizations.
                :"probe": returns two beam sets; the first uses only the first polarization and the second uses only 
                    the second polarization (doubling the number of beams)

            For single-polarized panels/arrays, this parameter is ignored. Default is "equal".

        Returns
        -------
        steeringVectors : NumPy array
            Complex steering vector matrix of shape ``(numPorts, numBeams)``, where ``numPorts`` is the number
            of antenna ports and ``numBeams`` is the number of sweeping beams. When `polStrategy="probe"`
            on a dual-pol panel, ``numBeams`` is doubled.
        beams : list
            Beam metadata as ``[thetas, phis, pols]``, where ``thetas`` and ``phis`` have length ``numBeams`` and
            ``pols`` is a string of length ``numBeams`` describing the polarization label per beam.

        Notes
        -----
        - All angles are in **local** panel coordinates. Use :py:meth:`AntennaBase.local2Global`
          if you need global angles for labeling/plotting.
        - Broadside is assumed at theta=90° and phi=0° in the local coordinate system.
        - For :py:class:`AntennaPanel`, ``numPorts`` is equal to number of antenna elements. For
          :py:class:`AntennaArray`, ``numPorts`` is equal to number of panels for single-polarization and 2 times
          number of panels for dual polarization.
        - The returned columns are **unnormalized steering vectors**, not power-normalized precoders. For
          single-polarized panels/arrays and for dual-polarized cases with ``polStrategy="equal"``, each column
          has :math:`\|w\|^2 = N_{ports}`. For ``polStrategy="probe"`` (dual-polarized only), one polarization
          is zeroed per column, so :math:`\|w\|^2 = N_{ports}/2` — a 3 dB power drop relative to the ``"equal"``
          mode. If you use these as transmit precoders, apply your own normalization to control total transmit
          power, and account for the 3 dB gap when comparing beam metrics across ``polStrategy`` modes.
        - ``angleMethod="sincos"`` samples uniformly in sin(φ) and cos(θ), which gives roughly equal angular
          separation in beam space — the same intuition behind a DFT-based beam codebook and the natural
          choice when the goal is uniform beam coverage. ``"linear"`` samples uniformly in degrees and is
          provided for the (less common) case where uniformly spaced angles are preferred over uniform beam
          spacing.
        - On a dual-polarized antenna, ``polStrategy="probe"`` returns ``2 * numTheta * numPhi`` steering-vector
          columns: the first half excites only the first polarization, the second half only the second.
          This doubles the number of beams compared to ``polStrategy="equal"``.

        Refer to the notebook :doc:`../Playground/Notebooks/Antenna/BeamSweeping` for examples of using this
        function. See also :py:meth:`getProbingBeams` for refining a beam around a known direction.
        """
        if self.isElement:  raise ValueError("'getSweepingBeams' should not be called on 'AntennaElement' objects!")
        if numTheta < 1 or numPhi < 1:
            raise ValueError(f"'numTheta' and 'numPhi' must both be >= 1 (got {numTheta} and {numPhi}).")
        # NOTE: Everything here uses local coordinates. Use local2Global() to convert for labeling/plotting.
        # In theory, this can be called on a 1x1 panel to sweep 2 beams based on polarization only.
        nV, nH = self.shape
        if nH == 1:     numPhi=1    # If this is a vertical linear panel/array, force azimuth sweep to 1 angle (0°).
        if numPhi == 1:
            phiAngles = np.array([0])
        elif angleMethod.lower() == "sincos":
            phiMin, phiMax = -phiSpan/2, phiSpan/2
            phiSins = np.linspace(np.sin(np.deg2rad(phiMin)), np.sin(np.deg2rad(phiMax)), numPhi)
            phiAngles = np.rad2deg(np.arcsin(np.clip(phiSins, -1, 1)))
        elif angleMethod.lower() == "linear":
            phiMin, phiMax = -phiSpan/2, phiSpan/2
            phiAngles = np.linspace(phiMin, phiMax, numPhi)
        else:
            raise ValueError(f"Invalid 'angleMethod' '{angleMethod}'! It must be 'sincos' or 'linear'.")

        if nV == 1:     numTheta=1  # If this is a horizontal linear panel/array, force zenith sweep to 1 angle (90°)
        if numTheta == 1:
            thetaAngles = np.array([90])
        elif angleMethod.lower() == "sincos":
            thetaMin, thetaMax = max(90-thetaSpan/2,0), min(90+thetaSpan/2,180)
            thetaCos = np.linspace(np.cos(np.deg2rad(thetaMin)), np.cos(np.deg2rad(thetaMax)), numTheta)
            thetaAngles = np.rad2deg(np.arccos(np.clip(thetaCos, -1, 1)))
        elif angleMethod.lower() == "linear":
            thetaMin, thetaMax = max(90-thetaSpan/2,0), min(90+thetaSpan/2,180)
            thetaAngles = np.linspace(thetaMin, thetaMax, numTheta)
        else:
            raise ValueError(f"Invalid 'angleMethod' '{angleMethod}'! It must be 'sincos' or 'linear'.")

        steeringVectors = self.getPortSteeringVector(thetaAngles,phiAngles)     # numPorts x numTheta x numPhi

        numPos = nV*nH                                  # Number of positions (AntennaElement or AntennaPanel objects)
        numBeams = numTheta*numPhi                      # Total number of beams
        
        spatialBeams = np.array(np.meshgrid(thetaAngles, phiAngles)).T.reshape(-1, 2)  # numBeams x 2
        beams = [ spatialBeams[:,0],                    # Zenith angles (theta)  Shape: (numBeams,)
                  spatialBeams[:,1],                    # Azimuth angles (phi)   Shape: (numBeams,)
                  numBeams*self.polarization]           # Polarizations ('|', '-', '+', or 'x')  (str)

        if self.polarization in ['|','-']:
            steeringVectors = steeringVectors.reshape(numPos,-1)                    # numPos x numBeams
        else:
            # In this case numPorts = 2*numPos
            steeringVectors = steeringVectors.reshape(2*numPos,-1)                  # 2*numPos x numBeams

            if polStrategy.lower()=="probe":
                steeringVectors = np.hstack([steeringVectors,steeringVectors])      # 2*numPos x 2*numBeams
                steeringVectors[numPos:,:numBeams] = 0      # 1st set: Turn off 2nd Polarization
                steeringVectors[:numPos,numBeams:] = 0      # 2nd set: Turn off 1st Polarization

                spatialBeams = np.vstack([spatialBeams,spatialBeams])               # 2*numBeams x 2
                angleToChar = {-45:'\\', 0:'-', 45:'/', 90:'|'}
                p0 = angleToChar[self.getElement(p=0).polAngle]                     # First polarization '/' or '-'
                p1 = angleToChar[self.getElement(p=1).polAngle]                     # Second polarization '\' or '|'
                beams = [ spatialBeams[:,0],                        # Zenith angles (theta)  Shape: (2*numBeams,)
                          spatialBeams[:,1],                        # Azimuth angles (phi)   Shape: (2*numBeams,)
                          numBeams*p0+numBeams*p1 ]                 # Polarizations ('/', '\', '-', or '|')  (str)

            elif polStrategy.lower()!="equal":
                raise ValueError(f"Invalid 'polStrategy' '{polStrategy}'! It must be 'equal' or 'probe'.")

        return steeringVectors, beams

    # ******************************************************************************************************************
    def getProbingBeams(self, theta0, phi0, numBeams, polStrategy=None, maxSeparation=6, deltaTheta=5):
        r"""
        This method generates a set of beam-probing steering vectors around a reference direction ``(theta0, phi0)``.
        It returns a matrix of beamforming weights and per-beam metadata (angles and polarization).

        Parameters
        ----------
        theta0 : float
            Reference zenith angle in degrees.
        phi0 : float
            Reference azimuth angle in degrees.
        numBeams : int
            Number of probe beams to generate. Must be one of {2, 4, 8}.
        polStrategy : {"equal", "probe"}, optional
            Polarization strategy:
            
                :"equal": each beam excites both polarizations.
                :"probe": split beams across polarizations (requires dual-polarization and ``numBeams >= 4``).

            If None, a default is chosen based on ``numBeams`` and panel/array polarization.
        maxSeparation : float, optional
            Maximum azimuth separation (degrees) for horizontal probing. Default is 6.
        deltaTheta : float, optional
            Zenith probing step size in degrees for vertical probing. Default is 5.


        Returns
        -------
        steeringVectors : NumPy array
            Complex steering vector matrix of shape ``(numPorts, numBeams)``, where ``numPorts`` is the number
            of antenna ports and ``numBeams`` is the number of probing beams. When `polStrategy="probe"`
            on a dual-pol panel, ``numBeams`` is doubled.
        beams : list
            Beam metadata as ``[thetas, phis, pols]``, where ``thetas`` and ``phis`` have length ``numBeams`` and
            ``pols`` is a string of length ``numBeams`` describing the polarization label per beam.

        Notes
        -----
        - All angles are in **local** panel coordinates (same convention as :py:meth:`getSweepingBeams`).
          Broadside is at theta=90°, phi=0°.
        - The returned columns are **unnormalized steering vectors**, not power-normalized precoders. For
          single-polarized panels/arrays and for dual-polarized cases with ``polStrategy="equal"``, each column
          has :math:`\|w\|^2 = N_{ports}`. For ``polStrategy="probe"`` (dual-polarized only), one polarization
          is zeroed per column, so :math:`\|w\|^2 = N_{ports}/2` — a 3 dB power drop relative to the ``"equal"``
          mode. If you use these as transmit precoders, apply your own normalization to control total transmit
          power, and account for the 3 dB gap when comparing beam metrics across ``polStrategy`` modes.
        - Probing is intended to refine a beam selection around a *known* reference direction
          ``(theta0, phi0)`` — for example, after a coarse sweep with :py:meth:`getSweepingBeams` has picked
          a best beam, ``getProbingBeams`` generates a small set of nearby beams to test for a finer choice.
        - The probe topology depends on ``numBeams``, the panel/array shape, and ``polStrategy``. The internal
          split into horizontal (``nH``), vertical (``nV``), and polarization (``nP``) probes is chosen so that
          ``nH + nV + nP == numBeams`` (or ``2 * (nH + nV)`` columns when polarization is split). See the
          source for the exact mapping.
        - ``maxSeparation`` clamps how far each horizontal probe deviates from ``phi0`` in degrees, keeping
          probes near the reference direction. ``deltaTheta`` plays the same role for the zenith probes;
          zenith uses a fixed step in degrees rather than uniform sin/cos sampling because the relevant
          range is centered around 90° where sin-domain sampling degenerates.
        - ``polStrategy="probe"`` requires a dual-polarized antenna and ``numBeams >= 4``. With this option
          the spatial beam count is halved so the total number of returned beams stays at ``numBeams``.

        See also :py:meth:`getSweepingBeams` for the coarse-grid version.
        """
        if phi0<-90 or phi0>90:         raise ValueError(f"'phi0' must be between -90 and 90 degrees.")
        if theta0<0 or theta0>180:      raise ValueError(f"'theta0' must be between 0 and 180 degrees.")
        if self.numPorts==1:            raise ValueError(f"Beam probing is not supported for 1-port antenna.")
        if numBeams not in [2,4,8]:     raise ValueError(f"'numBeams' must be 2, 4, or 8.")
        if polStrategy is None:     polStrategy='equal' if ((numBeams==2) or (self.polarization in "-|")) else 'probe'
        polStrategy = polStrategy.lower()
        if polStrategy=='probe':
            if self.polarization in "-|":   raise ValueError(f"Cannot probe the polarization with an unpolarized panel.")
            if numBeams==2:                 raise ValueError(f"Cannot probe the polarization with just 2 beams.")
        
        angleToChar = {-45:'\\', 0:'-', 45:'/', 90:'|'}
        if numBeams==2:                 nH,nV,nP = (2,0,0)
        elif numBeams==4:
            if self.shape[0]==1:        nH,nV,nP = (4,0,0) if polStrategy=='equal' else (2,0,2)
            else:                       nH,nV,nP = (3,1,0) if polStrategy=='equal' else (2,0,2)
        elif numBeams==8:
            if self.shape[0]==1:        nH,nV,nP = (8,0,0) if polStrategy=='equal' else (4,0,2)
            else:                       nH,nV,nP = (6,2,0) if polStrategy=='equal' else (2,2,2)
        
        # First horizontal beams:
        maxDeltaU = np.sin(np.deg2rad(maxSeparation))
        deltaU = min(2/max(nH,1), maxDeltaU)
        u = deltaU*np.sign(phi0)*np.array([0, -1, 1, -2, 2, -3, 3, -4])[:nH] + np.sin(np.deg2rad(phi0))
        phis = np.rad2deg(np.arcsin(np.clip(u, -1, 1)))                             # Shape: (nH,)
        thetas = np.ones(nH)*theta0                                                 # Shape: (nH,)
        steeringVectors = self.getPortSteeringVector(theta0,phis).squeeze(1)        # numPorts x nH
                                                           
        if nV>0:
            # For zenith, instead of deltaU in sin domain, we just use a fixed deltaTheta value. Since theta0
            # is usually around 90, the sin domain does not work well for zenith, so we handle this in angle domain.
            # Add/subtract deltaTheta if theta0 is below/above 90 degrees
            thetasV = (deltaTheta if theta0<=90 else -deltaTheta)*np.array([1, -1, 2, -2])[:nV] + theta0 # Shape=(nV,)
            wV = self.getPortSteeringVector(thetasV, phi0).squeeze(2)       # numPorts x nV
            steeringVectors = np.hstack([steeringVectors, wV])              # numPorts x (nH+nV)
            thetas = np.concatenate([thetas, thetasV])                      # Shape: (nH+nV,)
            phis = np.concatenate([phis, np.ones(nV)*phi0])                 # Shape: (nH+nV,)
        
        if nP>0:
            # Probing polarization
            steeringVectors = np.hstack([steeringVectors,steeringVectors])  # numPorts x numBeams
            numPos = self.numPorts//2
            steeringVectors[numPos:, :numBeams//2] = 0
            steeringVectors[:numPos, numBeams//2:] = 0
            p0 = angleToChar[self.getElement(p=0).polAngle]
            p1 = angleToChar[self.getElement(p=1).polAngle]
            beams = [ np.concatenate([thetas, thetas]),                 # Zenith angles (theta)  Shape: (2(nH+nV),)
                      np.concatenate([phis,phis]),                      # Azimuth angles (phi)   Shape: (2(nH+nV),)
                      (nH+nV)*p0 + (nH+nV)*p1 ]                         # Polarizations ('/', '\', '-', or '|')  (str)
        else:
            beams = [thetas, phis, (nH+nV)*self.polarization]

        return steeringVectors, beams

# **********************************************************************************************************************
class AntennaElement(AntennaBase):
    r"""
    This class implements the functionality of an antenna element. This implementation is based on **3GPP TR 38.901 
    Section 7.3**.
    """
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments. If you are creating a single antenna element object, most of the time you 
            do not need to specify any parameters; the default values are sufficient for normal functionality.
            Here is a list of supported parameters:

                :position: A list of 3 values (x, y, and z) specifying the position of this element in the 
                    :py:class:`AntennaPanel` containing this element.
                    
                :freqRange: A list of 2 values specifying the range of frequencies in which this antenna element
                    operates.
                    
                :polAngle: The polarization angle of this antenna element in degrees. A value of 0° means it is 
                    purely vertically polarized.
                    
                :polModel: The polarization model (1 or 2). The default is 1. Please refer to **TR38.901 Section
                    7.3.2** for more details.
                    
                :beamWidth: A list of 2 values specifying the beam width of this antenna element in degrees. The 
                    default is ``[65,65]``. These values correspond to :math:`\theta_{3dB}` and :math:`\phi_{3dB}` 
                    in **TR38.901-Table 7.3-1**.
                    
                    .. Note:: To make the antenna element omnidirectional, set :math:`\phi_{3dB}` to 360 degrees. The
                        following code shows how to create an omnidirectional antenna element:
                
                        .. code-block:: python
                        
                            import neoradium as nr
                            # Create an omnidirectional antenna element with θ(3dB)=75°
                            el = nr.AntennaElement(beamWidth=[75,360])
                    
                :verticalSidelobeAttenuation: Vertical side-lobe attenuation (:math:`SLA_V`). The default is 30.
                    Please refer to **TR38.901-Table 7.3-1** for more details.
                    
                :maxAttenuation: Maximum Attenuation (:math:`A_{max}`) in dB. The default is 30. Please refer 
                    to **TR38.901-Table 7.3-1** for more details.
                    
                :mainMaxGain: Maximum gain of main lobe in dBi. The default is 8. Please refer to **TR38.901-Table
                    7.3-1** for more details.
                    
                :panel: The :py:class:`AntennaPanel` object containing this element.
        """
        super().__init__(**kwargs)
        
        # The following is based on TR38.901
        self.position = np.float64(kwargs.get('position', [0,0,0])) # Position in the container (i.e. panel/array)
        self.freqRange = kwargs.get('freqRange', [0,100e9]) # Lower and upper bound of frequency
        
        self.polAngle = kwargs.get('polAngle', 0)           # Polarization slant angle in degrees (0°: vertical)
        validateRange(self.polAngle, [0,90,45,-45])

        # Note: The polModel default was changed from 2 to 1 starting in NeoRadium version 0.5
        self.polModel = kwargs.get('polModel', 1)           # Polarization model (1 or 2, Section 7.3.2)
        validateRange(self.polModel, [1,2])
        
        self.beamWidth = kwargs.get('beamWidth', [65,65])   # 3dB beamwidth in degrees [theta, phi]. (Table 7.3-1)
        valid = True
        if not isinstance(self.beamWidth, list):                    valid = False
        elif len(self.beamWidth) != 2:                              valid = False
        elif (self.beamWidth[0]<=0) or (self.beamWidth[0]>180):     valid = False
        elif (self.beamWidth[1]<=0):                                valid = False
        elif (self.beamWidth[1]>180) and (self.beamWidth[1]!=360):  valid = False
        if not valid:   raise ValueError("'beamWidth' must be list containining exactly 2 angles between 0° and 180°!")

        # Vertical side-lobe attenuation (SLAv) (Table 7.3-1)
        self.verticalSidelobeAttenuation = kwargs.get('verticalSidelobeAttenuation', 30)
        validateRange(self.verticalSidelobeAttenuation, (20,35))
        
        self.maxAttenuation = kwargs.get('maxAttenuation', 30)  # Maximum Attenuation (Amax) in dB (Table 7.3-1)
        validateRange(self.maxAttenuation, (20,35))

        self.mainMaxGain = kwargs.get('mainMaxGain', 8)         # Maximum gain of main lobe in dBi (Table 7.3-1)
        validateRange(self.mainMaxGain, (5,10))

        self.panel = kwargs.get('panel', None)                  # Owner

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`AntennaElement` object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "Antenna Element:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string. 
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Antenna Element:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        if self.panel is not None:
            repStr += indent*' ' + f"  position:               {self.position}\n"
        repStr += indent*' ' + f"  freqRange:              {freqStr(self.freqRange[0])} .. {freqStr(self.freqRange[1])}\n"
        repStr += indent*' ' + f"  polAngle:               {self.polAngle}°\n"
        repStr += indent*' ' + f"  polModel:               {self.polModel}\n"
        repStr += indent*' ' + f"  beamWidth:              {self.beamWidth[0]}°,{self.beamWidth[1]}°\n"
        repStr += indent*' ' + f"  Vertical Sidelobe Atten:{self.verticalSidelobeAttenuation} dB\n"
        repStr += indent*' ' + f"  maxAttenuation:         {self.maxAttenuation} dB\n"
        repStr += indent*' ' + f"  mainMaxGain:            {self.mainMaxGain} dBi\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    def posInArray(self):
        r"""
        Returns the position of this element in the :py:class:`AntennaArray` object.

        Returns
        -------
        NumPy array
            An array of 3 values (x, y, and z) specifying the position of this element in the :py:class:`AntennaArray`
            object.
        """
        return self.position + self.panel.position
    
    # ******************************************************************************************************************
    def clone(self, position, polAngle, panel):
        r"""
        Creates a copy of this :py:class:`AntennaElement` object and modifies the ``position``, polarization angle 
        (``polAngle``), and the ``panel`` object based on the parameters provided.

        Parameters
        ----------
        position : list or NumPy Array
            A list of 3 values (x, y, and z) specifying the position to be used for the cloned 
            :py:class:`AntennaElement`.
                        
        polAngle : float 
            The polarization angle of the cloned :py:class:`AntennaElement` in degrees.
    
        panel : :py:class:`AntennaPanel`
            The :py:class:`AntennaPanel` object containing the cloned :py:class:`AntennaElement`.
        
        Returns
        -------
        :py:class:`AntennaElement`
            The cloned :py:class:`AntennaElement`.
        """
        return AntennaElement(freqRange = self.freqRange,
                              polAngle = polAngle,
                              polModel = self.polModel,
                              beamWidth = self.beamWidth,
                              verticalSidelobeAttenuation = self.verticalSidelobeAttenuation,
                              maxAttenuation = self.maxAttenuation,
                              mainMaxGain = self.mainMaxGain,
                              position = position,
                              panel = panel)

    # ******************************************************************************************************************
    def verticalRadiationPower(self, theta=None):       # See TR38.901-Table 7.3-1 (1st row)
        # theta must be in [0, 180]. This is the 𝜃" in TR38.901-Table 7.3-1
        theta = self.anglesToNumpy(theta,0,181)
        # The calculation can be done in degrees because we are only calculating the ratios.
        return -np.minimum(12*np.square((theta-90)/self.beamWidth[0]), self.verticalSidelobeAttenuation)
        
    # ******************************************************************************************************************
    def horizontalRadiationPower(self, phi=None):          # See TR38.901-Table 7.3-1 (2nd row)
        if (phi is not None) and (self.beamWidth[1]==360):  return np.zeros(phi.shape)  # Special case: omnidirectional
        
        # phi must be in [-180, 180]. This is the 𝜙" in TR38.901-Table 7.3-1
        phi = self.anglesToNumpy(phi,-180,180)
        
        # The calculation can be done in degrees because we are only calculating the ratios.
        return -np.minimum(12*np.square(phi/self.beamWidth[1]), self.maxAttenuation)

    # ******************************************************************************************************************
    def allElements(self, polarization=True):                       # Undocumented - Not intended for direct use
        # Polarization is ignored. If you want a single antenna with dual polarization, you need to
        # create a 1x1 panel.
        return [self]

    # ******************************************************************************************************************
    def getAllPositions(self, polarization=True):                   # Undocumented - Not intended for direct use
        # Polarization is ignored. If you want a single antenna with dual polarization, you need to
        # create a 1x1 panel.
        return np.float64([self.position])

    # ******************************************************************************************************************
    def getPowerPatternDb(self, theta=None, phi=None, weights=None):  # See TR38.901-Table 7.3-1
        r"""
        This method calculates the field power pattern (in dB) in the directions specified by ``theta`` and ``phi``.
        This function is implemented based on **TR38.901-Table 7.3-1**.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the field
            powers.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the field power is calculated only for the single specified zenith angle
            (in degrees)

            If this is None, the field powers are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the field
            powers.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the field power is calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the field powers are calculated for all azimuth angles between -180 and 180 degrees.

        weights : None
            Ignored for AntennaElement objects.

        Returns
        -------
        NumPy array
            If ``theta`` and ``phi`` have the same shape, the returned value has the same shape as ``theta`` and 
            ``phi`` and contains the field powers in dB at the directions specified by ``theta`` and ``phi``. Otherwise,
            a NumPy array of shape (numTheta x numPhi) is returned, containing the field powers in dB at all 
            combinations of ``theta`` and ``phi``.
        """
        # This returns antenna power gain in dB. It is unitless relative to isotropic (dBi) which is the sum of the 3rd
        # and 4th rows in TR38.901 Table 7.3-1
        theta = self.anglesToNumpy(theta,0,180)
        phi = self.anglesToNumpy(phi,-180,180)

        if len(theta.shape)==1 and len(phi.shape)==1 and len(theta)!=len(phi):
            # Need to broadcast. The output will be a len(theta) x len(phi) matrix
            radPower = -np.minimum(-(self.verticalRadiationPower(theta).reshape(-1,1) +
                                   self.horizontalRadiationPower(phi).reshape(1,-1)), self.maxAttenuation) + self.mainMaxGain
            return np.float64(np.squeeze(radPower))

        # In this case, theta, phi, and the output have the same shape.
        return np.float64(-np.minimum(-(self.verticalRadiationPower(theta) + self.horizontalRadiationPower(phi)),
                                      self.maxAttenuation) + self.mainMaxGain)

    # ******************************************************************************************************************
    def getPowerPattern(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the field power pattern in the directions specified by ``theta`` and ``phi``. This
        function calls the :py:meth:`AntennaElement.getPowerPatternDb` and converts the results from dB to linear
        representation.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the field
            powers.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the field power is calculated only for the single specified zenith angle
            (in degrees)

            If this is None, the field powers are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the field
            powers.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the field power is calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the field powers are calculated for all azimuth angles between -180 and 180 degrees.

        weights : None
            Ignored for the AntennaElements.

        Returns
        -------
        NumPy array
            If ``theta`` and ``phi`` have the same shape, the returned value has the same shape as ``theta`` and 
            ``phi`` and contains the field powers at the directions specified by ``theta`` and ``phi``. Otherwise,
            a NumPy array of shape (numTheta x numPhi) is returned, containing the field powers at all 
            combinations of ``theta`` and ``phi``.
        """
        return toLinear(self.getPowerPatternDb(theta, phi))

    # ******************************************************************************************************************
    def getField(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the fields in specified directions, given by ``theta`` and ``phi``. It calls the 
        :py:meth:`AntennaElement.getPowerPatternDb` method and converts the results to field values. It’s important 
        to note that this function assumes vertically polarized antenna elements and returns the fields in vertical 
        orientations only. Use the :py:meth:`AntennaElement.getPolarizedFields` method to get the polarized fields.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the fields are calculated for all azimuth angles between -180 and 180 degrees.

        weights : None
            Ignored for the AntennaElements.
       
        Returns
        -------
        NumPy array
            If ``theta`` and ``phi`` have the same shape, the returned value has the same shape as ``theta`` and 
            ``phi`` and contains the electric field at the directions specified by ``theta`` and ``phi``. Otherwise,
            a NumPy array of shape (numTheta x numPhi) is returned, containing the electric field at all 
            combinations of ``theta`` and ``phi``.
        """
        # This assumes a polarization angle of 0 (pure vertical). In this case, the vertical (zenith) field=sqrt(power),
        # and the horizontal (azimuth) field = 0. If the polarization angle is not zero, use the "getPolarizedFields"
        # function below.
        # This is the field amplitude pattern. It is still unitless relative (to isotropic) electric field magnitude
        return toLinear(self.getPowerPatternDb(theta, phi)/2)

    # ******************************************************************************************************************
    def getPolarizedFields(self, theta, phi, weights=None):
        r"""
        This method calculates the polarized fields and outputs two matrices of the field values for vertical and 
        horizontal polarizations.
        
        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the fields are calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the fields are calculated for all azimuth angles between -180 and 180 degrees.

        weights : None
            Ignored for the AntennaElements.
       
        Returns
        -------
        2 NumPy arrays
            If ``theta`` and ``phi`` have the same shape, the following returned values are also the same shape as
            ``theta`` and ``phi``. Otherwise, two NumPy arrays of shape (numTheta x numPhi) are returned.

            * **arrayFieldV**:
                A NumPy array containing the field values with vertical polarization at the directions specified by
                ``theta`` and ``phi``.

            * **arrayFieldH**:
                A NumPy array containing the field values with horizontal polarization at the directions specified by
                ``theta`` and ``phi``.
        """
        field = self.getField(theta, phi)
        𝜁 = self.polAngle*np.pi/180         # Zeta: Polarization Angle in Radians

        if self.polModel == 1:
            # Model-1:
            # This treats polarization as a physical rotation of the radiating dipole in 3D space. If you rotate a
            # dipole by +45°, the projection of its electric field onto θ/φ basis changes with direction. So the
            # vertical/horizontal polarization split varies with (θ, φ). This is the physically correct model.
            # First calculate cos𝜓 and sin𝜓 (See TR38.901-Eq. 7.3-3)
            if self.polAngle == 0:              cos𝜓, sin𝜓 = 1, 0
            elif self.polAngle == 90:           cos𝜓, sin𝜓 = 0, 1
            else:
                # See the "Model-1" in "TR38.901 Section 7.3.2"
                if len(theta.shape)==1 and len(phi.shape)==1 and len(theta)!=len(phi):
                    # Need to broadcast. This is mostly used when calculating fields to draw radiation patterns.
                    𝜃 = theta[:,None]*np.pi/180
                    𝜑 = phi[None,:]*np.pi/180

                    denom = np.sqrt(1-np.square(np.cos(𝜁)*np.cos(𝜃)-np.sin(𝜁)*np.sin(𝜑)*np.sin(𝜃)))
                    cos𝜓 = (np.cos(𝜁)*np.sin(𝜃) + np.sin(𝜁)*np.sin(𝜑)*np.cos(𝜃))/(denom+1e-12)
                    sin𝜓 = np.sin(𝜁)*np.cos(𝜑)/(denom+1e-12)
                    # Use model-2 when denom becoms very close to zero:
                    cos𝜓[denom<1e-12] = np.cos(𝜁)
                    sin𝜓[denom<1e-12] = np.sin(𝜁)
                    cos𝜓 = np.squeeze(cos𝜓)
                    sin𝜓 = np.squeeze(sin𝜓)
                else:
                    # This used by channel models:
                    𝜃 = theta*np.pi/180
                    𝜑 = phi*np.pi/180
                    denom = np.sqrt(1-np.square(np.cos(𝜁)*np.cos(𝜃)-np.sin(𝜁)*np.sin(𝜑)*np.sin(𝜃)))
                    cos𝜓 = (np.cos(𝜁)*np.sin(𝜃) + np.sin(𝜁)*np.sin(𝜑)*np.cos(𝜃))/(denom+1e-12)
                    sin𝜓 = np.sin(𝜁)*np.cos(𝜑)/(denom+1e-12)
                    # Use model-2 when denom becoms very close to zero:
                    cos𝜓[denom<1e-12] = np.cos(𝜁)
                    sin𝜓[denom<1e-12] = np.sin(𝜁)

            fTheta = field * cos𝜓
            fPhi = field * sin𝜓
        else:
            # Model-2:
            # This assumes the polarization vector is already aligned with the spherical basis and simply splits
            # power according to slant angle. So, +45° → always 50% vertical, 50% horizontal (in power) independent
            # of direction. It is simpler, faster, less physically precise, but often good enough for system-level
            # simulations.
            fTheta = field * np.cos(𝜁)
            fPhi = field * np.sin(𝜁)
        
        return fTheta, fPhi

    # ******************************************************************************************************************
    def draw(self, ref="Array"):                                    # Undocumented - Not intended for direct use
        # This is called by panel or array objects to draw this element.
        if self.panel is None:
            raise ValueError("AntennaElement.draw() requires a parent panel; "
                             "call this on AntennaPanel or AntennaArray instead.")
        import matplotlib.pyplot as plt
        pos = self.position + (0 if ref=="Panel" else self.panel.position)
        points = { '|': np.float64([[0,1], [0,-1]]),
                   '-': np.float64([[-1,0], [1,0]]),
                   '+': np.float64([[-1,0], [1,0], [0,0], [0,-1], [0,1]]),
                   'x': np.float64([[-1,-1], [1,1], [0,0], [-1,1], [1,-1]]) }

        markerScale = 0.1
        elementPoints = pos[1:] + min(self.panel.spacing) * markerScale * points[self.panel.polarization]
        plt.plot(elementPoints[:,0], elementPoints[:,1], color="red", linewidth=1)

    # ******************************************************************************************************************
    def getDirectivity(self, theta=None, phi=None, weights=None):
        r"""
        Directivity at a specific direction is defined as:
        
        .. math::

            D = \frac {P} {P_{avg}}

        where :math:`P` is the power radiated at the specified angle and :math:`P_{avg}` is the average power 
        radiated in all directions. The average power is calculated by integrating the field values at all angles:
        (See `this web page <https://www.antenna-theory.com/basics/directivity.php>`_ for more details)
        
        .. math::

            P_{avg} = \frac {1} {4 \pi} \int_0^{2 \pi} \int_0^{\pi} |F(\theta, \phi)|^2 \sin \theta d\theta d\phi

        
        Directivity (without any specific direction) is defined as:
        
        .. math::

            D_{max} = \frac {P_{max}} {P_{avg}}
            
        where :math:`P_{max}` is the maximum power radiated at a given direction. Directivity is usually measured in dBi 
        which is the relative directivity in dB with respect to an "isotropic" radiator.
                
        This method calculates the directivity (in dBi) at directions specified by ``theta`` and ``phi``.

        Parameters
        ----------
        theta : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the zenith angles (in degrees) used to calculate the
            directivity.

            If this is a tuple, the values are assumed to specify the range of values used for zenith angles
            (in degrees)

            If this is a scalar value, the directivity is calculated only for the single specified zenith angle
            (in degrees)

            If this is None, the directivity is calculated for all zenith angles between 0 and 180 degrees.

        phi : list, tuple, NumPy array, scalar, or None
            If this is a list or NumPy array, it specifies the azimuth angles (in degrees) used to calculate the
            directivity.

            If this is a tuple, the values are assumed to specify the range of values used for azimuth angles
            (in degrees)

            If this is a scalar value, the directivity is calculated only for the single specified azimuth angle
            (in degrees)

            If this is None, the directivity is calculated for all azimuth angles between -180 and 180 degrees.

        weights : NumPy array
            This parameter is ignored by the :py:class:`AntennaElement` objects.

        Returns
        -------
        NumPy array
            If ``theta`` and ``phi`` have the same shape, the returned value has the same shape as ``theta`` and 
            ``phi`` and contains the directivity at the directions specified by ``theta`` and ``phi``. Otherwise,
            a NumPy array of shape (numTheta x numPhi) is returned, containing the directivity at all 
            combinations of ``theta`` and ``phi``.
        """
        theta = self.anglesToNumpy(theta,0,180)
        phi = self.anglesToNumpy(phi,-180,180)

        totalPower = self.getPowerPattern(theta, phi)       # Field Powers (A numTheta x numPhi matrix)

        # Now we calculate directivity based on the formula here:
        #       https://www.antenna-theory.com/basics/directivity.php
        # We first need to calculate the average power in all directions which is the denominator integral
        # in the directivity formula.
        angleStep = self.getIntegralAngleStep()
        allTheta = np.arange(0, 180, angleStep)
        allPhi = np.arange(-180, 180, angleStep)

        if (allTheta.shape!=theta.shape) or (allPhi.shape!=phi.shape):
            totalPowerAllD = self.getPowerPattern(allTheta, allPhi)
        elif np.any(allTheta!=theta) or np.any(allPhi!=phi):
            totalPowerAllD = self.getPowerPattern(allTheta, allPhi)
        else:
            totalPowerAllD = totalPower  # power pattern for the integral already calculated

        dTheta = dPhi = angleStep*np.pi/180
        integral = (totalPowerAllD*np.sin(allTheta*np.pi/180).reshape(-1,1)*dTheta*dPhi).sum()

        # Note that since totalPower is not normalized we have it in the numerator of directivity formula
        # instead of 1
        directivity = 4*np.pi*totalPower/integral
        directivityDbi = toDb(directivity)      # Convert to "dBi", which is dB with respect to an "isotropic" radiator
        return directivityDbi

# **********************************************************************************************************************
class AntennaPanel(AntennaBase):
    r"""
    This class implements the functionality of a rectangular antenna panel containing a set of antenna elements 
    (see :py:class:`AntennaElement`) organized in a 2-d grid. The elements are assumed to be on the Y-Z plane. An 
    antenna panel can be created individually or it can be grouped with other panels to form an
    :py:class:`AntennaArray`.
    """
    # ******************************************************************************************************************
    def __init__(self, shape=[1,1], **kwargs):
        r"""
        Parameters
        ----------
        shape : list
            A list of 2 integers specifying the number of antenna elements along ``z`` and ``y`` axes (the number of
            rows and columns of elements). The default is ``[1, 1]``.

        kwargs : dict
            A set of additional optional arguments. Here is a list of supported parameters:

                :spacing: A list of 2 values specifying the distance between neighboring elements in multiples of the 
                    wavelength. By default, the elements are half a wavelength away from each other, which 
                    means `spacing = [0.5, 0.5]`.
            
                :elements: This can be an :py:class:`AntennaElement` object, a 2-D array of :py:class:`AntennaElement`
                    objects, or None.
                
                    * If it is an :py:class:`AntennaElement` object, it will be used as a template to create all the
                      elements in this panel.
                    
                    * If it is a 2-D array of :py:class:`AntennaElement` objects, the specified elements are used for 
                      the elements of this panel.
                      
                    * If it is `None`, then antenna elements of the panel are created using the parameters in the 
                      ``kwargs`` (if any) and the default values.
                    
                :polarization: The polarization of antenna elements on this panel. The panel can be singly polarized
                    (P=1) or dually polarized (P=2). For singly polarized panels, the ``polarization`` can be either
                    "|" (Vertical), or "-" (Horizontal). For dually polarized panels, the ``polarization`` can be
                    either "+" (0 and 90 degree pairs), or "x" (-45 and 45 degree pairs). By default, 
                    ``polarization="|"`` (Vertically polarized).
                    
                :position: The position of the center point of this panel in the antenna array containing this panel.
                
                :array: The :py:class:`AntennaArray` object containing this antenna panel or `None` if this panel 
                    is not part of an antenna array.
                    
                :matlabOrder: The current implementation of the MATLAB toolkit uses a different order for the elements 
                    in a panel compared to the order specified in the 3-GPP standard (See **3GPP TR 38.901 - 
                    Section 7.3**). By default, this class uses the standard order (``matlabOrder=False``). If you need
                    to compare your results with the MATLAB implementation, you can set this parameter to `True`.
        """
        super().__init__(**kwargs)
        self.shape = np.int16(shape)                                # Number of antenna elements in columns and rows
        if self.shape.shape != (2,):        raise ValueError("'shape' must be a list or NumPy array of length 2.")

        self.spacing = np.float64(kwargs.get('spacing', [.5,.5]))   # [dv, dh] in multiples of wavelength.
        if self.spacing.shape != (2,):      raise ValueError("'spacing' must be a list or NumPy array of length 2.")
        if np.any(self.spacing <= 0):       raise ValueError("'spacing' values must be positive.")

        self.polarization = kwargs.get('polarization', "|")         # Can be one of "|", "-", "+", or "x"
        if self.polarization not in "|-+x":
            raise ValueError("'polarization' must be one of \"|\", \"-\", \"+\", or \"x\".")
        
        self.position = np.float64(kwargs.get('position', [0,0,0])) # Position in the array
        if self.position.shape != (3,):     raise ValueError("'position' must be a list or NumPy array of length 3.")
            
        self.array = kwargs.get('array', None)                      # The owner AntennaArray
        if self.array is not None:
            if type(self.array)!=AntennaArray:  raise ValueError("'array' must be an 'AntennaArray' object or None.")
        self.matlabOrder = kwargs.get('matlabOrder', False)         # If true, use MATLAB order in "allElements" method
        
        self.elements = kwargs.get('elements', None)                # A 2d array of AntennaElement objects
        if self.elements is None:
            elementTemplate = AntennaElement(**kwargs)              # Pass kwargs to the template element.
        elif type(self.elements)==list:
            elementTemplate = None
            if len(self.elements)!=self.shape[0]:
                raise ValueError("'elements' shape does not match the provided 'shape'!")
            for row in self.elements:
                if type(row)!=list:         raise ValueError("'elements' shape does not match the provided 'shape'!")
                if len(row)!=self.shape[1]: raise ValueError("'elements' shape does not match the provided 'shape'!")
        elif type(self.elements)==AntennaElement:
            elementTemplate = self.elements
        else:
            raise ValueError("'elements' must be an 'AntennaElement' object, a 2-D array of `AntennaElement` objects, "+
                             "or None.")
            
        # Antenna element orders with dual polarization for a panel with 2 rows and 4 columns:
        #         1st Polarization            2nd Polarization
        #         ----------------            ----------------
        #         |  4  5  6  7  |            |  12 13 14 15 |
        #   z↑    |  0  1  2  3  |            |  8  9  10 11 |
        #   y→    ----------------            ----------------
        if elementTemplate is not None:
            offsetZ, offsetY = (self.shape-1) * self.spacing / 2
            dz, dy = self.spacing

            # Polarization angles for each polarization (one angle for single and two angles for dual polarization)
            polAngles = [ {"|":0, "-": 90, "+":0, "x": 45}[self.polarization] ]
            if self.polarization in "+x": polAngles += [ {"+":90, "x": -45}[self.polarization] ]    # Dual polarization

            # elements[r][c][p] contains an AntennaElement in row 'r' and column 'c' for p'th polarization.
            self.elements = [self.shape[1]*[None] for r in range(self.shape[0])]    # Initialize all elements to None
            for r in range(self.shape[0]):
                for c in range(self.shape[1]):
                    position = [ 0, c*dy-offsetY, r*dz-offsetZ ]    # Position of the element for both polarizations
                    self.elements[r][c] = [ elementTemplate.clone(position, polAngle, self) for polAngle in polAngles ]
        self.numPorts = self.numEl

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`AntennaPanel` object.

        Parameters
        ----------
        indent : int 
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "Antenna Panel:" is used for the title.

        getStr : bool 
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string.
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Antenna Panel:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        if self.array is not None:
            repStr += indent*' ' + f"  position:               {self.position}*𝜆\n"
        repStr += indent*' ' + f"  Total Elements:         {self.numEl}\n"
        repStr += indent*' ' + f"  spacing:                {self.spacing[0]}𝜆, {self.spacing[1]}𝜆\n"
        repStr += indent*' ' + f"  shape:                  {self.shape[0]} rows x {self.shape[1]} columns\n"
        repStr += indent*' ' + f"  polarization:           {self.polarization}\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    # Returns the frequency range of the first element. It is assumed all elements have the same range.
    def freqRange(self):   return self.getElement(0).freqRange

    # ******************************************************************************************************************
    def clone(self, position, array):
        r"""
        Creates a copy of this :py:class:`AntennaPanel` object and modifies the ``position`` and ``polarization``
        angles, and the parent ``array`` object based on the parameters provided.
        
        Parameters
        ----------
        position : list or NumPy array
            The position of the center point of the cloned panel in the antenna array containing it.
            
        polarization : str
            The polarization of antenna elements on the cloned panel. The panel can be singly polarized
            (P=1) or dually polarized (P=2). For singly polarized panels, the ``polarization`` can be either
            "|" (Vertical), or "-" (Horizontal). For dually polarized panels, the ``polarization`` can be
            either "+" (0 and 90 degree pairs), or "x" (-45 and 45 degree pairs). By default, 
            ``polarization="|"`` (Vertically polarized).

        array : :py:class:`AntennaArray`
            The :py:class:`AntennaArray` object containing the cloned panel.

        Returns
        -------
        :py:class:`AntennaPanel`
            The cloned :py:class:`AntennaPanel` object.
        """
        return AntennaPanel(self.shape,
                            spacing = self.spacing,
                            polarization = self.polarization,
                            elements = self.elements[0][0][0],
                            position = position,
                            array = array)
    
    # ******************************************************************************************************************
    def getNumElements(self):
        r"""
        Returns the total number of antenna elements in this panel. For singly polarized panels, the total number
        of elements is ``shape[0] x shape[1]``. For dually polarized panels, the total number of elements is
        ``2 x shape[0] x shape[1]``.
        """
        return np.prod(self.shape)*(1 if self.polarization in "-|" else 2)      # Return total number of elements.

    # ******************************************************************************************************************
    def getElement(self, elementRC=(0,0), p=0):
        r"""
        Returns the specified :py:class:`AntennaElement` object from this panel.
        
        Parameters
        ----------
        elementRC : tuple or int
            If this is a tuple, the first and second integer values in the tuple specify the row and column of the
            desired element in the panel (0-based). If this is an integer, the allowed values are 0 or -1 which return
            the first or last element in the panel respectively. If ``elementRC`` is not specified, by default the
            first element is returned.
            
        p : int
            If this panel is singly polarized, this parameter is ignored. Otherwise, the first and second polarized
            antenna element is returned for ``p=0`` and ``p=1`` respectively.

        Returns
        -------
        :py:class:`AntennaElement`
            The specified :py:class:`AntennaElement` object from this panel.
        """
        if elementRC==0:  elementRC = (0,0)     # Get first
        if elementRC==-1: elementRC = (-1,-1)   # Get last
        return self.elements[ elementRC[0] ][ elementRC[1] ][ p ]

    # ******************************************************************************************************************
    def getElementPosition(self, elementRC=(0,0), ref="Array"):
        r"""
        Returns the position of the specified :py:class:`AntennaElement` object in this panel.
        
        Parameters
        ----------
        elementRC : tuple or int
            If this is a tuple, the first and second integer values in the tuple specify the row and column of the
            desired element in the panel (0-based). If this is an integer, the allowed values are 0 or -1 which return
            the first or last element in the array respectively. If ``elementRC`` is not specified, by default the
            first element is returned.
            
        ref : str
            If ``ref="Array"`` this function returns the element position with respect to the :py:class:`AntennaArray`
            object containing this panel. Otherwise, if ``ref=="Panel"``, the element position with respect to this
            panel is returned.

        Returns
        -------
        NumPy array
            An array of 3 values (x, y, and z) representing the position of the specified element. Note that the 
            values are in multiples of wavelength.
        """
        return self.getElement(elementRC).position + (0 if ref=="Panel" else self.position)
    
    # ******************************************************************************************************************
    def getAllPositions(self, polarization=True):
        r"""
        Returns the positions of all elements in this panel as a 2-D NumPy array.
        
        Parameters
        ----------
        polarization : bool
            If this is a dually polarized panel and this parameter is `True`, the positions of all elements are
            returned. In this case, there will be repeated positions in the returned array as the 2 polarized pairs
            of elements have the same position. Otherwise, if ``polarization=False``, only one position is returned
            for a pair of polarized antenna elements. If this is a singly polarized panel, this parameter is ignored.
            
        Returns
        -------
        NumPy array
            An ``n x 3`` NumPy array containing the positions of all ``n`` elements in this panel.
        """
        return np.float64([e.position for e in self.allElements(polarization)])

    # ******************************************************************************************************************
    def showElements(self, ref="Panel", maxSize=6.0, zeroTicks=False, title=None):
        r"""
        This is a visualization function that draws this antenna panel using the `matplotlib` library.
        
        Parameters
        ----------
        ref : str
            If ``ref="Panel"``, it means this is a standalone antenna panel that is visualized individually. Otherwise,
            if ``ref="Array"``, it means this is being visualized as part of an antenna array. (See 
            :py:meth:`AntennaArray.showElements`)
            
        maxSize : float
            This parameter specifies how large the output image of this panel should be. Depending on the number of
            antenna element rows and columns in this panel, the ``maxSize`` can specify the width or height of the
            resulting image.
        
        zeroTicks : bool
            If this is `True`, the zero positions on both axes are indicated by additional "ticks" to show
            the center of this panel. Otherwise, the "ticks" on the horizontal and vertical axes are only at the
            locations of antenna elements.
            
        title : str or None
            If specified, this will be used as the title for the image created for this panel. Otherwise, the title
            "Panel Elements" is used.
        """
        import matplotlib.pyplot as plt
        if ref=="Panel":
            s = self.shape*self.spacing
            figSize = [maxSize, maxSize*s[0]/s[1]] if s[0]<s[1] else [maxSize*s[1]/s[0], maxSize]
            plt.figure(figsize=figSize)
#            plt.rcParams['figure.figsize'] = figSize
#            plt.rcParams['figure.dpi'] = 100

        # Draw the rectangle around the panel
        rectPoints = np.array([ self.getElementPosition((-1,0),  ref)[1:] + self.spacing * [-0.3,+0.3],
                                self.getElementPosition((-1,-1), ref)[1:] + self.spacing * [+0.3,+0.3],
                                self.getElementPosition((0,-1),  ref)[1:] + self.spacing * [+0.3,-0.3],
                                self.getElementPosition((0,0),   ref)[1:] + self.spacing * [-0.3,-0.3],
                                self.getElementPosition((-1,0),  ref)[1:] + self.spacing * [-0.3,+0.3] ])
        panelRectStyle = '--' if ref=="Array" else '-'
        plt.plot(rectPoints[:,0], rectPoints[:,1], linestyle=panelRectStyle, color="black", linewidth=1)

        # Now draw the actual elements
        for element in self.allElements(False):  element.draw(ref)
        
        if ref=="Panel":
            plt.xlabel("$\\frac {Y}{\\lambda}$", size=15)
            plt.ylabel("$\\frac {Z}{\\lambda}$", size=15)
            plt.title("Panel Elements" if title is None else title, size=20)
            yTicks = [self.getElementPosition((0,e), ref)[1] for e in range(self.shape[1]) ] + ([0] if zeroTicks else [])
            plt.xticks(sorted(yTicks), size=10)
            zTicks = [self.getElementPosition((e,0), ref)[2] for e in range(self.shape[0]) ] + ([0] if zeroTicks else [])
            plt.yticks(sorted(zTicks), size=10)
            plt.axis('equal')

    # ******************************************************************************************************************
    def allElements(self, polarization=True):
        r"""
        This is a generator function that can be used to iterate through all elements of this panel. For example, the
        following code prints the position of every element in this panel:
        
        .. code-block::
        
            for element in myPanel.allElements():
                print( element.position )


        By default, this function iterates through elements in the order specified in **3GPP TR 38.901 Section
        7.3**. If the parameter ``matlabOrder`` is set to `True`, then the MATLAB order is used. Please refer to
        :py:class:`AntennaPanel` parameter documentation for more information about ``matlabOrder``.
        
        Parameters
        ----------
        polarization : bool
            If this is a dually polarized panel and this parameter is `True`, then all elements are included in
            the iteration. Otherwise, if ``polarization=False``, only the first element of the polarized pair of
            elements at each position is included in the iteration. If this is a singly polarized panel, this
            parameter is ignored.
            
        Yields
        ------
            The next :py:class:`AntennaElement` object in this panel.
        """
        numPol = 2 if (self.polarization in "+x") else 1
        rr, cc = self.shape
        if self.matlabOrder:
            if polarization:
                for p in range(numPol):
                    for c in range(cc):
                        for r in range(rr-1,-1,-1):
                            yield self.elements[ r ][ c ][ p ]
            else:
                for c in range(cc):
                    for r in range(rr-1,-1,-1):
                        yield self.elements[ r ][ c ][ 0 ]
        else:
            if polarization:
                for p in range(numPol):
                    for r in range(rr):
                        for c in range(cc):
                            yield self.elements[ r ][ c ][ p ]
            else:
                for r in range(rr):
                    for c in range(cc):
                        yield self.elements[ r ][ c ][ 0 ]

# **********************************************************************************************************************
class AntennaArray(AntennaBase):
    r"""
    This class implements the functionality of a rectangular antenna array containing a set of antenna panels (See
    :py:class:`AntennaPanel`) organized in a 2-D grid. The panels are assumed to be on the Y-Z plane.
    """
    # ******************************************************************************************************************
    def __init__(self, shape=[1,1], **kwargs):
        r"""
        Parameters
        ----------
        shape : list
            A list of 2 integers specifying the number of antenna panels along ``z`` and ``y`` axes (The number of
            rows and columns of panels)
            
        kwargs : dict
            A set of additional optional arguments. Here is a list of supported parameters:

                :spacing: A list of 2 values specifying the distance between the center point of neighboring panels 
                    in multiples of the wavelength. If not specified, by default the spacing is set such that the 
                    spacing between antenna elements across different panels is the same as that between antenna 
                    elements within panels.
                    
                :panels: This can be an :py:class:`AntennaPanel` object, a 2-D array of :py:class:`AntennaPanel` 
                    objects, or None.
                
                    * If it is an :py:class:`AntennaPanel` object, it will be used as a template to create all the 
                      panels in this array.
                      
                    * If it is a 2-D array of :py:class:`AntennaPanel` objects, the specified panels are used for the
                      panels of this array.
                      
                    * If it is `None`, then antenna panels and elements of this array are created using the default
                      values.
                      
                :internalB: A complex NumPy array of shape (nPos, numPorts) specifying the internal port-to-element 
                    precoding matrix. This matrix defines the fixed intra-panel beamforming that maps each CSI-RS 
                    antenna port to the antenna elements within a (panel, polarization) group. If not provided, a 
                    default broadside internal precoder is used. For example, a 2 x 3 array of 2 x 2 dual polarized 
                    panels has nPos=4 (e.g. 2x2 panels) and numPorts=12 (e.g. 2x3 panels with 2 polarizations). 
        """
        super().__init__(**kwargs)
        self.shape = np.int16(shape)    # Number of rows and columns of panels. ([M, N] in TR38.901-Section 7.3)
        if self.shape.shape != (2,):        raise ValueError("'shape' must be a list or NumPy array of length 2.")

        self.spacing = kwargs.get('spacing', None)  # [dgV, dgH] in TR38.901-Section 7.3 in wavelength
        self.panels = kwargs.get('panels', None)    # A 2-D shape[0]-by-shape[1] array of AntennaPanel objects.
        if self.panels is None:
            panelTemplate = AntennaPanel()
        elif type(self.panels)==list:
            panelTemplate = None
            if len(self.panels)!=self.shape[0]: raise ValueError("'panels' shape does not match the provided 'shape'!")
            for row in self.panels:
                if type(row)!=list:             raise ValueError("'panels' shape does not match the provided 'shape'!")
                if len(row)!=self.shape[1]:     raise ValueError("'panels' shape does not match the provided 'shape'!")
        elif type(self.panels)==AntennaPanel:
            panelTemplate = self.panels
        else:
            raise ValueError("'panels' must be an 'AntennaPanel', a 2-D array of 'AntennaPanel' objects, or None.")
        if panelTemplate is not None:
            numRows, numCols = self.shape                   # These are Mg and Ng in TR38.901-Section 7.3 respectively
            self.spacing = (panelTemplate.shape*panelTemplate.spacing) if self.spacing is None else \
                           np.float64(self.spacing)
            offsetZ, offsetY = (self.shape-1) * self.spacing / 2
            dz, dy = self.spacing
            
            allPanels = []
            for r in range(numRows):
                allPanels += [[]]
                for c in range(numCols):
                    # Assuming the x-axis is pointing toward us
                    position = [ 0, c*dy-offsetY, r*dz-offsetZ ]
                    allPanels[r] += [ panelTemplate.clone(position, self) ]
            
            self.panels = allPanels
        else:
            self.spacing = (self.panels[0][0].shape * self.panels[0][0].spacing) if self.spacing is None else \
                           np.float64(self.spacing)

        if self.spacing.shape != (2,):      raise ValueError("'spacing' must be a list or NumPy array of length 2.")
        if np.any(self.spacing <= 0):       raise ValueError("'spacing' values must be positive.")

        # The internal precoding matrix (B) maps each CSI-RS antenna port to the elements in a (panel,pol).
        # This assumes all panels have the same shape and polarization
        # nPos: number of positions in each panel. Note: panel.numEl is 2*nPos for polarized panels and nPos otherwise.
        self.numPorts = self.numPanels * (2 if self.polarization in '+x' else 1)        # number of CSI-RS ports
        numPos = np.prod(self.panels[0][0].shape)
        internalB = kwargs.get('internalB', None)                                       # nPos x numPorts
        if internalB is None:
            # By default use a "Panel Broadside" internal precoding matrix
            if self.polarization in '+x':
                sv = self.panels[0][0].getSteeringVector(90,0).flatten()                            # len: 2*nPos
                internalB = np.hstack( np.tile(sv[:,None],
                                               (1,self.numPorts//2)).reshape(2,-1,self.numPorts//2) ) # nPos x numPorts
            else:
                sv = self.panels[0][0].getSteeringVector(90,0).flatten()                            # len: nPos
                internalB = np.tile(sv[:,None], (1,self.numPorts))                                  # nPos x numPorts
        elif (not isinstance(internalB, np.ndarray)) or (internalB.shape != (numPos, self.numPorts)):
            raise ValueError(f"'internalB' must be a NumPy array of shape ({numPos},{self.numPorts}).")
                
        self.b = np.zeros((self.numEl,self.numPorts),dtype=np.complex128)                           # nt x numPorts
        rows = np.arange(self.numEl)                                                                # nt
        cols = np.repeat(np.arange(self.numPorts),internalB.shape[0])                               # nt
        self.b[(rows,cols)] = internalB.T.flatten()                                                 # update b
        portNorms = np.linalg.norm(self.b, axis=0, keepdims=True)                                   # Columm norms
        if (portNorms < 1e-9).any():
            raise ValueError("'internalB' has at least one column with effectively zero norm; "
                             "each CSI-RS port must have a non-trivial precoder.")
        self.b /= portNorms                                                                         # Normalize

    # ******************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`AntennaArray` object.

        Parameters
        ----------
        indent : int
            The number of indentation characters.
            
        title : str or None
            If specified, it is used as the title for the printed information. If `None` (the default), the text
            "Antenna Array:" is used for the title.

        getStr : bool
            If `True`, returns a string instead of printing it.

        Returns
        -------
        None or str
            If the ``getStr`` parameter is `True`, then this function returns the information in a string.
            Otherwise, nothing is returned.
        """
        if title is None:   title = "Antenna Array:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + f"  shape:                  {self.shape[0]} rows x {self.shape[1]} columns\n"
        repStr += indent*' ' + f"  Total Panels:           {np.prod(self.shape)}\n"
        repStr += indent*' ' + f"  Total Elements:         {self.numEl}\n"
        repStr += indent*' ' + f"  Panel Spacing:          {self.spacing[0]}𝜆, {self.spacing[1]}𝜆\n"
        repStr += indent*' ' + f"  Num. Ports:             {self.numPorts}\n"
        if getStr: return repStr
        print(repStr)

    # ******************************************************************************************************************
    @property
    # Returns the frequency range of the first element. It is assumed all elements have the same range.
    def freqRange(self):    return self.getElement(0).freqRange

    # ******************************************************************************************************************
    @property
    # Assuming all panels have the same type of polarization.
    def polarization(self): return self.panels[0][0].polarization

    # ******************************************************************************************************************
    def getElement(self, panelRC=(0,0), elementInPanelRC=(0,0), p=0):
        r"""
        Returns the :py:class:`AntennaElement` object from this array specified by row and column of panel in this
        array and row and column of the element in that panel.
        
        Parameters
        ----------
        panelRC : tuple or int
            If this is a tuple, the first and second integer values in the tuple specify the row and column of the
            desired panel in the array (0-based). If this is an integer, the allowed values are 0 or -1 which specify
            the first or last panel in the array respectively. If ``panelRC`` is not specified, by default the first
            panel is used.

        elementInPanelRC : tuple or int
            If this is a tuple, the first and second integer values in the tuple specify the row and column of the
            desired element in the panel (0-based). If this is an integer, the allowed values are 0 or -1 which return
            the first or last element in the specified panel respectively. If ``elementInPanelRC`` is not specified,
            by default the first element in the specified panel is returned.
            
        p : int
            If the panels of this array are singly polarized, this parameter is ignored. Otherwise, the first and 
            second polarized antenna element is returned for ``p=0`` and ``p=1`` respectively.

        Returns
        -------
        :py:class:`AntennaElement`
            The specified :py:class:`AntennaElement` object from this panel.
        """
        if panelRC==0:  panelRC, elementInPanelRC = (0,0),(0,0)         # Get first
        if panelRC==-1: panelRC, elementInPanelRC = (-1,-1),(-1,-1)     # Get last
        return self.panels[ panelRC[0] ][ panelRC[1] ].elements[ elementInPanelRC[0] ][ elementInPanelRC[1] ][ p ]
        
    # ******************************************************************************************************************
    def getElementPosition(self, panelRC=(0,0), elementInPanelRC=(0,0)):
        r"""
        Returns the position of the :py:class:`AntennaElement` object in this array specified by ``elementInPanelRC``
        in the panel specified by ``panelRC``.
        
        Parameters
        ----------
        panelRC : tuple or int
            If this is a tuple, the first and second integer values in the tuple specify the row and column of the
            desired panel in the array (0-based). If this is an integer, the allowed values are 0 or -1 which specify
            the first or last panel in the array respectively. If ``panelRC`` is not specified, by default the first
            panel is used.

        elementInPanelRC : tuple or int
            If this is a tuple, the first and second integer values in the tuple specify the row and column of the
            desired element in the specified panel (0-based). If this is an integer, the allowed values are 0 or -1
            which return the position of the first or last element in the panel respectively. If ``elementInPanelRC``
            is not specified, by default the position of the first element in the specified panel is returned.

        Returns
        -------
        NumPy array
            An array of 3 values (x, y, and z) representing the position of the specified element. Note that the
            values are in multiples of wavelength.
        """
        return self.getElement(panelRC,elementInPanelRC).posInArray
       
    # ******************************************************************************************************************
    def allPanels(self):
        r"""
        This is a generator function that can be used to iterate through all panels in this array.

        Yields
        ------
            The next :py:class:`AntennaPanel` object in this array.
        """
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                yield self.panels[ r ][ c ]

    # ******************************************************************************************************************
    @property
    def numPanels(self):    return np.prod(self.shape)
    
    # ******************************************************************************************************************
    def allElements(self, polarization=True):
        r"""
        This is a generator function that can be used to iterate through all elements of this array. For example, the
        following code prints the position of every element in this array:
        
        .. code-block::
        
            for element in myArray.allElements():
                print( element.position )


        This function uses the :py:meth:`AntennaPanel.allElements` to iterate through each panel.
        
        Parameters
        ----------
        polarization : bool
            If the panels of this array are dually polarized and this parameter is `True`, then all elements are
            included in the iteration. Otherwise, if ``polarization=False``, only the first element of the polarized
            pair of elements at each position is included in the iteration. If the panels of this array are singly
            polarized, this parameter is ignored.
            
        Yields
        ------
            The next :py:class:`AntennaElement` object in this array.
        """
        # Antenna element orders for a 2x2 array of 2x2 panels with dual polarization:
        #           1st Polarization                 2nd Polarization
        #         ---------------------            ---------------------
        #         |  10 11  |  14 15  |            |  26 27  |  30 31  |
        #         |  8  9   |  12 13  |            |  24 25  |  28 29  |
        #         ---------------------            ---------------------
        #         |  2  3   |  6  7   |            |  18 19  |  22 23  |
        #   z↑    |  0  1   |  4  5   |            |  16 17  |  20 21  |
        #   y→    ---------------------            ---------------------
        if polarization and (self.panels[0][0].polarization in "+x"):
            for panel in self.allPanels():
                for r in range(panel.shape[0]):
                    for c in range(panel.shape[1]):
                        yield panel.elements[r][c][0]

            for panel in self.allPanels():
                for r in range(panel.shape[0]):
                    for c in range(panel.shape[1]):
                        yield panel.elements[r][c][1]
        else:
            for panel in self.allPanels():
                for element in panel.allElements(False):
                    yield element

    # ******************************************************************************************************************
    def getAllPositions(self, polarization=True):
        r"""
        Returns the positions of all elements in this array as a 2-D NumPy array.

        Parameters
        ----------
        polarization : bool
            If the panels of this array are dually polarized and this parameter is `True`, then the positions of
            all elements are returned. Otherwise, if ``polarization=False``, only the position of the first element
            of the polarized pair of elements at each position is returned. If the panels of this array are singly
            polarized, this parameter is ignored.
            
        Returns
        -------
        NumPy array
            An ``n x 3`` NumPy array containing the positions of all ``n`` elements in this array.
        """
        return np.float64([e.posInArray for e in self.allElements(polarization)])

    # ******************************************************************************************************************
    def getNumElements(self): # Return total number of Antenna elements in all panels
        r"""
        Returns the total number of antenna elements in this array. It uses the :py:meth:`AntennaPanel.getNumElements`
        to get the number of elements in one panel (``Np``). Total number of elements in this array is then
        ``shape[0] x shape[1] * Np``.
        """
        return self.numPanels * self.panels[0][0].getNumElements()

    # ******************************************************************************************************************
    def showElements(self, maxSize=6.0, zeroTicks=False, title=None):
        r"""
        This is a visualization function that draws this antenna array using the `matplotlib` library.
        
        Parameters
        ----------
        maxSize : (float: 6.0)
            This parameter specifies how large the output image of this array should be. Depending on the number of
            antenna element/panel rows and columns in this array, the ``maxSize`` can specify the width or height of
            the resulting image.
        
        zeroTicks : bool
            If this is `True`, the zero positions on both axes are indicated by additional "ticks" to show
            the center of this array. Otherwise, the "ticks" on the horizontal and vertical axes are only at the
            locations of antenna elements.
            
        title : str or None
            If specified, this will be used as the title for the image created for this array. Otherwise, the title
            "Array Elements" is used.
        """
        import matplotlib.pyplot as plt
        s = self.shape*self.spacing
        figSize = [maxSize, maxSize*s[0]/s[1]] if s[0]<s[1] else [maxSize*s[1]/s[0], maxSize]
        plt.figure(figsize=figSize)

        for panel in self.allPanels():  panel.showElements("Array")
        
        plt.xlabel("$\\frac {Y}{\\lambda}$", size=15)
        plt.ylabel("$\\frac {Z}{\\lambda}$", size=15)
        plt.title("Array Elements" if title is None else title, size=20)
        yTicks = [self.getElementPosition((0,p),(0,e))[1] for p in range(self.shape[1])
                        for e in range(self.panels[0][p].shape[1]) ] + ([0] if zeroTicks else [])
        plt.xticks(sorted(yTicks), size=10)
        zTicks = [self.getElementPosition((p,0),(e,0))[2] for p in range(self.shape[0])
                        for e in range(self.panels[p][0].shape[0]) ] + ([0] if zeroTicks else [])
        plt.yticks(sorted(zTicks), size=10)
        plt.axis('equal')
