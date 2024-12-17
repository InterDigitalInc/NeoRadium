# Copyright (c) 2024 InterDigital AI Lab
"""
This file contains the implementation of Antenna Elements, Panels, and Arrays.
"""
# ****************************************************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------------------------------------
# 05/18/2023    Shahab Hamidi-Rad       First version of the file.
# ****************************************************************************************************************************************************
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import cm

# ****************************************************************************************************************************************************
# This file is based on 3GPP TR 38.901 V17.0.0 (2022-03)
# Other good reads:
# https://scholar.valpo.edu/engineering_oer/1/  (the last few chapters)
# https://www.antenna-theory.com

# ****************************************************************************************************************************************************
class AntennaBase:
    r"""
    This is the base class for all Antenna objects in **NeoRadium**. The
    classes :py:class:`AntennaElement`, :py:class:`AntennaArray`, and
    :py:class:`AntennaPanel` are all derived from this class.
    """
    # ************************************************************************************************************************************************
    def __init__(self, **kwargs):
        self.isElement = (self.__class__.__name__=='AntennaElement')

    # ************************************************************************************************************************************************
    def getMaxDim(self):
        # First get the diference between the 2 farthest elements, then return the maxumum value among all dimensions
        if self.isElement:      return 0
        return (self.getElementPosition(-1) - self.getElementPosition(0)).max()

    # ************************************************************************************************************************************************
    def anglesToNumpy(self, angle, minAngle=None, maxAngle=None):
        # Convertes/creates a numpy array of angle values based on the given arguments
        if angle is None:               angle = np.arange(minAngle,maxAngle)
        if type(angle) == np.ndarray:   return angle
        if type(angle) == list:         return np.float64(angle)
        if type(angle) == tuple:
            if angle[0]==angle[1]: angle = (angle[0], angle[0]+1)
            return np.float64(range(*angle))
        return np.float64([angle])

    # ************************************************************************************************************************************************
    def getNumElements(self):
        # This function returns the number of antenna element for the AntennaBase-driven classes.
        # This is overridden in panel and array classes
        return 1

    # ************************************************************************************************************************************************
    def getElementsDelays(self, theta, phi, frequency):
        # This function calculates the delay between different elements of a Panel or Array.
        # Currently this function is not used.
        if self.isElement:  raise ValueError("'getElementsDelays' should not be called on 'AntennaElement' objects!")
        
        𝜃 = theta.reshape(-1,1) *np.pi/180
        𝜑 = phi.reshape(1,-1)   *np.pi/180

        # This is a 3 x numTheta x numPhi matrix
        xyzFactors = -np.float64([ np.sin(𝝷) * np.cos(𝞅),
                                   np.sin(𝝷) * np.sin(𝞅),
                                   np.cos(𝝷) * np.ones_like(𝞅) ])
        # This is a numElements x 3 matrix
        elementPositions = self.getAllPositions()

        # This is a numElements x numTheta x numPhi matrix giving the delay for each element at each theta/phi combination
        delays = np.tensordot(elementPositions, xyzFactors, axes=1)/frequency
        return delays

    # ************************************************************************************************************************************************
    def getSteeringVector(self, theta, phi):
        r"""
        This method calculates the Steering vector (A.K.A. Array Response) of an Antenna
        Array or Antenna Panel for the given Azimuth and Zenith angles. Note that
        this function can only be called on the :py:class:`AntennaPanel` and
        :py:class:`AntennaArray` classes. An exception is thrown if it is called on
        :py:class:`AntennaElement` classes.
        
        Parameters
        ----------
        theta : numpy array
            A 1-D array of zenith angles in degrees. (between 0 and 180)
            
        phi: numpy array
            A 1-D array of azimuth angles in degrees. (between -180 and 180)
            
        Returns
        -------
        Numpy Array
            A 3-D complex numpy array containing steering vectors for every combination
            of `theta` and `phi`. The shape of the output is (numElements, numTheta, numPhi)
        """
        if self.isElement:  raise ValueError("'getSteeringVector' should not be called on 'AntennaElement' objects!")

        𝜃 = theta.reshape(-1,1) *np.pi/180
        𝜑 = phi.reshape(1,-1)   *np.pi/180

        xyzPhases = np.float64([ np.sin(𝝷) * np.cos(𝞅), np.sin(𝝷) * np.sin(𝞅), np.cos(𝝷) * np.ones_like(𝞅) ])   # Shape: 3 x numTheta x numPhi
        return np.exp( 2j * np.pi * np.tensordot(self.getAllPositions(), xyzPhases, axes=1))                    # Shape: numElements x numTheta x numPhi

    # ************************************************************************************************************************************************
    def getFieldPattern(self, theta=None, phi=None):
        r"""
        This method is used to calculate the field patterns around an antenna
        panel or array in the directions given by the arguments
        ``theta`` and ``phi``.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles
            between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the fields are calculated for all azimuth angles
            between -180 and 180 degrees.

        Returns
        -------
        Numpy Array
            A 3-D complex numpy array containing steering vectors for each combination
            of ``theta`` and ``phi``. The shape of the output is (numElements x numTheta x
            numPhi)
        """
        # Only used to calculater Directivity, We are interested in the power pattern so we ignore the polarization here.
        if self.isElement:  raise ValueError("'getFieldPattern' should not be called on 'AntennaElement' objects!")

        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)

        # We assume all elements have the same power pattern. (They may have different polarized field patterns)
        elementField = self.getElement(0).getField(theta, phi)                  # Field pattern for the first element (A numTheta x numPhi matrix)
        elementField *= self.taper                                              # Apply taper (Non-Scaler taper is not supported yet)
        
        stearingVector = self.getSteeringVector(theta, phi)                     # numElements x numTheta x numPhi matrix
        nEl,nTheta,nPhi = stearingVector.shape

        fieldPattern = ( elementField.reshape((1,nTheta,nPhi)) * stearingVector )    # Field pattern per element for the whole array (A numElements x numTheta x numPhi matrix)
        return fieldPattern

    # ************************************************************************************************************************************************
    def getRotationMatrix(self, orientation):
        r"""
        This method calculates the rotation matrix used to convert the
        coordinates from local to global system. For more information about the
        rotation matrix please refer to **3GPP TR 38.901 equation (7.1-4)**.
        
        Parameters
        ----------
        orientation : list or numpy array
            A list or numpy array containing the orientation angles :math:`\alpha`
            (bearing angle), :math:`\beta` (downtilt angle), and :math:`\gamma` (slant
            angle) in radians.

        Returns
        -------
        Numpy Array
            A 3x3 rotation matrix that is used to transform the local coordinates to
            global coordinates.
        """
        if not np.any(orientation): return np.eye(3)    # If all zeros, return Identity
        sinAlpha, sinBeta, sinGamma = np.sin(orientation)
        cosAlpha, cosBeta, cosGamma = np.cos(orientation)
        # See TR38.901 - Eq. 7.1-4
        return np.float64([[ cosAlpha*cosBeta, cosAlpha*sinBeta*sinGamma-sinAlpha*cosGamma, cosAlpha*sinBeta*cosGamma+sinAlpha*sinGamma ],
                           [ sinAlpha*cosBeta, sinAlpha*sinBeta*sinGamma+cosAlpha*cosGamma, sinAlpha*sinBeta*cosGamma-cosAlpha*sinGamma ],
                           [ -sinBeta,         cosBeta*sinGamma,                            cosBeta*cosGamma]])

    # ************************************************************************************************************************************************
    def getElementsFields(self, theta, phi, orientation=np.float64([0,0,0])):
        r"""
        This method calculates the fields used to calculate the channel response
        for CDL channel models. It returns polarized field values in the directions
        specified by the ``theta`` and ``phi`` values. This function also handles the
        conversion from local to global coordinates using the provided rotation
        angles. Please refer to **3GPP TR 38.901 sections 7.1 and 7.5** for more
        details.

        Parameters
        ----------
        theta : numpy array
            A 2-D numpy array containing the zenith angles (in radians) used to
            calculate the fields. For CDL channels, this is an ``n`` by ``m``
            matrix where ``n`` is the number of clusters and ``m`` is the number
            of rays per cluster.

        phi : numpy array
            A 2-D numpy array containing the azimuth angles (in radians) used to
            calculate the fields. For CDL channels, this is an ``n`` by ``m``
            matrix where ``n`` is the number of clusters and ``m`` is the number
            of rays per cluster.

        orientation : list or numpy array
            A list or numpy array containing the orientation angles :math:`\alpha`
            (bearing angle), :math:`\beta` (downtilt angle), and :math:`\gamma` (slant
            angle) in radians.

        Returns
        -------
        2 Numpy Arrays
            * **field**:
                A numpy array of shape (numAntenna x 2 x n x m) containing
                the field information for each antenna element and each one of ``m`` rays
                in each one of ``n`` clusters. The second dimension (2) is used to separate
                the vertical and horizontal polarization.

            * **locFactor**:
                A numpy array of shape (numAntenna x n x m) containing
                the location factor. For more information please refer to **3GPP TR 38.901
                equation 7.5-22**.
        """
        # This is called by the CDL channel model. theta and phi are n x m matrixes of path angles of arrival and departure.
        n, m = theta.shape
        sinTheta, cosTheta = np.sin(theta), np.cos(theta)
        sinPhi, cosPhi = np.sin(phi), np.cos(phi)
        
        # The spherical unit vector at theta,phi (See TR38.901 - Eq. 7.5-23 and 7.5-24). This is a 3 x n x m matrix
        rHat = np.array([ sinTheta * cosPhi,
                          sinTheta * sinPhi,
                          cosTheta ])

        r = self.getRotationMatrix(orientation)        # The rotation matrix. A 3x3 matrix

        # Cartesian Representation (See TR38.901 - Eq. 7.1-6). This is a 3 x n x m matrix
        rhoHat = rHat   # This is the same as rHat already calculated above

        # Calculating local theta values (See TR38.901 - Eq. 7.1-7). This is an n x m matrix (just like the global theta values)
        thetaLocal = np.arccos( (r[:,2].reshape(-1,1,1) * rHat).sum(0) )*180.0/np.pi

        # Calculating local phi values (See TR38.901 - Eq. 7.1-8). This is an n x m matrix (just like the global phi values)
        phiLocal  = np.arctan2( (r[:,1].reshape(-1,1,1) * rhoHat).sum(0),
                                (r[:,0].reshape(-1,1,1) * rhoHat).sum(0) )*180/np.pi
        # Phi does not make sense when theta is 0 or 180
        phiLocal[thetaLocal==0] = 0
        phiLocal[thetaLocal==180] = 0

        # Global unit vectors in theta and phi directions. (These are 3 x n x m tensors)
        thetaHat = np.float64([ cosTheta * cosPhi, cosTheta * sinPhi, -sinTheta ])      # See TR38.901 - Eq. 7.1-13
        phiHat = np.float64([ -sinPhi, cosPhi, np.zeros_like(cosPhi) ])                 # See TR38.901 - Eq. 7.1-14

        # Local unit vectors in theta directions. (A 3 x n x m tensor)
        cosThetaLocal = np.cos(thetaLocal*np.pi/180)
        thetaHatLocal = np.float64([ cosThetaLocal * np.cos(phiLocal*np.pi/180),
                                     cosThetaLocal * np.sin(phiLocal*np.pi/180),
                                     -np.sin(thetaLocal*np.pi/180) ])   # See TR38.901 - Eq. 7.1-13 (Applied to local phi and theta)

        # Calculating psi - the angular displacement between pairs of global and local unit vectors (See TR38.901 - Eq. 7.1-12)
        psi = np.arctan2( (phiHat.reshape(3,-1)   * r.dot(thetaHatLocal.reshape(3,-1))).sum(0),
                          (thetaHat.reshape(3,-1) * r.dot(thetaHatLocal.reshape(3,-1))).sum(0) ).reshape(phiHat.shape[1:])

        # The elements fields before polarization are the same for all elements. We create the fields for one element and broadcast it with all polarizations
        fieldsLocal = self.getElement(0).getField(thetaLocal, phiLocal)         # Local field pattern for the first element (n x m)
        
        # This is using Model-2 for polarized antenna (See See TR38.901 - Section 7.3.2)
        # TODO: Do this based on the polModel value. (See "getPolarizedFields" function)
        zeta = np.float64([e.polAngle for e in self.allElements()])*np.pi/180   # All zetas (An array of size numAntenna)
        fThetaLocal = fieldsLocal.reshape(1,n,m) * np.cos(zeta).reshape(-1,1,1) # Local vertical field pattern for all elements (numAntenna x n x m)
        fPhiLocal   = fieldsLocal.reshape(1,n,m) * np.sin(zeta).reshape(-1,1,1) # Local horizontal field pattern for all elements (numAntenna x n x m)

        # Global fields for all elements. (A numAntenna x 2 x n x m tensor)
        field = np.stack((fThetaLocal*np.cos(psi) - fPhiLocal*np.sin(psi),              # Global vertical field pattern for all elements
                          fThetaLocal*np.sin(psi) + fPhiLocal*np.cos(psi)), axis=1)     # Global horizontal field pattern for all elements

        positions = self.getAllPositions()                          # Local antenna positions. A numAntenna x 3 matrix
        positions = positions.dot(r.T).T                            # Global antenna positions. A 3 x numAntenna matrix
        
        # The location terms in TR38.901 - Eq. 7.5-22
        # Note that we simplified the location terms and using lambda based positions instead of "dBar" devided by "lambda"
        locFactor = np.exp(2j * np.pi * (rHat.reshape(3,1,n,m) * positions.reshape(3,-1,1,1)).sum(0))  # A numAntenna x n x m tensor
        
        return field, locFactor

    # ************************************************************************************************************************************************
    def getPolarizedFields(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the polarized fields and outputs 2 matrices for
        the field values for vertical and horizontal polarizations.
        
        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles
            between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the fields are calculated for all azimuth angles
            between -180 and 180 degrees.

        weights: numpy array
            A vector of weights to be applied to the field values. The weights
            can be used to steer the beams to the desiered direction. If this is
            ``None``, the field pattern is returned without any beamforming.

        Returns
        -------
        2 Numpy Arrays
            * **arrayFieldV**:
                A numpy array of shape (numElements x numTheta x numPhi) containing
                the field values with vertical polarization at the directions
                specified by ``theta`` and ``phi``.

            * **arrayFieldH**:
                A numpy array of shape (numElements x numTheta x numPhi) containing
                the field values with horizontal polarization at the directions
                specified by ``theta`` and ``phi``.
        """
        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)

        stearingVector = self.getSteeringVector(theta, phi)     # numElements x numTheta x numPhi matrix
        nEl,nTheta,nPhi = stearingVector.shape
        
        elementFieldV, elementFieldH = self.getElement(0,p=0).getPolarizedFields(theta, phi)
        elementFieldV = elementFieldV.reshape(nTheta,nPhi) * self.taper     # numTheta x numPhi
        elementFieldH = elementFieldH.reshape(nTheta,nPhi) * self.taper     # numTheta x numPhi
        if self.polarization in "+x":
            # The panel contains antenna with different polarizations. We need to get samples for both polarizations
            elementFieldVP2, elementFieldHP2 = self.getElement(0,p=1).getPolarizedFields(theta, phi)
            elementFieldVP2 = elementFieldVP2.reshape(nTheta,nPhi) * self.taper     # numTheta x numPhi
            elementFieldHP2 = elementFieldHP2.reshape(nTheta,nPhi) * self.taper     # numTheta x numPhi
            
            elementFieldV = np.array((nEl//2)*[elementFieldV] + (nEl//2)*[elementFieldVP2])
            elementFieldH = np.array((nEl//2)*[elementFieldH] + (nEl//2)*[elementFieldHP2])
        else:
            elementFieldV = np.array(nEl*[elementFieldV])       # Repeat nEl times (numElements x numTheta x numPhi)
            elementFieldH = np.array(nEl*[elementFieldH])       # Repeat nEl times (numElements x numTheta x numPhi)

        elementsFieldV = elementFieldV * stearingVector         # Steered Vertical Field pattern per element for the whole array (A numElements x numTheta x numPhi matrix)
        elementsFieldH = elementFieldH * stearingVector         # Steered Horizontal Field pattern per element for the whole array (A numElements x numTheta x numPhi matrix)

        if weights is not None:
            if len(weights)!=nEl:  raise ValueError( "'weights' must be a %d-dimensional vector!"%(nEl) )
            elementsFieldV *= weights.reshape((-1,1,1))
            elementsFieldH *= weights.reshape((-1,1,1))

        arrayFieldV = np.squeeze(elementsFieldV.sum(axis=0))
        arrayFieldH = np.squeeze(elementsFieldH.sum(axis=0))
        return arrayFieldV, arrayFieldH

    # ************************************************************************************************************************************************
    def getField(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the fields in directions specified by
        ``theta`` and ``phi``. It calls the :py:meth:`getPolarizedFields` method
        to get the vertical and horizontal polarized fields and combines them
        to get fields at the specified directions.

        .. math::

            F = \sqrt {F_v^2 + F_h^2}

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles
            between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the fields are calculated for all azimuth angles
            between -180 and 180 degrees.

        weights: numpy array
            A vector of weights to be applied to the field values. The weights
            can be used to steer the beams to the desiered direction. If this is
            ``None``, the field pattern is returned without any beamforming.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numElements x numTheta x numPhi) containing
            the field values at the directions specified by ``theta`` and ``phi``.
        """
        arrayFieldV, arrayFieldH = self.getPolarizedFields(theta, phi, weights)
        return np.hypot(np.abs(arrayFieldV),np.abs(arrayFieldH))
    
    # ************************************************************************************************************************************************
    def getPowerPattern(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the field power pattern in the directions specified
        by ``theta`` and ``phi``. It calls the :py:meth:`getField` method
        to get the fields then calculates the field powers.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the field power is calculated only
            for the single specified zenith angle (in degrees)

            If this is None, the field powers are calculated for all zenith
            angles between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the field power is calculated only
            for the single specified azimuth angle (in degrees)

            If this is None, the field powers are calculated for all azimuth
            angles between -180 and 180 degrees.

        weights: numpy array
            A vector of weights to be applied to the field values. The weights
            can be used to steer the beams to the desiered direction. If this is
            ``None``, the field pattern is returned without any beamforming.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numElements x numTheta x numPhi) containing
            the field powers at the directions specified by ``theta`` and ``phi``.
        """
        arrayField = self.getField(theta, phi, weights)
        power = np.square(np.abs(arrayField))
        return power

    # ************************************************************************************************************************************************
    def getPowerPatternDb(self, theta=None, phi=None, weights=None):
        r"""
        This method calculates the field power pattern (in dB) in the
        directions specified by ``theta`` and ``phi``. It calls the
        :py:meth:`getPowerPattern` method to get the field powers and
        then converts them to dB.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the field power is calculated only
            for the single specified zenith angle (in degrees)

            If this is None, the field powers are calculated for all zenith
            angles between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the field power is calculated only
            for the single specified azimuth angle (in degrees)

            If this is None, the field powers are calculated for all azimuth
            angles between -180 and 180 degrees.

        weights: numpy array
            A vector of weights to be applied to the field values. The weights
            can be used to steer the beams to the desiered direction. If this is
            ``None``, the field pattern is returned without any beamforming.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numElements x numTheta x numPhi) containing
            the field powers in dB at the directions specified by ``theta`` and ``phi``.
        """
        power = self.getPowerPattern(theta, phi, weights)
        power = np.maximum(1e-12, power)    # Make sure no zeros in power
        return 10*np.log10(power)           # Return the power in db

    # ************************************************************************************************************************************************
    def getIntegralAngleStep(self):
        # This function returns the angle step for the integral used to calculate the
        # directivity. See the getDirectivity function below.
        maxSpan = self.getMaxDim()  # Get the farthest distance between antenna elements
        if maxSpan == 0:    return 1
        
        # Using the approximation: beamWidth = 70 * wavelength / D
        # The maxSpan above is: D/wavelength, so:
        beamWidth = 70/maxSpan
        angleStep = beamWidth               # Making sure we have at least 3 angle steps per beamWidth
        
        # Pick one of 1, 0.5, 0.2, or 0.1 for the step. This makes it easier to handle the range of angles.
        if angleStep>=1:     return 1
        if angleStep>=0.5:   return 0.5
        if angleStep>=0.2:   return 0.2
        return 0.1

    # ************************************************************************************************************************************************
    def getDirectivity(self, theta=None, phi=None, weights=None):
        r"""
        Directivity at a specific direction is defined as:
        
        .. math::

            D = \frac {P} {P_{avg}}

        where :math:`P` is the power radiated at the specified angle and
        :math:`P_{avg}` is the average power radiated in all directions. The
        average power is calculated by integrating the field values at all
        angles: (See
        `this web page <https://www.antenna-theory.com/basics/directivity.php>`_
        for more details)
        
        .. math::

            P_{avg} = \frac {1} {4 \pi} \int_0^{2 \pi} \int_0^{\pi} |F(\theta, \phi)|^2 \sin \theta d\theta d\phi

        
        Directivity (without any specific direction) is defined as:
        
        .. math::

            D_{max} = \frac {P_{max}} {P_{avg}}
            
        where :math:`P_{max}` is maximum power radiated at a direction. Directivity
        is usually measured in dbi which is the relative directivity in db with
        respect to an "isotropic" radiator.
                
        This method calculates the directivity (in dbi) at directions specified by
        ``theta`` and ``phi``.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the directivity.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the directivity is calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the directivity is calculated for all
            zenith angles between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the directivity.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the directivity is calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the directivity is calculated for all
            azimuth angles between -180 and 180 degrees.

        weights: numpy array
            A vector of weights to be applied to the field values. The weights
            can be used to steer the beams to the desiered direction. If this is
            ``None``, the field pattern is returned without any beamforming.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numElements x numTheta x numPhi) containing
            the directivity in dbi at the directions specified by ``theta`` and ``phi``.
        """
        # Directivity:
        #   AKA Directive Gain
        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)

        elementsField = self.getFieldPattern(theta, phi)   # Fields for each element (A numElements x numTheta x numPhi matrix)

        # Now we calculate "Directivity" based on the formula here: https://www.antenna-theory.com/basics/directivity.php
        # To do that, we first need to calculate the average power in all directions which is the denominator integral in the
        # directivity formula.
        angleStep = self.getIntegralAngleStep()
        allTheta = np.arange(0, 180+angleStep, angleStep)
        allPhi = np.arange(-180, 180+angleStep, angleStep)
        
        if (allTheta.shape != theta.shape) or (allPhi.shape != phi.shape):  elementsFieldAllD = self.getFieldPattern(allTheta, allPhi)
        elif np.any(allTheta!=theta) or np.any(allPhi!=phi):                elementsFieldAllD = self.getFieldPattern(allTheta, allPhi)
        else:                                                               elementsFieldAllD = elementsField # elementsField already calculated

        # Shape of elementsFieldAllD: numElements x numTheta x numPhi
        n = elementsFieldAllD.shape[0]

        # Now calculating steering vector covariance matrix: (A numElements x numElements matrix)
        svCov = (elementsFieldAllD * np.sin(allTheta*np.pi/180).reshape(1,-1,1)).reshape(n,-1).dot(elementsFieldAllD.reshape(n,-1).conj().T)
        
        dTheta = dPhi = angleStep*np.pi/180
        if weights is not None:
            w = weights.reshape(1, n)
            integral = (w.dot(svCov).dot(w.T)*dTheta*dPhi)[0,0].real
            elementsField *= weights.reshape((-1,1,1))
        else:
            integral = svCov.real.sum()*dTheta*dPhi

        arrayField = elementsField.sum(axis=0)
        arrayPower = np.squeeze(np.square(np.abs(arrayField)))
        directivity = 4*np.pi*arrayPower/integral       # Note that since totalPower is not normalized we have it in the numerator of "Directivity" formula instead of 1
        directivity = np.maximum(1e-12, directivity)    # Make sure no zeros in Directivity so the log below works.
        directivityDbi = 10*np.log10(directivity)       # convert to "dbi", which is db with respect to "isotropic" radiator
        return directivityDbi

    # ************************************************************************************************************************************************
    def drawRadiation(self, theta=None, phi=None, radiationType="Directivity", normalize=True, title=None, viewAngles=(45,20), figSize=6.0):
        r"""
        This is a multi-purpose visualization function that shows the
        radiation around antenna elements, panels, and arrays in the directions
        specified by ``theta`` and ``phi``.
        
        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to visualize the radiations.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the radiations are visualized only for
            the single specified zenith angle (in degrees)

            If this is None, the radiations are visualized for all zenith
            angles between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to visualize the radiations.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the radiations are visualized only for
            the single specified azimuth angle (in degrees)

            If this is None, the radiations are visualized for all azimuth
            angles between -180 and 180 degrees.

        radiationType: str (default: "Directivity")
            This parameter specifies the type of radiation to plot. Here is a
            list of supported values:
                
                * **Directivity**
                * **Power**
                * **PowerDb**
                * **Field**
            
        normalize: Boolean (default: True)
            If ``True`` (default) all the values are normalized before being
            plotted.

        title: str (default: None)
            The title to be used for the plot. If not specified, then this function
            creates a title based on the given parameters.
            
        viewAngles: tuple (default: (45,20))
            For 3-D plots, you can use this parameter to specify your
            desired viewing angle. For non-3D plots, this parameter is ignored.
            
        figSize: float (default: 6.0)
            The figure size. Use this to control size of the plot.
            
        Returns
        -------
        Numpy Array
            A numpy array containing the actual data used for the visualization.


        Plot Types:
        ^^^^^^^^^^^
        Horizontal Cut at specified elevation
            For this case specify one ``theta`` value and include all azimuth
            angles (:math:`-\pi < \phi < \pi`). One common use case is the
            horizontal cut at zero elevation (:math:`\theta = \pi / 2`).

        Vertical Cut at specified azimuth
            For this case specify one ``phi`` value and include all zenith
            angles (:math:`0 < \theta < \pi`). One common use case is the
            vertical cut at zero azimuth (:math:`\phi = 0`).
            
        3-D pattern
            For this case specify the complete range for both ``theta`` and
            ``phi`` (:math:`0 < \theta < \pi` and :math:`-\pi < \phi < \pi`).
        """
        theta = self.anglesToNumpy(theta,0,180)
        phi   = self.anglesToNumpy(phi,-180,180)
                
        if radiationType=="Directivity":
            radValues = self.getDirectivity(theta, phi)
        elif radiationType=="PowerDb":
            radValues = self.getPowerPatternDb(theta, phi)
            if normalize:   radValues -= radValues.max()
        elif radiationType=="Power":
            radValues = self.getPowerPattern(theta, phi)
            if normalize:   radValues /= radValues.max()
        elif radiationType=="Field":
            radValues = self.getField(theta, phi)
            radValues = np.abs(radValues)   # We want to draw the magnitude of field
            if normalize:   radValues /= radValues.max()
        else:
            raise ValueError( "Unsupported 'radiationType' value \"%s\"!"%(radiationType) )

        # For logarithmic values limit the range otherwise the plot looks wierd! In this case we push all small values to the center for
        # the 2-D polar graphs.
        radRange = radValues.max() - radValues.min()
        if radiationType in ["Directivity", "PowerDb"]:
            plotValues = np.maximum(radValues, radValues.max()-60)
            plotRange = plotValues.max() - plotValues.min()
            plotMin = plotValues.min() if radRange > 60 else (plotValues.min()-plotRange/20)
        else:
            plotMin = 0
            plotValues = radValues
            plotRange = radRange

        if title is None:
            radTypeStr = {"Directivity":"Directivity", "PowerDb":"Radiation Power (db)", "Power":"Radiation Power", "Field": "Electric Field"}[radiationType]
            if normalize and (radiationType!="Directivity"):    radTypeStr = "Normalized "+radTypeStr
            if len(theta)==1:
                if theta[0]==90:    title = "Horizontal Cut of %s at zero elevation ($\\theta=\\pi/2$)"%(radTypeStr)
                else:               title = "Horizontal Cut of %s at $\\theta$=%d°"%(radTypeStr, int(theta[0]))

            elif len(phi)==1:
                if phi[0]==0:       title = "Vertical Cut of %s at zero azimuth ($\\phi=0$)"%(radTypeStr)
                else:               title = "Vertical Cut of %s at $\\phi$=%d°"%(radTypeStr, int(phi[0]))

            else:
                if max(theta)>=179 and min(theta)==0 and max(phi)>=179 and min(phi)==-180:
                    title = radTypeStr
                else:
                    title = "%s for %d°$\\leq\\theta\\leq$%d° and %d°$\\leq\\phi\\leq$%d°"%(radTypeStr, theta[0], theta[-1], phi[0], phi[-1])

        
        fig = plt.figure(figsize=(figSize,figSize))
        if len(theta)==1:
            plt.polar(phi*np.pi/180, plotValues)
            fig.axes[0].set_ylim(plotMin,plotValues.max())
#            if radiationType in ["Directivity", "PowerDb"]: fig.axes[0].set_ylim(plotValues.min()-plotRange/20,plotValues.max())
#            else:                                           fig.axes[0].set_ylim(0, plotValues.max())
            plt.title(title, size=16)
            plt.show()
            return radValues

        if len(phi)==1:
            plt.polar(theta*np.pi/180, plotValues)
            fig.axes[0].set_theta_zero_location("N")
            fig.axes[0].set_theta_direction(-1)
            fig.axes[0].set_thetamin(0)
            fig.axes[0].set_thetamax(180)
            fig.axes[0].set_ylim(plotMin,plotValues.max())
#            if radiationType in ["Directivity", "PowerDb"]: fig.axes[0].set_ylim(plotValues.min()-plotRange/20,plotValues.max())
#            else:                                           fig.axes[0].set_ylim(0, plotValues.max())
            plt.title(title, size=16)
            plt.show()
            return radValues

        # Now doing surface plot
        if type(viewAngles)!=tuple: raise ValueError( "'viewAngles' must be a tuple!" )
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=viewAngles[1], azim=viewAngles[0])

        # For logarithmic values, we shift values so that the radius of minimum value in polar coordinates is 10% of the range of all values.
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
                          facecolors=cm.winter(r), alpha=0.9,
                          rstride=1, cstride=1, linewidth=0, antialiased=False)

        # Draw our own axis:
        plt.axis('off')
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
                
        plt.title(title)
        plt.show()
        
        return radValues

# ****************************************************************************************************************************************************
class AntennaElement(AntennaBase):
    r"""
    This class implements the functionality of an antenna element. This 
    implementation is based on **3GPP TR 38.901 section 7.3**.
    """
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        kwargs : dict
            A set of optional arguments. If you are creating a single antenna
            element object, most of the time you don't need to specify any
            parameters and the default values are enough for normal functionality.
            Here is a list of supported parameters:

                * **position**: A list of 3 values (x, y, and z) specifying the position of this element in the :py:class:`AntennaPanel` containing this element.
                * **freqRange**: A list of 2 values specifying the range of frequencies in which this antenna element operates.
                * **polAngle**: The polarization angle of this antenna element in degrees. A value of 0° means it is purely vertically polarized.
                * **polModel**: The polarization model (1 or 2). The default is 2. Please refer to **TR38.901-Section 7.3.2** for more details.
                * **beamWidth**: A list of 2 values specifying the beam width of this antenna element in degrees. The default is ``[65,65]``. These values correspond to :math:`\theta_{3dB}` and :math:`\phi_{3dB}` in **TR38.901-Table 7.3-1**.
                * **verticalSidelobeAttenuation**: Vertical side-lobe attenuation (:math:`SLA_V`). The default is 30. Please refer to **TR38.901-Table 7.3-1** for more details.
                * **maxAttenuation**: Maximum Attenuation (:math:`A_{max}`) in dB. The default is 30. Please refer to **TR38.901-Table 7.3-1** for more details.
                * **mainMaxGain**: Maximum gain of main lobe in dBi. The default is 8. Please refer to **TR38.901-Table 7.3-1** for more details.
                * **panel**: The :py:class:`AntennaPanel` object containing this element.
        """
        super().__init__(**kwargs)
        self.position = np.float64(kwargs.get('position', [0,0,0]))     # Position in the container (i.e. panel/array)
        self.freqRange = kwargs.get('freqRange', [0,100e9])             # Lower and upper bound of frequency
        self.polAngle = kwargs.get('polAngle', 0)                       # Polarization slant angle in degrees (0° => Purely vertically polarized)
        self.polModel = kwargs.get('polModel', 2)                       # Polarization model (1 or 2). (TR38.901-Section 7.3.2)
        self.beamWidth = kwargs.get('beamWidth', [65,65])               # 3dB beam width angle in degrees [theta, phi]. (TR38.901-Table 7.3-1)
        self.verticalSidelobeAttenuation = kwargs.get('verticalSidelobeAttenuation', 30)    # Vertical side-lobe attenuation (SLAv) (TR 38.901-Table 7.3-1)
        self.maxAttenuation = kwargs.get('maxAttenuation', 30)                              # Maximum Attenuation (Amax) in dB (TR38.901-Table 7.3-1)
        self.mainMaxGain = kwargs.get('mainMaxGain', 8)                                     # Maximum gain of main lobe in dBi (TR38.901-Table 7.3-1)
        self.panel = kwargs.get('panel', None)                                              # Owner

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`AntennaElement` object.

        Parameters
        ----------
        indent : int (default: 0)
            The number of indentation characters.
            
        title : str (default: None)
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
        if title is None:   title = "Antenna Element:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        if self.panel is not None:
            repStr += indent*' ' + "  position: %d\n"%(str(self.position))
        repStr += indent*' ' + "  freqRange: %s..%s Hz\n"%(str(self.freqRange[0]), str(self.freqRange[1]))
        repStr += indent*' ' + "  polAngle: %s°\n"%(str(self.polAngle))
        repStr += indent*' ' + "  polModel: %s\n"%(str(self.polModel))
        repStr += indent*' ' + "  beamWidth: %s°,%s°\n"%(str(self.beamWidth[0]), str(self.beamWidth[1]))
        repStr += indent*' ' + "  verticalSidelobeAttenuation: %s\n"%(str(self.verticalSidelobeAttenuation))
        repStr += indent*' ' + "  maxAttenuation: %s dB\n"%(str(self.maxAttenuation))
        repStr += indent*' ' + "  mainMaxGain: %s dBi\n"%(str(self.mainMaxGain))
        if getStr: return repStr
        print(repStr)
            
    # ************************************************************************************************************************************************
    @property
    def posInArray(self):
        r"""
        Return the position of this element in the :py:class:`AntennaArray` object.

        Returns
        -------
        Numpy array
            An array of 3 values (x, y, and z) specifying the position of this element
            in the :py:class:`AntennaArray` object.
        """
        return self.position + self.panel.position
    
    # ************************************************************************************************************************************************
    def clone(self, position, polAngle, panel):
        r"""
        Creates a copy of this :py:class:`AntennaElement` object and modifies the
        ``position``, polarization angle (``polAngle``), and the ``panel`` object
        based on the parameters provided.
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

    # ************************************************************************************************************************************************
    def verticalRadiationPower(self, theta=None):       # See TR38.901-Table 7.3-1 (1st row)
        # theta must be in [0, 180]. This is the 𝜃" in TR38.901-Table 7.3-1
        theta = self.anglesToNumpy(theta,0,181)
        # The calculation can be done in degrees because we are only calculating the ratios
        return -np.minimum(12*np.square((theta-90)/self.beamWidth[0]), self.verticalSidelobeAttenuation)
        
    # ************************************************************************************************************************************************
    def horizonRadiationPower(self, phi=None):          # See TR38.901-Table 7.3-1 (2nd row)
        # phi must be in [-180, 180]. This is the 𝜙" in TR38.901-Table 7.3-1
        phi = self.anglesToNumpy(phi,-180,180)
        # The calculation can be done in degrees because we are only calculating the ratios
        return -np.minimum(12*np.square(phi/self.beamWidth[1]), self.maxAttenuation)

    # ************************************************************************************************************************************************
    def getPowerPatternDb(self, theta=None, phi=None):  # See TR38.901-Table 7.3-1
        r"""
        This method calculates the field power pattern (in dB) in the
        directions specified by ``theta`` and ``phi``. This function is
        implemented based on **TR38.901-Table 7.3-1**.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the field power is calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the field powers are calculated for all zenith
            angles between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the field power is calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the field powers are calculated for all azimuth
            angles between -180 and 180 degrees.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numTheta x numPhi) containing
            the field powers in dB at the directions specified by
            ``theta`` and ``phi``.
        """
        theta = self.anglesToNumpy(theta,0,180)
        phi = self.anglesToNumpy(phi,-180,180)

        # This returns sum of the 3rd and 4th rows in TR38.901-Table 7.3-1
        if len(theta.shape)==1 and len(phi.shape)==1 and len(theta)!=len(phi):
            # Need to broadcast. The output will be a len(theta) x len(phi) matrix
            radPower = -np.minimum(-(self.verticalRadiationPower(theta).reshape(-1,1) +
                                   self.horizonRadiationPower(phi).reshape(1,-1)), self.maxAttenuation) + self.mainMaxGain
        else:
            # In this case, theta, phi, and the output have the same shape.
            radPower = -np.minimum(-(self.verticalRadiationPower(theta) + self.horizonRadiationPower(phi)), self.maxAttenuation) + self.mainMaxGain
        
        return np.float64(np.squeeze(radPower))

    # ************************************************************************************************************************************************
    def getPowerPattern(self, theta=None, phi=None):
        r"""
        This method calculates the field power pattern in the
        directions specified by ``theta`` and ``phi``. This function
        calls the :py:meth:`AntennaElement.getPowerPatternDb` and
        converts the results from dB to linear representation.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the field power is calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the field powers are calculated for all zenith
            angles between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the field powers.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the field power is calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the field powers are calculated for all azimuth
            angles between -180 and 180 degrees.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numTheta x numPhi) containing
            the field powers at the directions specified by
            ``theta`` and ``phi``.
        """
        return 10**(self.getPowerPatternDb(theta, phi)/10)

    # ************************************************************************************************************************************************
    def getField(self, theta=None, phi=None):
        r"""
        This method calculates the fields in directions specified by
        ``theta`` and ``phi``. It calls the
        :py:meth:`AntennaElement.getPowerPatternDb` and converts the
        results to field values.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles
            between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the fields are calculated for all azimuth angles
            between -180 and 180 degrees.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numElements x numTheta x numPhi) containing
            the field values at the directions specified by ``theta``
            and ``phi``.
        """
        # This assumes polarization angle of 0 (Pure vertical). In this case vertical(Zenith) field=sqrt(power)
        # and horizontal(Azimuth) field=0. If polarization angle is not zero, use "getPolarizedFields" function below.
        return 10**(self.getPowerPatternDb(theta, phi)/20)

    # ************************************************************************************************************************************************
    def getPolarizedFields(self, theta, phi):
        r"""
        This method calculates the polarized fields and outputs 2 matrices for
        the field values for vertical and horizontal polarizations.
        
        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the fields are calculated for all zenith angles
            between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the fields.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the fields are calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the fields are calculated for all azimuth angles
            between -180 and 180 degrees.

        Returns
        -------
        2 Numpy Arrays
            * **arrayFieldV**:
                A numpy array of shape (numTheta x numPhi) containing the field
                values with vertical polarization at the directions
                specified by ``theta`` and ``phi``.

            * **arrayFieldH**:
                A numpy array of shape (numTheta x numPhi) containing the field
                values with horizontal polarization at the directions
                specified by ``theta`` and ``phi``.
        """
        field = self.getField(theta, phi)
        𝜁 = self.polAngle*np.pi/180         # Zeta: Polarization Angle in Radians

        if self.polModel == 1:
            # Model-1:
            # First calculate cos𝜓 and sin𝜓 (See TR38.901-Eq. 7.3-3)
            if self.polAngle == 0:              cos𝜓, sin𝜓 = 1, 0
            elif self.polAngle in [180, -180]:  cos𝜓, sin𝜓 = -1, 0
            else:
                # See the "Model-1 in TR38.901-Sec. 7.3.2
                𝜃 = theta.reshape(-1,1) *np.pi/180
                𝜑 = phi.reshape(1,-1)   *np.pi/180
                denom = np.sqrt(1-np.square(np.cos(𝜁)*np.cos(𝜃)-np.sin(𝜁)*np.sin(𝜑)*np.sin(𝜃)))
                cos𝜓 = (np.cos(𝜁)*np.sin(𝜃) + np.sin(𝜁)*np.sin(𝜑)*np.cos(𝜃))/denom
                sin𝜓 = np.sin(𝜁)*np.cos(𝜑)/denom
                
            fTheta = field * cos𝜓
            fPhi = field * sin𝜓
        else:
            # Model-2:
            fTheta = field * np.cos(𝜁)
            fPhi = field * np.sin(𝜁)
        
        return fTheta, fPhi

    # ************************************************************************************************************************************************
    def draw(self, ref="Array"):
        # This is called by panel or array objects to draw this element.
        pos = self.position + (0 if ref=="Panel" else self.panel.position)
        points = { '|': np.float64([[0,1], [0,-1]]),
                   '-': np.float64([[-1,0], [1,0]]),
                   '+': np.float64([[-1,0], [1,0], [0,0], [0,-1], [0,1]]),
                   'x': np.float64([[-1,-1], [1,1], [0,0], [-1,1], [1,-1]]) }
        
        markerScale = 0.1
        elementPoints = pos[1:] + min(self.panel.spacing) * markerScale * points[self.panel.polarization]
        plt.plot(elementPoints[:,0], elementPoints[:,1], color="red", linewidth=1)

    # ************************************************************************************************************************************************
    def getDirectivity(self, theta=None, phi=None, weights=None):
        r"""
        Directivity at a specific direction is defined as:
        
        .. math::

            D = \frac {P} {P_{avg}}

        where :math:`P` is the power radiated at the specified angle and
        :math:`P_{avg}` is the average power radiated in all directions. The
        average power is calculated by integrating the field values at all
        angles: (See
        `this web page <https://www.antenna-theory.com/basics/directivity.php>`_
        for more details)
        
        .. math::

            P_{avg} = \frac {1} {4 \pi} \int_0^{2 \pi} \int_0^{\pi} |F(\theta, \phi)|^2 \sin \theta d\theta d\phi

        
        Directivity (without any specific direction) is defined as:
        
        .. math::

            D_{max} = \frac {P_{max}} {P_{avg}}
            
        where :math:`P_{max}` is maximum power radiated at a direction. Directivity
        is usually measured in dbi which is the relative directivity in db with
        respect to an "isotropic" radiator.
                
        This method calculate the directivity (in dbi) at directions specified by
        ``theta`` and ``phi``.

        Parameters
        ----------
        theta : list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the zenith angles
            (in degrees) used to calculate the directivity.

            If this is a tuple, the values are assumed to specify the range
            of values used for zenith angles (in degrees)

            If this is a scaler value, the directivity is calculated only for
            the single specified zenith angle (in degrees)

            If this is None, the directivity is calculated for all zenith angles
            between 0 and 180 degrees.

        phi: list, tuple, numpy array, scaler, or None
            If this is a list or numpy array, it specifies the azimuth angles
            (in degrees) used to calculate the directivity.

            If this is a tuple, the values are assumed to specify the range
            of values used for azimuth angles (in degrees)

            If this is a scaler value, the directivity is calculated only for
            the single specified azimuth angle (in degrees)

            If this is None, the directivity is calculated for all azimuth angles
            between -180 and 180 degrees.

        weights: numpy array
            This parameter is not used for :py:class:`AntennaElement` objects.

        Returns
        -------
        Numpy Array
            A numpy array of shape (numTheta x numPhi) containing
            the directivity in dbi at the directions specified by
            ``theta`` and ``phi``.
        """
        theta = self.anglesToNumpy(theta,0,180)
        phi = self.anglesToNumpy(phi,-180,180)

        totalPower = self.getPowerPattern(theta, phi)       # Field Powers (A numTheta x numPhi matrix)

        # Now we calculate "Directivity" based on the formula here: https://www.antenna-theory.com/basics/directivity.php
        # To do that, we first need to calculate the average power in all directions which is the denominator integral in the
        # directivity formula.
        angleStep = self.getIntegralAngleStep()
        allTheta = np.arange(0, 180, angleStep)
        allPhi = np.arange(-180, 180, angleStep)

        if (allTheta.shape!=theta.shape) or (allPhi.shape!=phi.shape):  totalPowerAllD = self.getPowerPattern(allTheta, allPhi)
        elif np.any(allTheta!=theta) or np.any(allPhi!=phi):            totalPowerAllD = self.getPowerPattern(allTheta, allPhi)
        else:                                                           totalPowerAllD = totalPower  # power pattern for the integral already calculated

        dTheta = dPhi = angleStep*np.pi/180
        integral = (totalPowerAllD*np.sin(allTheta*np.pi/180).reshape(-1,1)*dTheta*dPhi).sum()

        directivity = 4*np.pi*totalPower/integral       # Note that since totalPower is not normalized we have it in the numerator of "Directivity" formula instead of 1
        directivityDbi = 10*np.log10(directivity)       # convert to "dbi", which is db with respect to "isotropic" radiator
        return directivityDbi

# ****************************************************************************************************************************************************
class AntennaPanel(AntennaBase):
    r"""
    This class implements the functionality of a rectangular antenna panel
    containing a set of antenna elements (See :py:class:`AntennaElement`)
    organized in a 2-d grid. The elements are assumed to be on the Y-Z plain.
    An antenna panel can be created individually or it can be grouped with
    other panels to form an antenna array (See :py:class:`AntennaArray`)
    """
    # ************************************************************************************************************************************************
    def __init__(self, shape=[2,2], **kwargs):
        r"""
        Parameters
        ----------
        shape : list
            A list of 2 integers specifying the number of antenna elements
            along ``z`` and ``y`` axis. (The number of rows and columns of
            elements)
            
        kwargs : dict
            A set of additional optional arguments. Here is a list of supported parameters:

                * **spacing**: A list of 2 values specifying the distance between neighboring elements in multiples of the wave-length. By default it is the elements are half the wave-length away from each other, which means `spacing = [0.5, 0.5]`.
                * **elements**: This can be an :py:class:`AntennaElement` object, a 2-D array of :py:class:`AntennaElement` objects, or None.
                
                    * If it is an :py:class:`AntennaElement` object, it will be used as a template to create all the elements in this panel.
                    * If it is a 2-D array of :py:class:`AntennaElement` objects, the specified elements are used for the elements of this panel.
                    * If it is ``None``, then antenna elements of the panel are created using the default values.
                    
                * **polarization**: The polarization of antenna elements on this panel. The panel can be singly polarized (P=1) or dually polarized (P=2). For singly polarized panels, the ``polarization`` can be either "|" (Vertical), or "-" (Horizontal). For dually polarized panels, the ``polarization`` can be either "+" (0 and 90 degree pairs), or "x" (-45 and 45 degree pairs). By default ``polarization="|"`` (Vertically polarized).
                * **position**: The position of the center point of this panel in the antenna array containing this panel.
                * **array**: The :py:class:`AntennaArray` object containing this antenna panel or ``None`` is this panel is not part of an antenna array.
                * **taper**: The `taper` value applied when calculating the fields patterns around this antenna panel. By default, this is set to ``1.0``.
                * **matlabOrder**: Current implementation of Matlab toolkit uses a different order for the elements in a panel compared to the order specified in the 3-GPP standard (See **3GPP TR 38.901 - Section 7.3**). By default this class uses the standard order (``matlabOrder=False``). If you need to compare your results with Matlab implementation, you can set this parameter to ``True``.
        """
        super().__init__(**kwargs)
        self.shape = np.int16(shape)                                # Number of antenna elements in each column and each row of the panel respectively
        self.spacing = np.float64(kwargs.get('spacing', [.5,.5]))   # [dv, dh] as defined in TR38.901-Section 7.3 in multiples of wave-length.
        self.elements = kwargs.get('elements', None)                # A 2d shape[0]-by-shape[1] array of AntennaElement objects
        self.polarization = kwargs.get('polarization', "|")         # Can be one of "|", "-", "+", or "x"
        self.position = np.float64(kwargs.get('position', [0,0,0])) # Position in the array
        self.array = kwargs.get('array', None)                      # Owner array
        self.taper = kwargs.get('taper', 1.0)                       # element taper.
        self.matlabOrder = kwargs.get('matlabOrder', False)         # If true, use matlab order in the "allElements" method
        
        if self.elements is None:       elementTemplate = AntennaElement()
        elif type(self.elements)==list: elementTemplate = None
        else:                           elementTemplate = self.elements

        if elementTemplate is not None:
            numRows, numCols = self.shape                           # These are M and N in TR38.901-Section 7.3 respectively
            offsetZ, offsetY = (self.shape-1) * self.spacing / 2
            dz, dy = self.spacing
            
            allElements = []
            for r in range(numRows):
                rowElements = []
                for c in range(numCols):
                    # Assuming x axis is pointing to us
                    position = [ 0, c*dy-offsetY, r*dz-offsetZ ]
                    if self.polarization in "-|":   # Single Polarization
                        rowElements += [[ elementTemplate.clone(position,  {"|":0,  "-": 90}[self.polarization], self) ]]
                    else:                           # Dual Polarization "+" or "x"
                        rowElements += [[ elementTemplate.clone(position, {"+":0,  "x": 45}[self.polarization], self),
                                          elementTemplate.clone(position, {"+":90, "x":-45}[self.polarization], self)]]
                allElements += [ rowElements ]
            
            self.elements = allElements

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`AntennaPanel` object.

        Parameters
        ----------
        indent : int (default: 0)
            The number of indentation characters.
            
        title : str (default: None)
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
        if title is None:   title = "Antenna Panel:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        if self.array is not None:
            repStr += indent*' ' + "  position: %d\n"%(str(self.position))
        repStr += indent*' ' + "  Total Elements: %d\n"%(self.getNumElements())
        repStr += indent*' ' + "  spacing: %s𝜆, %s𝜆\n"%(str(self.spacing[0]), str(self.spacing[1]))
        repStr += indent*' ' + "  shape: %d rows x %d columns\n"%(self.shape[0], self.shape[1])
        repStr += indent*' ' + "  polarization: %s\n"%(self.polarization)
        repStr += indent*' ' + "  taper: %s\n"%(str(self.taper))
        if getStr: return repStr
        print(repStr)

    # ************************************************************************************************************************************************
    @property
    # Returns the frequency range of the first element. It is assumed all elements have the same range.
    def freqRange(self):   return self.getElement(0).freqRange

    # ************************************************************************************************************************************************
    def clone(self, position, array):
        r"""
        Creates a copy of this :py:class:`AntennaPanel` object and modifies the
        ``position``, and ``polarization`` angles, and the parent ``array`` object
        based on the parameters provided.
        """
        return AntennaPanel(self.shape,
                            spacing = self.spacing,
                            polarization = self.polarization,
                            elements = self.elements[0][0][0],
                            position = position,
                            array = array)
    
    # ************************************************************************************************************************************************
    def getNumElements(self):
        r"""
        Returns the total number of antenna elements in this panel. For singly polarized
        panels, the total number of elements is ``shape[0] x shape[1]``. For dually polarized
        panels, the total number of elements is ``2 x shape[0] x shape[1]``.
        """
        return np.prod(self.shape)*(1 if self.polarization in "-|" else 2)      # Return total number of elements.

    # ************************************************************************************************************************************************
    def getElement(self, elementRC=(0,0), p=0):
        r"""
        Returns the specified :py:class:`AntennaElement` object from this panel.
        
        Parameters
        ----------
        elementRC : tuple (default: (0,0)) or int
            If this is a tuple, the first and second integer values in the tuple specify
            the row and column of the desired element in the panel (0-based).
            If this is an integer, the allowed values are 0 or -1 which return
            the first or last element in the panel respectively. If ``elementRC``
            is not specified, by default the first element is returned.
            
        p : int (default: 0)
            If this panel is singly polarized, this parameter is ignored. Otherwise,
            the first and second polarized antenna element is returned for ``p=0`` and
            ``p=1`` respectively.

        Returns
        -------
        :py:class:`AntennaElement`
            The specified :py:class:`AntennaElement` object from this panel.
        """
        if elementRC==0:  elementRC = (0,0)     # Get first
        if elementRC==-1: elementRC = (-1,-1)   # Get last
        return self.elements[ elementRC[0] ][ elementRC[1] ][ p ]

    # ************************************************************************************************************************************************
    def getElementPosition(self, elementRC=(0,0), ref="Array"):
        r"""
        Returns the position of the specified :py:class:`AntennaElement` object in
        this panel.
        
        Parameters
        ----------
        elementRC : tuple (default: (0,0)) or int
            If this is a tuple, the first and second integer values in the tuple specify
            the row and column of the desired element in the panel (0-based).
            If this is an integer, the allowed values are 0 or -1 which return
            the first or last element in the array respectively. If ``elementRC``
            is not specified, by default the first element is returned.
            
        ref : str (default: "Array")
            If ``ref="Array"`` this function returns the element position with respect to
            the :py:class:`AntennaArray` object containing this panel. Otherwise, if
            ``ref=="Panel"``, the element position with respect to this panel is returned.

        Returns
        -------
        Numpy Array
            An array of 3 values (x, y, and z) representing the position of the specified
            element. Note that the values are in multiples of wave-length.
        """
        return self.getElement(elementRC).position + (0 if ref=="Panel" else self.position)
    
    # ************************************************************************************************************************************************
    def getAllPositions(self, polarization=True):
        r"""
        Returns the positions of all elements in this panel in a 2-D numpy array.
        
        Parameters
        ----------
        polarization : Boolean (default: True)
            If this is a dually polarized panel and this parameter is
            ``True``, the positions of all elements is returned. In this case,
            there will be repeated positions in the returned array as the 2
            polarized pairs of elements have the same position. Otherwise, if
            ``polarization=False``, only one position is returned for a pair
            of polarized antenna elements. If this is a singly polarized panel,
            this parameter is ignored.
            
        Returns
        -------
        Numpy Array
            An ``n x 3`` numpy array containing the positions of all ``n``
            elements in this panel.
        """
        return np.float64([e.position for e in self.allElements(polarization)])

    # ************************************************************************************************************************************************
    def showElements(self, ref="Panel", maxSize=6.0, zeroTicks=False, title=None):
        r"""
        This is a visualization function that draws this antenna panel using
        the `matplotlib` library.
        
        Parameters
        ----------
        ref: str (default: "Panel")
            If ``ref="Panel"``, it means this is a standalone antenna panel
            that is visualized individually. Otherwise, if ``ref="Array"``,
            it means this is being visualized as part of an antenna array.
            (See :py:meth:`AntennaArray.showElements`)
            
        maxSize: (float: 6.0)
            This parameter specifies how large the output image of this panel
            should be. Depending on the number of antenna element rows and
            columns in this panel, the ``maxSize`` can specify the width or
            height of the resulting image.
        
        zeroTicks : Boolean (default: False)
            If this is ``True``, the 0 positions on both axes are indicated
            by additional "ticks" to somehow show the center of this panel.
            Otherwise the "ticks" on the horizontal and vertical axes are
            only at the locations of antenna elements.
            
        title : str (default: None)
            If specified, this will be used as the title for the image
            created for this panel. Otherwise the title "Panel Elements" is
            used.
        """
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

    # ************************************************************************************************************************************************
    def allElements(self, polarization=True):
        r"""
        This is a generator function that can be used to iterate through
        all elements of this panel. For example the following code prints
        the position of every element in this panel:
        
        .. code-block::
        
            for element in myPanel.allElements():
                print( element.position )


        By default, this function iterates through elements in the order
        specified in **3GPP TR 38.901 - Section 7.3**. If the parameter
        ``matlabOrder`` is set to ``True``, then the Matlab order is used.
        Please refer :py:class:`AntennaPanel` parameter documentation for
        more information about ``matlabOrder``.
        
        Parameters
        ----------
        polarization : Boolean (default: True)
            If this is a dually polarized panel and this parameter is
            ``True``, then all elements are included in the iteration.
            Otherwise, if ``polarization=False``, only the first element
            of the polarized pair of elements at each position is included
            in the iteration. If this is a singly polarized panel, this
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

# ****************************************************************************************************************************************************
class AntennaArray(AntennaBase):
    r"""
    This class implements the functionality of a rectangular antenna array
    containing a set of antenna panels (See :py:class:`AntennaPanel`)
    organized in a 2-d grid. The panels are assumed to be on the Y-Z plain.
    """
    # ************************************************************************************************************************************************
    def __init__(self, shape=[1,1], **kwargs):
        r"""
        Parameters
        ----------
        shape : list
            A list of 2 integers specifying the number of antenna panels
            along ``z`` and ``y`` axis (The number of rows and columns of
            panels)
            
        kwargs : dict
            A set of additional optional arguments. Here is a list of supported parameters:

                * **spacing**: A list of 2 values specifying the distance between the center point of neighboring panels in multiples of the wave-length. If not specified, by default the spacing is set such that the specing between antenna elements across different panels is the same as the antenna elements within panels.
                * **panels**: This can be an :py:class:`AntennaPanel` object, a 2-D array of :py:class:`AntennaPanel` objects, or None.
                
                    * If it is an :py:class:`AntennaPanel` object, it will be used as a template to create all the panels in this array.
                    * If it is a 2-D array of :py:class:`AntennaPanel` objects, the specified panels are used for the panels of this array.
                    * If it is ``None``, then antenna panels and elements of this array are created using the default values.
                    
                * **taper**: The `taper` value applied when calculating the fields patterns around this antenna array. By default, this is set to ``1.0``.
        """
        super().__init__(**kwargs)
        self.shape = np.int16(shape)                                # Number of rows and columns of panels in the array. ([M, N] as defined in TR38.901-Section 7.3)
        self.spacing = kwargs.get('spacing', None)                  # [dgV, dgH] as defined in TR38.901-Section 7.3 in multiples of wave-length, or None.
        self.taper = kwargs.get('taper', 1.0)                       # element taper.
        self.panels = kwargs.get('panels', None)                    # An array 2d shape[0]-by-shape[1] array of AntennaPanel objects

        if self.panels is None:         panelTemplate = AntennaPanel()
        elif type(self.panels)==list:   panelTemplate = None
        else:                           panelTemplate = self.panels

        if panelTemplate is not None:
            numRows, numCols = self.shape                           # These are Mg and Ng in TR38.901-Section 7.3 respectively
            self.spacing = np.float64(self.spacing) if self.spacing is not None else (panelTemplate.shape * panelTemplate.spacing)
            offsetZ, offsetY = (self.shape-1) * self.spacing / 2
            dz, dy = self.spacing
            
            allPanels = []
            for r in range(numRows):
                allPanels += [[]]
                for c in range(numCols):
                    # Assuming x axis is pointing to us
                    position = [ 0, c*dy-offsetY, r*dz-offsetZ ]
                    allPanels[r] += [ panelTemplate.clone(position, self) ]
            
            self.panels = allPanels
        else:
            self.spacing = np.float64(self.spacing) if self.spacing is not None else (panel[0][0].shape * panel[0][0].spacing)

    # ************************************************************************************************************************************************
    def __repr__(self):     return self.print(getStr=True)
    def print(self, indent=0, title=None, getStr=False):
        r"""
        Prints the properties of this :py:class:`AntennaArray` object.

        Parameters
        ----------
        indent : int (default: 0)
            The number of indentation characters.
            
        title : str (default: None)
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
        if title is None:   title = "Antenna Array:"
        repStr = "\n" if indent==0 else ""
        repStr += indent*' ' + title + "\n"
        repStr += indent*' ' + "  Total Panels: %d\n"%(np.prod(self.shape))
        repStr += indent*' ' + "  Total Element: %d\n"%(self.getNumElements())
        repStr += indent*' ' + "  Pannel Spacing: %s𝜆, %s𝜆\n"%(str(self.spacing[0]), str(self.spacing[1]))
        repStr += indent*' ' + "  shape: %d rows x %d columns\n"%(self.shape[0], self.shape[1])
        repStr += indent*' ' + "  taper: %s\n"%(str(self.taper))
        if getStr: return repStr
        print(repStr)

    # ************************************************************************************************************************************************
    @property
    # Returns the frequency range of the first element. It is assumed all elements have the same range.
    def freqRange(self):    return self.getElement(0).freqRange

    # ************************************************************************************************************************************************
    @property
    # Assuming all panels have the same type of polarization.
    def polarization(self): return self.panels[0][0].polarization

    # ************************************************************************************************************************************************
    def getElement(self, panelRC=(0,0), elementInPanelRC=(0,0), p=0):
        r"""
        Returns the :py:class:`AntennaElement` object from this array specified
        by row and column of panel in this array and row and column of the element
        in that panel.
        
        Parameters
        ----------
        panelRC : tuple (default: (0,0)) or int
            If this is a tuple, the first and second integer values in the tuple
            specify the row and column of the desired panel in the array (0-based).
            If this is an integer, the allowed values are 0 or -1 which specify
            the first or last panel in the array respectively. If ``panelRC``
            is not specified, by default the first panel is used.

        elementInPanelRC : tuple (default: (0,0)) or int
            If this is a tuple, the first and second integer values in the
            tuple specify the row and column of the desired element in the
            panel (0-based). If this is an integer, the allowed values are 0
            or -1 which return the first or last element in the specified panel
            respectively. If ``elementInPanelRC`` is not specified, by default
            the first element in the specified panel is returned.
            
        p : int (default: 0)
            If the panels of this array are singly polarized, this parameter is
            ignored. Otherwise, the first and second polarized antenna element
            is returned for ``p=0`` and ``p=1`` respectively.

        Returns
        -------
        :py:class:`AntennaElement`
            The specified :py:class:`AntennaElement` object from this panel.
        """
        if panelRC==0:  panelRC, elementInPanelRC = (0,0),(0,0)         # Get first
        if panelRC==-1: panelRC, elementInPanelRC = (-1,-1),(-1,-1)     # Get last
        return self.panels[ panelRC[0] ][ panelRC[1] ].elements[ elementInPanelRC[0] ][ elementInPanelRC[1] ][ p ]
        
    # ************************************************************************************************************************************************
    def getElementPosition(self, panelRC=(0,0), elementInPanelRC=(0,0)):
        r"""
        Returns the position of the :py:class:`AntennaElement` object in
        this array specified by ``elementInPanelRC`` in the panel specified by
        ``panelRC``.
        
        Parameters
        ----------
        panelRC : tuple (default: (0,0)) or int
            If this is a tuple, the first and second integer values in the tuple
            specify the row and column of the desired panel in the array (0-based).
            If this is an integer, the allowed values are 0 or -1 which specify
            the first or last panel in the array respectively. If ``panelRC``
            is not specified, by default the first panel is used.

        elementInPanelRC : tuple (default: (0,0)) or int
            If this is a tuple, the first and second integer values in the
            tuple specify the row and column of the desired element in the
            specified panel (0-based). If this is an integer, the allowed values
            are 0 or -1 which return the position of the first or last element
            in the panel respectively. If ``elementInPanelRC`` is not specified,
            by default the position of the first element in the specified
            panel is returned.

        Returns
        -------
        Numpy Array
            An array of 3 values (x, y, and z) representing the position of the
            specified element. Note that the values are in multiples of wave-length.
        """
        return self.getElement(panelRC,elementInPanelRC).posInArray
       
    # ************************************************************************************************************************************************
    def allPanels(self):
        r"""
        This is a generator function that can be used to iterate through
        all panels of this array.
                    
        Yields
        ------
            The next :py:class:`AntennaPanel` object in this array.
        """
        for r in range(self.shape[0]):
            for c in range(self.shape[1]):
                yield self.panels[ r ][ c ]

    # ************************************************************************************************************************************************
    def allElements(self, polarization=True):
        r"""
        This is a generator function that can be used to iterate through
        all elements of this array. For example the following code prints
        the position of every element in this array:
        
        .. code-block::
        
            for element in myArray.allElements():
                print( element.position )


        This function uses the :py:meth:`AntennaPanel.allElements` to
        iterate through each panel.
        
        Parameters
        ----------
        polarization : Boolean (default: True)
            If the panels of this array are dually polarized and this
            parameter is ``True``, then all elements are included in the
            iteration. Otherwise, if ``polarization=False``, only the first
            element of the polarized pair of elements at each position is
            included in the iteration. If the panels of this array are
            singly polarized, this parameter is ignored.
            
        Yields
        ------
            The next :py:class:`AntennaElement` object in this array.
        """
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

    # ************************************************************************************************************************************************
    def getAllPositions(self, polarization=True):
        r"""
        Returns the positions of all elements in this array in a 2-D numpy array.
        
        Parameters
        ----------
        polarization : Boolean (default: True)
            If the panels of this array are dually polarized and this
            parameter is ``True``, then the positions of all elements are
            returned. Otherwise, if ``polarization=False``, only the position of
            the first element of the polarized pair of elements at each position is
            returned. If the panels of this array are singly polarized, this
            parameter is ignored.
            
        Returns
        -------
        Numpy Array
            An ``n x 3`` numpy array containing the positions of all ``n``
            elements in this array.
        """
        return np.float64([e.posInArray for e in self.allElements(polarization)])

    # ************************************************************************************************************************************************
    def getNumElements(self): # Return total number of Antenna elements in all panels
        r"""
        Returns the total number of antenna elements in this array. It uses
        the :py:meth:`AntennaPanel.getNumElements` to get the number of elements
        in one panel (``Np``). Total number of elements in this array is then
        ``shape[0] x shape[1] * Np``.
        """
        return np.prod(self.shape) * self.panels[0][0].getNumElements()

    # ************************************************************************************************************************************************
    def showElements(self, maxSize=6.0, zeroTicks=False, title=None):
        r"""
        This is a visualization function that draws this antenna array using
        the `matplotlib` library.
        
        Parameters
        ----------
        maxSize: (float: 6.0)
            This parameter specifies how large the output image of this array
            should be. Depending on the number of antenna element/panel
            rows and columns in this array, the ``maxSize`` can specify the
            width or height of the resulting image.
        
        zeroTicks : Boolean (default: False)
            If this is ``True``, the 0 positions on both axes are indicated
            by additional "ticks" to somehow show the center of this array.
            Otherwise the "ticks" on the horizontal and vertical axes are
            only at the locations of antenna elements.
            
        title : str (default: None)
            If specified, this will be used as the title for the image
            created for this array. Otherwise the title "Array Elements" is
            used.
        """
        s = self.shape*self.spacing
        figSize = [maxSize, maxSize*s[0]/s[1]] if s[0]<s[1] else [maxSize*s[1]/s[0], maxSize]
        plt.figure(figsize=figSize)

        for panel in self.allPanels():  panel.showElements("Array")
        
        plt.xlabel("$\\frac {Y}{\\lambda}$", size=15)
        plt.ylabel("$\\frac {Z}{\\lambda}$", size=15)
        plt.title("Array Elements" if title is None else title, size=20)
        yTicks = [self.getElementPosition((0,p),(0,e))[1] for p in range(self.shape[1]) for e in range(self.panels[0][p].shape[1]) ] + ([0] if zeroTicks else [])
        plt.xticks(sorted(yTicks), size=10)
        zTicks = [self.getElementPosition((p,0),(e,0))[2] for p in range(self.shape[0]) for e in range(self.panels[p][0].shape[0]) ] + ([0] if zeroTicks else [])
        plt.yticks(sorted(zTicks), size=10)
        plt.axis('equal')
