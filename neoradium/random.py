# Copyright (c) 2024-2026, InterDigital AI Lab
"""
The module ``random.py`` provides **NeoRadium**'s random number generation utilities.
It defines the global :py:data:`random` object, which is the recommended entry point
for all random operations in **NeoRadium**, and a small set of helper generator classes
used internally.

The global ``random`` object
----------------------------
The global ``random`` object is an instance of :py:class:`RanGen`. It is initialized
with **NeoRadium**'s default random generator configuration and can be used directly to
generate random values using the methods of NumPy's random generators, such as
`choice <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy-random-generator-choice>`_
and
`shuffle <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.shuffle.html#numpy-random-generator-shuffle>`_.

.. code-block:: python

    >>> from neoradium import random
    >>> random.choice(5, 3)
    array([1, 1, 4])

    >>> a = np.arange(10)
    >>> random.shuffle(a)
    >>> a
    array([5, 8, 0, 1, 6, 9, 7, 2, 3, 4])

In addition to standard NumPy generator methods, **NeoRadium** generators also provide:

    :bits(size): Generates a bitstream of random bits.

        .. code-block:: python

            >>> from neoradium import random
            >>> random.bits(8)
            array([0, 1, 1, 0, 1, 1, 0, 1], dtype=int8)

    :awgn(shape, noiseStd): Generates complex additive white Gaussian noise
        with standard deviation ``noiseStd``.

        .. code-block:: python

            >>> from neoradium import random
            >>> random.awgn((2,2), 0.5)
            array([[-0.38382838+0.35261486j,  0.10004801-0.5325556j ],
                   [-0.20456608+0.58387099j, -0.85796067-0.15164351j]])


Creating additional generators
------------------------------
New random generators should be created using the global ``random`` object's
:py:meth:`~RanGen.getGenerator` method:

.. code-block:: python

    >>> from neoradium import random
    >>> myGen = random.getGenerator(123, "PCG64")
    >>> myGen.integers(0, 10, 5)
    array([0, 6, 5, 0, 9])

The :py:class:`RanGen` class is not intended to be instantiated directly by users.
Instead, use :py:meth:`~RanGen.getGenerator` to create new independent generators.
This ensures consistent initialization and makes the intended generator type and seed
explicit.


.. _SupportedRanGens:

Supported random generator types
--------------------------------
The :py:meth:`~RanGen.getGenerator` method supports the following generator types:

    :DEFAULT: NumPy's ``default_rng`` generator. This is **NeoRadium**'s default choice.
        At the time of writing, NumPy's default generator is based on PCG64.

    :PCG64: NumPy's PCG64 bit generator.

    :MT19937: NumPy's Mersenne Twister bit generator.

    :PCG64DXSM: NumPy's PCG64DXSM bit generator.

    :PHILOX: NumPy's Philox counter-based bit generator.

    :SFC64: NumPy's SFC64 bit generator.

    :RANDOMSTATE: NumPy's legacy ``RandomState`` generator.

    :MATLAB: Alias for ``RANDOMSTATE``. This option is provided as a convenient way
        to create generators whose output matches MATLAB's default random number
        generator for the same seed.

        .. code-block:: python
            :caption: Predictable random generator in **NeoRadium**

            >>> from neoradium import random
            >>> myGen = random.getGenerator(123, "MATLAB")
            >>> myGen.random(size=5)
            array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897])

        .. code-block:: matlab
            :caption: Predictable random generator in MATLAB

            >> rng(123);
            >> rand(1,5)

            ans =

                0.6965    0.2861    0.2269    0.5513    0.7195

Reproducibility
---------------
Each call to :py:meth:`~RanGen.getGenerator` creates a new generator instance.
For a given ``seed`` and generator type, the returned generator always starts from
the beginning of the same sequence. Using the same ``seed`` with different generator
types may still produce different sequences.
"""
# **********************************************************************************************************************
# Revision History:
# Date Changed  By                      Description
# ------------  --------------------    --------------------------------------------------------------------------------
# 07/18/2023    Shahab Hamidi-Rad       First version of the file.
# 01/10/2024    Shahab Hamidi-Rad       Completed the documentation
# 08/01/2025    Shahab Hamidi-Rad       - Some minor improvements to the RanGen class.
#                                       - Added the "integers" function to NrGen1 for consistency.
#                                       - Added the "randint" function to NrGen2 for consistency.
# 04/12/2026    Shahab Hamidi-Rad       Changes in NeoRadium version 0.5.0:
#                                       * Trying to prevent the users from creating RanGen objects directly. They must
#                                         be created using 'getGenerator' function only.
#                                       * Fixed issues with random states of the internal objects which could cause
#                                         reproducibility issues.
#                                       * The 'getGenerator' now gets a 'seed' and a 'genType'.
#                                       * The 'reset' function can be used to restart the random sequence.
# **********************************************************************************************************************
import numpy as np

from .utils import validateRange

# **********************************************************************************************************************
class NrGen1(np.random.RandomState):                            # Undocumented - Not called directly by the user
    # NrGen1 is the same as NumPy's RandomState with one more method: "bits"
    def __init__(self, seed): super().__init__(seed)
    def integers(self, low, high=None, size=None, dtype=np.int64):  return self.randint(low, high, size, dtype)
    def bits(self, size):            return self.randint(0,2,size,dtype=np.int8)
    def awgn(self, shape, noiseStd): return (self.normal(0, noiseStd/np.sqrt(2), shape+(2,))*[1,1j]).sum(-1)

# **********************************************************************************************************************
class NrGen2(np.random.Generator):                              # Undocumented - Not called directly by the user
    # NrGen2 is the same as NumPy's Generator with one more method: "bits"
    def __init__(self, bitGen): super().__init__(bitGen)
    def randint(self, low, high=None, size=None, dtype=int):        return self.integers(low, high, size, dtype)
    def bits(self, size):               return self.integers(0,2,size,dtype=np.int8)
    def awgn(self, shape, noiseStd):    return (self.normal(0, noiseStd/np.sqrt(2), shape+(2,))*[1,1j]).sum(-1)

# **********************************************************************************************************************
class RanGen:
    r"""
    **NeoRadium** random generator wrapper.

    This class wraps a NumPy random generator and exposes both the standard NumPy
    random-generation methods and **NeoRadium**-specific helper methods such as
    :py:meth:`bits` and :py:meth:`awgn`.

    The :py:class:`RanGen` class is primarily used through **NeoRadium**'s global
    :py:data:`random` object. Users are not expected to instantiate :py:class:`RanGen`
    directly. Instead, new generators should be created by calling
    :py:meth:`getGenerator` on the global ``random`` object:

    .. code-block:: python

        from neoradium import random
        myGen = random.getGenerator(123, "PCG64")

    This design ensures that all **NeoRadium** generators are created in a consistent
    way and that their seed and generator type are tracked correctly.
    """
    def __init__(self, generator=None, seed=None, genType="DEFAULT",  *, _internal=False):
        """
        Parameters
        ----------
        generator : object or None
            Internal NumPy-based generator object used by this wrapper. This parameter is
            intended for internal use only.

        seed : int or None
            The seed associated with this generator. If specified, it can be used later
            with :py:meth:`reset` to restart the same sequence from the beginning.

        genType : str
            The generator type used to create this object. Supported values are the same
            as those accepted by :py:meth:`getGenerator`.
        """
        if not _internal:
            raise RuntimeError("RanGen objects must not be created directly. Use 'random.getGenerator' instead.")

        self.seed = seed
        self.genType = genType.upper() if isinstance(genType, str) else genType
        if generator is None:   self.generator = self.getGenerator(seed, genType).generator
        else:                   self.generator = generator

    # ******************************************************************************************************************
    def __repr__(self):
        return f"RanGen(seed={self.seed}, genType={self.genType!r})"

    # ******************************************************************************************************************
    def getGenerator(self, seed=None, genType="DEFAULT"):
        r"""
        Creates and returns a new random number generator with a specified generator type
        and seed. The returned generator is independent of the global ``random`` object and
        always starts a new deterministic sequence for a given ``seed`` and ``genType``.

        Parameters
        ----------
        seed : int or None
            Seed used to initialize the random generator.

            - If an integer is provided, the generator produces a deterministic and reproducible sequence.
            - If ``None`` (default), the generator is initialized in a non-deterministic manner.

        genType : str
            Specifies the type of random generator to create. The value is case-insensitive.
            Supported generator types are:

            =================  ============================================================
            genType            Description
            =================  ============================================================
            "DEFAULT"          NumPy default generator (default_rng, currently PCG64-based)
            "PCG64"            NumPy PCG64 generator (recommended default)
            "MT19937"          Mersenne Twister generator
            "PCG64DXSM"        PCG64DXSM generator
            "PHILOX"           Philox counter-based generator
            "SFC64"            SFC64 generator
            "RANDOMSTATE"      NumPy legacy RandomState generator
            "MATLAB"           Alias for RandomState, intended to match MATLAB behavior
            =================  ============================================================

            The ``"MATLAB"`` option provides a convenient way to generate sequences that
            match MATLAB's default random number generator for the same seed, which is
            useful for cross-validation and comparison with MATLAB simulations.

        Returns
        -------
        RanGen
            A new :py:class:`RanGen` object wrapping the selected NumPy random generator.

        Notes
        -----
        - Each call to this function returns a *new* generator instance. The sequence always
          starts from the beginning for the given ``seed`` and ``genType``.
        - Generators created with the same ``seed`` and ``genType`` will produce identical
          sequences, ensuring reproducibility.
        - Different generator types may produce different sequences even when using the same seed.
        """
        # See https://numpy.org/doc/stable/reference/random/index.html
        if not isinstance(genType, str):            raise ValueError(f"'genType' must be a string")
        genType = genType.upper()
        validateRange(genType, ['DEFAULT', 'PCG64', 'MT19937', 'PCG64DXSM', 'PHILOX', 'SFC64',
                                'RANDOMSTATE', 'MATLAB'], varName="genType")
        if genType in ["RANDOMSTATE", "MATLAB"]:    return RanGen(NrGen1(seed), seed, genType, _internal=True)

        if genType =="DEFAULT":                     ranObj = np.random.default_rng(seed).bit_generator
        elif genType =="PCG64":                     ranObj = np.random.PCG64(seed)
        elif genType =="MT19937":                   ranObj = np.random.MT19937(seed)
        elif genType =="PCG64DXSM":                 ranObj = np.random.PCG64DXSM(seed)
        elif genType =="PHILOX":                    ranObj = np.random.Philox(seed)
        else:                                       ranObj = np.random.SFC64(seed)      # genType == "SFC64"

        return RanGen(NrGen2(ranObj), seed, genType, _internal=True)

    # ******************************************************************************************************************
    def __getattr__(self, attrName):
        return getattr(self.generator, attrName)

    # ******************************************************************************************************************
    def setSeed(self, seed):
        r"""
        Re-initializes this generator with a new seed while keeping the current
        generator type unchanged.

        After calling this method, the underlying generator is recreated from scratch
        using the specified ``seed`` and the current ``genType``. This means the random
        sequence restarts from the beginning for that seed and generator type.

        If the new ``seed`` is the same as the current one, this method has the same
        effect as :py:meth:`reset`.

        Parameters
        ----------
        seed : int or None
            The new seed used to initialize the generator.

            - If an integer is provided, the generator becomes deterministic and
              reproducible.
            - If ``None``, the generator is reinitialized in a non-deterministic manner.

        Notes
        -----
        This method does not preserve the current generator state. It always creates a
        new generator instance starting at the beginning of the sequence defined by the
        given ``seed`` and the current generator type.
        """
        self.seed = seed
        self.generator = self.getGenerator(seed, self.genType).generator

    # ******************************************************************************************************************
    def reset(self):
        r"""
        Resets this generator to the beginning of its current random sequence.

        This method recreates the underlying generator using the stored ``seed`` and
        current ``genType``. As a result, subsequent random values will match the values
        produced when this generator was first created, provided the seed is not `None`.

        Notes
        -----
        - If this generator was created with a fixed integer seed, calling ``reset()``
          makes the sequence reproducible from the beginning.
        - If this generator was created with ``seed=None``, calling ``reset()`` creates
          a new non-deterministic generator, so the sequence will generally not match
          the previous one.
        """
        self.setSeed(self.seed)

random = RanGen(_internal=True)
