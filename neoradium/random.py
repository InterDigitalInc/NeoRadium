# Copyright (c) 2024 InterDigital AI Lab
"""
The module ``random.py`` contains the random generator classes that can be used to generate random numbers and 
control the random behavior of your code. This module also provides the global ``random`` object which is an instance
of the :py:class:`RanGen` class.

The ``random`` object
---------------------
The ``random`` object by default is initialized as a *Permuted Congruential Generator* based on NumPy's
`PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_ class. It can be used to generate
random numbers using the methods defined for NumPy's
`Generator <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`__ class such as
`choice <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.choice.html#numpy-random-generator-choice>`_
or
`shuffle <https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.shuffle.html#numpy-random-generator-shuffle>`_.

.. code-block:: python

    >>> from neoradium import random
    >>> random.choice(5, 3)
    array([1, 1, 4])
    
    >>> a = np.arange(10)
    >>> random.shuffle(a)
    >>> a
    array([5, 8, 0, 1, 6, 9, 7, 2, 3, 4])

In addition to the methods defined for the
`Generator <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`__ class,
**NeoRadium**'s ``random`` object also supports the following methods:

    :bits(size): This method creates a bitstream of ``size`` random bits.
    
        .. code-block:: python

            >>> from neoradium import random
            >>> random.bits(8)
            array([0, 1, 1, 0, 1, 1, 0, 1], dtype=int8)


    :awgn(shape, noiseStd): This method creates *Additive White Gaussian Noise*
        with standard deviation specified by ``noiseStd``. The result will be a complex NumPy array of shape ``shape``.

        .. code-block:: python

            >>> from neoradium import random
            >>> random.awgn((2,2),0.5)
            array([[-0.38382838+0.35261486j,  0.10004801-0.5325556j ],
                   [-0.20456608+0.58387099j, -0.85796067-0.15164351j]])
                   

.. _SupportedRanGens:

Supported Random Generators
---------------------------
Using the ``random`` object's :py:meth:`getGenerator` method you can create all types of random generators supported
by NumPy. Here is a list of supported generators:

    :Default Random Generator: This is based on the NumPy class
        `default_rng <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.default_rng>`_, which
        internally uses the default bit generator (PCG64). The generator can be made deterministic by providing a
        ``seed`` value.
        
        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating a predictable random generator based on "default_rng"
            >>> myGen = random.getGenerator(np.random.default_rng(123))
            >>> myGen.integers(0,10,5)
            array([0, 6, 5, 0, 9])
            
            >>> # The same as above since both use "PCG64" internally
            >>> myGen = random.getGenerator(123)
            >>> myGen.integers(0,10,5)
            array([0, 6, 5, 0, 9])

    :PCG64: Permuted Congruential Generator (64-bit, PCG64). This is currently
        **NeoRadium**'s (and Python's) default bit generator object. It is based on NumPy's
        `PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_
        class. As you can see the results match those above because all use the same bit generator with the same seed.

        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating a predictable random generator based on "PCG64"
            >>> myGen = random.getGenerator(np.random.PCG64(123))
            >>> myGen.integers(0,10,5)
            array([0, 6, 5, 0, 9])
            
    :MT19937: Mersenne Twister. This is based on NumPy's
        `MT19937 <https://numpy.org/doc/stable/reference/random/bit_generators/mt19937.html#mersenne-twister-mt19937>`_
        class. The generator can be made deterministic by providing a ``seed`` value.
       
        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating an unpredictable random generator based on "MT19937"
            >>> myGen = random.getGenerator(np.random.MT19937()) # No seed specified -> Unpredictable
            >>> myGen.integers(0,10,5)
            array([1, 8, 6, 8, 9])

    :PCG64DXSM: Permuted Congruential Generator (64-bit, PCG64 DXSM). This is based on NumPy's
        `PCG64DXSM <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html#permuted-congruential-generator-64-bit-pcg64-dxsm>`_
        class. The generator can be made deterministic by providing a ``seed`` value.
        
        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating a predictable random generator based on "PCG64DXSM"
            >>> myGen = random.getGenerator(np.random.PCG64DXSM(123))
            >>> myGen.integers(0,10,5)
            array([9, 7, 0, 9, 7])


    :Philox: Philox Counter-based RNG. This is based on NumPy's
        `Philox <https://numpy.org/doc/stable/reference/random/bit_generators/philox.html#philox-counter-based-rng>`_
        class. The generator can be made deterministic by providing a ``seed`` value.

        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating an unpredictable random generator based on "Philox"
            >>> myGen = random.getGenerator(np.random.Philox()) # No seed specified -> Unpredictable
            >>> myGen.integers(0,10,5)
            array([4, 0, 1, 6, 4])


    :SFC64: SFC64 Small Fast Chaotic PRNG. This is based on NumPy's
        `SFC64 <https://numpy.org/doc/stable/reference/random/bit_generators/sfc64.html#sfc64-small-fast-chaotic-prng>`_
        class. The generator can be made deterministic by providing a ``seed`` value.

        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating an unpredictable random generator based on "SFC64"
            >>> myGen = random.getGenerator(np.random.SFC64()) # No seed specified -> Unpredictable
            >>> myGen.integers(0,10,5)
            array([0, 7, 8, 1, 8])

    :RandomState: Python's legacy random generator. This is based on NumPy's
        `RandomState <https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState>`_
        class. The generator can be made deterministic by providing a ``seed`` value.

        .. code-block:: python

            >>> from neoradium import random
            >>> # Creating an unpredictable random generator based on "RandomState"
            >>> myGen = random.getGenerator(np.random.RandomState(123))
            >>> myGen.randint(0,10,5)
            array([2, 2, 6, 1, 3])


        .. Important::
            The ``RandomState`` can be used to create a random generator that matches Matlab's default random 
            generator. For example, this is used by **NeoRadium** when comparing the simulation results with Matlab's
            implementation.
           
        .. code-block:: python
            :caption: predictable random generator based on ``RandomState`` in **NeoRadium**

            >>> from neoradium import random
            >>> # Creating a predictable random generator based on "RandomState"
            >>> myGen = random.getGenerator(np.random.RandomState(123))
            >>> myGen.random(size=5)
            array([0.69646919, 0.28613933, 0.22685145, 0.55131477, 0.71946897])

        .. code-block:: matlab
            :caption: predictable random generator in **Matlab**

            >> rng(123);
            >> rand(1,5)
            
            ans =
            
                0.6965    0.2861    0.2269    0.5513    0.7195
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
# **********************************************************************************************************************
import numpy as np

# **********************************************************************************************************************
class NrGen1(np.random.RandomState):                            # Not documented - Not called directly by the user
    # NrGen1 is the same as NumPy's RandomState with one more method: "bits"
    def __init__(self, seed): super().__init__(seed)
    def integers(self, low, high=None, size=None, dtype=np.int64):  return self.randint(low, high, size, dtype)
    def bits(self, size):            return self.randint(0,2,size,dtype=np.int8)
    def awgn(self, shape, noiseStd): return (self.normal(0, noiseStd/np.sqrt(2), shape+(2,))*[1,1j]).sum(-1)

# **********************************************************************************************************************
class NrGen2(np.random.Generator):                              # Not documented - Not called directly by the user
    # NrGen2 is the same as NumPy's Generator with one more method: "bits"
    def __init__(self, bitGen): super().__init__(bitGen)
    def randint(self, low, high=None, size=None, dtype=int):        return self.integers(low, high, size, dtype)
    def bits(self, size):               return self.integers(0,2,size,dtype=np.int8)
    def awgn(self, shape, noiseStd):    return (self.normal(0, noiseStd/np.sqrt(2), shape+(2,))*[1,1j]).sum(-1)

# **********************************************************************************************************************
class RanGen:
    r"""
    This is **NeoRadium**'s random number generator class. This class is used internally to create **NeoRadium**'s
    ``random`` object. It is strongly recommended to use only **NeoRadium**'s ``random`` object for all random
    operations.
    """
    def __init__(self, generator=None):
        self.generator = self.getGenerator() if generator is None else generator

    # ******************************************************************************************************************
    def getGenerator(self, seed=None):
        r"""
        This function creates a new random generator object that can be used to create new random values. See
        :ref:`Supported Random Generators <SupportedRanGens>` for examples of how to use this function.

        Parameters
        ----------
        seed: int, BitGenerator, Generator, RandomState, or None
            This parameter specifies how the new random generator should
            be created. It can be one of the following:
            
                :int: If ``seed`` is an integer, it is used as the
                    ``seed`` to generate a
                    `PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_
                    random number generator. The returned random number
                    generator is predictable.
                   
                :BitGenerator: If ``seed`` is one of the
                    `Supported BitGenerators <https://numpy.org/doc/stable/reference/random/bit_generators/index.html#supported-bitgenerators>`_
                    the returned random generator is based on the specified
                    bit generator.
                
                :Generator: If ``seed`` is a
                    `Generator <https://numpy.org/doc/stable/reference/random/generator.html>`__
                    object, the returned random generator is based on ``default_rng``.
                    Note that internally, ``default_rng`` uses the
                    `PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_
                    bit generator.
                    
                :RandomState: If ``seed`` is a
                    `RandomState <https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState>`_
                    object, the returned random generator is based on ``RandomState``.
                    
                :None: If ``seed`` is `None` (default), a default
                    `PCG64 <https://numpy.org/doc/stable/reference/random/bit_generators/pcg64.html>`_
                    bit generator is used without specifying the seed, which results
                    in an unpredictable random generator.

        Returns
        -------
        RanGen
            The returned :py:class:`RanGen` object is based on a **NeoRadium** internal class which is derived from
            NumPy's `Generator <https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator>`__
            or `RandomState <https://numpy.org/doc/stable/reference/random/legacy.html#numpy.random.RandomState>`_
            class.
        """
        # See https://numpy.org/doc/stable/reference/random/index.html
        if seed is None:                                gen = NrGen2(np.random.PCG64())       # PCG64, unpredictable
        elif isinstance(seed, RanGen):                  return seed                           # Already a RanGen object
        elif isinstance(seed, np.random.BitGenerator):  gen = NrGen2(seed)                    # bit generator object
        elif isinstance(seed, np.random.Generator):     gen = NrGen2(seed.bit_generator)      # generator object
        elif isinstance(seed, np.random.RandomState):   gen = NrGen1(seed.get_state()[1][0])  # getting seed
        else:                                           gen = NrGen2(np.random.PCG64(seed))   # PCG64, predictable
        return RanGen(gen)

    # ******************************************************************************************************************
    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return getattr(self.generator,attr)
        return super().__getattr__(self, attr)

    # ******************************************************************************************************************
    def setSeed(self, seed):
        r"""
        This function changes the random generator used by this :py:class:`RanGen` object.
        
        .. Important::
            It is recommended to avoid using this method on **NeoRadium**'s global ``random`` object. Since the
            ``random`` object is a single instance used globally by all of the codebase using **NeoRadium**, changing
            it affects other parts of your code which may depend on the original ``random`` object.
            
            A better approach is to create a new random generator using the :py:meth:`getGenerator` method and use
            that for the part of your code that needs a specific random generator.
            
            However, for smaller programs where you want to control the random behavior of your code, this function
            provides a quick and easy way to make your results reproducible by passing a constant integer value
            to this function.
            
            .. code-block:: python

                from neoradium import random
                
                # Changing the "random" object to be predictable with "MT19937" bit generator
                # This is not recommended because it changes the "random" object which is
                # used globally in NeoRadium.
                random.setSeed(np.random.MT19937(123))
                myRandInts = random.integers(0,10,5)

                # Creating a new predictable random generator with "MT19937" bit generator
                # This is the preferred approach as the original "random" object remains unchanged
                # The results are the same as the above code.
                myGen = random.getGenerator(np.random.MT19937(123))
                myRandInts = myGen.integers(0,10,5)
                

        Parameters
        ----------
        seed: int, BitGenerator, Generator, RandomState, or None Please refer to :py:meth:`getGenerator` method
            for details about the ``seed`` parameter.
        """
        self.generator = self.getGenerator(seed)

random = RanGen()
