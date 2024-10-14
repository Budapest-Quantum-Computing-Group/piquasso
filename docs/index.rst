======================
Piquasso documentation
======================


|

.. container:: mission-statement

   Piquasso is an open source Python package, which allows you to simulate a photonic
   quantum computer.

|

.. grid:: 4

   .. grid-item-card::  Installation
      :link: installation
      :link-type: doc
      :class-title: cardclass

      Instructions on the installation of the Piquasso package

   .. grid-item-card::  Tutorials
      :link: tutorials/index
      :link-type: doc

      Basic tutorials for using Piquasso

   .. grid-item-card::  Simulators
      :link: simulators/index
      :link-type: doc

      The built-in simulators in Piquasso

   .. grid-item-card::  API reference
      :link: api/index
      :link-type: doc

      API reference for Piquasso


|

Code example
============

.. code-block::

   import numpy as np
   import piquasso as pq

   # Program definition
   with pq.Program() as program:
      # Prepare a Gaussian vacuum state
      pq.Q() | pq.Vacuum()

      # Displace the state on mode 0
      pq.Q(0) | pq.Displacement(r=np.sqrt(2), phi=np.pi / 4)

      # Use a beamsplitter gate on modes 0, 1
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 2)

      # Measurement on mode 0
      pq.Q(0) | pq.HomodyneMeasurement(phi=0)

   # Creating the Gaussian simulator
   simulator = pq.GaussianSimulator(d=3)

   # Apply the program with 10 shots
   result = simulator.execute(program, shots=10)


How to cite us
==============

If you are doing research using Piquasso, please cite us as follows:

.. code-block::

   @misc{piquasso2024,
   title         = {Piquasso: A Photonic Quantum Computer Simulation Software Platform},
   author        = {Z. Kolarovszki and
                     T. Rybotycki and
                     P. Rakyta and
                     Á. Kaposi and
                     B. Poór and
                     Sz. Jóczik and
                     D. T. R. Nagy and
                     H. Varga and
                     K. H. El-Safty and
                     G. Morse and
                     M. Oszmaniec and
                     T. Kozsik and
                     Z. Zimborás
                     },
   year          = {2024},
   eprint        = {2403.04006},
   archiveprefix = {arXiv},
   primaryclass  = {quant-ph}
   }



.. toctree::
   :hidden:
   :caption: Basics

   installation
   tutorials/index


.. toctree::
   :hidden:
   :caption: Features

   simulators/index
   states/index
   instructions/index

.. toctree::
   :hidden:
   :caption: Advanced

   api/index
   advanced/connectors
   advanced/decompositions
   advanced/cvqnn

.. toctree::
   :hidden:
   :caption: Experimental

   experimental/fermionic
