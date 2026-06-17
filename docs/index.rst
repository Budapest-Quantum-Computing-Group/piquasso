

.. figure:: _static/logo_only_text.svg
   :alt: Piquasso logo
   :width: 200%
   :align: center
   :class: no-scaled-link


|

.. container:: mission-statement

   Piquasso is an open-source Python package for simulating photonic quantum
   computers.


.. container:: landing-intro

   Build photonic programs from preparations, gates, measurements, and channels,
   then execute them with task-specific high-performance simulators. Piquasso supports
   Gaussian, Fock-space, and Boson Sampling-based workflows, from small examples to
   adaptive and differentiable simulations.

Get started
===========

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Install Piquasso and verify your local setup.

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc

      Learn the basic program structure through guided examples.

   .. grid-item-card:: Simulators
      :link: simulators/index
      :link-type: doc

      Choose the simulator that matches your circuit and representation.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Browse the public classes, functions, and configuration options.

Explore Piquasso
================

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Instructions
      :link: instructions/index
      :link-type: doc

      Use preparations, gates, measurements, channels, and adaptive execution.

   .. grid-item-card:: States
      :link: states/index
      :link-type: doc

      Inspect the state representations returned by the simulators.

   .. grid-item-card:: Connectors
      :link: advanced/connectors
      :link-type: doc

      Use alternative numerical backends, including JAX-based workflows.

   .. grid-item-card:: Advanced topics
      :link: advanced/decompositions
      :link-type: doc

      Learn about decompositions and specialized simulation workflows.

Code example
============

The example below prepares a Gaussian state, applies a displacement and a
beamsplitter, then performs a homodyne measurement.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q() | pq.Vacuum()
      pq.Q(0) | pq.Displacement(r=np.sqrt(2), phi=np.pi / 4)
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 3, phi=np.pi / 2)
      pq.Q(0) | pq.HomodyneMeasurement(phi=0)

   simulator = pq.GaussianSimulator(d=3)
   result = simulator.execute(program, shots=10)

   print(result.samples)


How to cite us
==============

If you use Piquasso in your research, please cite:

.. code-block:: text

   Piquasso: A Photonic Quantum Computer Simulation Software Platform,
   Quantum 9, 1708 (2025).

.. dropdown:: BibTeX entry

   .. code-block:: bibtex

      @article{Kolarovszki_2025,
         title={Piquasso: A Photonic Quantum Computer Simulation Software Platform},
         volume={9},
         ISSN={2521-327X},
         url={http://dx.doi.org/10.22331/q-2025-04-15-1708},
         DOI={10.22331/q-2025-04-15-1708},
         journal={Quantum},
         publisher={Verein zur Forderung des Open Access Publizierens in den Quantenwissenschaften},
         author={
            Kolarovszki, Zoltán
            and Rybotycki, Tomasz
            and Rakyta, Péter
            and Kaposi, Ágoston
            and Poór, Boldizsár
            and Jóczik, Szabolcs
            and Nagy, Dániel T. R.
            and Varga, Henrik
            and El-Safty, Kareem H.
            and Morse, Gregory
            and Oszmaniec, Michał
            and Kozsik, Tamás
            and Zimborás, Zoltán
         },
         year={2025},
         month=apr,
         pages={1708}
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
   advanced/dual_rail_encoding
   advanced/lxeb

.. toctree::
   :hidden:
   :caption: Experimental

   experimental/fermionic


Notable features
================

Piquasso is designed for concise examples, adaptive photonic programs, and
research workflows that need performance, differentiability, or realistic
imperfections.

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Task-specific simulators
      :link: simulators/index
      :link-type: doc

      Choose Gaussian, Fock-space, or Boson Sampling simulators depending on the
      circuit and the representation you need.

   .. grid-item-card:: Adaptive execution
      :link: instructions/index
      :link-type: doc

      Use mid-circuit measurements, postselection, and conditional instructions
      to build adaptive photonic programs.

   .. grid-item-card:: Runtime-resolved parameters
      :link: instructions/index
      :link-type: doc

      Let instruction parameters depend on previous measurement outcomes using
      Python callables or expression strings.

   .. grid-item-card:: Losses and imperfections
      :link: instructions/channels
      :link-type: doc

      Model non-unitary effects such as photon loss and other realistic
      imperfections through channel instructions.

   .. grid-item-card:: Differentiable workflows
      :link: advanced/cvqnn
      :link-type: doc

      Build optimization workflows for trainable photonic circuits and quantum
      neural-network-style simulations.

   .. grid-item-card:: JAX, GPU, and performance
      :link: advanced/connectors
      :link-type: doc

      Use connector-based workflows and optimized backends for demanding
      simulations and accelerator-ready execution.
