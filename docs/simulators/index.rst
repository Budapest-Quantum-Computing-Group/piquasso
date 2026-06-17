Simulators
==========

Piquasso provides several built-in simulators for different photonic simulation
tasks. The right choice depends on the representation you need, the instructions
in your program, and whether you want finite-shot samples or the final state.

.. toctree::
   :maxdepth: 3
   :caption: Simulators:
   :hidden:

   gaussian
   fock
   sampling

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Gaussian simulator
      :link: gaussian
      :link-type: doc

      Efficiently simulate Gaussian states, Gaussian gates, Gaussian channels,
      and compatible measurements.

   .. grid-item-card:: Fock space-based simulators
      :link: fock
      :link-type: doc

      Simulate states in the truncated Fock basis, including non-Gaussian
      operations such as Kerr-type gates.

   .. grid-item-card:: Boson Sampling simulator
      :link: sampling
      :link-type: doc

      Sample from interferometer-based Boson Sampling circuits with terminal
      particle number measurements.

Choosing a simulator
--------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: GaussianSimulator
      :link: gaussian
      :link-type: doc

      Use this for Gaussian circuits and efficient phase-space simulations based
      on first and second moments.

   .. grid-item-card:: PureFockSimulator
      :link: fock
      :link-type: doc

      Use this for pure-state simulations in a truncated Fock basis, including
      supported non-Gaussian operations.

   .. grid-item-card:: FockSimulator
      :link: fock
      :link-type: doc

      Use this when you need the more general Fock representation, for example
      for mixed states or supported noisy operations.

   .. grid-item-card:: SamplingSimulator
      :link: sampling
      :link-type: doc

      Use this for Boson Sampling-style circuits with Fock-state inputs,
      interferometers, losses, and particle number measurements.

Quick examples
--------------

Gaussian simulation
~~~~~~~~~~~~~~~~~~~

Use the Gaussian simulator for circuits that can be represented in phase space.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q() | pq.Vacuum()
      pq.Q(0) | pq.Displacement(r=0.2)
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)
      pq.Q(all) | pq.ParticleNumberMeasurement()

   simulator = pq.GaussianSimulator(d=2)
   result = simulator.execute(program, shots=10)

   print(result.samples)

Fock simulation
~~~~~~~~~~~~~~~

Use a Fock simulator when the circuit contains non-Gaussian operations. The
``cutoff`` controls the size of the truncated Fock basis.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q() | pq.Vacuum()
      pq.Q(0) | pq.Displacement(r=0.4)
      pq.Q(0) | pq.Kerr(xi=0.05)

   simulator = pq.PureFockSimulator(
      d=1,
      config=pq.Config(cutoff=8),
   )
   result = simulator.execute(program)

   print(result.state.fock_probabilities)

Boson Sampling simulation
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the Sampling simulator for particle-number input states followed by a
linear optical circuit and terminal particle number measurements.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   interferometer = np.array(
      [
         [1 / np.sqrt(2), 1 / np.sqrt(2)],
         [1 / np.sqrt(2), -1 / np.sqrt(2)],
      ]
   )

   with pq.Program() as program:
      pq.Q() | pq.StateVector([1, 1])
      pq.Q() | pq.Interferometer(interferometer)
      pq.Q(all) | pq.ParticleNumberMeasurement()

   simulator = pq.SamplingSimulator(d=2)
   result = simulator.execute(program, shots=10)

   print(result.samples)

Automatic simulator selection with ``simulate``
-----------------------------------------------

For small examples and quick experiments, the top-level ``pq.simulate`` function
can choose a compatible built-in simulator for the program.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q() | pq.Vacuum()
      pq.Q(0) | pq.Displacement(r=0.2)
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)
      pq.Q(all) | pq.ParticleNumberMeasurement()

   result = pq.simulate(
      program,
      number_of_modes=2,
      shots=10,
   )

   print(result.samples)

The function accepts the same common simulation options as explicit simulator
usage, including ``config``, ``connector``, and ``shots``. Use an explicit
simulator class when you want to document or control the exact representation
used by the simulation.
