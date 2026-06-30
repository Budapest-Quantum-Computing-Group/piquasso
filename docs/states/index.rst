States
======

Piquasso state classes represent the result of a simulation. In most workflows,
you do not choose a state class directly. Instead, you choose a simulator, execute a
program, and inspect the returned :class:`~piquasso.api.result.Result` (e.g.,
``result.state`` when no measurements are performed, otherwise ``result.samples`` or
``result.branches``).

.. toctree::
   :maxdepth: 3
   :hidden:

   gaussian
   fock
   passive

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: GaussianState
      :link: gaussian
      :link-type: doc

      Returned by Gaussian simulations. It represents Gaussian states through
      phase-space data such as first and second moments.

   .. grid-item-card:: FockState and PureFockState
      :link: fock
      :link-type: doc

      Returned by Fock-space simulations. They represent mixed or pure photonic
      states in a truncated Fock basis.

   .. grid-item-card:: PassiveState
      :link: passive
      :link-type: doc

      Used by Boson Sampling simulations to represent the state before terminal
      particle number measurements.

How states are obtained
-----------------------

The state type follows from the simulator used to execute the program.

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: GaussianSimulator

      Returns a :class:`~piquasso._simulators.gaussian.state.GaussianState`
      when the program is executed without terminal sampling-only output.

   .. grid-item-card:: PureFockSimulator

      Returns a :class:`~piquasso._simulators.fock.pure.state.PureFockState`
      represented by state-vector coefficients in the Fock basis.

   .. grid-item-card:: FockSimulator

      Returns a :class:`~piquasso._simulators.fock.general.state.FockState`
      represented by a density matrix in the Fock basis.

   .. grid-item-card:: PassiveSimulator

      Uses a :class:`~piquasso._simulators.passive.state.PassiveState` for
      Boson Sampling-style circuits and produces particle-number samples.

Basic usage
-----------

After executing a program without measurements, the final state can be accessed through
``result.state``. For programs with measurements, inspect ``result.samples`` (finite shots)
or ``result.branches`` (exact distribution with ``shots=None``).

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q() | pq.Vacuum()
      pq.Q(0) | pq.Displacement(r=0.2)
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

   simulator = pq.GaussianSimulator(d=2)
   result = simulator.execute(program)

   state = result.state

   print(type(state))
   print(state.mean_photon_number())

Fock-state example
------------------

For non-Gaussian circuits, use a Fock-space simulator. The state is represented
in a truncated Fock basis, controlled by ``Config(cutoff=...)``.

.. code-block:: python

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

   state = result.state

   print(type(state))
   print(state.fock_probabilities)

Boson Sampling example
----------------------

For Boson Sampling-style simulations, the main output is usually the collection
of measurement samples.

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
      pq.Q() | pq.NumberState([1, 1])
      pq.Q() | pq.Interferometer(interferometer)
      pq.Q(all) | pq.ParticleNumberMeasurement()

   simulator = pq.PassiveSimulator(d=2)
   result = simulator.execute(program, shots=10)

   print(result.samples)
