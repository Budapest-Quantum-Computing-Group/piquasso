Instructions
============

Instructions are the operations placed inside a :class:`~piquasso.api.program.Program`.
They describe how selected modes are prepared, transformed, measured, or affected
by noise. In most examples, an instruction is attached to one or more modes with
the ``pq.Q(...) | instruction`` syntax.

.. toctree::
   :maxdepth: 3
   :hidden:

   preparations
   gates
   measurements
   channels

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Preparations
      :link: preparations
      :link-type: doc

      Define the initial state or add state-vector components to the program.

   .. grid-item-card:: Gates
      :link: gates
      :link-type: doc

      Apply transformations such as displacements, squeezings, beamsplitters,
      and interferometers.

   .. grid-item-card:: Measurements
      :link: measurements
      :link-type: doc

      Extract samples or measurement results from selected modes.

   .. grid-item-card:: Channels
      :link: channels
      :link-type: doc

      Model non-unitary effects such as loss and other noisy processes.

How instructions are used
-------------------------

A program is built by applying instructions to modes. The instruction itself
describes the operation, while ``pq.Q(...)`` selects the modes on which the
operation acts.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      # Preparation
      pq.Q() | pq.Vacuum()

      # Gates
      pq.Q(0) | pq.Displacement(r=0.2)
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

      # Measurement
      pq.Q(all) | pq.ParticleNumberMeasurement()

   simulator = pq.GaussianSimulator(d=2)
   result = simulator.execute(program, shots=10)

   print(result.samples)

Instruction categories
----------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Preparations usually come first

      A preparation instruction initializes the simulated state before gates,
      channels, or measurements are applied.

   .. grid-item-card:: Gates transform the state

      Gate instructions describe reversible transformations of the selected
      modes. Many common optical operations are represented as gates.

   .. grid-item-card:: Channels model noise

      Channel instructions describe non-unitary effects, such as losses. Their
      availability can depend on the chosen simulator.

   .. grid-item-card:: Measurements usually come last

      Measurement instructions produce classical information, such as samples.
      When shots are requested, the result contains measurement samples.

Simulator compatibility
-----------------------

Instructions are declarative: they describe the program to be executed. The
chosen simulator determines which instructions can be evaluated efficiently and
which state representation is used. For example, Gaussian circuits are naturally
handled by Gaussian simulators, while many non-Gaussian operations require a
Fock-space simulator. Boson Sampling-style programs are handled by the Sampling
simulator.


Dynamic and adaptive programs
-----------------------------

Piquasso programs do not have to be simple straight-line circuits. In supported
simulators, measurements can appear before the end of the program, later
instructions can be conditioned on previous measurement outcomes, and some
instruction parameters can be resolved at runtime.

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Mid-circuit measurements

      Measure or postselect modes before the program ends, then continue the
      simulation on the remaining modes.

   .. grid-item-card:: Conditional execution

      Attach a condition to an instruction with ``Instruction.when(...)`` so it
      is applied only for selected measurement histories.

   .. grid-item-card:: Runtime parameters

      Use previous measurement outcomes to determine instruction parameters
      during execution.

Mid-circuit measurements
~~~~~~~~~~~~~~~~~~~~~~~~

Mid-circuit measurements are useful when a measurement result should affect the
state before later instructions are applied. A common example is heralding: one
mode is measured or postselected, and the unmeasured modes continue in the
post-measurement state.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q() | pq.Vacuum()

      # Entangle modes 0 and 1.
      pq.Q(0, 1) | pq.Squeezing2(r=np.log(1 + np.sqrt(2)))

      # Herald on detecting exactly one photon in mode 1.
      pq.Q(1) | pq.PostSelectPhotons((1,))

      # Continue the program on the heralded state.
      pq.Q(0, 2) | pq.Beamsplitter5050()

   result = pq.PureFockSimulator(d=3).execute(program)

Conditional instruction execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Instruction.when(...)`` method makes an instruction conditional. The
condition receives the previous measurement outcomes and decides whether the
instruction should be applied in the current branch.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   with pq.Program() as program:
      pq.Q(0, 1, 2) | pq.StateVector((1, 0, 0))
      pq.Q(0, 1) | pq.Beamsplitter(theta=np.pi / 4)

      # Measure mode 0 before the end of the program.
      pq.Q(0) | pq.ParticleNumberMeasurement()

      # If the last detected value was 0, route the remaining photon.
      pq.Q(1, 2) | pq.Beamsplitter(theta=np.pi / 2).when(
         lambda outcomes: outcomes[-1] == 0
      )

      pq.Q(1, 2) | pq.ParticleNumberMeasurement()

   result = pq.PureFockSimulator(d=3).execute(program, shots=20)

Runtime-resolved parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instruction parameters can also depend on earlier measurement outcomes. This is
useful for adaptive programs where a later gate is calibrated from information
obtained during the same execution.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   program = pq.Program(
      instructions=[
         pq.StateVector([0, 2]) * np.sqrt(1 / 3),
         pq.StateVector([1, 1]) * np.sqrt(1 / 3),
         pq.StateVector([2, 0]) * np.sqrt(1 / 3),
         pq.ParticleNumberMeasurement().on_modes(1),
         pq.Squeezing(r=lambda x: 0.1 * x[-1] ** 2).on_modes(0),
      ]
   )

   result = pq.PureFockSimulator(d=2).execute(program, shots=10)

Using an expression string:

.. code-block:: python

   import numpy as np
   import piquasso as pq

   program = pq.Program(
      instructions=[
         pq.StateVector([0, 2]) * np.sqrt(1 / 3),
         pq.StateVector([1, 1]) * np.sqrt(1 / 3),
         pq.StateVector([2, 0]) * np.sqrt(1 / 3),
         pq.ParticleNumberMeasurement().on_modes(1),
         pq.Squeezing(r="0.1 * x[-1] ** 2").on_modes(0),
      ]
   )

   result = pq.PureFockSimulator(d=2).execute(program, shots=10)

Program construction styles
---------------------------

Most examples use the context-manager style, but the same program can also be
created from an explicit instruction list. This can be convenient when programs
are generated by another tool or assembled dynamically.

.. code-block:: python

   import numpy as np
   import piquasso as pq

   program = pq.Program(
      instructions=[
         pq.Vacuum(),
         pq.Squeezing(r=0.5).on_modes(0),
         pq.Beamsplitter(theta=np.pi / 4).on_modes(0, 1),
      ]
   )

   result = pq.GaussianSimulator(d=2).execute(program)