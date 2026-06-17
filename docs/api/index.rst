API reference
=============

.. container:: mission-statement

   Explore the public building blocks of Piquasso, from programs and simulators
   to states, connectors, and exceptions.

.. toctree::
   :maxdepth: 1
   :hidden:

   program
   simulator
   config
   mode
   state
   instruction
   result
   branch
   exceptions
   computer
   connector

Core workflow
-------------

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Program
      :link: program
      :link-type: doc

      Define and compose photonic quantum programs.

   .. grid-item-card:: Simulator
      :link: simulator
      :link-type: doc

      Execute programs using the available simulation backends.

   .. grid-item-card:: Config
      :link: config
      :link-type: doc

      Configure numerical precision, cutoffs, validation, and execution options.

Quantum objects
---------------

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Mode
      :link: mode
      :link-type: doc

      Address individual modes and mode registers.

   .. grid-item-card:: State
      :link: state
      :link-type: doc

      Inspect the quantum states produced by simulations.

   .. grid-item-card:: Instruction
      :link: instruction
      :link-type: doc

      Apply gates, preparations, measurements, and channels.

Execution results
-----------------

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Result
      :link: result
      :link-type: doc

      Access samples, states, and metadata returned by simulator runs.

   .. grid-item-card:: Branch
      :link: branch
      :link-type: doc

      Work with conditional execution branches and measurement outcomes.

   .. grid-item-card:: Exceptions
      :link: exceptions
      :link-type: doc

      Understand the errors raised by Piquasso.

Advanced components
-------------------

.. grid:: 1 1 2 3
   :gutter: 2

   .. grid-item-card:: Computer
      :link: computer
      :link-type: doc

      Use lower-level execution interfaces.

   .. grid-item-card:: Connector
      :link: connector
      :link-type: doc

      Connect Piquasso to numerical backends and autodiff frameworks.