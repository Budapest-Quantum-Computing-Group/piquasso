Piquasso
********


Piquasso is an open source Python package, which allows you to simulate a photonic
quantum computer. One could use Gaussian or Fock state corresponding to different
representations to run simulations.


Installation
============

One could easily install Piquasso with the following command:

.. code-block:: bash

   pip install piquasso

For a basic example, check out :doc:`tutorials/getting-started`.

One can also use Piquasso along with TensorFlow, see, e.g.,
:doc:`tutorials/cvqnn-with-tensorflow`. To install Piquasso with TensorFlow, just enter

.. code-block:: bash

   pip install piquasso[tensorflow]


Similarly, Piquasso admits a JAX support, as described in :doc:`tutorials/jax-example`
To install Piquasso with JAX is done by

.. code-block:: bash

   pip install piquasso[jax]


.. toctree::
   :maxdepth: 3
   :caption: Tutorials:
   :hidden:

   tutorials/getting-started
   tutorials/separating-programs
   tutorials/boson-sampling
   tutorials/gaussian-boson-sampling
   tutorials/cvqnn-with-tensorflow
   tutorials/jax-example

.. toctree::
   :maxdepth: 3
   :caption: Simulators:
   :hidden:

   simulators/gaussian
   simulators/fock
   simulators/sampling

.. toctree::
   :maxdepth: 3
   :caption: States:
   :hidden:

   states/gaussian
   states/fock
   states/sampling

.. toctree::
   :maxdepth: 3
   :caption: Instructions:
   :hidden:

   instructions/preparations
   instructions/gates
   instructions/measurements
   instructions/channels
   instructions/misc

.. toctree::
   :maxdepth: 3
   :caption: API:
   :hidden:

   api/program
   api/simulator
   api/config
   api/mode
   api/state
   api/instruction
   api/result
   api/exceptions
   api/computer
   api/calculator

.. toctree::
   :maxdepth: 3
   :caption: Advanced:
   :hidden:

   advanced/cvqnn
   advanced/calculators
