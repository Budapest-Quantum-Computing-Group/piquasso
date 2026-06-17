Installation
============

Install Piquasso from PyPI with:

.. code-block:: bash

   pip install piquasso

For a first example, see :doc:`tutorials/getting-started`.

Optional dependencies
---------------------

TensorFlow support
^^^^^^^^^^^^^^^^^^

Piquasso can also be used with TensorFlow, for example in continuous-variable
quantum neural network workflows. See :doc:`tutorials/cvqnn-with-tensorflow`
for a tutorial.

To install Piquasso with TensorFlow support, run:

.. code-block:: bash

   pip install "piquasso[tensorflow]"

JAX support
^^^^^^^^^^^

Piquasso also supports JAX-based workflows, as described in
:doc:`tutorials/jax-example`.

To install Piquasso with JAX support, run:

.. code-block:: bash

   pip install "piquasso[jax]"