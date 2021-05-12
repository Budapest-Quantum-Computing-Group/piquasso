Piquasso
********


Piquasso is an open source Python package, which allows you to perform simulations on a
photonic quantum circuit. One could use Gaussian and Fock states corresponding to
different representations to run simulations.

Installation
============

One could easily install Piquasso with the following command:

.. code-block:: bash

   pip install piquasso


CPiquasso
---------

Firstly, you need to install `MPICH <https://www.mpich.org/>`_.

Then issue

.. code-block:: bash

   pip install cpiquasso


To use it:

.. code-block:: Python

   import piquasso as pq
   from cpiquasso import patch

   patch()


The `patch` function will patch the usual pure Python code with extensions written in
C++.


.. toctree::
   :maxdepth: 3
   :caption: Tutorials:
   :hidden:

   tutorials/getting-started
   tutorials/separating-programs
   tutorials/boson-sampling
   tutorials/gaussian-boson-sampling

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

.. toctree::
   :maxdepth: 3
   :caption: API:
   :hidden:

   api/program
   api/mode
   api/state
   api/instruction
   api/result
   api/circuit
   api/errors
