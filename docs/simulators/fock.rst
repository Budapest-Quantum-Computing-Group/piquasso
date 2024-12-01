Fock Simulators
===============

.. warning::
   The ordering of the Fock basis is increasing with particle numbers, and in each
   particle number conserving subspace, anti-lexicographic ordering is used.

   Example for 3 modes:

   .. math ::

      \ket{000},
      \ket{100}, \ket{010}, \ket{001},
      \ket{200}, \ket{110}, \ket{101}, \ket{020}, \ket{011}, \ket{002}, \dots

General Fock Simulator
----------------------

.. automodule:: piquasso._simulators.fock.general.simulator
   :members:
   :inherited-members:


Pure Fock Simulator
-------------------

.. automodule:: piquasso._simulators.fock.pure.simulator
   :members:
   :inherited-members:
