Fock States
===========

.. warning::
   The ordering of the Fock basis is increasing with particle numbers, and in each
   particle number conserving subspace, lexicographic ordering is used.

   Example for 3 modes:

   .. math ::

      | 000 \rangle,
      | 001 \rangle, | 010 \rangle, | 100 \rangle,
      | 002 \rangle, | 011 \rangle, | 020 \rangle, | 101 \rangle, | 110 \rangle,
      | 200 \rangle \dots

General Fock State
------------------

.. automodule:: piquasso._backends.fock.general.state
   :members:


Pure Fock State
---------------

.. automodule:: piquasso._backends.fock.pure.state
   :members:


Particle number conserving Fock State
-------------------------------------

.. automodule:: piquasso._backends.fock.pnc.state
   :members:
