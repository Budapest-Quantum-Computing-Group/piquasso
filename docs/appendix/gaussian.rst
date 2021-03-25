Gaussian Transformations
------------------------

.. _passive_gaussian_transformations:

Passive Gaussian Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\vec{i}` denote an index set. E.g. for 2 modes denoted by :math:`n` and
:math:`m`:, one could write

.. math::
   \vec{i} = \{n, m\} \times \{n, m\}.

Let :math:`P \in \mathbb{C}^{k \times k},\, k \in [d]` be a transformation
which transforms the vector of annihilation operators in the following manner:

.. math::
   \mathbf{a}_{\vec{m}} \mapsto P \mathbf{a}_{\vec{m}},

or in terms of vector elements:

.. math::
   a_{i} \mapsto \sum_{j \in \vec{m}} P^{ij} a_j

Application to :math:`m` is done by matrix multiplication.

The canonical commutation relations can be written as

.. math::
   [a^\dagger_i, a_j] = \delta_{i j},

and then applying the transformation :math:`P` we get

.. math::
   \sum_{i, j \in \vec{m}} [P^*_{ki} a^\dagger_i, P_{lj} a_j]
         &= \sum_{i, j \in \vec{m}} P^*_{ki} P_{lj}
            [a^\dagger_i, a_j] \\
         &= \sum_{i, j \in \vec{m}} P^*_{ki} P_{lj} \delta_{i j} \\
         &= \sum_{i \in \vec{m}} P^*_{ki} P_{li} \\
         &= \sum_{i \in \vec{m}} (P^\dagger)_{ik} P_{li} \\
         &= \delta_{k l},

where the last line imposes, that any transformation should leave the canonical
commutation relations invariant.
The last line of the equation means, that :math:`P` should actually be a
unitary matrix.

From now on, I will use the notation
:math:`\{n, m\} := \mathrm{modes}`.

The transformation by :math:`P` can be prescribed in the following
manner:

.. math::
   C_{\vec{i}} \mapsto P^* C_{\vec{i}} P^T \\
   G_{\vec{i}} \mapsto P G_{\vec{i}} P^T

Let us denote :math:`\vec{k}` the following:

.. math::
   \vec{k} = \mathrm{modes}
         \times \big (
                  [d]
                  - \mathrm{modes}
         \big ).

For all the remaining modes, the following is applied regarding the
elements, where the **first** index corresponds to
:math:`\mathrm{modes}`:

.. math::
   C_{\vec{k}} \mapsto T^* C_{\vec{k}} \\
   G_{\vec{k}} \mapsto T G_{\vec{k}}

Regarding the case where the **second** index corresponds to
:math:`\mathrm{modes}`, i.e. where we use
:math:`\big ( [d] - \mathrm{modes} \big )
\times \mathrm{modes}`, the same has to be applied.

For :math:`n \in \mathrm{modes}` and :math:`m \in [d]`, we could
use

.. math::
   C_{nm} := C^*_{mn} \\
   G_{nm} := G_{mn}.


.. _active_gaussian_transformations:

Active Gaussian Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let :math:`\vec{m}` denote an index set, which corresponds to the parameter
`modes`.

Let :math:`P, A \in \mathbb{C}^{k \times k},\, k \in [d]` be a passive and an
active transformation, respectively. An active transformation transforms the vector
of annihilation operators in the following manner:

.. math::
      \mathbf{a}_{\vec{m}}
         P \mathbf{a}_{\vec{m}} + A \mathbf{a}_{\vec{m}^*},

or in terms of vector elements:

.. math::
   a_{i} \mapsto
         \sum_{j \in \vec{m}} P^{ij} a_j
         + \sum_{j \in \vec{m}} A^{ij} a_j^\dagger

The vector of the means of the :math:`i`-th mode
:math:`m_i = \langle \hat{a}_i \rangle_\rho` is evolved as follows:

.. math::
   m_i \mapsto P m_i + A m_i^*


The transformations in the terms of the transformation matrices are defined by

.. math::
   G_{i j} \mapsto
         (P G P^T + A G^\dagger A^T + P (1 + C^T) A^T + A C P^T)_{i j}

.. math::
   C_{i j} \mapsto
         (P^* C P^T + A^* (1 + C^T) P^T + P^* G^\dagger A^T + A^* G P^T)_{i j}


Then each row of the mode :math:`i` will be updated according to the fact that
:math:`C_{ij} = C_{ij}^*` and :math:`G_{ij} = G_{ij}^T`.


.. _gaussian_displacement:

Displacement
~~~~~~~~~~~~

Applies the displacement instruction to the state.

.. math::
    D(\alpha) = \exp(\alpha \hat{a}_i^\dagger - \alpha^* \hat{a}_i),

where :math:`\alpha \in \mathbb{C}` is a parameter, :math:`\hat{a}_i` and
:math:`\hat{a}_i^\dagger` are the annihilation and creation operators on the
:math:`i`-th mode, respectively.

The displacement instruction acts on the annihilation and creation operators
in the following way:

.. math::
    D^\dagger (\alpha) \hat{a}_i D (\alpha) = \hat{a}_i + \alpha \mathbb{1}.

:attr:`GaussianState.m` is defined by

.. math::
    m = \langle \hat{a}_i \rangle_{\rho}.

By using the displacement, one acquires

.. math::
    m_{\mathrm{displaced}}
        = \langle D^\dagger (\alpha) \hat{a}_i D (\alpha) \rangle_{\rho}
        = \langle \hat{a}_i + \alpha \mathbb{1}) \rangle_{\rho}
        = m + \alpha.

The :math:`\hat{a}_i \mapsto hat{a}_i + \alpha \mathbb{1}` transformation
should be applied to the `GaussianState.C` and `GaussianState.G` matrices as
well:

.. math::
    C_{ii} \mapsto C_{ii} + \alpha m_i^* + \alpha^* m_i + | \alpha |^2, \\
    G_{ii} \mapsto G_{ii} + 2 \alpha m_i + \alpha^2, \\

and to the corresponding `i, j (i \neq j)` modes.

Note, that :math:`\alpha` is often written in the form

.. math::
    \alpha = r \exp(i \phi),

where :math:`r \geq 0` and :math:`\phi \in [ 0, 2 \pi )`. When two parameters
are specified for this instruction, the first is interpreted as :math:`r`, and the
second one as :math:`\phi`.

Also note, that the displacement cannot be categorized as an active or passive
linear transformation, because the unitary transformation does not strictly
produce a linear combination of the field operators.