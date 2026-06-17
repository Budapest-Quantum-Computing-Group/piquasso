Tutorials
=========

These tutorials introduce the main workflows in Piquasso, from basic program
construction to sampling, differentiable simulations, and application-oriented
examples.

First steps
-----------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Getting started
      :link: getting-started
      :link-type: doc

      Build and execute a first Piquasso program.

   .. grid-item-card:: Modularity in Piquasso
      :link: separating-programs
      :link-type: doc

      Compose programs, reuse circuit fragments, and define custom gates.

Sampling experiments
--------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Boson Sampling
      :link: boson-sampling
      :link-type: doc

      Simulate Boson Sampling circuits with particle-number input states.

   .. grid-item-card:: Gaussian Boson Sampling
      :link: gaussian-boson-sampling
      :link-type: doc

      Simulate Gaussian Boson Sampling circuits and inspect the resulting
      samples.

Differentiable workflows
------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Quantum Neural Network layers
      :link: cvqnn-with-tensorflow
      :link-type: doc

      Use Piquasso with TensorFlow in a trainable continuous-variable workflow.

   .. grid-item-card:: Using JAX with Piquasso
      :link: jax-example
      :link-type: doc

      Learn a simple state-learning example using JAX and Piquasso.

Applications and benchmarking
-----------------------------

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: Dense k-subgraph problem
      :link: dense-subgraph-gbs
      :link-type: doc

      Use Gaussian Boson Sampling to study the dense k-subgraph problem.

   .. grid-item-card:: Linear cross-entropy benchmarking
      :link: lxeb
      :link-type: doc

      Run a simple linear cross-entropy benchmarking example.

.. toctree::
   :maxdepth: 3
   :hidden:

   getting-started
   separating-programs
   boson-sampling
   gaussian-boson-sampling
   cvqnn-with-tensorflow
   jax-example
   dense-subgraph-gbs
   lxeb