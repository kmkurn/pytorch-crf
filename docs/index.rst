.. pytorch-crf documentation master file, created by
   sphinx-quickstart on Sat Feb  2 13:37:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytorch-crf
===========

*Conditional random fields in PyTorch.*

This package provides an implementation of a `conditional random fields (CRF)`_ layer
in PyTorch. The implementation borrows mostly from `AllenNLP CRF module`_ with some
modifications.

.. toctree::
   :maxdepth: 2

Minimal requirements
====================

* Python 3.6
* PyTorch 1.0.0

Installation
============

Install with pip::

    pip install pytorch-crf

Or, install from Github for the latest version::

    pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

Getting started
===============

.. currentmodule:: torchcrf

**pytorch-crf** exposes a single `CRF` class which inherits from PyTorch's
`nn.Module <torch.nn.Module>`. This class provides an implementation of a CRF layer.

.. code-block:: python

   >>> import torch
   >>> from torchcrf import CRF
   >>> num_tags = 5  # number of tags is 5
   >>> model = CRF(num_tags)

Computing log likelihood
------------------------

Once created, you can compute the log likelihood of a sequence of tags given some emission
scores.

.. code-block:: python

   >>> seq_length = 3  # maximum sequence length in a batch
   >>> batch_size = 2  # number of samples in the batch
   >>> emissions = torch.randn(seq_length, batch_size, num_tags)
   >>> tags = torch.tensor([
   ...   [0, 1], [2, 4], [3, 1]
   ... ], dtype=torch.long)  # (seq_length, batch_size)
   >>> model(emissions, tags)
   tensor(-12.7431, grad_fn=<SumBackward0>)

If you have some padding in your input tensors, you can pass a mask tensor.

.. code-block:: python

   >>> # mask size is (seq_length, batch_size)
   >>> # the last sample has length of 1
   >>> mask = torch.tensor([
   ...   [1, 1], [1, 1], [1, 0]
   ... ], dtype=torch.uint8)
   >>> model(emissions, tags, mask=mask)
   tensor(-10.8390, grad_fn=<SumBackward0>)

Note that the returned value is the *log likelihood* so you'll need to make this value
negative as your loss. By default, the log likelihood is summed over batches. For other
options, consult the API documentation of `CRF.forward`.

Decoding
--------

To obtain the most probable sequence of tags, use the `CRF.decode` method.

.. code-block:: python

   >>> model.decode(emissions)
   [[3, 1, 3], [0, 1, 0]]

This method also accepts a mask tensor, see `CRF.decode` for details.

API documentation
=================

.. autoclass:: torchcrf.CRF
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`


.. _`conditional random fields (CRF)`: https://en.wikipedia.org/wiki/Conditional_random_field
.. _`AllenNLP CRF module`: https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
