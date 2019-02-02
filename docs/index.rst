.. pytorch-crf documentation master file, created by
   sphinx-quickstart on Sat Feb  2 13:37:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

pytorch-crf
===========

*Conditional random fields for PyTorch.*

This package provides an implementation of a `conditional random fields (CRF)`_ layer
for PyTorch. The implementation borrows mostly from `AllenNLP CRF module`_ with some
modifications.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Minimal requirements
====================

* Python 3.6
* PyTorch 1.0.0

Installation
============

Install with pip::

    pip install pytorch-crf

Getting started
===============

**pytorch-crf** provides a single `~torchcrf.CRF` module which implements a CRF layer.

API documentation
=================

.. autoclass:: torchcrf.CRF
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. _`conditional random fields (CRF)`: https://en.wikipedia.org/wiki/Conditional_random_field
.. _`AllenNLP CRF module`: https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py
