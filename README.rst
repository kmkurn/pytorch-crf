pytorch-crf
+++++++++++

.. image:: https://badge.fury.io/py/pytorch-crf.svg
   :target: https://badge.fury.io/py/pytorch-crf

.. image:: https://travis-ci.org/kmkurn/pytorch-crf.svg?branch=master
   :target: https://travis-ci.org/kmkurn/pytorch-crf

.. image:: https://coveralls.io/repos/github/kmkurn/pytorch-crf/badge.svg?branch=master
   :target: https://coveralls.io/github/kmkurn/pytorch-crf?branch=master

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org

Conditional random field in `PyTorch <http://pytorch.org/>`_.

Description
===========

This package provides an implementation of `conditional random field
<https://en.wikipedia.org/wiki/Conditional_random_field>`_ (CRF) in PyTorch.
This implementation borrows mostly from `AllenNLP CRF module
<https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_ra
ndom_field.py>`_ with some modifications.

Requirements
============

- Python 3.6
- PyTorch 1.0.0

Installation
============

You can install with pip ::

    pip install pytorch-crf

Or, you can install from Github directly ::

    pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

Examples
========

In the examples below, we will assume that these lines have been executed

.. code-block:: python

    >>> import torch
    >>> from torchcrf import CRF
    >>> seq_length, batch_size, num_tags = 3, 2, 5
    >>> emissions = torch.randn(seq_length, batch_size, num_tags)
    >>> tags = torch.tensor([
    ...   [0, 1], [2, 4], [3, 1]
    ... ], dtype=torch.long)  # (seq_length, batch_size)
    >>> model = CRF(num_tags)

Computing log likelihood
------------------------

.. code-block:: python

    >>> model(emissions, tags)
    tensor(-12.7431, grad_fn=<SumBackward0>)

Computing log likelihood with mask
----------------------------------

.. code-block:: python

    >>> mask = torch.tensor([
    ...   [1, 1], [1, 1], [1, 0]
    ... ], dtype=torch.uint8)  # (seq_length, batch_size)
    >>> model(emissions, tags, mask=mask)
    tensor(-10.8390, grad_fn=<SumBackward0>)

Decoding
--------

.. code-block:: python

    >>> model.decode(emissions)
    [[3, 1, 3], [0, 1, 0]]

Decoding with mask
------------------

.. code-block:: python

    >>> model.decode(emissions, mask=mask)
    [[3, 1, 3], [0, 1]]

See ``tests/test_crf.py`` for more examples.

License
=======

MIT. See `LICENSE`_ for details.

Contributing
============

Contributions are welcome! Please follow these instructions to install
dependencies and running the tests and linter. Make a pull request once your
contribution is ready.

Installing dependencies
-----------------------

Make sure you setup a virtual environment with Python and PyTorch
installed. Then, install all the dependencies in ``requirements.txt`` file and
install this package in development mode. ::

    pip install -r requirements.txt
    pip install -e .

Setup pre-commit hook
---------------------

Simply run ::

    ln -s ../../pre-commit.sh .git/hooks/pre-commit

Running tests
-------------

Run ``pytest`` in the project root directory.

Running linter
--------------

Run ``flake8`` in the project root directory. This will also run ``mypy``,
thanks to ``flake8-mypy`` package.

.. _`LICENSE`: https://github.com/kmkurn/pytorch-crf/blob/master/LICENSE.txt
