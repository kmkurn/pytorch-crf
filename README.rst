pytorch-crf
+++++++++++

Conditional random field in `PyTorch <http://pytorch.org/>`_.

Description
===========

This package provides an implementation of `conditional random field <https://en.wikipedia.org/wiki/Conditional_random_field>`_ (CRF) in PyTorch. This implementation borrows mostly from `AllenNLP CRF module <https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py>`_ with some modifications.

NOTE: This software is still in alpha version; every minor version change introduces backward incompatibility.

Requirements
============

- Python 3.6
- PyTorch 0.2

Installation
============

You can install with pip ::

    pip install pytorch-crf

Or, you can install from Github directly ::

    pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

Examples
========

In the examples below, we will assume that these lines have been executed ::

    >>> import torch
    >>> from torchcrf import CRF
    >>> seq_length, batch_size, num_tags = 3, 2, 5
    >>> emissions = torch.autograd.Variable(torch.randn(seq_length, batch_size, num_tags), requires_grad=True)
    >>> tags = torch.autograd.Variable(torch.LongTensor([[0, 1], [2, 4], [3, 1]]))  # (seq_length, batch_size)
    >>> model = CRF(num_tags)
    >>> # Initialize model parameters
    ... for p in model.parameters():
    ...    _ = torch.nn.init.uniform(p, -1, 1)
    ...
    >>>

Forward computation
-------------------

::

    >>> model(emissions, tags)
    Variable containing:
    -10.0635
    [torch.FloatTensor of size 1]

Forward computation with mask
-----------------------------

::

    >>> mask = torch.autograd.Variable(torch.ByteTensor([[1, 1], [1, 1], [1, 0]]))  # (seq_length, batch_size)
    >>> model(emissions, tags, mask=mask)
    Variable containing:
    -8.4981
    [torch.FloatTensor of size 1]

Decoding
--------

::

    >>> model.decode(emissions)
    [[3, 1, 3], [0, 1, 0]]

Decoding with mask
------------------

::

    >>> model.decode(emissions, mask=mask)
    [[3, 1, 3], [0, 1, 0]]

See ``tests/test_crf.py`` for more examples.

License
=======

MIT. See ``LICENSE.txt`` for details.

Contributing
============

Contributions are welcome! Please follow these instructions to setup dependencies and running the tests and linter. Make a pull request once your contribution is ready.

Installing dependencies
-----------------------

Make sure you setup a virtual environment with Python 3.6 and PyTorch installed. Then, install all the dependencies in ``requirements.txt`` file and install this package in development mode. ::

    pip install -r requirements.txt
    pip install -e .

Running tests
-------------

Run ``pytest`` in the project root directory.

Running linter
--------------

Run ``flake8`` in the project root directory. This will also run ``mypy``, thanks to ``flake8-mypy`` package.
