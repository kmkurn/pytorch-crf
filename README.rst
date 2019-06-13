pytorch-crf
===========

Conditional random field in `PyTorch <http://pytorch.org/>`_.

.. image:: https://img.shields.io/pypi/pyversions/pytorch-crf.svg?style=flat
   :target: https://img.shields.io/pypi/pyversions/pytorch-crf.svg?style=flat
   :alt: Python versions

.. image:: https://img.shields.io/pypi/v/pytorch-crf.svg?style=flat
   :target: https://pypi.org/project/pytorch-crf
   :alt: PyPI project

.. image:: https://img.shields.io/travis/kmkurn/pytorch-crf.svg?style=flat
   :target: https://travis-ci.org/kmkurn/pytorch-crf
   :alt: Build status

.. image:: https://img.shields.io/readthedocs/pytorch-crf.svg?style=flat
   :target: https://pytorch-crf.readthedocs.io
   :alt: Documentation status

.. image:: https://img.shields.io/coveralls/github/kmkurn/pytorch-crf.svg?style=flat
   :target: https://coveralls.io/github/kmkurn/pytorch-crf
   :alt: Code coverage

.. image:: https://img.shields.io/pypi/l/pytorch-crf.svg?style=flat
   :target: https://choosealicense.com/licenses/mit/
   :alt: License

.. image:: https://cdn.rawgit.com/syl20bnr/spacemacs/442d025779da2f62fc86c2082703697714db6514/assets/spacemacs-badge.svg
   :target: http://spacemacs.org
   :alt: Built with Spacemacs

This package provides an implementation of `conditional random field
<https://en.wikipedia.org/wiki/Conditional_random_field>`_ (CRF) in PyTorch.
This implementation borrows mostly from `AllenNLP CRF module
<https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_ra
ndom_field.py>`_ with some modifications.

Documentation
=============

https://pytorch-crf.readthedocs.io/

License
=======

MIT

Contributing
============

Contributions are welcome! Please follow these instructions to install
dependencies and running the tests and linter.

Installing dependencies
-----------------------

Make sure you setup a virtual environment with Python. Then, install all
the dependencies in ``requirements.txt`` file and install this package in
development mode.

::

    pip install -r requirements.txt
    pip install -e .

Setup pre-commit hook
---------------------

Simply run::

    ln -s ../../pre-commit.sh .git/hooks/pre-commit

Running tests
-------------

Run ``pytest`` in the project root directory.

Running linter
--------------

Run ``flake8`` in the project root directory. This will also run ``mypy``,
thanks to ``flake8-mypy`` package.
