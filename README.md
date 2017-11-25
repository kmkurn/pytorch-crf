# pytorch-crf

Conditional random field in [PyTorch](http://pytorch.org/).

## Description

This package provides an implementation of [conditional random field](https://en.wikipedia.org/wiki/Conditional_random_field) (CRF) in PyTorch. This implementation borrows mostly from [AllenNLP's CRF module](https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py) with some modifications, most notably is using PyTorch's fancy indexing instead of `torch.gather` and different way of doing Viterbi decoding.

## Requirements

- Python 3.6
- PyTorch 0.2

## Installation

Install from Github directly:

    pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

## License

MIT. See LICENSE.md for details.
