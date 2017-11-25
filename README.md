# pytorch-crf

Conditional random field in [PyTorch](http://pytorch.org/).

## Description

This package provides an implementation of [conditional random field](https://en.wikipedia.org/wiki/Conditional_random_field) (CRF) in PyTorch. This implementation borrows mostly from [AllenNLP CRF module](https://github.com/allenai/allennlp/blob/master/allennlp/modules/conditional_random_field.py) with some modifications, most notably is using PyTorch's fancy indexing instead of `torch.gather` and different way of doing Viterbi decoding.

## Requirements

- Python 3.6
- PyTorch 0.2

## Installation

Install from Github directly:

    pip install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf

## License

MIT. See LICENSE.md for details.

## Contributing

Contributions are welcome! Please follow these instructions to setup dependencies and running the tests and linter. Make a pull request once your contribution is ready.

### Installing dependencies

Make sure you setup a virtual environment with Python 3.6 and PyTorch installed. Then, install all the dependencies in `requirements.txt` file.

    pip install -r requirements.txt

### Running tests

Run `pytest` in the project root directory.

### Running linter

Run `flake8` in the project root directory. This will also run `mypy`, thanks to `flake8-mypy` package.
