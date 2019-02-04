import os
import re
from setuptools import setup, find_packages

here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    readme = f.read()
with open(os.path.join(here, 'torchcrf', '__init__.py')) as f:
    version = re.search(r'__version__ = (["\'])([^"\']*)\1', f.read())[2]

setup(
    name='pytorch-crf',
    version=version,
    description='Conditional random field in PyTorch',
    long_description=readme,
    url='https://github.com/kmkurn/pytorch-crf',
    author='Kemal Kurniawan',
    author_email='kemal@kkurniawan.com',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='torch',
    packages=find_packages(),
    python_requires='>=3.6, <4')
