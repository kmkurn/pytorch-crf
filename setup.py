import os
from setuptools import setup, find_packages


here = os.path.dirname(os.path.realpath(__file__))
with open(os.path.join(here, 'README.rst')) as f:
    readme = f.read()


setup(name='pytorch-crf',
      version='0.6.0',
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
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      keywords='torch',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      python_requires='>=3.6, <4')
