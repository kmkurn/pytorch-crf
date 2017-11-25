from setuptools import setup, find_packages


setup(name='pytorch-crf',
      version='0.1.0',
      description='Conditional random field in PyTorch',
      url='https://github.com/kmkurn/pytorch-crf',
      author='Kemal Kurniawan',
      author_email='kemal@kkurniawan.com',
      license='MIT',
      classifiers=[
          'Development Status :: 3 - Alpha',
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
