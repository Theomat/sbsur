"""Installs SBS+UR."""
import setuptools
from Cython.Build import cythonize
from Cython.Distutils import build_ext

def run_setup():
  """Installs SBSUR."""

  with open('README.md', 'r') as fh:
    long_description = fh.read()

  setuptools.setup(
      name='sbsur',
      version="0.0.1",
      author='Théo Matricon, Nathanal Fijalkow',
      author_email='theomatricon@gmail.com',
      description='Stochastic Beam Search + UniqueRandomizer: Fast Incremental Sampling Without Replacement',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/Theomat/sbsur',
      packages=setuptools.find_packages(),
      install_requires=[
        'cython'
      ],
      extras_require={'dev': ['pytest', 'numpy']},
      python_requires='>=3.6',
      ext_modules=cythonize("sbsur/*.pyx"),
      zip_safe=False,
      cmdclass = {'build_ext':build_ext}
  )


if __name__ == '__main__':
  run_setup()
