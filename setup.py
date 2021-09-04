"""Installs SBS+UR."""
import setuptools
try:
  from Cython.Build import cythonize
  from Cython.Distutils import build_ext
except ImportError:
  import sys
  print("You must install Cython before running the setup!", file=sys.stderr)
  print("You can do so with: pip install cython.", file=sys.stderr)
  sys.exit(1)

def run_setup():
  """Installs SBSUR."""

  with open('README.md', 'r') as fh:
    long_description = fh.read()

  setuptools.setup(
      name='sbsur',
      version="0.2.0",
      author='Théo Matricon',
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
      cmdclass = {'build_ext':build_ext},
      extra_compile_args=["-O3"]
  )


if __name__ == '__main__':
  run_setup()
