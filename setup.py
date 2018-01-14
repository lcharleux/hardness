from setuptools import setup

setup(name='Hardness',
      version='0.1',
      description="Python Indentation Toolbox",
      long_description="",
      author='Ludovic Charleux',
      author_email='ludovic.charleux@univ-smb.fr',
      license='GPL v3',
      packages=['hardness'],
      zip_safe=False,
      install_requires=[
          "numpy",
          "scipy",
          "matplotlib",
          "pandas",
          "jupyter",
          "nbconvert"
          ],
      )
