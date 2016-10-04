from setuptools import setup

setup(name='hardness',
      version='0.0',
      description="Indentation toolbox",
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
          "pandas"
          ],
      )
