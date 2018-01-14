import numpy as np
import pandas as pd
from argiope import mesh as Mesh
import argiope
import os, subprocess, inspect

# PATH TO MODULE
import hardness
MODPATH = os.path.dirname(inspect.getfile(hardness))
