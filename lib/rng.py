import random
import os
import torch
import numpy as np
from random import Random
from numpy.random import RandomState

seed = 42

py_rng = Random(seed)
np_rng = RandomState(seed)
