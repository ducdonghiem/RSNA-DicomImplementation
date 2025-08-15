import math
import numpy as np
from scipy.special import expit

def logit(p): return np.log(p/(1-p))

a = 0.9512
b = -3.1175
t_orig = 0.35
t_cal = expit(a * logit(t_orig) + b)
print(t_cal)   # ~0.0424
