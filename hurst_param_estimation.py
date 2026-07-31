import numpy as np
import math
import matplotlib.pyplot as plt
import fbm_simulation as fbms

def k_mom(k: float, incr: np.ndarray) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    return sum(abs(incr)**k) / incr.shape[0]

def H_estimation(k:float, fgns: dict) -> float:
    if k <= 0:
        raise ValueError('k must be positive')
    m = len(fgns.keys())
    logsum = 0
    for i in fgns.keys():
        logsum += math.log(i)
    numerator = 0
    denominator = 0
    for i in fgns.keys():
        a_i = math.log(i) - 1/m * logsum
        numerator += a_i * math.log(k_mom(k, fgns[i]))
        denominator += a_i**2

    return 1/k * numerator/denominator



rng = np.random.default_rng()
timepoints = np.linspace(1, 2000, 1999)
noise = rng.normal(0, 1, len(timepoints))

H = 0.1
fgns = {}
for i in range(1,5):
    fgns[i] = fbms.simulate_fgn(timepoints, H, noise, i)

res = H_estimation(2, fgns)
print(res)