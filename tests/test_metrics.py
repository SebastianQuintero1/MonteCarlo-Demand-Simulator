import numpy as np
from Metrics import MAE, MSE, R2, Corr

perfect = np.array([0.1, 0.5, 0.8, 0.3, 0.6])

#ideal reactor
print(MAE(perfect, perfect))  # → 0.0
print(MSE(perfect, perfect))  # → 0.0
print(R2(perfect, perfect))  # → 1.0
print(Corr(perfect, perfect))  # → 1.0

#reactor - 0
zeros = np.zeros(5)
print(MAE(perfect, zeros))  # → ~0.46
print(R2(perfect, zeros))  # → negative (worse then stupid predictor)
