import numpy as np
from ControlModule import ControlModule

C = ControlModule.generate_R(
    demand=0.5,     # хочемо 50% потужності
    n_states=100,
    n_actions=3
)

print("Shape:", C.shape)  # має бути (3, 100, 100)

# Стан 50 (рівно на цілі) має мати cost=0 для будь-якої дії
print("Cost at target (s'=50):", C[1, 0, 50])  # → 0.0

# Стан 99 далеко від 0.5 → велика ціна
print("Cost far from target (s'=99):", C[1, 0, 99])  # → ~0.49