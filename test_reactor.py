import numpy as np

# Import your Reactor class (adjust the name/file depending on your project)
from Reactor import Reactor  


# Create a reactor instance (replace with your real constructor parameters)
reactor = Reactor(
    model= "RBMK",
    effective_section=1.0,
    neutron_flux=1.0,
    core_volume=1.0,
    fision_energy=1.0,
    probabilities={ 
        "decrease": [0.55, 0.20, 0.25], 
        "maintain": [0.95, 0.025, 0.025], 
        "increase": [0.65, 0.25, 0.1]
    } 
)

# --- Compute max power and k ---
reactor.max_power = reactor.compute_max_power()
reactor.k = reactor.compute_k()

print("=== REACTOR BASIC TESTS ===")
print("Pmax =", reactor.max_power)
print("k =", reactor.k)

# --- Test compute_power ---
print("\n=== POWER TESTS ===")
print("Power at B=0 (should be Pmax):", reactor.compute_power(0.0))
print("Power at B=1 (should be ~1e-6):", reactor.compute_power(1.0))

# --- Test monotonicity ---
print("\n=== MONOTONICITY TEST ===")
B_vals = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
P_vals = np.array([reactor.compute_power(b) for b in B_vals])

print("B values:", B_vals)
print("P values:", P_vals)

# Should decrease
print("Check decreasing:", np.all(P_vals[:-1] >= P_vals[1:]))

# --- Optional inverse test (if implemented) ---
if hasattr(reactor, "compute_control_bars_insertion"):
    print("\n=== INVERSE FUNCTION TEST ===")
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p = reactor.compute_power(b)
        b_back = reactor.compute_control_bars_insertion(p)
        print(f"B={b:.2f} -> P={p:.6e} -> B_back={b_back:.6f}")