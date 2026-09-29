# Monte Carlo Stochastic Demand Simulator 📈

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Testing](https://img.shields.io/badge/Tests-PyTest-green.svg)]()

## Abstract
A stochastic simulation engine implementing Monte Carlo methods to model variable demand and distribute load across a multi-reactor system. The architecture is decoupled to isolate data ingestion, routing logic, and state management.

## System Architecture
* **`DemandGenerator.py`**: Ingests and models stochastic processes to feed the core system.
* **`ControlModule.py`**: Routing logic that allocates demand across active reactors based on predefined constraints.
* **`Reactor.py` & JSON Configs**: State management is isolated. Reactor parameters are loaded dynamically via `/Reactors` JSON schemas, avoiding hardcoded variables.
* **`Metrics.py`**: Captures execution latency and tracks system state for post-simulation analysis.

## Quantitative Benchmarking
*Performance and latency tracking are critical. Current benchmarks indicate:*
* **Throughput:** Processes 100,000 iterations successfully.
* **Time Complexity:** Optimized for multi-state simulation routing.

## Installation & Execution

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/SebastianQuintero1/MonteCarlo-Demand-Simulator.git](https://github.com/SebastianQuintero1/MonteCarlo-Demand-Simulator.git)
cd MonteCarlo-Demand-Simulator
pip install -r requirements.txt