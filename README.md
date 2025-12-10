## `README.md` for `engine-digital-twin`

# Engine Digital Twin

An advanced real-time, crank-synchronous simulation framework for Internal Combustion Engines (ICE), designed to serve as a high-fidelity environment for the development and training of Engine Control Unit (ECU) algorithms, particularly using Reinforcement Learning (RL).

## ⚠️ Important Attribution and Licensing

### Foundation and Acknowledgement

The core physics functions responsible for the thermodynamic and mechanical modeling within `engine_model.py` (specifically in **`physics_functions.py`**) are derived from the foundational work of the repository:

> **`octarine123/automotive-ICE-modelling`**

We extend our sincere thanks to the original author for providing a robust starting point for the engine physics simulation.

### Licensing

This project is licensed under the **MIT License**.

> **This is a permissive license that allows for great freedom in how the software is used, modified, and distributed, provided the original copyright and license notice are included.  The full license if provided in the LICENSE file**

-----

## 🚀 Getting Started

### Prerequisites

To run this simulation, you will need:

  * Python 3.x
  * The required dependencies (e.g., NumPy, Matplotlib for the dashboard).

> **[Placeholder: Add exact instructions on how to install dependencies, e.g., `pip install -r requirements.txt`]**

### How to Run

The simulation is executed via `main.py`, which uses `argparse` to set the running mode.

#### Basic Execution

Run the simulation in the default **Wide Open Throttle (WOT)** mode:

```bash
python main.py
```

#### Running in Different Modes

Use the `--mode` flag to select the simulation scenario:

| Mode Option | Description |
| :--- | :--- |
| `idle` | Simulates the engine at idle conditions. |
| `wot` | Simulates the engine at Wide Open Throttle (default). |
| `dyno` | Simulates a typical chassis or engine dynamometer run. |
| `circuit` | Simulates driver inputs on a defined race circuit profile. |
| `rl` | **(Future Use)** Sets up the environment for Reinforcement Learning training. |

**Example (Dyno Mode):**

```bash
python main.py --mode dyno
```

#### Debug Mode

To run the simulation without the overhead of logging and the graphical dashboard (for speed and testing purposes), use the `--debug` flag:

```bash
python main.py --debug
```

-----

## 🏗️ System Architecture

The simulation is managed by the `SimulationManager` class, which orchestrates the flow of data between three primary components every crank degree.

### Core Components

| Component | File | Class | Description |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | `main.py` | `SimulationManager` | Manages the time step and the 720-degree engine cycle loop. |
| **Driver Input** | `driver_input.py` | `DriverInput` | Generates throttle, load, and environmental inputs based on the selected `--mode`. |
| **Controller** | `ecu_controller.py` | `ECUController` | The control logic (the ECU) that takes sensor readings and outputs commands (fuel, spark, etc.). |
| **Engine Model** | `engine_model.py` | `EngineModel` | The physics-based simulation of the ICE, calculating the engine's new state. |
| **Telemetry** | `logger.py` | `Logger` | Records all simulation outputs to a CSV file. |
| **Visualization** | `dashboard.py` | `Dashboard` | Provides real-time graphical feedback (dyno, gauges). |

-----

## 🔌 Data & Modeling Details

### Engine Model Parameters

> **[Placeholder: Define the key engine specifications used for the simulation, e.g., bore, stroke, number of cylinders, compression ratio, displacement, etc.]**

### ECU Calibration and Maps

> **[Placeholder: Detail how the ECU determines its outputs. Describe the default fuel/spark maps or tables it uses (before RL optimization).]**

### Data Output

> **[Placeholder: Detail the format of the output log files created by `Logger`, including column headers and units.]**