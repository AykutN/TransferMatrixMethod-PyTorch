# Transfer Matrix Method (TMM) - Physics-Informed AI in PyTorch

## 🚀 Overview

This project implements a **Differentiable Transfer Matrix Method (TMM)** engine using **PyTorch**. It is designed for the rapid simulation and optimization of thin-film optoelectronic devices (e.g., Organic Solar Cells, Perovskites).

By leveraging PyTorch's automatic differentiation (`autograd`), this engine allows for:
1.  **Inverse Design:** Directly optimizing layer thicknesses for target optical properties (AVT, Jph) using Gradient Descent.
2.  **Physics-Informed AI:** Integrating rigorous optical physics directly into Machine Learning pipelines (RL, generative models) without relying on slow external solvers like MATLAB.
3.  **GPU Acceleration:** Batch processing thousands of simulations simultaneously.

**Key Features:**
*   **Speed:** ~25x-100x faster than traditional MATLAB implementations.
*   **Differentiability:** Enables gradient-based optimization of physical parameters.
*   **Dynamic Configuration:** Easily change materials and layer stacks via configuration files.
*   **MATLAB Equivalence:** Verified to be mathematically equivalent to standard MATLAB TMM codes (Error < 1e-4).

## 🛠 Installation

### Prerequisites
*   Python 3.8+
*   PyTorch
*   NumPy
*   Pandas
*   SciPy

### Setup
1.  Clone the repository:
    ```bash
    git clone https://github.com/AykutN/TransferMatrixMethod-PyTorch.git
    cd TransferMatrixMethod-PyTorch
    ```

2.  (Optional) Create a Conda environment:
    ```bash
    conda create -n tmm_torch python=3.9
    conda activate tmm_torch
    ```

3.  Install dependencies:
    ```bash
    pip install torch numpy pandas scipy matplotlib
    ```

## 📖 Usage Guide

### 1. Verification (MATLAB vs PyTorch)
To demonstrate that the PyTorch engine produces the exact same results as the legacy MATLAB code:

```bash
python demo_comparison.py
```
*This script runs a side-by-side comparison for a specific 7-layer structure and outputs the Average Visible Transmittance (AVT) and Short-Circuit Current Density (Jph).*

### 2. AI-Driven Optimization (Inverse Design)
To finding the optimal layer thicknesses for a specific goal (e.g., AVT > 25% and Maximize Jph):

```bash
python optimize_design.py
```
*This uses Gradient Descent optimization to "design" the device structure in seconds.*

### 3. Training Reinforcement Learning Agent (DQN)
To train the request Deep Q-Network agent using the new fast engine:

```bash
python src/main.py
```

## ⚙️ Configuration

### Changing Layer Structure
You can define your material stack in `config/settings.py`.

```python
# Material Stack Definition (Order matters!)
# 'Vac' (Air) is added automatically at top and bottom.
LAYERS = ['MoO3', 'Ag', 'ZnO', 'PTB7_PCBM', 'MoO3', 'Ag']
```

### Changing Bounds
Constraints for layer thicknesses (in nanometers) can be set in `config/settings.py`:

```python
BOUNDS = {
    "d1": (10, 100),  # MoO3
    "d2": (5, 20),    # Ag
    # ...
}
```

### Adding New Materials
1.  Add your material's refractive index file (`.txt`) to `src/matlab/Materials/Properties/` (Format: `Wavelength(um)  n  k`).
2.  Add the material name to the `LAYERS` list in `config/settings.py`.

## 📂 Project Structure

*   `src/tmm_torch.py`: **Core Physics Engine.** The differentiable TMM implementation.
*   `src/environment_torch.py`: RL Environment wrapper using the PyTorch engine.
*   `optimize_design.py`: Script for gradient-based inverse design.
*   `config/settings.py`: Central configuration for physical constants and hyperparameters.
*   `src/matlab/`: Contains legacy MATLAB codes and Material data.

## 🔬 Scientific Validation
The engine has been validated against established Transfer Matrix Method implementations. Discrepancies in `Jph` are negligible (< 0.01%) and attributed to differences in interpolation methods between SciPy and MATLAB. `AVT` calculations are mathematically identical.

## 📝 License
[MIT License](LICENSE)
