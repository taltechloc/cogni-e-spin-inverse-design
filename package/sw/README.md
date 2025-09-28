# 🔬 eSpinID: Inverse Design Framework for Electrospun Nanofibers

**eSpinID** (**E**lectro**Spin**ning **I**nverse **D**esign) is a Python package that implements a robust, data-driven framework for predicting and controlling the diameter of electrospun nanofibers. It moves beyond traditional trial-and-error methods by integrating a high-fidelity surrogate model with advanced optimization techniques to determine the optimal process parameters for a specified target fiber diameter.

This package provides the core code supporting the findings in the manuscript:
*Inverse Design of Electrospun Nanofibers: An XGBoost and PSO Framework for Targeted Diameter Control*

## 🌟 Core Methodology

The framework relies on a two-stage approach:

1.  **Surrogate Modeling:** An **Extreme Gradient Boosting (XGBoost)** model acts as a reliable "digital twin," predicting the fiber diameter from four key process parameters. XGBoost delivered the highest predictive accuracy with a testing $R^{2}$ of $0.896 \pm 0.06$.
2.  **Inverse Optimization:** The trained XGBoost model is integrated with **Particle Swarm Optimization (PSO)**. The optimization routine finds the experimentally feasible input parameters that minimize the error between the model's prediction and the desired target diameter ($d_{target}$). PSO achieved the highest accuracy with an $R^{2}$ of $0.9942$ and a Mean Absolute Error (MAE) of $\approx 1.16\text{ nm}$.

## 💻 Installation

### Prerequisites

Ensure you have Python (version 3.8 or higher) installed. Using a virtual environment is strongly recommended to manage dependencies.

### Step 1: Clone the Repository

The package is hosted under the larger project name but is installed as `eSpinID`.

```bash
git clone [https://github.com/taltechloc/cogni-e-spin-inverse-design.git](https://github.com/taltechloc/cogni-e-spin-inverse-design.git)
cd cogni-e-spin-inverse-design

```

### Step 2: Install the Package
Navidate to "install" folder, and install the package.

``` bach
cd install
./install_package.sh
```

### 🔗 Code and Data Availability

#### Code Repository
The code for this study is publicly available on GitHub at:

https://github.com/taltechloc/cogni-e-spin-inverse-design.git
#### Data Source
The experimental data (96 trials on poly(vinyl alcohol) or PVA) used to train the surrogate model was sourced from:

Ziabari, M., Mottaghitalab, V. & Haghi, A. process. A new approach for optimization of electrospun nanofiber formation process. Korean J. Chem. Eng.

 27(1), 340–354 (2010). https://doi.org/10.1007/s11814-009-0309-1 

### 📜 Authors and Funding
Authors: Mehrab Mahdian, Ferenc Ender, and Tamas Pardy
Funding: This study was funded by Estonian Research Council PSG897.
