# Numerical Methods Visualizer

This is an interactive Streamlit web app to **visualize and compare numerical methods** for solving mathematical problems like:

- **Gauss-Legendre Quadrature** (for numerical integration)
- **IVP/BVP Solvers** (Initial/Boundary Value Problems using Euler and Finite Difference methods)

---

## Features

## Gauss-Legendre Quadrature
- Computes roots and weights using:
  - **Jacobi method**
  - **Companion matrix method**
- Plots **Weights vs Roots**
- Displays numerical values for easy comparison

## IVP/BVP Solver
- Solves \( \frac{d^2u}{dy^2} = -P \) using:
  - **Explicit Euler Method**
  - **Implicit Euler Method**
  - **Finite Difference Method**
- Compares all with the **Analytical Solution**
- Interactive slider to adjust **P parameter**

---

## Tech Stack

- **Python**
- **Streamlit** – for frontend app
- **NumPy**, **Matplotlib**, **SciPy**, **SymPy** – for numerical methods

---
