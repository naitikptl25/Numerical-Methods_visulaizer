import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.linalg import eigh
from scipy.special import legendre
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set page config
st.set_page_config(page_title="Math Methods Combo App", layout="centered")

# Common styling
st.markdown("""
    <style>
        .stApp {
            background-color: #f9f9f9;
        }
        .stButton>button {
            background-color: #4b4beb;
            color: white;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📘 Numerical Methods Visualizer")

# Sidebar for global sliders
st.sidebar.header("Global Settings")
selected_project = st.sidebar.radio("Select Project:", ["Gauss-Legendre Quadrature", "IVP/BVP Solver"])

# ---- Project 1: Gauss-Legendre ----
if selected_project == "Gauss-Legendre Quadrature":
    st.header("📐 Gauss-Legendre Quadrature")
    n = st.slider("Select n (number of points)", 1, 64, 8)

    def compute_jacobi_method(n):
        k = np.arange(1., n)
        b = k / np.sqrt(4 * k * k - 1)
        A = np.diag(b, -1) + np.diag(b, 1)
        roots, eigenvectors = eigh(A)
        weights = 2 * (eigenvectors[0, :] ** 2)
        return roots, weights

    def companion_matrix(coeffs):
        n = len(coeffs) - 1
        C = np.zeros((n, n))
        for i in range(n - 1):
            C[i + 1, i] = 1
        for i in range(n):
            C[i, -1] = -(coeffs[n - i] / coeffs[0])
        return C

    def legendre_polynomial_derivative(coeffs):
        return np.polyder(coeffs)

    def compute_gauss_legendre_weights(n, coeffs, roots):
        derivative_coeffs = legendre_polynomial_derivative(coeffs)
        derivative_values = np.polyval(derivative_coeffs, roots)
        weights = 2 / ((1 - roots**2) * derivative_values**2)
        return weights

    roots_jacobi, weights_jacobi = compute_jacobi_method(n)
    coeffs = legendre(n).coefficients
    roots_companion = np.sort(np.linalg.eigvals(companion_matrix(coeffs)).real)
    weights_companion = compute_gauss_legendre_weights(n, coeffs, roots_companion)

    st.subheader("📊 Weights vs Roots Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(roots_jacobi, weights_jacobi, 'o-', label="Jacobi Method", color='blue')
    ax.plot(roots_companion, weights_companion, 'x--', label="Companion Method", color='green')
    ax.set_xlabel("Roots")
    ax.set_ylabel("Weights")
    ax.set_title(f"Weights vs Roots for n = {n}")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Jacobi Roots")
        st.text("\n".join([f"{r:.6f}" for r in roots_jacobi]))
    with col2:
        st.subheader("Companion Roots")
        st.text("\n".join([f"{r:.6f}" for r in roots_companion]))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Jacobi Weights")
        st.text("\n".join([f"{w:.6f}" for w in weights_jacobi]))
    with col4:
        st.subheader("Companion Weights")
        st.text("\n".join([f"{w:.6f}" for w in weights_companion]))

# ---- Project 2: IVP/BVP Solver ----
else:
    st.header("🧮 IVP/BVP Solver")
    P = st.slider("Select value of P", -5.0, 10.0, 5.0, 0.1)

    h = 0.01
    N = 100
    s_values = np.arange(0, 10, 0.1)

    def euler_method(P, s):
        u, v = 0, s
        for _ in range(N):
            u += h * v
            v += h * (-P)
        return u

    def calculate_optimal_s(P):
        return min(s_values, key=lambda s: abs(euler_method(P, s) - 1))

    def analytical_solution(P, y):
        return (-P / 2) * y**2 + (1 + P / 2) * y

    def explicit_euler(P, h, N, s):
        u, v = np.zeros(N + 1), np.zeros(N + 1)
        v[0] = s
        for n in range(N):
            u[n + 1] = u[n] + h * v[n]
            v[n + 1] = v[n] + h * (-P)
        return u

    def implicit_euler(P, h, s):
        steps = int(1 / h)
        u, v = np.zeros(steps + 1), np.zeros(steps + 1)
        v[0] = s
        for n in range(steps):
            v_new = v[n] - h * P
            u[n + 1] = u[n] + h * v_new
            v[n + 1] = v_new
        return u

    def finite_difference_bvp(P, N, u0, u1):
        h = 1.0 / N
        A = np.zeros((N, N))
        b = np.zeros(N)
        for i in range(1, N - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = 1 / h**2, -2 / h**2, 1 / h**2
        b[1:N - 1] = -P
        A[0, 0], b[0] = 1, u0
        A[-1, -1], b[-1] = 1, u1
        return np.linalg.solve(A, b)

    s_opt = calculate_optimal_s(P)
    st.success(f"For d²u/dy² = -{P}, the optimal u'(0) such that u(1)=1 (boundary condition is satisfied) is approximately {s_opt:.4f}")

    y_vals = np.linspace(0, 1, N + 1)
    y_fd = np.linspace(0, 1, 100)

    u_ana = analytical_solution(P, y_vals)
    u_exp = explicit_euler(P, h, N, s_opt)
    u_imp = implicit_euler(P, h, s_opt)
    u_fd = finite_difference_bvp(P, len(y_fd), 0, 1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(u_ana, y_vals, label="Analytical", linestyle='--', color='black')
    ax2.plot(u_exp, y_vals, label="Explicit Euler", color='blue')
    ax2.plot(u_imp, y_vals, label="Implicit Euler", linestyle='-', marker='x', color='green', markersize=4)
    ax2.plot(u_fd, y_fd, label="Finite Difference", linestyle='None', marker='o', color='red', markersize=3)
    ax2.set_xlabel("u(y)")
    ax2.set_ylabel("y")
    ax2.set_title(f"Solution Comparison for P = {P}")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Footer
st.markdown("---")

