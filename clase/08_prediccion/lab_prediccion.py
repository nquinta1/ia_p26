#!/usr/bin/env python3
"""
Laboratorio: Predicción (imágenes para las notas)

Uso:
    cd clase/08_prediccion
    python lab_prediccion.py

Genera imágenes en:
    clase/08_prediccion/images/

Dependencias: numpy, matplotlib, scipy
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.stats import norm
from scipy.optimize import minimize

# -----------------------------------------------------------------------------
# Styling (same vibe as lab_optimization.py)
# -----------------------------------------------------------------------------

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11

COLORS = {
    "blue": "#2E86AB",
    "red": "#E94F37",
    "green": "#27AE60",
    "gray": "#7F8C8D",
    "orange": "#F39C12",
    "purple": "#8E44AD",
}

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def _save(fig, name: str) -> None:
    out = IMAGES_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Generada: {out.name}")


# -----------------------------------------------------------------------------
# 1) Three points, infinite curves
# -----------------------------------------------------------------------------


def plot_restriccion_tres_puntos():
    """3 data points with multiple different curves passing through all 3."""
    # Three data points
    xp = np.array([1.0, 3.0, 5.0])
    yp = np.array([2.0, 4.5, 3.0])

    x = np.linspace(0.2, 5.8, 500)

    # Curve 1: Polynomial (degree 2) through 3 points — unique parabola
    coeffs2 = np.polyfit(xp, yp, 2)
    y_parabola = np.polyval(coeffs2, x)

    # Curve 2: Cubic spline (natural) — smooth interpolation
    cs = CubicSpline(xp, yp, bc_type="natural")
    y_spline = cs(x)

    # Curve 3: High-degree polynomial wiggle (degree 8 via Lagrange + perturbation)
    # Build a degree-6 poly that passes through the 3 points but wiggles
    # Use: p(x) = parabola(x) + (x-1)(x-3)(x-5)*q(x)
    vanish = (x - 1) * (x - 3) * (x - 5)
    y_wiggle1 = y_parabola + vanish * 0.3 * np.sin(2.0 * x)

    # Curve 4: Another wiggle variant
    y_wiggle2 = y_parabola + vanish * 0.2 * np.cos(3.0 * x)

    # Curve 5: Step-like interpolation with sigmoid transitions
    y_step = y_parabola + vanish * 0.15 * np.sin(5.0 * x)

    fig, ax = plt.subplots(figsize=(10, 6))

    curves = [
        (y_parabola, COLORS["blue"], "Parábola", 2.5),
        (y_spline, COLORS["green"], "Spline cúbico", 2.0),
        (y_wiggle1, COLORS["orange"], "Polinomio oscilante A", 1.5),
        (y_wiggle2, COLORS["purple"], "Polinomio oscilante B", 1.5),
        (y_step, COLORS["red"], "Polinomio oscilante C", 1.5),
    ]

    for y_curve, color, label, lw in curves:
        ax.plot(x, y_curve, color=color, linewidth=lw, label=label, alpha=0.85)

    # Data points on top
    ax.scatter(xp, yp, color="black", s=120, zorder=10, edgecolors="white",
               linewidths=2, label="Datos observados")

    ax.set_title("Datos finitos, infinitas hipótesis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 7)
    ax.text(0.5, 0.5, "Todas pasan por los 3 puntos.\n¿Cuál es la \"correcta\"?",
            transform=ax.transAxes, fontsize=11, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffcc", alpha=0.9))
    _save(fig, "01_restriccion_tres_puntos.png")


# -----------------------------------------------------------------------------
# 2) Hierarchy of prediction objectives (tree diagram)
# -----------------------------------------------------------------------------


def plot_objetivos_jerarquia():
    """Tree diagram showing P(X,Y,Z) branching to sub-objectives."""
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 9)
    ax.axis("off")

    # Color scheme by type
    c_joint = "#2E86AB"    # blue — joint
    c_sup = "#E94F37"      # red — supervised
    c_unsup = "#27AE60"    # green — unsupervised
    c_causal = "#F39C12"   # orange — causal

    # Node positions: (x, y, label, color, fontsize)
    nodes = [
        (5, 8, "P(X, Y, Z)", c_joint, 13),
        # Level 2
        (2, 6, "P(Y|X,Z)", c_sup, 11),
        (5, 6, "P(Y)", c_sup, 11),
        (8, 6, "P(X,Z)", c_unsup, 11),
        # Level 3
        (0.8, 4, "E[Y|X,Z]", c_sup, 10),
        (3.2, 4, "Qα[Y|X,Z]", c_sup, 10),
        (5, 4, "E[Y]", c_sup, 10),
        (7, 4, "P(X)", c_unsup, 10),
        (9.2, 4, "$\\phi$(X)", c_unsup, 10),
        # Level 4 (causal — separate)
        (5, 1.5, "P(Y|do(X))", c_causal, 11),
    ]

    # Draw nodes as boxes
    box_style = dict(boxstyle="round,pad=0.4", alpha=0.9)
    for x, y, label, color, fs in nodes:
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9))

    # Edges: (from_idx, to_idx)
    edges = [
        (0, 1), (0, 2), (0, 3),  # level 1 -> 2
        (1, 4), (1, 5),          # P(Y|X,Z) -> E, Q
        (2, 6),                   # P(Y) -> E[Y]
        (3, 7), (3, 8),          # P(X,Z) -> P(X), phi(X)
    ]

    for i, j in edges:
        x0, y0 = nodes[i][0], nodes[i][1]
        x1, y1 = nodes[j][0], nodes[j][1]
        ax.annotate("", xy=(x1, y1 + 0.4), xytext=(x0, y0 - 0.4),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                    color=COLORS["gray"], alpha=0.7))

    # Causal node — dashed line from joint + structure
    ax.annotate("", xy=(5, 2.0), xytext=(5, 5.5),
                arrowprops=dict(arrowstyle="-|>", lw=1.5,
                                color=c_causal, linestyle="--", alpha=0.8))
    ax.text(6.3, 3.5, "+ Estructura\n   Causal (DAG)", fontsize=10, color=c_causal,
            fontstyle="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", alpha=0.9))

    # Legend
    legend_items = [
        mpatches.Patch(color=c_joint, label="Distribución conjunta"),
        mpatches.Patch(color=c_sup, label="Supervisado (con Y)"),
        mpatches.Patch(color=c_unsup, label="No supervisado (sin Y)"),
        mpatches.Patch(color=c_causal, label="Causal (requiere DAG)"),
    ]
    ax.legend(handles=legend_items, loc="lower left", fontsize=10,
              framealpha=0.9)

    ax.set_title("Jerarquía de objetivos de predicción", fontsize=14, pad=15)
    _save(fig, "02_objetivos_jerarquia.png")


# -----------------------------------------------------------------------------
# 3) Deductive vs Inductive approaches
# -----------------------------------------------------------------------------


def plot_deductivo_vs_inductivo():
    """Left: physics trajectory (clean parabola), Right: data-driven (scatter + fit)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Left panel: Deductive (physics-based) --
    t = np.linspace(0, 3, 300)
    v0, angle, g = 15, 55, 9.81
    angle_rad = np.radians(angle)
    x_phys = v0 * np.cos(angle_rad) * t
    y_phys = v0 * np.sin(angle_rad) * t - 0.5 * g * t**2

    mask = y_phys >= 0
    ax1.plot(x_phys[mask], y_phys[mask], color=COLORS["blue"], linewidth=2.5,
             label="Trayectoria teórica")
    ax1.fill_between(x_phys[mask], 0, y_phys[mask], alpha=0.08, color=COLORS["blue"])

    # Equations annotation
    ax1.text(0.05, 0.95, "Ecuaciones de Newton:\n"
             r"$x = v_0 \cos\theta \cdot t$" + "\n"
             r"$y = v_0 \sin\theta \cdot t - \frac{1}{2}gt^2$",
             transform=ax1.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F4F8", alpha=0.9))

    ax1.scatter([0], [0], color=COLORS["orange"], s=80, zorder=5)
    ax1.annotate("Lanzamiento", xy=(0, 0), xytext=(3, 4),
                 arrowprops=dict(arrowstyle="->"), fontsize=9)
    ax1.set_title("Deductivo: la teoría define el modelo", fontsize=12)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.5, 12)

    # -- Right panel: Inductive (data-driven) --
    # Generate noisy data from same trajectory
    t_data = np.linspace(0.1, 2.4, 15)
    x_data = v0 * np.cos(angle_rad) * t_data + np.random.normal(0, 0.4, len(t_data))
    y_data = (v0 * np.sin(angle_rad) * t_data - 0.5 * g * t_data**2
              + np.random.normal(0, 0.5, len(t_data)))

    ax2.scatter(x_data, y_data, color=COLORS["red"], s=50, zorder=5,
                edgecolors="black", label="Datos observados")

    # Fit a polynomial (degree 2) — purely data-driven
    coeffs = np.polyfit(x_data, y_data, 2)
    x_fit = np.linspace(min(x_data) - 1, max(x_data) + 2, 300)
    y_fit = np.polyval(coeffs, x_fit)
    y_fit_data = np.polyval(coeffs, x_data)
    residuals = y_data - y_fit_data
    sigma = np.std(residuals)

    ax2.plot(x_fit, y_fit, color=COLORS["green"], linewidth=2.5,
             label="Ajuste polinomial (grado 2)")
    ax2.fill_between(x_fit, y_fit - 2 * sigma, y_fit + 2 * sigma,
                     alpha=0.15, color=COLORS["green"], label="±2σ (banda de confianza)")

    ax2.text(0.05, 0.95, "Sin ecuaciones previas:\n"
             "Solo ajustar f(x) a los datos\n"
             r"$\hat{y} = a x^2 + bx + c$",
             transform=ax2.transAxes, fontsize=9, va="top",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#FEF9E7", alpha=0.9))

    ax2.set_title("Inductivo: los datos definen el modelo", fontsize=12)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.legend(fontsize=9, loc="upper right")
    ax2.set_ylim(-0.5, 12)

    fig.suptitle("D1: Dos enfoques al mismo fenómeno (trayectoria de un proyectil)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "03_deductivo_vs_inductivo.png")


# -----------------------------------------------------------------------------
# 4) Bayesian updating: Prior → Likelihood → Posterior
# -----------------------------------------------------------------------------


def plot_bayesiano_vs_frequentist():
    """3-panel: Prior, Likelihood, Posterior with MLE annotation."""
    theta = np.linspace(0, 1, 500)

    # Prior: Beta(2, 5) — belief that θ is low
    from scipy.stats import beta as beta_dist
    prior = beta_dist.pdf(theta, 2, 5)

    # Data: 6 heads out of 10 flips
    n, k = 10, 6
    likelihood = theta**k * (1 - theta)**(n - k)
    likelihood = likelihood / likelihood.max() * prior.max()  # scale for visualization

    # Posterior: Beta(2+6, 5+4) = Beta(8, 9)
    posterior = beta_dist.pdf(theta, 2 + k, 5 + (n - k))

    # MLE
    mle = k / n

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Prior
    ax = axes[0]
    ax.fill_between(theta, prior, alpha=0.3, color=COLORS["blue"])
    ax.plot(theta, prior, color=COLORS["blue"], linewidth=2)
    ax.set_title("Prior: P(θ)", fontsize=12)
    ax.set_xlabel("θ (probabilidad de cara)")
    ax.set_ylabel("Densidad")
    ax.text(0.55, 0.85, "Beta(2, 5)\n\"Creo que θ es bajo\"",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4F8", alpha=0.9))

    # Panel 2: Likelihood
    ax = axes[1]
    # Show actual likelihood (unscaled)
    lik_true = theta**k * (1 - theta)**(n - k)
    lik_scaled = lik_true / lik_true.max()
    ax.fill_between(theta, lik_scaled * prior.max(), alpha=0.3, color=COLORS["orange"])
    ax.plot(theta, lik_scaled * prior.max(), color=COLORS["orange"], linewidth=2)
    ax.axvline(mle, color=COLORS["red"], linestyle="--", linewidth=2, label=f"MLE = {mle:.1f}")
    ax.scatter([mle], [prior.max()], color=COLORS["red"], s=100, zorder=5, edgecolors="black")
    ax.annotate(f"Estimador frecuentista\nθ̂ = k/n = {mle:.1f}",
                xy=(mle, prior.max()), xytext=(mle + 0.15, prior.max() * 0.7),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["red"]),
                fontsize=9, color=COLORS["red"],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    ax.set_title("Likelihood: P(datos|θ)", fontsize=12)
    ax.set_xlabel("θ")
    ax.text(0.05, 0.85, f"Datos: {k} caras\nen {n} lanzamientos",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", alpha=0.9))
    ax.legend(fontsize=9)

    # Panel 3: Posterior
    ax = axes[2]
    # Show prior ghost
    ax.plot(theta, prior, color=COLORS["blue"], linewidth=1, alpha=0.3,
            linestyle="--", label="Prior")
    ax.fill_between(theta, posterior, alpha=0.3, color=COLORS["green"])
    ax.plot(theta, posterior, color=COLORS["green"], linewidth=2.5, label="Posterior")
    post_mode = (2 + k - 1) / (2 + k + 5 + (n - k) - 2)
    ax.axvline(post_mode, color=COLORS["green"], linestyle=":", linewidth=1.5)
    ax.axvline(mle, color=COLORS["red"], linestyle="--", linewidth=1.5, alpha=0.5,
               label=f"MLE = {mle:.1f}")
    ax.set_title("Posterior: P(θ|datos)", fontsize=12)
    ax.set_xlabel("θ")
    ax.text(0.4, 0.85, "Beta(8, 9)\nPrior + datos →\ncreencia actualizada",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8E8", alpha=0.9))
    ax.legend(fontsize=9)

    fig.suptitle("D2: Actualización Bayesiana vs estimación Frequentist",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "04_bayesiano_vs_frequentist.png")


# -----------------------------------------------------------------------------
# 5) Epistemic vs Aleatoric uncertainty
# -----------------------------------------------------------------------------


def plot_incertidumbre_epistemica_vs_aleatoria():
    """Two subplots: few data (wide band) vs many data (narrow band, irreducible noise)."""
    # True function
    f_true = lambda x: 2.0 * np.sin(1.5 * x) + 0.5 * x
    noise_std = 0.8
    x_plot = np.linspace(0, 6, 300)
    y_true = f_true(x_plot)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # -- Left: Few data points (epistemic dominates) --
    n_few = 6
    x_few = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    y_few = f_true(x_few) + np.random.normal(0, noise_std, n_few)

    # Simple polynomial fit
    coeffs_few = np.polyfit(x_few, y_few, 3)
    y_pred_few = np.polyval(coeffs_few, x_plot)

    # Bootstrap for uncertainty bands
    n_boot = 200
    y_boots_few = np.zeros((n_boot, len(x_plot)))
    for i in range(n_boot):
        idx = np.random.choice(n_few, n_few, replace=True)
        c = np.polyfit(x_few[idx], y_few[idx], 3)
        y_boots_few[i] = np.polyval(c, x_plot)

    y_lo_few = np.percentile(y_boots_few, 5, axis=0)
    y_hi_few = np.percentile(y_boots_few, 95, axis=0)

    ax1.fill_between(x_plot, y_lo_few, y_hi_few, alpha=0.25, color=COLORS["blue"],
                     label="Incertidumbre (epistémica + aleatoria)")
    ax1.plot(x_plot, y_pred_few, color=COLORS["blue"], linewidth=2, label="Ajuste")
    ax1.plot(x_plot, y_true, "--", color=COLORS["gray"], linewidth=1, alpha=0.6,
             label="Función real")
    ax1.scatter(x_few, y_few, color=COLORS["red"], s=70, zorder=5,
                edgecolors="black", label=f"Datos (n={n_few})")

    ax1.set_title("Pocos datos → Incertidumbre epistémica alta", fontsize=11)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.text(0.5, 0.05, "Banda ancha: no sabemos\ndónde está la función real",
             transform=ax1.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF3CD", alpha=0.9))

    # -- Right: Many data points (aleatoric dominates) --
    n_many = 100
    x_many = np.random.uniform(0, 6, n_many)
    y_many = f_true(x_many) + np.random.normal(0, noise_std, n_many)

    coeffs_many = np.polyfit(x_many, y_many, 5)
    y_pred_many = np.polyval(coeffs_many, x_plot)

    y_boots_many = np.zeros((n_boot, len(x_plot)))
    for i in range(n_boot):
        idx = np.random.choice(n_many, n_many, replace=True)
        c = np.polyfit(x_many[idx], y_many[idx], 5)
        y_boots_many[i] = np.polyval(c, x_plot)

    y_lo_many = np.percentile(y_boots_many, 5, axis=0)
    y_hi_many = np.percentile(y_boots_many, 95, axis=0)

    ax2.fill_between(x_plot, y_lo_many, y_hi_many, alpha=0.25, color=COLORS["green"],
                     label="Incertidumbre (mayormente aleatoria)")
    ax2.plot(x_plot, y_pred_many, color=COLORS["green"], linewidth=2, label="Ajuste")
    ax2.plot(x_plot, y_true, "--", color=COLORS["gray"], linewidth=1, alpha=0.6,
             label="Función real")
    ax2.scatter(x_many, y_many, color=COLORS["red"], s=15, zorder=3,
                alpha=0.5, label=f"Datos (n={n_many})")

    ax2.set_title("Muchos datos → Solo incertidumbre aleatoria", fontsize=11)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.text(0.5, 0.05, "Banda estrecha: sabemos dónde está\nla función, pero el ruido es irreducible",
             transform=ax2.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F8E8", alpha=0.9))

    fig.suptitle("Incertidumbre epistémica (reducible) vs aleatoria (irreducible)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "05_incertidumbre_epistemica_vs_aleatoria.png")


# -----------------------------------------------------------------------------
# 6) Discriminative vs Generative
# -----------------------------------------------------------------------------


def plot_discriminativo_vs_generativo():
    """Left: decision boundary only. Right: density contours + Bayes boundary."""
    # Generate two 2D Gaussian classes
    n_per_class = 150
    mu0, mu1 = np.array([2, 2]), np.array([4, 4])
    cov0 = np.array([[0.8, 0.3], [0.3, 0.6]])
    cov1 = np.array([[0.6, -0.2], [-0.2, 0.8]])

    X0 = np.random.multivariate_normal(mu0, cov0, n_per_class)
    X1 = np.random.multivariate_normal(mu1, cov1, n_per_class)

    # Grid for decision boundary / contours
    xg = np.linspace(-0.5, 7, 200)
    yg = np.linspace(-0.5, 7, 200)
    Xg, Yg = np.meshgrid(xg, yg)
    grid = np.c_[Xg.ravel(), Yg.ravel()]

    # Compute Bayes optimal boundary from known distributions
    from scipy.stats import multivariate_normal
    rv0 = multivariate_normal(mu0, cov0)
    rv1 = multivariate_normal(mu1, cov1)
    p0 = rv0.pdf(grid).reshape(Xg.shape)
    p1 = rv1.pdf(grid).reshape(Xg.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # -- Left: Discriminativo (boundary only) --
    ax1.scatter(X0[:, 0], X0[:, 1], color=COLORS["blue"], s=20, alpha=0.5, label="Clase 0")
    ax1.scatter(X1[:, 0], X1[:, 1], color=COLORS["red"], s=20, alpha=0.5, label="Clase 1")
    # Decision boundary: p0 = p1
    ax1.contour(Xg, Yg, p0 - p1, levels=[0], colors=["black"], linewidths=2.5)
    # Fill regions
    ax1.contourf(Xg, Yg, p0 - p1, levels=[-100, 0, 100],
                 colors=[COLORS["red"], COLORS["blue"]], alpha=0.08)
    ax1.set_title("Discriminativo: P(Y|X)\nSolo la frontera de decisión", fontsize=11)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.text(0.5, 0.05, "No modela cómo se distribuyen\nlos datos, solo los separa",
             transform=ax1.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # -- Right: Generativo (density contours + boundary) --
    ax2.scatter(X0[:, 0], X0[:, 1], color=COLORS["blue"], s=20, alpha=0.3)
    ax2.scatter(X1[:, 0], X1[:, 1], color=COLORS["red"], s=20, alpha=0.3)
    # Density contours for each class
    ax2.contour(Xg, Yg, p0, levels=5, colors=[COLORS["blue"]], linewidths=1.2, alpha=0.8)
    ax2.contour(Xg, Yg, p1, levels=5, colors=[COLORS["red"]], linewidths=1.2, alpha=0.8)
    # Boundary emerges from Bayes
    ax2.contour(Xg, Yg, p0 - p1, levels=[0], colors=["black"], linewidths=2.5,
                linestyles="--")
    ax2.set_title("Generativo: P(X|Y) para cada clase\nFrontera emerge de Bayes", fontsize=11)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    # Legend patches
    ax2.plot([], [], color=COLORS["blue"], linewidth=1.5, label="Contornos P(X|Y=0)")
    ax2.plot([], [], color=COLORS["red"], linewidth=1.5, label="Contornos P(X|Y=1)")
    ax2.plot([], [], color="black", linewidth=2, linestyle="--", label="Frontera de Bayes")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.text(0.5, 0.05, "Modela la densidad de cada clase;\nla frontera sale de P(Y|X) $\\propto$ P(X|Y)P(Y)",
             transform=ax2.transAxes, ha="center", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    fig.suptitle("D4: Discriminativo vs Generativo", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "06_discriminativo_vs_generativo.png")


# -----------------------------------------------------------------------------
# 7) Regularization: No reg vs L2 vs L1
# -----------------------------------------------------------------------------


def plot_regularizacion_l1_l2():
    """Polynomial regression: overfit, L2, L1 with coefficient bar charts."""
    # Generate noisy data from a simple function
    n = 20
    x_data = np.sort(np.random.uniform(0, 1, n))
    y_true_fn = lambda x: np.sin(2 * np.pi * x)
    y_data = y_true_fn(x_data) + np.random.normal(0, 0.3, n)

    x_plot = np.linspace(0, 1, 300)
    degree = 12

    # Build Vandermonde matrix
    def vander(x, d):
        return np.column_stack([x**i for i in range(d + 1)])

    V_data = vander(x_data, degree)
    V_plot = vander(x_plot, degree)

    # 1) No regularization (OLS)
    coeffs_ols = np.linalg.lstsq(V_data, y_data, rcond=None)[0]
    y_ols = V_plot @ coeffs_ols

    # 2) L2 regularization (Ridge)
    lam_l2 = 1e-2
    I = np.eye(degree + 1)
    I[0, 0] = 0  # don't penalize intercept
    coeffs_l2 = np.linalg.solve(V_data.T @ V_data + lam_l2 * I, V_data.T @ y_data)
    y_l2 = V_plot @ coeffs_l2

    # 3) L1 regularization (Lasso via scipy minimize)
    def lasso_loss(w, lam):
        residuals = y_data - V_data @ w
        return 0.5 * np.mean(residuals**2) + lam * np.sum(np.abs(w[1:]))

    lam_l1 = 5e-3
    res = minimize(lasso_loss, np.zeros(degree + 1), args=(lam_l1,), method="L-BFGS-B")
    coeffs_l1 = res.x
    y_l1 = V_plot @ coeffs_l1

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 9),
                              gridspec_kw={"height_ratios": [2, 1]})

    configs = [
        ("Sin regularización (overfit)", y_ols, coeffs_ols, COLORS["red"]),
        ("L2 (Ridge): funciones suaves", y_l2, coeffs_l2, COLORS["blue"]),
        ("L1 (Lasso): coeficientes sparse", y_l1, coeffs_l1, COLORS["green"]),
    ]

    for j, (title, y_pred, coeffs, color) in enumerate(configs):
        ax_top = axes[0, j]
        ax_bot = axes[1, j]

        # Top: fit
        ax_top.scatter(x_data, y_data, color="black", s=30, zorder=5, label="Datos")
        ax_top.plot(x_plot, y_true_fn(x_plot), "--", color=COLORS["gray"],
                    linewidth=1, label="f(x) real")
        ax_top.plot(x_plot, y_pred, color=color, linewidth=2, label="Ajuste")
        ax_top.set_title(title, fontsize=11)
        ax_top.set_ylim(-2, 2)
        ax_top.legend(fontsize=8, loc="upper right")
        if j == 0:
            ax_top.set_ylabel("y")

        # Bottom: coefficient bar chart
        idx = np.arange(len(coeffs))
        ax_bot.bar(idx, coeffs, color=color, edgecolor="black", linewidth=0.5, alpha=0.8)
        ax_bot.set_xlabel("Grado del coeficiente")
        ax_bot.axhline(0, color="black", linewidth=0.5)
        if j == 0:
            ax_bot.set_ylabel("Valor del coeficiente")
        # Show max coeff magnitude
        ax_bot.set_title(f"|max coef| = {np.max(np.abs(coeffs)):.1f}", fontsize=9)

    fig.suptitle("D5: Efecto de la regularización en regresión polinomial (grado 12)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, "07_regularizacion_l1_l2.png")


# -----------------------------------------------------------------------------
# 8) Methods positioned in 2D space (conceptual map)
# -----------------------------------------------------------------------------


def plot_metodos_5d_mapa():
    """~15 methods on a 2D scatter, colored by D1, shaped by D4."""
    # Manually placed methods in a 2D conceptual space:
    # x-axis ~ "Data-driven ← → Theory-driven" (D1)
    # y-axis ~ "Simple/Flat ← → Complex/Structured" (D4)
    methods = [
        # (name, x, y, D1_color, D4_marker)
        # D1: Inductivo=blue, Deductivo=orange, Híbrido=green
        # D4: Flat=o, Latente=D, Grafo=s, Causal=*
        ("Reg. Lineal", 0.8, 0.3, "blue", "o"),
        ("Ridge/Lasso", 0.75, 0.4, "blue", "o"),
        ("Logistic Reg.", 0.7, 0.35, "blue", "o"),
        ("Random Forest", 0.85, 0.5, "blue", "o"),
        ("XGBoost", 0.9, 0.55, "blue", "o"),
        ("DNN", 0.95, 0.7, "blue", "o"),
        ("Gaussian Process", 0.6, 0.6, "blue", "o"),
        ("K-Means", 0.85, 0.2, "blue", "o"),
        ("PCA", 0.75, 0.25, "blue", "D"),
        ("VAE", 0.9, 0.75, "blue", "D"),
        ("GAN", 0.95, 0.8, "blue", "D"),
        ("GPT/LLM", 1.0, 0.95, "blue", "D"),
        ("BERT", 0.95, 0.85, "blue", "D"),
        ("HMM", 0.4, 0.65, "green", "s"),
        ("Kalman Filter", 0.3, 0.7, "green", "D"),
        ("DSGE", 0.1, 0.8, "orange", "s"),
        ("Bayesian Net", 0.35, 0.75, "green", "s"),
        ("Physics-NN", 0.5, 0.7, "green", "o"),
        ("Causal Forest", 0.7, 0.85, "blue", "*"),
        ("Double ML", 0.65, 0.8, "blue", "*"),
    ]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Background shading for D1 regions
    ax.axvspan(-0.05, 0.25, alpha=0.06, color=COLORS["orange"])
    ax.axvspan(0.25, 0.55, alpha=0.06, color=COLORS["green"])
    ax.axvspan(0.55, 1.1, alpha=0.06, color=COLORS["blue"])
    ax.text(0.1, -0.06, "Deductivo", fontsize=10, ha="center", color=COLORS["orange"],
            fontweight="bold")
    ax.text(0.4, -0.06, "Híbrido", fontsize=10, ha="center", color=COLORS["green"],
            fontweight="bold")
    ax.text(0.82, -0.06, "Inductivo", fontsize=10, ha="center", color=COLORS["blue"],
            fontweight="bold")

    marker_map = {"o": "o", "D": "D", "s": "s", "*": "*"}
    marker_size = {"o": 80, "D": 80, "s": 80, "*": 150}
    marker_label = {"o": "Flat", "D": "Latente", "s": "Grafo/PGM", "*": "Causal"}

    plotted_markers = set()
    for name, mx, my, d1_color, d4_marker in methods:
        color = COLORS[d1_color]
        label = marker_label[d4_marker] if d4_marker not in plotted_markers else None
        ax.scatter(mx, my, color=color, marker=marker_map[d4_marker],
                   s=marker_size[d4_marker], edgecolors="black", linewidths=0.8,
                   zorder=5, label=label)
        plotted_markers.add(d4_marker)

        # Label
        ax.annotate(name, xy=(mx, my), xytext=(5, 5), textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7))

    ax.set_xlabel("← Basado en teoría          Basado en datos →", fontsize=11)
    ax.set_ylabel("← Simple / Flat          Complejo / Estructurado →", fontsize=11)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Mapa conceptual de métodos de predicción (D1 × D4)", fontsize=13)

    # Custom legend for markers (D4)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gray"],
               markersize=8, label="Flat"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=COLORS["gray"],
               markersize=8, label="Latente"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLORS["gray"],
               markersize=8, label="Grafo/PGM"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=COLORS["gray"],
               markersize=12, label="Causal"),
        mpatches.Patch(color=COLORS["blue"], alpha=0.3, label="D1: Inductivo"),
        mpatches.Patch(color=COLORS["green"], alpha=0.3, label="D1: Híbrido"),
        mpatches.Patch(color=COLORS["orange"], alpha=0.3, label="D1: Deductivo"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              title="D4: Arquitectura / D1: Fuente", title_fontsize=10,
              framealpha=0.9)
    _save(fig, "08_metodos_5d_mapa.png")


# =============================================================================
# Main
# =============================================================================


def main():
    print("Generando imágenes para módulo 08: Predicción\n")
    plot_restriccion_tres_puntos()
    plot_objetivos_jerarquia()
    plot_deductivo_vs_inductivo()
    plot_bayesiano_vs_frequentist()
    plot_incertidumbre_epistemica_vs_aleatoria()
    plot_discriminativo_vs_generativo()
    plot_regularizacion_l1_l2()
    plot_metodos_5d_mapa()
    print(f"\n✓ Todas las imágenes generadas en {IMAGES_DIR}")


if __name__ == "__main__":
    main()
