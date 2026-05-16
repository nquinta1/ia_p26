#!/usr/bin/env python3
"""
lab_ml.py — Genera las imágenes pedagógicas para clase/22_machine_learning/

Ejecución:
    cd clase/22_machine_learning && python lab_ml.py

Genera 16 imágenes en:
    clase/22_machine_learning/images/

Dependencias: numpy, matplotlib, scipy
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_fn
from numpy.polynomial.legendre import legvander

# ── Styling ───────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 160,
})

COLORS = {
    "blue":   "#2E86AB",
    "red":    "#E94F37",
    "green":  "#27AE60",
    "gray":   "#7F8C8D",
    "orange": "#F39C12",
    "purple": "#8E44AD",
    "light":  "#ECF0F1",
    "dark":   "#2C3E50",
    "teal":   "#1ABC9C",
    "yellow": "#F1C40F",
}

ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
SIGMA = 0.3
SIGMA2 = SIGMA ** 2  # = 0.09  Bayes error floor


def _save(fig, name: str) -> None:
    out = IMAGES_DIR / name
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Generada: {out.name}")


# ── Mathematical helpers ──────────────────────────────────────────────────────
def true_f(x: np.ndarray) -> np.ndarray:
    return np.sin(np.pi * x)


def make_data(m: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.random.uniform(-1, 1, m)
    y = true_f(x) + np.random.normal(0, SIGMA, m)
    return x, y


def fit_legendre(x_tr: np.ndarray, y_tr: np.ndarray,
                 deg: int, lam: float = 0.0) -> np.ndarray:
    """Ridge regression with Legendre basis.

    Objective:  (1/m) ||X w - y||^2  +  lam ||w||^2
    Solution:   w* = (X^T X/m + lam I)^{-1}  X^T y/m
    """
    X = legvander(x_tr, deg)
    m = len(x_tr)
    A = X.T @ X / m + lam * np.eye(deg + 1)
    b_vec = X.T @ y_tr / m
    return np.linalg.solve(A, b_vec)


def predict_legendre(x: np.ndarray, w: np.ndarray, deg: int) -> np.ndarray:
    return legvander(x, deg) @ w


def mse(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean((y_hat - y_true) ** 2))


# Dense reference grid (used across many plots)
X_GRID = np.linspace(-1, 1, 500)
Y_GRID_TRUE = true_f(X_GRID)

# Reproducible large test set for generalization error estimates
_x_big, _y_big = make_data(2000)


# ── 01 — Población vs. muestra ───────────────────────────────────────────────
def plot_poblacion_vs_muestra() -> None:
    """Joint density p(x,y) (left) and 30 i.i.d. samples (right)."""
    x_g = np.linspace(-1.05, 1.05, 120)
    y_g = np.linspace(-1.6, 1.6, 120)
    XX, YY = np.meshgrid(x_g, y_g)

    # p(y|x) = N(y; sin(πx), σ²);  p(x) = Uniform → constant weight
    mu_grid = true_f(XX)
    density = np.exp(-0.5 * ((YY - mu_grid) / SIGMA) ** 2) / (SIGMA * np.sqrt(2 * np.pi))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    cf = ax.contourf(XX, YY, density, levels=20, cmap="Blues")
    ax.plot(X_GRID, Y_GRID_TRUE, color=COLORS["orange"], lw=2.5,
            label=r"$f^*(x) = \sin(\pi x)$")
    ax.set_title(r"Distribución generadora $p_\mathrm{data}(x,y)$ — desconocida",
                 fontsize=12)
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.legend(loc="upper right", fontsize=10)
    plt.colorbar(cf, ax=ax, shrink=0.8, label="densidad")

    np.random.seed(0)
    x_s, y_s = make_data(30)
    ax2 = axes[1]
    ax2.scatter(x_s, y_s, color=COLORS["blue"], s=55, alpha=0.8, zorder=3,
                label=r"$\mathcal{D}$: $m=30$ muestras")
    ax2.plot(X_GRID, Y_GRID_TRUE, color=COLORS["orange"], lw=2.5, linestyle="--",
             label=r"$f^*(x)$ — no observada")
    ax2.set_xlim(-1.05, 1.05); ax2.set_ylim(-1.6, 1.6)
    ax2.set_title(r"Lo que observamos: $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}$",
                  fontsize=12)
    ax2.set_xlabel("$x$"); ax2.set_ylabel("$y$")
    ax2.legend(loc="upper right", fontsize=10)

    fig.suptitle("Distribución generadora vs. muestra observada", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "01_poblacion_vs_muestra.png")


# ── 02 — Error stack ─────────────────────────────────────────────────────────
def plot_error_stack() -> None:
    """Stacked bars: Bayes error + ε_approx + ε_estim for 3 λ values."""
    # Compute with degree-5 polynomial on fixed train/test split
    np.random.seed(1)
    x_tr, y_tr = make_data(30)
    x_te, y_te = make_data(5000)  # large test → ≈ true risk
    y_true_te = true_f(x_te)

    deg = 5
    lambdas = [5.0, 0.01, 0.00001]

    # With huge data: estimate R(θ*_F(λ)) ≈ best-in-class risk
    np.random.seed(2)
    x_big, y_big = make_data(5000)
    y_true_big = true_f(x_big)

    eps_approx, eps_estim = [], []
    for lam in lambdas:
        # R(θ̂) with m=30
        w_small = fit_legendre(x_tr, y_tr, deg, lam)
        r_hat = mse(predict_legendre(x_te, w_small, deg), y_true_te)
        # R(θ*_F) ≈ test error with huge data (estimation error → 0)
        w_big = fit_legendre(x_big, y_big, deg, lam)
        r_star_f = mse(predict_legendre(x_te, w_big, deg), y_true_te)

        eps_approx.append(max(r_star_f - SIGMA2, 0.0))
        eps_estim.append(max(r_hat - r_star_f, 0.0))

    labels = [r"$\lambda$ grande" "\n(subajuste)", r"$\lambda^*$" "\n(óptimo)",
              r"$\lambda$ pequeño" "\n(sobreajuste)"]
    x_pos = np.arange(3)
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(8, 5))
    b_bayes = ax.bar(x_pos, [SIGMA2] * 3, bar_w,
                     label=r"$R^*$ (error de Bayes, $\sigma^2$)",
                     color=COLORS["gray"], alpha=0.7)
    b_approx = ax.bar(x_pos, eps_approx, bar_w, bottom=[SIGMA2] * 3,
                      label=r"$\varepsilon_\mathrm{approx}$ (sesgo del modelo)",
                      color=COLORS["blue"], alpha=0.85)
    b_estim = ax.bar(x_pos, eps_estim, bar_w,
                     bottom=[SIGMA2 + ea for ea in eps_approx],
                     label=r"$\varepsilon_\mathrm{estim}$ (datos finitos)",
                     color=COLORS["red"], alpha=0.85)

    ax.set_xticks(x_pos); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"$R(\hat\theta) - R^*$ desglosado", fontsize=11)
    ax.set_title("Descomposición del error en tres estados de regularización", fontsize=12)
    ax.legend(loc="upper center", fontsize=10)
    ax.axhline(SIGMA2, color=COLORS["gray"], lw=1.2, linestyle="--")

    for bar_group in [b_bayes, b_approx, b_estim]:
        for rect in bar_group:
            h = rect.get_height()
            if h > 0.005:
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + h / 2, f"{h:.3f}",
                        ha="center", va="center", fontsize=9, color="white",
                        fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_error_stack.png")


# ── 03 — Underfitting / overfitting ──────────────────────────────────────────
def plot_underfitting_overfitting() -> None:
    """1×3 panels: degree 1, 4, 14 polynomial fits on the same 15 training points."""
    np.random.seed(7)
    m = 15
    x_tr, y_tr = make_data(m)
    x_te, y_te = make_data(300)
    x_plot = np.linspace(-1, 1, 400)

    degrees = [1, 4, 14]
    titles = ["Grado 1\n(subajuste)", "Grado 4\n(buen ajuste)", "Grado 14\n(sobreajuste)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    for ax, deg, title in zip(axes, degrees, titles):
        w = fit_legendre(x_tr, y_tr, deg, lam=0.0)
        y_pred_tr = predict_legendre(x_tr, w, deg)
        y_pred_te = predict_legendre(x_te, w, deg)
        y_plot = predict_legendre(x_plot, w, deg)

        mse_tr = mse(y_pred_tr, y_tr)
        mse_te = mse(y_pred_te, y_te)

        ax.scatter(x_tr, y_tr, color=COLORS["blue"], s=55, zorder=4,
                   label="Entrenamiento", alpha=0.85)
        ax.plot(x_plot, true_f(x_plot), color=COLORS["gray"], lw=2,
                linestyle="--", label=r"$f^*(x)$")
        y_clip = np.clip(y_plot, -3, 3)
        ax.plot(x_plot, y_clip, color=COLORS["red"], lw=2.2,
                label=f"Grado {deg}")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-2.5, 2.5)
        ax.set_title(f"{title}\nTrain MSE={mse_tr:.3f}  Test MSE={mse_te:.3f}",
                     fontsize=11)
        ax.set_xlabel("$x$")
        if ax is axes[0]:
            ax.set_ylabel("$y$")
            ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Subajuste, buen ajuste y sobreajuste (mismos 15 puntos)", fontsize=13)
    fig.tight_layout()
    _save(fig, "03_underfitting_overfitting.png")


# ── 04 — Curva de capacidad ───────────────────────────────────────────────────
def plot_curva_capacidad() -> None:
    """Train and test MSE vs. polynomial degree. Log y-scale, Bayes floor."""
    np.random.seed(10)
    x_tr, y_tr = make_data(20)
    x_te, y_te = make_data(500)

    degrees = list(range(0, 19))
    train_mses, test_mses = [], []

    for deg in degrees:
        w = fit_legendre(x_tr, y_tr, deg, lam=1e-10)  # tiny reg for stability
        train_mses.append(mse(predict_legendre(x_tr, w, deg), y_tr))
        test_mses.append(mse(predict_legendre(x_te, w, deg), y_te))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(degrees, train_mses, "o-", color=COLORS["blue"], lw=2,
                markersize=6, label="Error de entrenamiento $\\hat{R}$")
    ax.semilogy(degrees, test_mses, "s-", color=COLORS["red"], lw=2,
                markersize=6, label="Error de prueba $R(\\hat{\\theta})$")
    ax.axhline(SIGMA2, color=COLORS["gray"], lw=1.8, linestyle="--",
               label=f"Error de Bayes $\\sigma^2 = {SIGMA2}$")

    best_deg = int(np.argmin(test_mses))
    ax.axvline(best_deg, color=COLORS["green"], lw=1.2, linestyle=":",
               alpha=0.8, label=f"Grado óptimo $p^*={best_deg}$")

    ax.set_xlabel("Grado del polinomio $p$ (capacidad →)")
    ax.set_ylabel("MSE (escala log)")
    ax.set_title("Curva de capacidad: error de entrenamiento y generalización", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xticks(degrees)
    fig.tight_layout()
    _save(fig, "04_curva_capacidad.png")


# ── 05 — Ridge curves ─────────────────────────────────────────────────────────
def plot_ridge_curves() -> None:
    """Effect of λ: overlaid fitted curves (top) and U-curve in λ (bottom)."""
    np.random.seed(7)
    m = 15
    x_tr, y_tr = make_data(m)
    x_te, y_te = make_data(500)
    x_plot = np.linspace(-1, 1, 400)
    deg = 14

    # Top panel: overlaid curves
    lambdas_curves = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    lam_labels = [r"$\lambda=10^{-4}$", r"$\lambda=10^{-3}$", r"$\lambda=10^{-2}$",
                  r"$\lambda=0.1$", r"$\lambda=1.0$"]
    curve_colors = [COLORS["red"], COLORS["orange"], COLORS["green"],
                    COLORS["blue"], COLORS["purple"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    ax1.scatter(x_tr, y_tr, color=COLORS["dark"], s=55, zorder=5,
                label="Datos de entrenamiento", alpha=0.8)
    ax1.plot(x_plot, true_f(x_plot), color=COLORS["gray"], lw=2,
             linestyle="--", label=r"$f^*(x)$", zorder=4)
    for lam, label, col in zip(lambdas_curves, lam_labels, curve_colors):
        w = fit_legendre(x_tr, y_tr, deg, lam)
        y_pred = np.clip(predict_legendre(x_plot, w, deg), -3, 3)
        ax1.plot(x_plot, y_pred, color=col, lw=1.8, label=label, alpha=0.85)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")
    ax1.set_title(f"Efecto de $\\lambda$ en el modelo (grado {deg})", fontsize=12)
    ax1.legend(fontsize=9, ncol=2)

    # Bottom panel: U-curve in λ
    lambdas_sweep = np.logspace(-5, 2, 60)
    tr_mses = [mse(predict_legendre(x_tr, fit_legendre(x_tr, y_tr, deg, l), deg), y_tr)
               for l in lambdas_sweep]
    te_mses = [mse(predict_legendre(x_te, fit_legendre(x_tr, y_tr, deg, l), deg), y_te)
               for l in lambdas_sweep]

    best_idx = int(np.argmin(te_mses))
    lam_star = lambdas_sweep[best_idx]

    ax2.semilogx(lambdas_sweep, tr_mses, color=COLORS["blue"], lw=2,
                 label="Error de entrenamiento")
    ax2.semilogx(lambdas_sweep, te_mses, color=COLORS["red"], lw=2,
                 label="Error de prueba (generalización)")
    ax2.axhline(SIGMA2, color=COLORS["gray"], lw=1.5, linestyle="--",
                label=f"$\\sigma^2 = {SIGMA2}$")
    ax2.axvline(lam_star, color=COLORS["green"], lw=1.5, linestyle=":",
                label=f"$\\lambda^* \\approx {lam_star:.4f}$")
    ax2.set_ylim(0, min(max(te_mses) * 1.1, 2.0))
    ax2.set_xlabel(r"$\lambda$ (escala log)")
    ax2.set_ylabel("MSE")
    ax2.set_title(r"Curva U de regularización: MSE vs. $\lambda$", fontsize=12)
    ax2.legend(fontsize=10)

    fig.tight_layout(pad=2.5)
    _save(fig, "05_ridge_curves.png")


# ── 06 — Generalization bound ────────────────────────────────────────────────
def plot_generalization_bound() -> None:
    """VC bound sqrt(d_VC/m) vs m for several model classes."""
    m_vals = np.logspace(2, 5, 200)
    d_vc_list = [3, 11, 21, 100]
    model_labels = [
        r"Regresión lineal en $\mathbb{R}^2$  ($d_{VC}=3$)",
        r"Regresión logística en $\mathbb{R}^{10}$  ($d_{VC}=11$)",
        r"Polinomio grado 5 en $\mathbb{R}^2$  ($d_{VC}=21$)",
        r"Red neuronal chica  ($d_{VC}=100$)",
    ]
    line_colors = [COLORS["green"], COLORS["blue"], COLORS["orange"], COLORS["red"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    for d_vc, label, col in zip(d_vc_list, model_labels, line_colors):
        bound = np.sqrt(d_vc / m_vals)
        ax.loglog(m_vals, bound, lw=2, color=col, label=label)

    ax.axhline(0.10, color=COLORS["gray"], lw=1.2, linestyle="--", alpha=0.7,
               label="Umbral 0.10")
    ax.axhline(0.05, color=COLORS["dark"], lw=1.2, linestyle=":", alpha=0.7,
               label="Umbral 0.05")
    ax.set_xlabel("Tamaño de entrenamiento $m$")
    ax.set_ylabel(r"Penalización $\sqrt{d_{VC}/m}$")
    ax.set_title(r"Bound de generalización $\approx \sqrt{d_{VC}/m}$ para distintos modelos",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    _save(fig, "06_generalization_bound.png")


# ── 07 — k-Fold diagram ──────────────────────────────────────────────────────
def plot_kfold_diagrama() -> None:
    """Colored grid showing 5-fold cross-validation split."""
    k = 5
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.3, k + 2.2)
    ax.set_ylim(-0.5, k + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")

    cell_w, cell_h = 1.0, 0.7

    for row in range(k):
        fold_idx = k - 1 - row
        for col in range(k):
            is_val = (col == fold_idx)
            fc = COLORS["orange"] if is_val else COLORS["blue"]
            ec = COLORS["dark"]
            rect = Rectangle((col * cell_w, row * cell_h), cell_w, cell_h,
                              facecolor=fc, edgecolor=ec, lw=1.5)
            ax.add_patch(rect)
            label = "Val" if is_val else "Train"
            ax.text(col * cell_w + cell_w / 2, row * cell_h + cell_h / 2,
                    label, ha="center", va="center", fontsize=9.5,
                    color="white", fontweight="bold")

        fold_label = f"Pliegue {fold_idx + 1}"
        ax.text(-0.2, row * cell_h + cell_h / 2, fold_label,
                ha="right", va="center", fontsize=10)
        err_label = f"$e_{fold_idx + 1}$"
        ax.text(k * cell_w + 0.2, row * cell_h + cell_h / 2, err_label,
                ha="left", va="center", fontsize=11, color=COLORS["red"])

    # Average annotation
    ax.text(k * cell_w + 0.2, -0.35,
            r"$\hat{e} = \frac{1}{5}\sum_{i=1}^{5} e_i$",
            ha="left", va="center", fontsize=11, color=COLORS["dark"])

    # Legend patches
    patch_tr = mpatches.Patch(facecolor=COLORS["blue"], edgecolor=COLORS["dark"],
                               label="Entrenamiento")
    patch_val = mpatches.Patch(facecolor=COLORS["orange"], edgecolor=COLORS["dark"],
                                label="Validación")
    ax.legend(handles=[patch_tr, patch_val], loc="upper center",
              bbox_to_anchor=(0.45, 1.08), ncol=2, fontsize=10)
    ax.set_title("Validación cruzada de 5 pliegues (k-Fold CV)", fontsize=12, pad=20)
    fig.tight_layout()
    _save(fig, "07_kfold_diagrama.png")


# ── 08 — Optimismo del error de entrenamiento ────────────────────────────────
def plot_optimismo_entrenamiento() -> None:
    """KDE + scatter showing training error is biased below test error."""
    np.random.seed(20)
    B = 500
    deg = 5
    m = 25
    lam = 0.0

    x_te_fixed, y_te_fixed = make_data(1000)
    train_errs, test_errs = [], []

    for _ in range(B):
        x_tr, y_tr = make_data(m)
        w = fit_legendre(x_tr, y_tr, deg, lam)
        train_errs.append(mse(predict_legendre(x_tr, w, deg), y_tr))
        test_errs.append(mse(predict_legendre(x_te_fixed, w, deg), y_te_fixed))

    train_errs = np.array(train_errs)
    test_errs = np.array(test_errs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: KDE
    x_range = np.linspace(0, max(test_errs.max(), train_errs.max()) * 1.1, 300)
    kde_tr = stats.gaussian_kde(train_errs)(x_range)
    kde_te = stats.gaussian_kde(test_errs)(x_range)
    ax1.fill_between(x_range, kde_tr, alpha=0.45, color=COLORS["blue"],
                     label=f"Error entrenamiento  (media={train_errs.mean():.3f})")
    ax1.fill_between(x_range, kde_te, alpha=0.45, color=COLORS["red"],
                     label=f"Error prueba  (media={test_errs.mean():.3f})")
    ax1.plot(x_range, kde_tr, color=COLORS["blue"], lw=2)
    ax1.plot(x_range, kde_te, color=COLORS["red"], lw=2)
    ax1.axvline(SIGMA2, color=COLORS["gray"], lw=1.5, linestyle="--",
                label=f"$\\sigma^2={SIGMA2}$")
    ax1.set_xlabel("MSE"); ax1.set_ylabel("Densidad")
    ax1.set_title("Distribución del error: entrenamiento vs. prueba\n"
                  f"($B={B}$ experimentos, grado {deg}, $m={m}$)", fontsize=11)
    ax1.legend(fontsize=9)

    # Right: scatter
    lim = max(train_errs.max(), test_errs.max()) * 1.05
    ax2.scatter(train_errs, test_errs, alpha=0.25, s=18, color=COLORS["blue"])
    ax2.plot([0, lim], [0, lim], color=COLORS["red"], lw=2, linestyle="--",
             label="$y = x$ (sin sesgo)")
    ax2.set_xlabel("Error de entrenamiento")
    ax2.set_ylabel("Error de prueba (generalización)")
    ax2.set_title("Error de entrenamiento vs. error de prueba\n"
                  "Puntos encima de la diagonal → optimismo", fontsize=11)
    ax2.legend(fontsize=10)
    frac_above = np.mean(test_errs > train_errs)
    ax2.text(0.05, 0.92, f"Puntos encima diagonal: {frac_above:.0%}",
             transform=ax2.transAxes, fontsize=10, color=COLORS["dark"])

    fig.tight_layout()
    _save(fig, "08_optimismo_entrenamiento.png")


# ── 09 — Varianza del estimador k-fold ───────────────────────────────────────
def plot_varianza_kfold() -> None:
    """Variance of k-fold estimate vs. k across B=300 independent datasets."""
    np.random.seed(30)
    B = 300
    m = 40
    deg = 3
    lam = 0.01
    k_vals = [2, 5, 10, 20, 40]  # k=40 is LOO for m=40

    def kfold_estimate(x: np.ndarray, y: np.ndarray, k: int) -> float:
        idx = np.random.permutation(len(x))
        fold_errors = []
        for i in range(k):
            val_idx = idx[i * m // k: (i + 1) * m // k]
            tr_idx = np.concatenate([idx[:i * m // k], idx[(i + 1) * m // k:]])
            if len(tr_idx) < deg + 2:
                continue
            w = fit_legendre(x[tr_idx], y[tr_idx], deg, lam)
            fold_errors.append(mse(predict_legendre(x[val_idx], w, deg), y[val_idx]))
        return float(np.mean(fold_errors))

    all_estimates = {k: [] for k in k_vals}
    for _ in range(B):
        x_d, y_d = make_data(m)
        for k in k_vals:
            all_estimates[k].append(kfold_estimate(x_d, y_d, k))

    data_for_box = [all_estimates[k] for k in k_vals]
    labels = [f"$k={k}$" if k < m else f"$k={k}$\n(LOO)" for k in k_vals]
    stdevs = [np.std(all_estimates[k]) for k in k_vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    bp = ax1.boxplot(data_for_box, labels=labels, patch_artist=True,
                     medianprops=dict(color=COLORS["dark"], lw=2))
    for patch, k in zip(bp["boxes"], k_vals):
        patch.set_facecolor(COLORS["blue"])
        patch.set_alpha(0.6)
    ax1.axhline(SIGMA2, color=COLORS["gray"], lw=1.5, linestyle="--",
                label=f"$\\sigma^2={SIGMA2}$")
    ax1.set_xlabel("$k$"); ax1.set_ylabel("Estimado del error de generalización")
    ax1.set_title(f"Distribución del estimado k-fold\n($B={B}$ datasets, $m={m}$)",
                  fontsize=11)
    ax1.legend(fontsize=10)

    ax2.plot(k_vals, stdevs, "o-", color=COLORS["red"], lw=2, markersize=8)
    ax2.set_xlabel("$k$"); ax2.set_ylabel("Desviación estándar del estimado")
    ax2.set_title("Variabilidad del estimado k-fold vs. $k$", fontsize=11)
    ax2.set_xticks(k_vals)
    ax2.set_xticklabels(labels)

    fig.suptitle("Estimado de validación cruzada: variabilidad vs. número de pliegues",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "09_varianza_kfold.png")


# ── 10 — Bias² + Var + σ² vs. degree ────────────────────────────────────────
def plot_bias_varianza_componentes() -> None:
    """MSE = sigma² + Bias² + Var decomposed, for degree 0–12."""
    np.random.seed(40)
    B = 500
    m_tr = 30
    deg_max = 12
    lam = 0.0
    n_test = 200
    x_te = np.linspace(-1, 1, n_test)
    y_te_true = true_f(x_te)

    degrees = list(range(0, deg_max + 1))
    bias2_list, var_list, mse_list = [], [], []

    for deg in degrees:
        preds = np.zeros((B, n_test))
        for b in range(B):
            x_tr, y_tr = make_data(m_tr)
            w = fit_legendre(x_tr, y_tr, deg, lam + 1e-10)
            preds[b] = predict_legendre(x_te, w, deg)

        mean_pred = preds.mean(axis=0)
        bias2 = float(np.mean((mean_pred - y_te_true) ** 2))
        var = float(np.mean(preds.var(axis=0)))
        bias2_list.append(bias2)
        var_list.append(var)
        mse_list.append(SIGMA2 + bias2 + var)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(degrees, [SIGMA2] * len(degrees), "--", color=COLORS["gray"],
            lw=2, label=f"$\\sigma^2 = {SIGMA2}$ (error de Bayes)")
    ax.plot(degrees, bias2_list, "o-", color=COLORS["blue"], lw=2.2,
            markersize=6, label=r"Sesgo$^2$ (↓ con capacidad)")
    ax.plot(degrees, var_list, "s-", color=COLORS["orange"], lw=2.2,
            markersize=6, label=r"Varianza (↑ con capacidad)")
    ax.plot(degrees, mse_list, "^-", color=COLORS["red"], lw=2.5,
            markersize=7, label=r"MSE $= \sigma^2 + $ Sesgo$^2$ + Varianza")

    best = int(np.argmin(mse_list))
    ax.axvline(best, color=COLORS["green"], lw=1.5, linestyle=":",
               label=f"Grado óptimo $p^*={best}$")

    ax.set_xlabel("Grado del polinomio $p$")
    ax.set_ylabel("Error cuadrático")
    ax.set_title(f"Descomposición MSE = $\\sigma^2$ + Sesgo² + Varianza\n"
                 f"($B={B}$ muestras, $m={m_tr}$ puntos de entrenamiento)", fontsize=12)
    ax.legend(fontsize=10, loc="upper center")
    ax.set_xticks(degrees)
    ax.set_ylim(0, min(1.8, max(mse_list) * 1.1))
    fig.tight_layout()
    _save(fig, "10_bias_varianza_componentes.png")


# ── 11 — Efecto de m en la estimación ────────────────────────────────────────
def plot_efecto_m_estimacion() -> None:
    """Bias² stays flat; Var decays as C/m as m grows (fixed degree=5)."""
    np.random.seed(50)
    B = 500
    deg = 5
    lam = 0.0
    n_test = 200
    x_te = np.linspace(-1, 1, n_test)
    y_te_true = true_f(x_te)
    m_vals = [10, 20, 40, 80, 160, 320]

    bias2_list, var_list = [], []

    for m in m_vals:
        preds = np.zeros((B, n_test))
        for b in range(B):
            x_tr, y_tr = make_data(m)
            w = fit_legendre(x_tr, y_tr, deg, lam + 1e-10)
            preds[b] = predict_legendre(x_te, w, deg)
        mean_pred = preds.mean(axis=0)
        bias2_list.append(float(np.mean((mean_pred - y_te_true) ** 2)))
        var_list.append(float(np.mean(preds.var(axis=0))))

    # Fit C/m to variance
    m_arr = np.array(m_vals, dtype=float)
    C_fit = var_list[0] * m_vals[0]
    var_fit = C_fit / m_arr

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.loglog(m_arr, bias2_list, "o-", color=COLORS["blue"], lw=2.2,
              markersize=7, label=r"Sesgo$^2$ (≈ constante en $m$)")
    ax.loglog(m_arr, var_list, "s-", color=COLORS["red"], lw=2.2,
              markersize=7, label="Varianza (empírica)")
    ax.loglog(m_arr, var_fit, "--", color=COLORS["orange"], lw=1.8,
              label=f"Ajuste $C/m$ ($C={C_fit:.3f}$)")
    ax.axhline(SIGMA2, color=COLORS["gray"], lw=1.5, linestyle=":",
               label=f"$\\sigma^2={SIGMA2}$")

    ax.set_xlabel("Tamaño de entrenamiento $m$ (escala log)")
    ax.set_ylabel("Error (escala log)")
    ax.set_title(f"Más datos → menos varianza, pero no menos sesgo\n"
                 f"(polinomio grado {deg}, $B={B}$ muestras)", fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "11_efecto_m_estimacion.png")


# ── 12 — MLE y regresión lineal (Gaussiana) ───────────────────────────────────
def plot_mle_gaussiano() -> None:
    """Left: Gaussian likelihood at training points. Right: NLL = MSE + const."""
    np.random.seed(60)
    m = 12
    x_tr, y_tr = make_data(m)
    deg = 1
    w = fit_legendre(x_tr, y_tr, deg, lam=0.0)
    x_plot = np.linspace(-1, 1, 300)
    y_hat_plot = predict_legendre(x_plot, w, deg)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: scatter + fitted line + Gaussian bells
    ax1.scatter(x_tr, y_tr, color=COLORS["blue"], s=65, zorder=5,
                label="Puntos de entrenamiento")
    ax1.plot(x_plot, y_hat_plot, color=COLORS["red"], lw=2.5,
             label=r"$\hat{f}(x;\mathbf{w})$ ajustado", zorder=4)
    ax1.plot(x_plot, true_f(x_plot), color=COLORS["gray"], lw=1.5,
             linestyle="--", label=r"$f^*(x)$", zorder=3)

    # Draw Gaussian bells at 3 specific points
    bell_xs = [-0.6, 0.0, 0.6]
    y_g_range = np.linspace(-1.8, 1.8, 300)
    for x_b in bell_xs:
        y_center = predict_legendre(np.array([x_b]), w, deg)[0]
        bell = stats.norm.pdf(y_g_range, y_center, SIGMA)
        bell_scaled = bell / bell.max() * 0.18
        ax1.fill_betweenx(y_g_range, x_b, x_b + bell_scaled,
                          alpha=0.35, color=COLORS["orange"])
        ax1.plot(x_b + bell_scaled, y_g_range, color=COLORS["orange"], lw=1)
        # Find closest training point to x_b
        closest = np.argmin(np.abs(x_tr - x_b))
        ax1.plot([x_b, x_b + stats.norm.pdf(y_tr[closest], y_center, SIGMA) /
                  stats.norm.pdf(y_center, y_center, SIGMA) * 0.18],
                 [y_tr[closest], y_tr[closest]],
                 color=COLORS["red"], lw=1.2, linestyle=":")

    ax1.set_xlim(-1.1, 1.2)
    ax1.set_xlabel("$x$"); ax1.set_ylabel("$y$")
    ax1.set_title("Verosimilitud: $p(y_i \\mid x_i; \\mathbf{w}) = "
                  "\\mathcal{N}(y_i;\\, \\hat{f}(x_i), \\sigma^2)$", fontsize=11)
    ax1.legend(fontsize=9)

    # Right: algebraic derivation as text
    ax2.axis("off")
    derivation = [
        r"MLE: maximizar $\mathcal{L}(\mathbf{w}) = \sum_i \log p(y_i \mid x_i; \mathbf{w})$",
        "",
        r"$= \sum_i \log \mathcal{N}(y_i;\, \mathbf{w}^T x_i, \sigma^2)$",
        "",
        r"$= -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}"
        r"\sum_i (y_i - \mathbf{w}^T x_i)^2$",
        "",
        r"Minimizar $-\mathcal{L}$ es equivalente a minimizar:",
        "",
        r"$\frac{1}{m}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$ (MSE)",
        "",
        r"$\Rightarrow$ ¿Por qué MSE en regresión?",
        r"Porque es MLE bajo ruido Gaussiano.",
        "",
        r"Para clasificación: $p(y \mid x;\mathbf{w}) = \text{Bernoulli}(\sigma(\mathbf{w}^T x))$",
        r"$\Rightarrow -\mathcal{L}$ = entropía cruzada (binary cross-entropy)",
    ]
    y_pos = 0.97
    for line in derivation:
        if line == "":
            y_pos -= 0.035
            continue
        ax2.text(0.05, y_pos, line, transform=ax2.transAxes,
                 fontsize=10.5, va="top", ha="left", wrap=True,
                 color=COLORS["dark"])
        y_pos -= 0.072

    ax2.set_title("Derivación: MLE = MSE (bajo supuesto Gaussiano)", fontsize=11)
    fig.tight_layout()
    _save(fig, "12_mle_gaussiano.png")


# ── 13 — Distribución muestral del MLE ───────────────────────────────────────
def plot_sampling_distribution_mle() -> None:
    """KDE of sample mean (MLE of μ) for m=5,20,100; overlay N(0,1/m)."""
    np.random.seed(70)
    B = 1000
    m_vals = [5, 20, 100]
    true_mu = 0.0
    true_sigma = 1.0
    colors = [COLORS["red"], COLORS["orange"], COLORS["blue"]]
    labels = [f"$m={m}$" for m in m_vals]

    fig, ax = plt.subplots(figsize=(10, 5))

    x_range = np.linspace(-1.5, 1.5, 400)
    for m, col, label in zip(m_vals, colors, labels):
        means = np.array([np.mean(np.random.normal(true_mu, true_sigma, m))
                          for _ in range(B)])
        kde = stats.gaussian_kde(means)(x_range)
        ax.fill_between(x_range, kde, alpha=0.25, color=col)
        ax.plot(x_range, kde, color=col, lw=2, label=f"{label} (empírica)")

        # Theoretical: N(0, 1/m) = N(0, sigma²/m)
        theoretical = stats.norm.pdf(x_range, true_mu, true_sigma / np.sqrt(m))
        ax.plot(x_range, theoretical, color=col, lw=1.5, linestyle="--",
                alpha=0.8, label=f"{label} (teórica: $\\mathcal{{N}}(0, 1/{m})$)")

    ax.axvline(true_mu, color=COLORS["dark"], lw=1.5, linestyle=":",
               label=r"$\mu^* = 0$")
    ax.set_xlabel(r"$\bar{x}_m$ (estimado MLE de $\mu$)")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución muestral del estimador MLE: $\\bar{x}_m \\sim "
                 "\\mathcal{N}(\\mu^*, \\sigma^2/m)$\n"
                 f"Líneas discontinuas = distribución teórica ({B} muestras por $m$)",
                 fontsize=11)
    ax.legend(fontsize=9, ncol=2)
    fig.tight_layout()
    _save(fig, "13_sampling_distribution_mle.png")


# ── 14 — Maldición de la dimensionalidad ─────────────────────────────────────
def plot_volumen_hiperbola() -> None:
    """Left: hypersphere volume fraction. Right: samples needed for k-NN."""
    d_vals = np.arange(1, 21)

    # Fraction: V_d(r=0.5) / 1^d   where V_d(r) = pi^(d/2) * r^d / Gamma(d/2+1)
    r = 0.5
    fractions = np.array([
        (np.pi ** (d / 2) * r ** d) / gamma_fn(d / 2 + 1)
        for d in d_vals
    ])
    # Clip to [0,1]: for d=1 it equals 1 exactly
    fractions = np.clip(fractions, 0, 1)

    # Samples needed: for k-NN at resolution eps=0.1, need (1/eps)^d = 10^d
    eps = 0.1
    samples_needed = (1 / eps) ** d_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(d_vals, fractions, "o-", color=COLORS["blue"], lw=2.2, markersize=7)
    ax1.fill_between(d_vals, fractions, alpha=0.2, color=COLORS["blue"])
    ax1.set_xlabel("Dimensión $d$")
    ax1.set_ylabel("Fracción del volumen del hipercubo")
    ax1.set_title("Volumen de la hiperesfera inscrita en el hipercubo unitario\n"
                  r"$V_d(0.5) \,/\, 1^d \;=\; \pi^{d/2} \cdot 0.5^d \,/\, \Gamma(d/2+1)$",
                  fontsize=11)
    ax1.set_xticks(d_vals)
    ax1.set_ylim(0, 1.05)
    for d, f in zip(d_vals[::3], fractions[::3]):
        ax1.text(d, f + 0.03, f"{f:.2f}", ha="center", fontsize=8.5)

    ax2.semilogy(d_vals, samples_needed, "s-", color=COLORS["red"], lw=2.2, markersize=7)
    ax2.axhline(1e3, color=COLORS["blue"], lw=1.5, linestyle="--", alpha=0.8,
                label="$m = 10^3$ (dataset pequeño)")
    ax2.axhline(1e6, color=COLORS["orange"], lw=1.5, linestyle="--", alpha=0.8,
                label="$m = 10^6$ (dataset grande)")
    ax2.axhline(1e9, color=COLORS["gray"], lw=1.5, linestyle=":", alpha=0.8,
                label="$m = 10^9$")
    ax2.set_xlabel("Dimensión $d$")
    ax2.set_ylabel("Muestras necesarias (escala log)")
    ax2.set_title(f"Muestras para k-NN con $\\varepsilon={eps}$:\n"
                  r"$m \sim (1/\varepsilon)^d = 10^d$", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.set_xticks(d_vals)

    fig.suptitle("La maldición de la dimensionalidad", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, "14_volumen_hiperbola.png")


# ── 15 — Local constancy failure ─────────────────────────────────────────────
def plot_local_constancy_failure() -> None:
    """True sin(5πx), vs. linear fit vs. 5-NN: local methods fail."""
    np.random.seed(80)
    m = 30
    x_tr = np.random.uniform(0, 1, m)
    y_tr = np.sin(5 * np.pi * x_tr) + np.random.normal(0, 0.1, m)
    x_plot = np.linspace(0, 1, 500)
    y_true_plot = np.sin(5 * np.pi * x_plot)

    # Linear fit
    w_lin = fit_legendre(x_tr, y_tr, deg=1, lam=0.0)
    y_lin = predict_legendre(x_plot, w_lin, deg=1)

    # 5-NN regression (manual, 1D)
    def knn_pred(x_query, x_train, y_train, k=5):
        out = []
        for xq in x_query:
            dists = np.abs(x_train - xq)
            nn_idx = np.argpartition(dists, k)[:k]
            out.append(y_train[nn_idx].mean())
        return np.array(out)

    y_knn = knn_pred(x_plot, x_tr, y_tr, k=5)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x_plot, y_true_plot, color=COLORS["dark"], lw=2.5, linestyle="--",
            label=r"$f^*(x) = \sin(5\pi x)$ (verdadera)", zorder=5)
    ax.scatter(x_tr, y_tr, color=COLORS["gray"], s=45, alpha=0.7, zorder=4,
               label=f"Datos de entrenamiento ($m={m}$)")
    ax.plot(x_plot, y_lin, color=COLORS["blue"], lw=2,
            label="Regresión lineal (grado 1)")
    ax.plot(x_plot, y_knn, color=COLORS["red"], lw=2,
            label="5-NN regression")
    ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    ax.set_title("Falla de la suposición de constancia local\n"
                 "($m=30$ puntos no son suficientes para capturar 5 oscilaciones)", fontsize=12)
    ax.legend(fontsize=10)

    n_needed = int((1 / 0.2) ** 1 * 10)
    ax.text(0.02, 0.05,
            f"En 1D con esta función: se necesitan ≈{n_needed} puntos.\n"
            "En $d$ dimensiones: $(10)^d$ puntos.",
            transform=ax.transAxes, fontsize=9.5, color=COLORS["dark"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["light"], alpha=0.8))
    fig.tight_layout()
    _save(fig, "15_local_constancy_failure.png")


# ── 16 — Manifold illustration ───────────────────────────────────────────────
def plot_manifold_ilustracion() -> None:
    """Spiral manifold: Euclidean NN (some wrong) vs. manifold NN (all correct)."""
    np.random.seed(90)
    n_pts = 250
    theta_all = np.linspace(0, 3 * np.pi, n_pts)
    r_all = 0.5 + 0.1 * theta_all / np.pi
    noise = 0.025
    x1 = r_all * np.cos(theta_all) + np.random.normal(0, noise, n_pts)
    x2 = r_all * np.sin(theta_all) + np.random.normal(0, noise, n_pts)

    # Query point at θ ≈ 1.5π (middle of spiral)
    q_theta_idx = int(n_pts * 0.50)
    q_x1, q_x2 = x1[q_theta_idx], x2[q_theta_idx]

    k = 5
    dists_euclidean = np.sqrt((x1 - q_x1) ** 2 + (x2 - q_x2) ** 2)
    dists_euclidean[q_theta_idx] = np.inf  # exclude self
    eu_nn_idx = np.argpartition(dists_euclidean, k)[:k]

    dists_manifold = np.abs(np.arange(n_pts) - q_theta_idx).astype(float)
    dists_manifold[q_theta_idx] = np.inf
    man_nn_idx = np.argpartition(dists_manifold, k)[:k]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, nn_idx, title, col_nn in [
        (ax1, eu_nn_idx, "Vecinos más cercanos\nen distancia euclidiana", COLORS["red"]),
        (ax2, man_nn_idx, "Vecinos más cercanos\nen la variedad (manifold)", COLORS["green"]),
    ]:
        ax.scatter(x1, x2, c=COLORS["blue"], s=15, alpha=0.4, zorder=2)
        ax.scatter(q_x1, q_x2, s=180, marker="*", color=COLORS["dark"],
                   zorder=5, label="Punto consulta")
        ax.scatter(x1[nn_idx], x2[nn_idx], s=90, color=col_nn,
                   zorder=4, label="5 vecinos seleccionados", edgecolors="white")
        for idx in nn_idx:
            ax.plot([q_x1, x1[idx]], [q_x2, x2[idx]], color=col_nn,
                    lw=1.2, alpha=0.7, linestyle="--")

        # Indicate wrong neighbors (Euclidean: check if they're far in θ)
        if ax is ax1:
            wrong = [i for i in nn_idx if abs(i - q_theta_idx) > n_pts // 8]
            if wrong:
                ax.scatter(x1[wrong], x2[wrong], s=130, color=COLORS["red"],
                           marker="x", lw=2.5, zorder=6, label="Vecinos incorrectos ✗")

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.legend(fontsize=9, loc="upper right")
        ax.set_aspect("equal")

    fig.suptitle("Hipótesis del manifold: distancia euclidiana ≠ distancia en la variedad",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "16_manifold_ilustracion.png")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Generando imágenes para módulo 22: Machine Learning Basics")
    print(f"  Directorio: {IMAGES_DIR}")
    print()

    plot_poblacion_vs_muestra()
    plot_error_stack()
    plot_underfitting_overfitting()
    plot_curva_capacidad()
    plot_ridge_curves()
    plot_generalization_bound()
    plot_kfold_diagrama()
    plot_optimismo_entrenamiento()
    plot_varianza_kfold()
    plot_bias_varianza_componentes()
    plot_efecto_m_estimacion()
    plot_mle_gaussiano()
    plot_sampling_distribution_mle()
    plot_volumen_hiperbola()
    plot_local_constancy_failure()
    plot_manifold_ilustracion()

    imgs = sorted(IMAGES_DIR.glob("*.png"))
    print(f"\nTotal: {len(imgs)} imágenes generadas.")


if __name__ == "__main__":
    main()
