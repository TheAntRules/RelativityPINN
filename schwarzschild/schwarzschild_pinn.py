"""
schwarzschild_pinn.py
=====================
Physics-Informed Neural Network to recover the Schwarzschild metric
from the vacuum Einstein Field Equations.

Paper: "Recovering the Schwarzschild Metric through PINNs"
       Anant Krishna & Sujoge Dua — QLab, TJHSST (2026)

Units: Geometrized (G = c = 1). Mass M has units of length.
       Schwarzschild radius r_s = 2M.

The network learns alpha(r) defined by the metric ansatz:
    ds^2 = -e^{2*alpha(r)} dt^2 + e^{2*beta(r)} dr^2 + r^2 dOmega^2

The constraint beta = -alpha is imposed architecturally, derived
analytically from R_tt - R_rr = 0. The network therefore outputs
only alpha(r); beta(r) = -alpha(r) everywhere by construction.

Exact (benchmark) solution:
    alpha(r) =  (1/2) ln(1 - r_s/r)
    beta(r)  = -(1/2) ln(1 - r_s/r)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────
# 0. Reproducibility & Device
# ─────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 1. Physical Parameters
# ─────────────────────────────────────────────
M = 1.0  # Mass in geometrized units (G = c = 1)
r_s = 2.0 * M  # Schwarzschild radius (event horizon)

# Training domain: exterior region only. r_min is set above r_s to avoid
# the coordinate singularity; r_max is the far-field boundary where
# asymptotic flatness is enforced.
r_min = 1.1 * r_s  # ~2.2 M  — near-horizon boundary
r_max = 20.0 * M  # far-field boundary (training)

# Extrapolation regions (never seen during training)
r_extrap_near_min = 1.01 * r_s  # deep near-horizon: [1.01 r_s, r_min)
r_extrap_near_max = r_min  #
r_extrap_far_min = r_max  # far-field:         (r_max, 50 M]
r_extrap_far_max = 50.0 * M

# Loss weights
LAMBDA_PDE = 10.0
LAMBDA_BC = 0.0  # asymptotic flatness anchor (currently off; mass term handles it)
LAMBDA_MASS = 1.0


# ─────────────────────────────────────────────
# 2. Exact Schwarzschild Solution (benchmarking only)
# ─────────────────────────────────────────────
def exact_alpha(r: torch.Tensor) -> torch.Tensor:
    """alpha_exact(r) = (1/2) ln(1 - r_s/r)"""
    return 0.5 * torch.log(1.0 - r_s / r)


def exact_beta(r: torch.Tensor) -> torch.Tensor:
    """beta_exact(r) = -alpha_exact(r)"""
    return -exact_alpha(r)


def misner_sharp_mass(alpha: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Quasi-local Misner-Sharp mass from the learned metric.
    m(r) = (r/2)(1 - e^{2*alpha})
    Should equal M everywhere in the exterior vacuum.
    """
    return 0.5 * r * (1.0 - torch.exp(2.0 * alpha))


def ricci_residuals_np(alpha_vals, alpha_r_vals, alpha_rr_vals, r_vals):
    """
    Compute Ricci residuals from numpy arrays (for post-hoc diagnostics).
    R_tt  = alpha'' + 2*(alpha')^2 + (2/r)*alpha'
    R_thth = e^{4*alpha} * (2*r*alpha' + 1) - 1
    """
    R_tt = alpha_rr_vals + 2.0 * alpha_r_vals**2 + (2.0 / r_vals) * alpha_r_vals
    R_thth = np.exp(4.0 * alpha_vals) * (2.0 * r_vals * alpha_r_vals + 1.0) - 1.0
    return R_tt, R_thth


# ─────────────────────────────────────────────
# 3. Neural Network Architecture
# ─────────────────────────────────────────────
class SchwarzschildPINN(nn.Module):
    """
    Fully-connected network: R^1 -> R^1
      Input:  r (normalized to [-1, 1] over the TRAINING domain)
      Output: alpha(r)
      beta(r) = -alpha(r) is enforced architecturally.

    Activation: tanh throughout. This is required — ReLU has zero
    second derivative almost everywhere and breaks the second-order
    automatic differentiation needed for the Ricci residuals.
    """

    def __init__(self, hidden_layers: int = 5, hidden_dim: int = 64):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization — recommended for tanh networks."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def normalize(self, r: torch.Tensor) -> torch.Tensor:
        """
        Map r in [r_min, r_max] to [-1, 1] for numerical stability.
        NOTE: this normalization uses the TRAINING domain endpoints.
        When r is outside [r_min, r_max], the normalized value falls
        outside [-1, 1] — this is intentional for extrapolation tests.
        """
        return 2.0 * (r - r_min) / (r_max - r_min) - 1.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r: shape (N, 1), any positive value (including outside training domain)
        Returns:
            alpha: shape (N, 1)
        """
        return self.net(self.normalize(r))


# ─────────────────────────────────────────────
# 4. Automatic Differentiation Helper
# ─────────────────────────────────────────────
def grad(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """
    d(output)/d(input) via autograd.
    create_graph=True keeps the graph alive for second-order differentiation.
    """
    return torch.autograd.grad(
        output,
        input,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


# ─────────────────────────────────────────────
# 5. PDE Residuals
# ─────────────────────────────────────────────
def compute_ricci_residuals(
    model: SchwarzschildPINN, r: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vacuum Ricci residuals after imposing the hard constraint beta = -alpha.

    Substituting beta = -alpha into R_munu = 0:

      R_tt = R_rr (identical — enforcing one covers both):
          R_tt = alpha'' + 2*(alpha')^2 + (2/r)*alpha'

      R_thth (e^{2(alpha-beta)} = e^{4*alpha}):
          R_thth = e^{4*alpha} * (2*r*alpha' + 1) - 1

    Args:
        model: the PINN
        r:     collocation points, shape (N, 1)
    Returns:
        R_tt, R_thth: residual tensors, each shape (N, 1)
    """
    r = r.requires_grad_(True)
    alpha = model(r)
    alpha_r = grad(alpha, r)
    alpha_rr = grad(alpha_r, r)

    R_tt = alpha_rr + 2.0 * alpha_r**2 + (2.0 / r) * alpha_r
    R_thth = torch.exp(2.0 * alpha) * (2.0 * r * alpha_r + 1.0) - 1.0

    return R_tt, R_thth


def compute_derivatives_np(model: SchwarzschildPINN, r_np: np.ndarray) -> tuple:
    """
    Compute alpha, alpha', alpha'' at arbitrary r values using autograd.
    Returns numpy arrays suitable for plotting and post-hoc PDE checks.
    """
    r_t = torch.tensor(r_np[:, None], dtype=torch.float32, device=DEVICE)
    r_t.requires_grad_(True)
    alpha = model(r_t)
    alpha_r = grad(alpha, r_t)
    alpha_rr = grad(alpha_r, r_t)
    return (
        alpha.detach().cpu().numpy().squeeze(),
        alpha_r.detach().cpu().numpy().squeeze(),
        alpha_rr.detach().cpu().numpy().squeeze(),
    )


# ─────────────────────────────────────────────
# 6. Loss Function
# ─────────────────────────────────────────────
def loss_function(
    model: SchwarzschildPINN, r_coll: torch.Tensor, r_bc: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    """
    Total loss = L_pde + L_bc + L_mass

    L_pde:  Mean squared residual of R_tt and R_thth at collocation
            points. Enforces the vacuum EFE.

    L_bc:   Asymptotic flatness: alpha(r_max) = 0.
            Target is 0 (Minkowski limit) — not a solution value.
            beta(r_max) = -alpha(r_max) = 0 follows automatically.

    L_mass: Misner-Sharp quasi-local mass:
                m(r) = (r/2)(1 - e^{-2*beta}) = (r/2)(1 - e^{2*alpha}) = M
            Holds exactly at every r in the exterior vacuum — no
            asymptotic approximation. Selects which Schwarzschild
            solution (i.e. which M) without using solution values.
    """
    # PDE loss
    R_tt, R_thth = compute_ricci_residuals(model, r_coll)
    loss_pde = torch.mean(R_tt**2 + R_thth**2)

    # BC loss: alpha(r_max) = 0
    loss_bc = model(r_bc).squeeze() ** 2

    # Misner-Sharp mass loss
    alpha_ms = model(r_coll)
    M_ms = misner_sharp_mass(alpha_ms, r_coll)
    loss_mass = torch.mean((M_ms - M) ** 2)

    total = LAMBDA_PDE * loss_pde + LAMBDA_BC * loss_bc + LAMBDA_MASS * loss_mass

    return total, {
        "total": total.item(),
        "pde": loss_pde.item(),
        "bc": loss_bc.item(),
        "mass": loss_mass.item(),
    }


# ─────────────────────────────────────────────
# 7. Collocation Point Sampling
# ─────────────────────────────────────────────
def sample_collocation(n: int, strategy: str = "log") -> torch.Tensor:
    """
    Sample n collocation points in [r_min, r_max].

    'log'    — log-uniform: denser near the horizon where
               gradients are steepest. Recommended.
    'random' — uniform random.
    'fixed'  — evenly spaced (deterministic).
    """
    if strategy == "log":
        r = torch.exp(torch.FloatTensor(n, 1).uniform_(np.log(r_min), np.log(r_max)))
    elif strategy == "fixed":
        r = torch.linspace(r_min, r_max, n).unsqueeze(1)
    else:
        r = torch.FloatTensor(n, 1).uniform_(r_min, r_max)
    return r.to(DEVICE)


# ─────────────────────────────────────────────
# 8. Training
# ─────────────────────────────────────────────
def train(
    model: SchwarzschildPINN,
    n_epochs: int = 20_000,
    n_coll: int = 1_000,
    lr_adam: float = 1e-3,
    use_lbfgs: bool = True,
    lbfgs_steps: int = 1_000,
    log_every: int = 500,
) -> dict:
    """
    Two-phase training:
      Phase 1 — Adam with cosine-annealing warm restarts.
                Broad convergence; robust to poor initialization.
      Phase 2 — L-BFGS with strong Wolfe line search.
                Quasi-Newton refinement; drives residuals orders of
                magnitude lower than Adam alone. Standard in PINN
                literature (Raissi et al. 2019).
    """
    model.to(DEVICE)

    # Warm up CUDA context before any autograd calls to suppress the
    # "no current CUDA context" UserWarning on the first backward pass.
    _ = torch.zeros(1, device=DEVICE) + 1.0

    # Persistent boundary tensor — created once after GPU is warm.
    r_bc = torch.tensor([[r_max]], dtype=torch.float32, device=DEVICE)

    history = {k: [] for k in ["total", "pde", "bc", "mass", "l2_alpha", "l2_beta"]}

    def log_metrics(epoch: int, components: dict):
        for key in ["total", "pde", "bc", "mass"]:
            history[key].append(components[key])

        with torch.no_grad():
            r_test = torch.linspace(r_min, r_max, 500, device=DEVICE).unsqueeze(1)
            alpha_p = model(r_test)
            beta_p = -alpha_p
            l2_a = torch.sqrt(torch.mean((alpha_p - exact_alpha(r_test)) ** 2)).item()
            l2_b = torch.sqrt(torch.mean((beta_p - exact_beta(r_test)) ** 2)).item()
        history["l2_alpha"].append(l2_a)
        history["l2_beta"].append(l2_b)

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Total {components['total']:.3e} | "
                f"PDE {components['pde']:.3e} | "
                f"BC {components['bc']:.3e} | "
                f"Mass {components['mass']:.3e} | "
                f"L2α {l2_a:.3e}"
            )

    # ── Phase 1: Adam ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("Phase 1: Adam")
    print("=" * 65)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=1, eta_min=1e-6
    )

    for epoch in range(n_epochs):
        r_coll = sample_collocation(n_coll, strategy="log")
        optimizer.zero_grad()
        loss, components = loss_function(model, r_coll, r_bc)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step(epoch)
        log_metrics(epoch, components)

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────────
    if use_lbfgs:
        print("=" * 65)
        print("Phase 2: L-BFGS")
        print("=" * 65)

        r_coll_fixed = sample_collocation(n_coll * 2, strategy="log")
        optimizer_lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_steps,
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        step_counter = [0]

        def closure():
            optimizer_lbfgs.zero_grad()
            loss, components = loss_function(model, r_coll_fixed, r_bc)
            loss.backward()
            step_counter[0] += 1
            if step_counter[0] % 50 == 0:
                log_metrics(n_epochs + step_counter[0], components)
            return loss

        optimizer_lbfgs.step(closure)

    print("\nTraining complete.")
    return history


# ─────────────────────────────────────────────
# 9. Evaluation & Plotting
# ─────────────────────────────────────────────
def evaluate_and_plot(model: SchwarzschildPINN, history: dict):
    model.eval()
    r_test = torch.linspace(r_min, r_max, 1000, device=DEVICE).unsqueeze(1)

    with torch.no_grad():
        alpha_pred = model(r_test).squeeze().cpu().numpy()
        beta_pred = -alpha_pred

    r_np = r_test.squeeze().cpu().numpy()
    alpha_ex = exact_alpha(r_test.squeeze()).cpu().numpy()
    beta_ex = exact_beta(r_test.squeeze()).cpu().numpy()

    # ── Figure 1: Metric functions & pointwise error ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("Schwarzschild PINN — Metric Recovery", fontsize=13)

    axes[0].plot(r_np, alpha_pred, lw=2, color="#1f77b4", label="PINN")
    axes[0].plot(r_np, alpha_ex, lw=2, ls="--", color="#d62728", label="Exact")
    axes[0].set(xlabel="r / M", ylabel="α(r)", title="Metric function α(r)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(r_np, beta_pred, lw=2, color="#1f77b4", label="PINN")
    axes[1].plot(r_np, beta_ex, lw=2, ls="--", color="#d62728", label="Exact")
    axes[1].set(xlabel="r / M", ylabel="β(r)", title="Metric function β(r)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].semilogy(
        r_np, np.abs(alpha_pred - alpha_ex), color="#1f77b4", label="|α error|"
    )
    axes[2].semilogy(
        r_np, np.abs(beta_pred - beta_ex), color="#ff7f0e", label="|β error|"
    )
    axes[2].set(
        xlabel="r / M", ylabel="Absolute error", title="Pointwise error (log scale)"
    )
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("metric_recovery.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── Figure 2: Training history ─────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
    fig2.suptitle("Training History", fontsize=13)

    for key, label in [
        ("total", "Total"),
        ("pde", "PDE"),
        ("bc", "BC"),
        ("mass", "Mass"),
    ]:
        axes2[0].semilogy(history[key], label=label, lw=1.5)
    axes2[0].set(xlabel="Epoch", ylabel="Loss", title="Loss components")
    axes2[0].legend()
    axes2[0].grid(alpha=0.3)

    axes2[1].semilogy(history["l2_alpha"], lw=2, label="L2 error α")
    axes2[1].semilogy(history["l2_beta"], lw=2, label="L2 error β")
    axes2[1].set(xlabel="Epoch", ylabel="L2 error", title="L2 error vs exact solution")
    axes2[1].legend()
    axes2[1].grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("training_history.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# 10. Extrapolation Diagnostic (Figure 3)
# ─────────────────────────────────────────────
def extrapolation_diagnostic(model: SchwarzschildPINN):
    """
    Tests whether the physical constraints learned by the PINN generalise
    beyond the training domain [r_min, r_max] = [1.1 r_s, 20 M].

    Two extrapolation zones are examined:
      • Near-horizon:  r ∈ [1.01 r_s, 1.1 r_s)   — deep gravitational well,
                       never seen during training, steepest gradients.
      • Far-field:     r ∈ (20 M, 50 M]            — asymptotically flat tail,
                       network must generalise Minkowski behaviour.

    Four diagnostics are shown for each zone:

    Panel 1 — α(r): PINN vs exact.
      Checks whether the metric function is recovered correctly outside
      the training window.

    Panel 2 — Pointwise |α| error (log scale).
      Quantifies how much accuracy degrades as r moves away from the
      training domain.

    Panel 3 — Misner-Sharp mass m(r).
      The quasi-local mass m(r) = (r/2)(1 - e^{2α}) should equal M = 1
      everywhere in the exterior vacuum. Deviation from M = 1 indicates
      the network has lost the conservation law outside training.

    Panel 4 — Ricci residuals |R_tt| and |R_thθ| (log scale).
      The most direct test: do the vacuum field equations R_μν = 0
      continue to hold outside the training domain? A well-constrained
      PINN should maintain small residuals even in extrapolation regions,
      because the PDE structure is physically enforced, not memorised.
    """
    model.eval()

    # ── Build evaluation grids ─────────────────────────────────────────────────
    # Near-horizon extrapolation (below training r_min)
    r_near_np = np.linspace(r_extrap_near_min, r_extrap_near_max, 300)
    # Far-field extrapolation (above training r_max)
    r_far_np = np.linspace(r_extrap_far_min, r_extrap_far_max, 300)
    # Full combined range for context panels
    r_full_np = np.linspace(r_extrap_near_min, r_extrap_far_max, 1200)

    zones = {
        "Near-horizon\n" r"$r \in [1.01\,r_s,\;1.1\,r_s)$": r_near_np,
        "Far-field\n" r"$r \in (20\,M,\;50\,M]$": r_far_np,
    }

    # ── Compute PINN outputs and derivatives via autograd ─────────────────────
    def evaluate_zone(r_np_arr):
        alpha_p, alpha_r_p, alpha_rr_p = compute_derivatives_np(model, r_np_arr)
        R_tt, R_thth = ricci_residuals_np(alpha_p, alpha_r_p, alpha_rr_p, r_np_arr)
        r_t = torch.tensor(r_np_arr[:, None], dtype=torch.float32, device=DEVICE)
        alpha_t = torch.tensor(alpha_p[:, None], dtype=torch.float32, device=DEVICE)
        mass_p = misner_sharp_mass(alpha_t, r_t).detach().cpu().numpy().squeeze()
        alpha_ex = 0.5 * np.log(1.0 - r_s / r_np_arr)
        return {
            "alpha_p": alpha_p,
            "alpha_ex": alpha_ex,
            "mass_p": mass_p,
            "R_tt": np.abs(R_tt),
            "R_thth": np.abs(R_thth),
        }

    results = {label: evaluate_zone(r_np) for label, r_np in zones.items()}

    # Also evaluate the full range for the overview context panel
    alpha_full_p, _, _ = compute_derivatives_np(model, r_full_np)
    alpha_full_ex = 0.5 * np.log(np.maximum(1.0 - r_s / r_full_np, 1e-10))

    # ── Colors & style ────────────────────────────────────────────────────────
    C_PINN = "#1f77b4"
    C_EXACT = "#d62728"
    C_MASS = "#2ca02c"
    C_RTT = "#9467bd"
    C_RTHTH = "#ff7f0e"

    TRAIN_COLOR = "#b0c4de"  # shaded training band color
    EXTRAP_NEAR = "#ffe4b5"  # near-horizon extrapolation shading
    EXTRAP_FAR = "#e0ffe0"  # far-field extrapolation shading

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Figure 3 — Extrapolation Diagnostic:\nPhysical Constraint Generalisation Beyond the Training Domain",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    gs_top = GridSpec(1, 1, figure=fig, top=0.88, bottom=0.72, left=0.07, right=0.93)
    gs_bottom = GridSpec(
        2,
        4,
        figure=fig,
        top=0.65,
        bottom=0.06,
        left=0.06,
        right=0.97,
        hspace=0.55,
        wspace=0.38,
    )

    # ── Panel 0 (top): Full-range overview ────────────────────────────────────
    ax0 = fig.add_subplot(gs_top[0, 0])
    ax0.fill_betweenx(
        [-1.5, 0.1], r_min, r_max, color=TRAIN_COLOR, alpha=0.6, label="Training domain"
    )
    ax0.fill_betweenx(
        [-1.5, 0.1],
        r_extrap_near_min,
        r_min,
        color=EXTRAP_NEAR,
        alpha=0.7,
        label="Near-horizon extrap.",
    )
    ax0.fill_betweenx(
        [-1.5, 0.1],
        r_max,
        r_extrap_far_max,
        color=EXTRAP_FAR,
        alpha=0.7,
        label="Far-field extrap.",
    )
    ax0.plot(r_full_np, alpha_full_p, color=C_PINN, lw=2, label="PINN α(r)")
    ax0.plot(r_full_np, alpha_full_ex, color=C_EXACT, lw=2, ls="--", label="Exact α(r)")
    ax0.axvline(r_min, color="gray", lw=1, ls=":")
    ax0.axvline(r_max, color="gray", lw=1, ls=":")
    ax0.set(
        xlabel="r / M",
        ylabel="α(r)",
        title=r"Overview: PINN $\alpha(r)$ across full range "
        r"$r \in [1.01\,r_s,\;50\,M]$  —  shaded regions never seen during training",
        xlim=(r_extrap_near_min, r_extrap_far_max),
        ylim=(-1.55, 0.1),
    )
    ax0.legend(ncol=5, fontsize=8, loc="lower right")
    ax0.grid(alpha=0.3)

    # ── Panels 1–8: Four diagnostics × two zones ──────────────────────────────
    row_labels = list(zones.keys())
    panel_titles = [
        "α(r):  PINN vs exact",
        "Pointwise |α| error",
        "Misner-Sharp mass m(r)",
        "Ricci residuals |R_μν|",
    ]

    for col, panel_title in enumerate(panel_titles):
        for row, (zone_label, r_np) in enumerate(zones.items()):
            d = results[zone_label]
            ax = fig.add_subplot(gs_bottom[row, col])
            is_near = row == 0

            # -- background shading --
            shade_color = EXTRAP_NEAR if is_near else EXTRAP_FAR
            ax.set_facecolor(shade_color + "55")  # very light tint

            if col == 0:
                # α(r) comparison
                ax.plot(r_np, d["alpha_p"], color=C_PINN, lw=2, label="PINN")
                ax.plot(
                    r_np, d["alpha_ex"], color=C_EXACT, lw=2, ls="--", label="Exact"
                )
                ax.set_ylabel("α(r)")
                ax.legend(fontsize=7)

            elif col == 1:
                # Pointwise absolute error (log scale)
                err = np.abs(d["alpha_p"] - d["alpha_ex"])
                err = np.clip(err, 1e-12, None)
                ax.semilogy(r_np, err, color=C_PINN, lw=2)
                ax.set_ylabel("|α_PINN − α_exact|")
                # Annotate mean error
                ax.axhline(np.mean(err), color="gray", lw=1, ls="--")
                ax.text(
                    r_np[-1],
                    np.mean(err) * 1.5,
                    f"mean={np.mean(err):.1e}",
                    fontsize=7,
                    ha="right",
                    color="gray",
                )

            elif col == 2:
                # Misner-Sharp mass
                ax.plot(r_np, d["mass_p"], color=C_MASS, lw=2, label="m(r) PINN")
                ax.axhline(M, color=C_EXACT, lw=1.5, ls="--", label=f"M = {M}")
                ax.set_ylabel("m(r)")
                ax.legend(fontsize=7)
                # Shade deviation band
                ax.fill_between(
                    r_np, d["mass_p"], M, alpha=0.2, color=C_MASS, label="deviation"
                )

            elif col == 3:
                # Ricci residuals
                ax.semilogy(
                    r_np,
                    np.clip(d["R_tt"], 1e-12, None),
                    color=C_RTT,
                    lw=2,
                    label=r"|$R_{tt}$|",
                )
                ax.semilogy(
                    r_np,
                    np.clip(d["R_thth"], 1e-12, None),
                    color=C_RTHTH,
                    lw=2,
                    label=r"|$R_{\theta\theta}$|",
                )
                ax.set_ylabel("Residual magnitude")
                ax.legend(fontsize=7)

            # Shared formatting
            ax.set_xlabel("r / M")
            ax.grid(alpha=0.3)

            # Row label on leftmost column
            if col == 0:
                ax.set_title(f"{zone_label}\n{panel_title}", fontsize=8.5)
            else:
                ax.set_title(panel_title, fontsize=8.5)

    # ── Legend strip for zone shading ─────────────────────────────────────────
    near_patch = mpatches.Patch(
        color=EXTRAP_NEAR, alpha=0.8, label="Near-horizon extrapolation"
    )
    far_patch = mpatches.Patch(
        color=EXTRAP_FAR, alpha=0.8, label="Far-field extrapolation"
    )
    train_patch = mpatches.Patch(
        color=TRAIN_COLOR, alpha=0.8, label="Training domain (reference)"
    )
    fig.legend(
        handles=[train_patch, near_patch, far_patch],
        loc="lower center",
        ncol=3,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
        frameon=True,
    )

    fig.savefig("extrapolation_diagnostic.png", dpi=150, bbox_inches="tight")
    print("Figure 3 saved → extrapolation_diagnostic.png")
    plt.show()


# ─────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = SchwarzschildPINN(hidden_layers=5, hidden_dim=64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network parameters: {n_params:,}")
    print(f"Training domain: r ∈ [{r_min:.2f}, {r_max:.2f}] M   (r_s = {r_s:.2f} M)")
    print(
        f"Near-horizon extrapolation: r ∈ [{r_extrap_near_min:.3f}, {r_extrap_near_max:.3f}] M"
    )
    print(
        f"Far-field extrapolation:    r ∈ [{r_extrap_far_min:.1f},  {r_extrap_far_max:.1f}] M"
    )

    history = train(
        model,
        n_epochs=20_000,
        n_coll=1_000,
        lr_adam=1e-3,
        use_lbfgs=True,
        lbfgs_steps=1_000,
        log_every=100,
    )

    evaluate_and_plot(model, history)
    extrapolation_diagnostic(model)
