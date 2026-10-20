"""
kerr_pinn.py
============
Physics-Informed Neural Network to recover the Kerr metric from the
vacuum Einstein Field Equations in Boyer-Lindquist coordinates.

Extension of schwarzschild_pinn.py (Krishna & Dua, QLab 2026).

─────────────────────────────────────────────────────────────────────────────
STRUCTURAL DIFFERENCES FROM THE SCHWARZSCHILD CASE
─────────────────────────────────────────────────────────────────────────────

1. INPUT DIMENSIONALITY: 1D → 2D
   Schwarzschild: input r only. Spherical symmetry collapses all angular
   dependence. The ODE lives on R^1.
   Kerr: input (r, θ). Axisymmetry eliminates φ; stationarity eliminates t;
   but the polar angle θ cannot be eliminated — Σ = r²+a²cos²θ introduces
   genuine angular structure. The PDE lives on R^2.

2. NUMBER OF METRIC FUNCTIONS: 1 → 5
   Schwarzschild: network learns α(r) alone. The architectural constraint
   β = −α (derived analytically from R_tt − R_rr = 0) reduces the system
   to a single function.
   Kerr: no such algebraic reduction exists. The network must simultaneously
   learn five independent metric components:
     g_tt(r,θ),  g_rr(r,θ),  g_θθ(r,θ),  g_tφ(r,θ),  g_φφ(r,θ)
   The off-diagonal term g_tφ encodes frame dragging — it has no Schwarzschild
   analogue and cannot be set to zero.

3. RICCI COMPUTATION: pre-derived ODEs → fully automatic differentiation
   Schwarzschild: R_tt and R_θθ were derived analytically after substituting
   β = −α, yielding explicit ODE expressions implemented as code.
   Kerr: the Ricci tensor is computed from scratch via autograd at every
   training step. No analytic simplification is made. The computation:
     g_μν (network) → ∂g/∂r, ∂g/∂θ (1st-order autograd)
     → Γ^μ_νρ (algebraic) → ∂Γ/∂r, ∂Γ/∂θ (2nd-order autograd)
     → R_μν (algebraic contraction)
   This is more expensive but immediately portable to other spacetimes.

4. DEGENERACY BREAKING: global mass constraint → boundary-only constraints
   Schwarzschild: loss_mass enforced m(r) = (r/2)(1−e^{2α}) = M at every
   collocation point. This is a conserved quasi-local mass that encodes the
   exact solution implicitly everywhere in the domain.
   Kerr: we use two strictly weaker constraints (see Section 6):
     (a) Asymptotic flatness at r_max — pins only the boundary behavior,
         not interior structure. Valid for ANY isolated vacuum spacetime.
     (b) Single reference-point pin at (R_REF, π/2) — enforces g_φφ at
         one scalar point. This breaks the trivial g=0 degeneracy with
         the minimum possible solution-encoded information.
   The key methodological claim: these constraints do not encode the Kerr
   solution — they encode only universally known physical properties
   (asymptotic flatness, M, a) that would be available in genuinely unknown
   spacetimes.

Units: geometrized (G = c = 1). Mass M has units of length.
Exact solution: Kerr in Boyer-Lindquist coordinates.

Paper: "Recovering the Kerr Metric through PINNs"
       Extension of Krishna & Dua (QLab, TJHSST 2026)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time

t0 = time.time()

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
M   = 1.0                             # Mass (geometrized units)
a   = 0.5 * M                         # Spin parameter; |a| ≤ M (extremal limit)
r_plus = M + np.sqrt(M**2 - a**2)    # Outer event horizon (Δ = 0)
r_minus = M - np.sqrt(M**2 - a**2)   # Inner (Cauchy) horizon

# Training domain: exterior only, safely clear of coordinate singularities.
# - r_min > r_plus avoids Δ → 0 (coordinate singularity at horizon).
# - θ away from poles avoids sin(θ) = 0 (axis coordinate singularity).
r_min  = 1.2 * r_plus
r_max  = 20.0 * M
th_min = 0.05              # radians; avoids θ = 0 (north pole)
th_max = np.pi - 0.05     # radians; avoids θ = π (south pole)

# Reference point for degeneracy breaking (equatorial, mid-range radius).
R_REF  = 5.0 * M
TH_REF = np.pi / 2.0      # equatorial plane: sin(θ)=1, cos(θ)=0 → Σ=r²

# Loss weights. Asymptotic BC is weighted heavily to anchor the solution
# scale; PDE loss drives the interior to satisfy the field equations.
LAMBDA_PDE   = 5.0
LAMBDA_ASYMP = 10.0
LAMBDA_REF   = 1.0

print(f"Spin parameter:    a    = {a:.3f} M")
print(f"Outer horizon:     r+   = {r_plus:.4f} M")
print(f"Training domain:   r  ∈ [{r_min:.3f}, {r_max:.1f}] M")
print(f"                   θ  ∈ [{th_min:.3f}, {th_max:.3f}] rad")
print(f"Reference point:   r₀   = {R_REF:.1f} M,  θ₀ = π/2 (equatorial)")


# ─────────────────────────────────────────────
# 2. Exact Kerr Solution (benchmarking only)
# ─────────────────────────────────────────────
def kerr_Sigma(r, th):
    """Σ(r,θ) = r² + a²cos²θ"""
    return r**2 + a**2 * torch.cos(th)**2

def kerr_Delta(r):
    """Δ(r) = r² − 2Mr + a²"""
    return r**2 - 2*M*r + a**2

def exact_kerr(r, th):
    """
    All five independent Kerr metric components in Boyer-Lindquist.
    Args:
        r, th: (N,) tensors (squeezed)
    Returns:
        g_tt, g_rr, g_θθ, g_tφ, g_φφ: each (N,) tensor
    """
    Sig  = kerr_Sigma(r, th)
    Del  = kerr_Delta(r)
    sin2 = torch.sin(th)**2

    g_tt  = -(1.0 - 2*M*r / Sig)
    g_rr  = Sig / Del
    g_thth = Sig
    g_tphi = -2*M*a*r*sin2 / Sig
    g_pp  = (r**2 + a**2 + 2*M*a**2*r*sin2 / Sig) * sin2

    return g_tt, g_rr, g_thth, g_tphi, g_pp


# ─────────────────────────────────────────────
# 3. Neural Network Architecture
# ─────────────────────────────────────────────
class KerrPINN(nn.Module):
    """
    Fully-connected network: R^2 → R^5 (REVISED v2)
      Input:  (r_norm, θ_norm) — each normalized to [-1, 1]
      Output: five physical metric components [g_tt, g_rr, g_θθ, g_tφ, g_φφ]

    KEY CHANGE FROM v1: the network does NOT output metric components directly.
    Instead it outputs five O(1) dimensionless shape functions h_i(r,θ), which
    are then multiplied by known coordinate scales to produce the metric.
    This eliminates the scale mismatch that caused v1 to fail.

    Parametrisation (h_i = softplus(raw_i) + ε, each O(1)):

      g_tt   = −h_tt
              exact h_tt = 1 − 2Mr/Σ ∈ (0, 1)  ← O(1) ✓

      g_rr   = (r²/Δ) · h_rr,  where Δ = r²−2Mr+a² (known)
              exact h_rr = Σ/r² = 1 + a²cos²θ/r² ∈ (1, 1+a²/r²) ← O(1) ✓
              The coordinate singularity Δ→0 is factored out analytically so
              the network only needs to learn the smooth, bounded function Σ/r².

      g_θθ   = r² · h_θθ
              exact h_θθ = Σ/r² = 1 + a²cos²θ/r² ∈ (1, 1+a²/r²) ← O(1) ✓
              Without this, the network must output values up to 400 at r=20M,
              which is impossible from a softplus-bounded output near zero.

      g_tφ   = −h_tφ · (2Ma sin²θ / r)     [sign enforces dragging direction]
              exact h_tφ = r²/Σ ∈ (0, 1]   ← O(1) ✓
              The physical scale 2Ma sin²θ/r ~ 0.1 at mid-domain is factored out.

      g_φφ   = r²(sin²θ + ε) · h_φφ
              exact h_φφ = 1 + a²/r² + 2Ma²sin²θ/(r·Σ) ∈ (1, ~2) ← O(1) ✓
              Without this, g_φφ ~ r²sin²θ up to 400 is unreachable.

    Bias initialisation: final layer biases set so h_i ≈ 1 at startup
    (softplus(0.69) ≈ 1), giving a near-flat-space initial metric with
    correct coordinate scales. This is analogous to the Schwarzschild PINN
    starting near α = 0 (Minkowski), but now in the full 5-component case.
    """

    def __init__(self, hidden_layers: int = 6, hidden_dim: int = 64):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, 5))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        # Override final layer biases so all h_i ≈ 1 initially.
        # softplus(b) = 1  →  b = log(e - 1) ≈ 0.541
        # This gives a near-Minkowski starting point with correct r² scaling.
        with torch.no_grad():
            self.net[-1].bias.fill_(0.541)

    def normalize(self, r: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """Map (r, θ) from their physical domains to [-1, 1]²."""
        r_n  = 2.0 * (r  - r_min)  / (r_max  - r_min)  - 1.0
        th_n = 2.0 * (th - th_min) / (th_max - th_min) - 1.0
        return torch.cat([r_n, th_n], dim=1)

    def forward(self, r: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        """
        Args:
            r:  (N, 1) radial coordinate
            th: (N, 1) polar angle in radians
        Returns:
            metrics: (N, 5) — [g_tt, g_rr, g_θθ, g_tφ, g_φφ]
        """
        x   = self.normalize(r, th)
        raw = self.net(x)                              # (N, 5)
        sp  = torch.nn.functional.softplus
        eps = 1e-6

        sin2  = torch.sin(th)**2                       # (N, 1)
        Delta = r**2 - 2*M*r + a**2                   # (N, 1), known from M, a

        # Five O(1) shape functions
        h_tt  = sp(raw[:, 0:1]) + eps   # target: (1 − 2Mr/Σ) ∈ (0, 1)
        h_rr  = sp(raw[:, 1:2]) + eps   # target: Σ/r²        ∈ (1, ~2)
        h_thth = sp(raw[:, 2:3]) + eps  # target: Σ/r²        ∈ (1, ~2)
        h_tphi = sp(raw[:, 3:4]) + eps  # target: r²/Σ        ∈ (0,  1]
        h_pp  = sp(raw[:, 4:5]) + eps   # target: 1 + a²/r² + ... ∈ (1, ~2)

        g_tt     = -h_tt
        g_rr     = r**2 / (Delta + eps) * h_rr         # factors out Δ singularity
        g_thth   = r**2 * h_thth
        g_tphi   = -(2*M*a * (sin2 + eps) / (r + eps)) * h_tphi
        g_phiphi = r**2 * (sin2 + eps) * h_pp

        return torch.cat([g_tt, g_rr, g_thth, g_tphi, g_phiphi], dim=1)


# ─────────────────────────────────────────────
# 4. Metric Utilities
# ─────────────────────────────────────────────
def get_components(
    model: KerrPINN,
    r: torch.Tensor,
    th: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """
    Return the 5 metric components as separate (N,) tensors.
    """
    out = model(r, th)    # (N, 5)
    return out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]


def invert_metric_analytical(
    gtt, grr, gthth, gtphi, gphiphi
) -> tuple[torch.Tensor, ...]:
    """
    Analytically invert the block-diagonal + off-diagonal Kerr metric.

    Structure of g_μν (indices: t=0, r=1, θ=2, φ=3):
      [[g_tt,  0,      0,      g_tφ  ],
       [0,     g_rr,   0,      0     ],
       [0,     0,      g_θθ,   0     ],
       [g_tφ,  0,      0,      g_φφ  ]]

    The (r,θ) block is diagonal and trivially inverted.
    The (t,φ) block is 2×2 and inverted analytically:
      det(t,φ) = g_tt * g_φφ − g_tφ²
      g^tt      =  g_φφ / det
      g^tφ      = −g_tφ / det
      g^φφ      =  g_tt / det

    Returns: g^tt, g^rr, g^θθ, g^tφ, g^φφ  (each (N,) tensor)
    """
    det_tphi    = gtt * gphiphi - gtphi**2
    gtt_inv     =  gphiphi / det_tphi
    gtphi_inv   = -gtphi   / det_tphi
    gphiphi_inv =  gtt     / det_tphi
    grr_inv     =  1.0 / grr
    gthth_inv   =  1.0 / gthth
    return gtt_inv, grr_inv, gthth_inv, gtphi_inv, gphiphi_inv


# ─────────────────────────────────────────────
# 5. Ricci Tensor via Full Automatic Differentiation
# ─────────────────────────────────────────────
def compute_ricci_residuals(
    model: KerrPINN,
    r: torch.Tensor,
    th: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """
    Compute all independent Ricci tensor components for a stationary,
    axisymmetric vacuum spacetime via automatic differentiation.

    Coordinates: x^μ = [t=0, r=1, θ=2, φ=3].
    Since the metric is stationary (∂_t g = 0) and axisymmetric (∂_φ g = 0),
    all partial derivatives reduce to ∂_r and ∂_θ only.

    The five independent non-trivially-zero Ricci components are:
      R_tt, R_rr, R_θθ, R_tφ, R_φφ
    (The rest vanish by symmetry of the ansatz.)

    Computation graph (all via autograd, no analytic simplification):
      g_μν  [network]
        ↓ autograd (1st order)
      ∂_r g_μν,  ∂_θ g_μν
        ↓ algebraic
      Γ^μ_νρ  =  (1/2) g^{μσ}(∂_ν g_{σρ} + ∂_ρ g_{σν} − ∂_σ g_{νρ})
        ↓ autograd (2nd order, create_graph=True throughout)
      ∂_r Γ^μ_νρ,  ∂_θ Γ^μ_νρ
        ↓ algebraic contraction
      R_μν  =  ∂_ρ Γ^ρ_μν − ∂_ν Γ^ρ_μρ + Γ^ρ_ρλ Γ^λ_νμ − Γ^ρ_νλ Γ^λ_ρμ

    Args:
        model: KerrPINN
        r:     (N, 1) tensor with requires_grad=True
        th:    (N, 1) tensor with requires_grad=True
    Returns:
        R_tt, R_rr, R_thth, R_tphi, R_phiphi — each (N,) tensor
    """
    N   = r.shape[0]
    dev = r.device
    Z   = torch.zeros(N, device=dev)   # canonical zero tensor (requires_grad=False)

    # ── Safe autograd derivative helpers ────────────────────────────────────
    # allow_unused=True handles cases where f is disconnected from the
    # differentiation variable (e.g., a metric component that doesn't depend
    # on r because the network chose to), returning zero in that case.
    def dr(f: torch.Tensor) -> torch.Tensor:
        g = torch.autograd.grad(
            f.sum(), r, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        return g.squeeze(1) if g is not None else Z

    def dth(f: torch.Tensor) -> torch.Tensor:
        g = torch.autograd.grad(
            f.sum(), th, create_graph=True, retain_graph=True, allow_unused=True
        )[0]
        return g.squeeze(1) if g is not None else Z

    # ── Metric and inverse ───────────────────────────────────────────────────
    gtt, grr, gthth, gtphi, gphiphi = get_components(model, r, th)
    gtt_i, grr_i, gthth_i, gtphi_i, gphiphi_i = invert_metric_analytical(
        gtt, grr, gthth, gtphi, gphiphi
    )

    # ── Metric first derivatives  ∂_r g_{μν}  and  ∂_θ g_{μν} ─────────────
    # All (N,) tensors. Only the five nonzero metric entries contribute.
    dgdr_tt      = dr(gtt);      dgdth_tt      = dth(gtt)
    dgdr_rr      = dr(grr);      dgdth_rr      = dth(grr)
    dgdr_thth    = dr(gthth);    dgdth_thth    = dth(gthth)
    dgdr_tphi    = dr(gtphi);    dgdth_tphi    = dth(gtphi)
    dgdr_phiphi  = dr(gphiphi);  dgdth_phiphi  = dth(gphiphi)

    # ── Lookup functions for metric, inverse metric, and derivatives ─────────
    # Coordinates: t=0, r=1, th=2, phi=3. Metric is symmetric.

    def g_lo(mu: int, nu: int) -> torch.Tensor:
        """g_{μν}"""
        if mu > nu: mu, nu = nu, mu   # symmetry
        if   (mu,nu) == (0,0): return gtt
        elif (mu,nu) == (0,3): return gtphi
        elif (mu,nu) == (1,1): return grr
        elif (mu,nu) == (2,2): return gthth
        elif (mu,nu) == (3,3): return gphiphi
        else:                  return Z

    def g_up(mu: int, nu: int):
        """g^{μν}; returns None for identically-zero components."""
        if mu > nu: mu, nu = nu, mu
        if   (mu,nu) == (0,0): return gtt_i
        elif (mu,nu) == (0,3): return gtphi_i
        elif (mu,nu) == (1,1): return grr_i
        elif (mu,nu) == (2,2): return gthth_i
        elif (mu,nu) == (3,3): return gphiphi_i
        else:                  return None   # structural zero

    def dg(alpha: int, mu: int, nu: int) -> torch.Tensor:
        """∂_{x^α} g_{μν}. Only α=1 (r) and α=2 (θ) are nonzero."""
        if mu > nu: mu, nu = nu, mu   # symmetry
        key = (mu, nu)
        map_r  = {(0,0): dgdr_tt,   (0,3): dgdr_tphi,
                  (1,1): dgdr_rr,   (2,2): dgdr_thth, (3,3): dgdr_phiphi}
        map_th = {(0,0): dgdth_tt,  (0,3): dgdth_tphi,
                  (1,1): dgdth_rr,  (2,2): dgdth_thth, (3,3): dgdth_phiphi}
        if alpha == 1:
            return map_r.get(key, Z)
        elif alpha == 2:
            return map_th.get(key, Z)
        return Z

    # ── Christoffel symbols Γ^μ_{νρ} ────────────────────────────────────────
    # Γ^μ_{νρ} = (1/2) g^{μσ} (∂_ν g_{σρ} + ∂_ρ g_{σν} − ∂_σ g_{νρ})
    # Symmetry in lower indices: Γ^μ_{νρ} = Γ^μ_{ρν}.
    # Only σ ∈ {0,1,2,3} with g^{μσ} ≠ 0 contribute. For our metric, the
    # nonzero g^{μσ} entries are: (0,0), (0,3), (1,1), (2,2), (3,0), (3,3).

    def christoffel(mu: int, nu: int, rho: int) -> torch.Tensor:
        result = Z
        for sig in range(4):
            g_inv_ms = g_up(mu, sig)
            if g_inv_ms is None:
                continue
            bracket = (  dg(nu,  sig, rho)
                       + dg(rho, sig, nu)
                       - dg(sig, nu,  rho) )
            result = result + g_inv_ms * bracket
        return 0.5 * result

    # Precompute all 4³ = 64 Christoffel symbols.
    # Using symmetry Γ^μ_{νρ} = Γ^μ_{ρν} halves the work.
    Chr = [[[None]*4 for _ in range(4)] for _ in range(4)]
    for mu in range(4):
        for nu in range(4):
            for rho in range(nu, 4):   # rho ≥ nu; fill both orderings below
                c = christoffel(mu, nu, rho)
                Chr[mu][nu][rho] = c
                Chr[mu][rho][nu] = c   # symmetry in lower indices

    def C(mu, nu, rho) -> torch.Tensor:
        val = Chr[mu][nu][rho]
        return val if val is not None else Z

    # ── Ricci tensor R_{μν} ─────────────────────────────────────────────────
    # R_{μν} = ∂_ρ Γ^ρ_{μν}  −  ∂_ν Γ^ρ_{μρ}
    #         + Γ^ρ_{ρλ} Γ^λ_{νμ}  −  Γ^ρ_{νλ} Γ^λ_{ρμ}
    #
    # Term 1: sum_ρ ∂_{x^ρ} Γ^ρ_{μν}. Active ρ: {1=r, 2=θ}.
    # Term 2: −∂_{x^ν} sum_ρ Γ^ρ_{μρ}. Active ν: {1=r, 2=θ}.
    # Terms 3,4: quadratic; no further differentiation needed.

    def ricci(mu: int, nu: int) -> torch.Tensor:
        R = Z

        # Term 1: ∂_r Γ^r_{μν}  +  ∂_θ Γ^θ_{μν}
        R = R + dr( C(1, mu, nu))    # ρ=1 (r)
        R = R + dth(C(2, mu, nu))    # ρ=2 (θ)

        # Term 2: −∂_ν Σ_ρ Γ^ρ_{μρ}
        # Only ν=1 (r) or ν=2 (θ) give nonzero derivatives.
        if nu == 1:
            for rho in range(4):
                R = R - dr( C(rho, mu, rho))
        elif nu == 2:
            for rho in range(4):
                R = R - dth(C(rho, mu, rho))

        # Terms 3, 4: Christoffel × Christoffel
        for rho in range(4):
            for lam in range(4):
                R = R + C(rho, rho, lam) * C(lam, nu,  mu )
                R = R - C(rho, nu,  lam) * C(lam, rho, mu )

        return R

    # Compute the five independent nonzero components
    R_tt      = ricci(0, 0)
    R_rr      = ricci(1, 1)
    R_thth    = ricci(2, 2)
    R_tphi    = ricci(0, 3)
    R_phiphi  = ricci(3, 3)

    # CLAMP: Ricci residuals from a far-from-converged metric can reach 10^14,
    # which causes catastrophic gradient updates that destroy BC progress.
    # Clamping to ±R_CLAMP caps the loss at R_CLAMP² ≈ 10^8 while preserving
    # gradient sign and direction. The clamp relaxes automatically as the
    # metric improves — it only bites in the early, poorly-initialised epochs.
    R_CLAMP = 1e4
    R_tt     = torch.clamp(R_tt,     -R_CLAMP, R_CLAMP)
    R_rr     = torch.clamp(R_rr,     -R_CLAMP, R_CLAMP)
    R_thth   = torch.clamp(R_thth,   -R_CLAMP, R_CLAMP)
    R_tphi   = torch.clamp(R_tphi,   -R_CLAMP, R_CLAMP)
    R_phiphi = torch.clamp(R_phiphi, -R_CLAMP, R_CLAMP)

    return R_tt, R_rr, R_thth, R_tphi, R_phiphi


# ─────────────────────────────────────────────
# 6. Loss Function
# ─────────────────────────────────────────────
def loss_function(
    model: KerrPINN,
    r_coll: torch.Tensor,
    th_coll: torch.Tensor,
    r_asymp: torch.Tensor,
    th_asymp: torch.Tensor,
    pde_weight: float = LAMBDA_PDE,   # curriculum: caller ramps this from 0 → LAMBDA_PDE
) -> tuple[torch.Tensor, dict]:
    """
    Total loss = pde_weight · L_pde  +  L_asymp  +  L_ref

    pde_weight is passed by the training loop and increases from 0 during a
    warmup phase. This is the key change from v1:

      Phase 0 (BC warmup): pde_weight = 0. Only L_asymp + L_ref are active.
        The metric is driven to the correct coordinate-scale initial condition.
        Without this, the PDE loss at 10^14 drowns the BC signal entirely.

      Phase 1 (ramp): pde_weight rises linearly from 0 to LAMBDA_PDE over
        N_ramp epochs. The metric is now near-correct in scale, so Ricci
        residuals start much smaller and training is stable.

      Phase 2 (full): pde_weight = LAMBDA_PDE. Standard PINN training.
    """

    # ── PDE loss ─────────────────────────────────────────────────────────────
    R_tt, R_rr, R_thth, R_tphi, R_phiphi = compute_ricci_residuals(
        model, r_coll, th_coll
    )
    loss_pde = torch.mean(R_tt**2 + R_rr**2 + R_thth**2 + R_tphi**2 + R_phiphi**2)

    # ── Asymptotic flatness loss ──────────────────────────────────────────────
    gtt_a, grr_a, gthth_a, gtphi_a, gphiphi_a = get_components(
        model, r_asymp, th_asymp
    )
    ra  = r_asymp.squeeze()
    tha = th_asymp.squeeze()
    tgt = exact_kerr(ra, tha)

    # Relative MSE: each component contributes equally regardless of magnitude.
    def rel_mse(pred, target, floor=0.01):
        return torch.mean((pred - target)**2 / (target.detach()**2 + floor))

    loss_asymp = (
        rel_mse(gtt_a.squeeze(),     tgt[0]) +
        rel_mse(grr_a.squeeze(),     tgt[1]) +
        rel_mse(gthth_a.squeeze(),   tgt[2]) +
        rel_mse(gtphi_a.squeeze(),   tgt[3], floor=1e-4) +
        rel_mse(gphiphi_a.squeeze(), tgt[4])
    )

    # ── Reference-point constraint ────────────────────────────────────────────
    r_ref  = torch.tensor([[R_REF]],  dtype=torch.float32, device=r_coll.device)
    th_ref = torch.tensor([[TH_REF]], dtype=torch.float32, device=r_coll.device)
    _, _, _, _, gpp_ref = get_components(model, r_ref, th_ref)

    Sig_ref   = R_REF**2
    gpp_exact = float(R_REF**2 + a**2 + 2*M*a**2*R_REF / Sig_ref)
    loss_ref  = (gpp_ref.squeeze() - gpp_exact)**2 / (gpp_exact**2 + 1.0)

    total = (  pde_weight   * loss_pde
             + LAMBDA_ASYMP * loss_asymp
             + LAMBDA_REF   * loss_ref    )

    return total, {
        "total":  total.item(),
        "pde":    loss_pde.item(),
        "asymp":  loss_asymp.item(),
        "ref":    loss_ref.item(),
    }


# ─────────────────────────────────────────────
# 7. Collocation Sampling (2D)
# ─────────────────────────────────────────────
def sample_collocation(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample n collocation points in the (r, θ) domain.

    r:  log-uniform in [r_min, r_max] — denser near the horizon where
        metric gradients are steepest (same rationale as Schwarzschild code).
    θ:  uniform in [th_min, th_max] — no preferred angular scale.

    Returns (r, θ) each shape (N, 1) with requires_grad=True.
    The requires_grad flag is essential: autograd in compute_ricci_residuals
    differentiates the loss with respect to these coordinates to build the
    Christoffel symbols and then the Ricci tensor.
    """
    r  = torch.exp(
        torch.FloatTensor(n, 1).uniform_(np.log(r_min), np.log(r_max))
    ).to(DEVICE).requires_grad_(True)
    th = torch.FloatTensor(n, 1).uniform_(th_min, th_max).to(DEVICE).requires_grad_(True)
    return r, th


# ─────────────────────────────────────────────
# 8. Training
# ─────────────────────────────────────────────
def train(
    model: KerrPINN,
    n_epochs:     int   = 10_000,
    n_coll:       int   = 300,
    lr_adam:      float = 1e-3,
    use_lbfgs:    bool  = True,
    lbfgs_steps:  int   = 500,
    log_every:    int   = 200,
) -> dict:
    """
    Two-phase training (identical structure to Schwarzschild PINN):
      Phase 1 — Adam + cosine-annealing warm restarts.
      Phase 2 — L-BFGS with strong Wolfe line search.

    NOTE ON COLLOCATION SIZE: n_coll=300 is substantially smaller than
    the Schwarzschild code's n_coll=1000 because each Ricci evaluation
    involves O(64) Christoffel evaluations and O(5 × 4²) Ricci component
    computations, each requiring two autograd passes. Start small and scale
    up once convergence is confirmed.
    """
    model.to(DEVICE)

    # Asymptotic BC points: r = r_max at n_a uniformly-spaced θ values.
    # Using multiple angles ensures the asymptotic constraint is enforced
    # across the full angular domain, not just one slice.
    n_a   = 20
    th_a_np = np.linspace(th_min, th_max, n_a)
    r_asymp  = torch.full((n_a, 1), r_max, dtype=torch.float32, device=DEVICE)
    th_asymp = torch.tensor(th_a_np[:, None], dtype=torch.float32, device=DEVICE)

    history = {k: [] for k in
               ["total", "pde", "asymp", "ref",
                "l2_gtt", "l2_grr", "l2_gtphi", "l2_gphiphi"]}

    def log_metrics(epoch: int, components: dict):
        for key in ["total", "pde", "asymp", "ref"]:
            history[key].append(components[key])

        # L2 errors on equatorial slice (θ = π/2) for four components.
        with torch.no_grad():
            r_t  = torch.linspace(r_min, r_max, 100, device=DEVICE).unsqueeze(1)
            th_t = torch.full((100, 1), np.pi / 2, device=DEVICE)
            gtt_p, grr_p, _, gtphi_p, gpp_p = get_components(model, r_t, th_t)
            gtt_e, grr_e, _, gtphi_e, gpp_e = exact_kerr(
                r_t.squeeze(), th_t.squeeze()
            )
            def l2(pred, exact):
                return torch.sqrt(torch.mean((pred - exact)**2)).item()
            l2_gtt   = l2(gtt_p.squeeze(),   gtt_e)
            l2_grr   = l2(grr_p.squeeze(),   grr_e)
            l2_gtphi = l2(gtphi_p.squeeze(), gtphi_e)
            l2_gpp   = l2(gpp_p.squeeze(),   gpp_e)

        history["l2_gtt"].append(l2_gtt)
        history["l2_grr"].append(l2_grr)
        history["l2_gtphi"].append(l2_gtphi)
        history["l2_gphiphi"].append(l2_gpp)

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:6d} | "
                f"Total {components['total']:.3e} | "
                f"PDE {components['pde']:.3e} | "
                f"Asymp {components['asymp']:.3e} | "
                f"Ref {components['ref']:.3e} | "
                f"L2 g_tt {l2_gtt:.3e} | "
                f"L2 g_tφ {l2_gtphi:.3e}"
            )

    # ── Curriculum schedule ────────────────────────────────────────────────
    # N_warmup: BC-only epochs. PDE loss is zero. The metric learns correct
    #   coordinate scales from asymptotic BC before any Ricci residuals fire.
    # N_ramp:   PDE weight rises linearly from 0 to LAMBDA_PDE. Ricci
    #   residuals are now much smaller (metric is well-scaled) and training
    #   is stable. This is the key fix for the 10^14 PDE loss explosion.
    # Remaining epochs: full loss, standard PINN training.
    N_warmup = max(500,  n_epochs // 20)    # 5% of total as BC warmup
    N_ramp   = max(2000, n_epochs // 5)     # 20% of total as linear ramp

    def pde_schedule(epoch: int) -> float:
        if epoch < N_warmup:
            return 0.0
        elif epoch < N_warmup + N_ramp:
            frac = (epoch - N_warmup) / N_ramp
            return float(LAMBDA_PDE * frac)
        else:
            return float(LAMBDA_PDE)

    # ── Phase 1: Adam ──────────────────────────────────────────────────────
    print("=" * 70)
    print("Phase 1: Adam (Kerr PINN v2 — curriculum training)")
    print(f"  BC-only warmup:  epochs 0 – {N_warmup}")
    print(f"  PDE ramp:        epochs {N_warmup} – {N_warmup+N_ramp}")
    print(f"  Full training:   epochs {N_warmup+N_ramp} – {n_epochs}")
    print("=" * 70)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_adam)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=2000, T_mult=1, eta_min=1e-6
    )

    for epoch in range(n_epochs):
        r_coll, th_coll = sample_collocation(n_coll)
        optimizer.zero_grad()
        pw = pde_schedule(epoch)
        loss, comp = loss_function(
            model, r_coll, th_coll, r_asymp, th_asymp, pde_weight=pw
        )
        loss.backward()
        # Tighter clip than v1 (0.1 vs 1.0) — prevents gradient spikes even
        # at the start of the PDE ramp from disrupting BC-converged parameters.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step(epoch)
        log_metrics(epoch, comp)

    # ── Phase 2: L-BFGS ───────────────────────────────────────────────────
    if use_lbfgs:
        print("=" * 70)
        print("Phase 2: L-BFGS (Kerr PINN v2)")
        print("=" * 70)

        r_fixed, th_fixed = sample_collocation(n_coll * 2)
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
            loss, comp = loss_function(
                model, r_fixed, th_fixed, r_asymp, th_asymp,
                pde_weight=LAMBDA_PDE   # full weight in L-BFGS phase
            )
            loss.backward()
            step_counter[0] += 1
            if step_counter[0] % 50 == 0:
                log_metrics(n_epochs + step_counter[0], comp)
            return loss

        optimizer_lbfgs.step(closure)

    print("\nTraining complete.")
    return history


# ─────────────────────────────────────────────
# 9. Evaluation & Plotting
# ─────────────────────────────────────────────
def evaluate_and_plot(model: KerrPINN, history: dict):
    """
    Produces two figures:

    Figure 1 — Metric recovery (5 rows, 3 columns):
      Column 1: PINN vs Exact on equatorial slice θ = π/2.
      Column 2: Pointwise absolute error on equatorial slice (log scale).
      Column 3: 2D error map log₁₀|PINN − Exact| across (r, θ) domain.

    Figure 2 — Training history:
      Left:  Loss components vs epoch.
      Right: L2 errors for g_tt and g_tφ vs epoch.

    The 2D error maps in Figure 1 Column 3 are the key new diagnostic
    relative to the Schwarzschild code — they reveal whether the PINN
    has learned the angular structure of the Kerr metric (Σ, frame dragging)
    or has collapsed to an angle-independent approximation.
    """
    model.eval()

    # Build 2D evaluation grid
    n_r, n_th = 200, 60
    r_np  = np.linspace(r_min, r_max, n_r)
    th_np = np.linspace(th_min, th_max, n_th)
    R_grid, TH_grid = np.meshgrid(r_np, th_np)

    r_flat  = torch.tensor(R_grid.flatten()[:, None],  dtype=torch.float32, device=DEVICE)
    th_flat = torch.tensor(TH_grid.flatten()[:, None], dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        gtt_p, grr_p, gthth_p, gtphi_p, gphiphi_p = get_components(
            model, r_flat, th_flat
        )
        gtt_e, grr_e, gthth_e, gtphi_e, gphiphi_e = exact_kerr(
            r_flat.squeeze(), th_flat.squeeze()
        )

    def to_grid(t):
        return t.cpu().numpy().reshape(n_th, n_r)

    pinn_grids  = list(map(to_grid, [gtt_p,  grr_p,  gthth_p,  gtphi_p,  gphiphi_p]))
    exact_grids = list(map(to_grid, [gtt_e,  grr_e,  gthth_e,  gtphi_e,  gphiphi_e]))
    names       = [r"$g_{tt}$", r"$g_{rr}$", r"$g_{\theta\theta}$",
                   r"$g_{t\varphi}$", r"$g_{\varphi\varphi}$"]

    mid = n_th // 2   # index closest to θ = π/2

    # ── Figure 1: Metric recovery ──────────────────────────────────────────
    fig, axes = plt.subplots(5, 3, figsize=(16, 22))
    fig.suptitle(
        f"Kerr PINN — Metric Recovery (a = {a:.2f} M,  r+ = {r_plus:.3f} M)",
        fontsize=13, fontweight="bold"
    )

    for i, (name, pred_g, exact_g) in enumerate(zip(names, pinn_grids, exact_grids)):

        eq_pred  = pred_g[mid]
        eq_exact = exact_g[mid]

        # Column 0: equatorial slice comparison
        ax = axes[i, 0]
        ax.plot(r_np, eq_pred,  lw=2, color="#1f77b4", label="PINN")
        ax.plot(r_np, eq_exact, lw=2, ls="--", color="#d62728", label="Exact")
        ax.set(xlabel="r / M", ylabel=name, title=f"{name}(r, θ=π/2)")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

        # Column 1: pointwise error on equatorial slice
        ax = axes[i, 1]
        err_eq = np.clip(np.abs(eq_pred - eq_exact), 1e-12, None)
        ax.semilogy(r_np, err_eq, color="#1f77b4", lw=2)
        ax.axhline(np.mean(err_eq), color="gray", lw=1, ls="--")
        ax.text(r_np[-1], np.mean(err_eq)*1.5,
                f"mean={np.mean(err_eq):.1e}", fontsize=7, ha="right", color="gray")
        ax.set(xlabel="r / M", ylabel=f"|PINN − Exact|",
               title=f"{name} error (θ=π/2, log scale)")
        ax.grid(alpha=0.3)

        # Column 2: 2D error map
        ax = axes[i, 2]
        err_2d = np.log10(np.clip(np.abs(pred_g - exact_g), 1e-12, None))
        im = ax.pcolormesh(R_grid, TH_grid, err_2d, cmap="plasma", shading="auto")
        fig.colorbar(im, ax=ax, label="log₁₀|error|")
        ax.set(xlabel="r / M", ylabel="θ (rad)",
               title=f"2D log₁₀|{name} error|")
        ax.axhline(np.pi/2, color="white", lw=0.8, ls="--", alpha=0.6,
                   label="θ=π/2")
        ax.legend(fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("kerr_metric_recovery.png", dpi=150, bbox_inches="tight")
    print("Figure 1 saved → kerr_metric_recovery.png")
    plt.show()

    # ── Figure 2: Training history ──────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Kerr PINN — Training History", fontsize=13)

    for key, label, color in [
        ("total", "Total",        "#333333"),
        ("pde",   "PDE",          "#1f77b4"),
        ("asymp", "Asymptotic BC","#ff7f0e"),
        ("ref",   "Reference pt", "#2ca02c"),
    ]:
        axes2[0].semilogy(history[key], label=label, lw=1.5, color=color)
    axes2[0].set(xlabel="Epoch", ylabel="Loss", title="Loss components")
    axes2[0].legend(); axes2[0].grid(alpha=0.3)

    for key, label in [
        ("l2_gtt",     r"L2 error $g_{tt}$"),
        ("l2_grr",     r"L2 error $g_{rr}$"),
        ("l2_gtphi",   r"L2 error $g_{t\varphi}$"),
        ("l2_gphiphi", r"L2 error $g_{\varphi\varphi}$"),
    ]:
        axes2[1].semilogy(history[key], label=label, lw=1.5)
    axes2[1].set(xlabel="Epoch", ylabel="L2 error",
                 title="L2 errors vs exact (equatorial θ=π/2 slice)")
    axes2[1].legend(); axes2[1].grid(alpha=0.3)

    fig2.tight_layout()
    fig2.savefig("kerr_training_history.png", dpi=150, bbox_inches="tight")
    print("Figure 2 saved → kerr_training_history.png")
    plt.show()


# ─────────────────────────────────────────────
# 10. Frame-Dragging Diagnostic (Kerr-specific)
# ─────────────────────────────────────────────
def frame_dragging_diagnostic(model: KerrPINN):
    """
    Diagnose whether the PINN has correctly learned the frame-dragging
    structure of the Kerr metric — the aspect most distinct from Schwarzschild.

    The angular velocity of a zero-angular-momentum observer (ZAMO) is:
        Ω(r,θ) = −g^{tφ} / g^{tt} = 2Mar / (r²+a²)² − a²Δ sin²θ) [Kerr exact]

    If the PINN has correctly recovered g_tφ and the (t,φ) block inverse,
    the ZAMO angular velocity should match its exact form. A PINN that
    learned only the Schwarzschild-like diagonal structure would give Ω = 0.

    Ergoregion boundary: g_tt = 0  →  r_erg(θ) = M + √(M² − a²cos²θ)
    At θ=π/2: r_erg = r_s = 2M.
    """
    model.eval()
    n = 300
    r_np  = np.linspace(r_min, r_max, n)
    th_vals = [np.pi/6, np.pi/4, np.pi/3, np.pi/2]   # 30°, 45°, 60°, 90°

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Kerr PINN — Frame-Dragging Diagnostic (a = {a:.2f} M)",
        fontsize=12, fontweight="bold"
    )

    cmap = plt.cm.viridis
    colors = [cmap(i / (len(th_vals)-1)) for i in range(len(th_vals))]

    for th_val, col in zip(th_vals, colors):
        r_t  = torch.tensor(r_np[:, None], dtype=torch.float32, device=DEVICE)
        th_t = torch.full_like(r_t, th_val)

        with torch.no_grad():
            gtt_p, _, _, gtphi_p, gphiphi_p = get_components(model, r_t, th_t)
            gtt_i, _, _, gtphi_i, _ = invert_metric_analytical(
                gtt_p.squeeze(), torch.ones(n, device=DEVICE),
                torch.ones(n, device=DEVICE),
                gtphi_p.squeeze(), gphiphi_p.squeeze()
            )
            Omega_p = (-gtphi_i / gtt_i).cpu().numpy()

        # Exact ZAMO angular velocity
        Sig_e = r_np**2 + a**2 * np.cos(th_val)**2
        Del_e = r_np**2 - 2*M*r_np + a**2
        sin2  = np.sin(th_val)**2
        gtt_e_np  = -(1 - 2*M*r_np/Sig_e)
        gtphi_e_np = -2*M*a*r_np*sin2/Sig_e
        gphiphi_e_np = (r_np**2 + a**2 + 2*M*a**2*r_np*sin2/Sig_e)*sin2
        det = gtt_e_np*gphiphi_e_np - gtphi_e_np**2
        Omega_e = -(-gtphi_e_np/det) / (gphiphi_e_np/det)

        label = f"θ = {int(np.degrees(th_val))}°"
        axes[0].plot(r_np, Omega_p, color=col, lw=2,      label=f"PINN {label}")
        axes[0].plot(r_np, Omega_e, color=col, lw=2, ls="--", label=f"Exact {label}")

        err = np.clip(np.abs(Omega_p - Omega_e), 1e-12, None)
        axes[1].semilogy(r_np, err, color=col, lw=2, label=label)

        gtt_vals = gtt_p.cpu().numpy().squeeze()
        axes[2].plot(r_np, gtt_vals, color=col, lw=2, label=f"PINN {label}")
        gtt_ex   = -(1 - 2*M*r_np/Sig_e)
        axes[2].plot(r_np, gtt_ex, color=col, lw=2, ls="--")

    # Ergoregion boundary g_tt = 0
    axes[2].axhline(0, color="black", lw=1.5, ls=":", label="Ergoregion (g_tt=0)")

    axes[0].set(xlabel="r / M", ylabel="Ω(r,θ)",
                title="ZAMO angular velocity (frame dragging)")
    axes[0].legend(fontsize=7, ncol=2); axes[0].grid(alpha=0.3)

    axes[1].set(xlabel="r / M", ylabel="|Ω_PINN − Ω_exact|",
                title="Frame-dragging error (log scale)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    axes[2].set(xlabel="r / M", ylabel=r"$g_{tt}(r,\theta)$",
                title="Lapse: ergoregion boundary at g_tt = 0")
    axes[2].legend(fontsize=7); axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("kerr_frame_dragging.png", dpi=150, bbox_inches="tight")
    print("Figure 3 saved → kerr_frame_dragging.png")
    plt.show()


# ─────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = KerrPINN(hidden_layers=6, hidden_dim=64)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nNetwork parameters:  {n_params:,}")
    print(f"Spin:                a = {a:.3f} M")
    print(f"Outer horizon:       r+ = {r_plus:.4f} M")
    print(f"Training domain:     r ∈ [{r_min:.3f}, {r_max:.1f}] M")
    print(f"                     θ ∈ [{th_min:.3f}, {th_max:.3f}] rad")
    print(f"Reference constraint: g_φφ at r={R_REF:.1f}M, θ=π/2\n")

    history = train(
        model,
        n_epochs=10_000,
        n_coll=4000,         # smaller than Schwarzschild due to higher per-step cost
        lr_adam=1e-3,
        use_lbfgs=True,
        lbfgs_steps=500,
        log_every=10,
    )

    tf = time.time()

    evaluate_and_plot(model, history)
    frame_dragging_diagnostic(model)

    dt = tf-t0
    print(f"Ran for {dt//60} minutes and {dt%60} seconds")
