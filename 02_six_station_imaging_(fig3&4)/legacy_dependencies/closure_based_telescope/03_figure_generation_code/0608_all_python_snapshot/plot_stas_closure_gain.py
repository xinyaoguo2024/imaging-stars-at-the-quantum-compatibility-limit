import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ROOT = Path(__import__("os").environ["VLBI_HOMODYNE_ROOT"])
OUTDIR = ROOT / "output" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def fcond_opt(nu12, nu23, nu31):
    a = nu12
    b = nu23
    c = nu31
    delta0 = 4.0 - a * a - b * b - c * c - a * b * c
    poly = (
        4.0 * (a * a + b * b + c * c)
        - (a**4 + b**4 + c**4)
        + 2.0 * (a * a * b * b + b * b * c * c + c * c * a * a)
        + 12.0 * a * b * c
    )
    # Physical closure phase is Phi_cl = phi12 + phi23 + phi31 = 3 q on the
    # equal-edge gauge slice used in the local SLD construction, so the
    # conditioned Fisher information for Phi_cl is smaller than that for q by 9.
    return 2.0 * poly / (27.0 * delta0)


def g0(nu12, nu23, nu31):
    return 3.0 * fcond_opt(nu12, nu23, nu31) * (
        1.0 / nu12**2 + 1.0 / nu23**2 + 1.0 / nu31**2
    )


nu12 = 0.3
alpha = np.logspace(-2, 0, 220)
beta = np.logspace(-2, 0, 220)
A, B = np.meshgrid(alpha, beta)
NU23 = nu12 * A
NU31 = nu12 * B
G0 = g0(nu12, NU23, NU31)
SNR_GAIN0 = np.sqrt(G0)
SNR_GAIN_LOWU = np.sqrt(3.0 * G0)

r = np.logspace(-3, 3, 500)
noise_factor = np.sqrt(3.0 * (1.0 + r) / (3.0 + r))

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    }
)

fig = plt.figure(figsize=(13.5, 4.8), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 0.9, 1.05])

ax1 = fig.add_subplot(gs[0, 0])
pcm1 = ax1.pcolormesh(A, B, SNR_GAIN0, shading="auto", cmap="viridis")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel(r"$\alpha=\nu_{23}/\nu_{12}$")
ax1.set_ylabel(r"$\beta=\nu_{31}/\nu_{12}$")
ax1.set_title(r"Lossless gain $\mathcal{G}_0$")
cbar1 = fig.colorbar(pcm1, ax=ax1)
cbar1.set_label("SNR gain")

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(r, noise_factor, color="#d04f2b", lw=2.5)
ax2.axhline(1.0, color="0.5", ls="--", lw=1)
ax2.axhline(np.sqrt(3.0), color="0.5", ls=":", lw=1)
ax2.set_xscale("log")
ax2.set_xlabel(r"$r=\epsilon/(\eta u)$")
ax2.set_ylabel(r"factor $\sqrt{3(1+r)/(3+r)}$")
ax2.set_title("Common-noise correction")
ax2.grid(True, alpha=0.25)

ax3 = fig.add_subplot(gs[0, 2])
pcm3 = ax3.pcolormesh(A, B, SNR_GAIN_LOWU, shading="auto", cmap="magma")
ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel(r"$\alpha=\nu_{23}/\nu_{12}$")
ax3.set_ylabel(r"$\beta=\nu_{31}/\nu_{12}$")
ax3.set_title(r"Low-flux noisy gain $\sqrt{3}\,\mathcal{G}_0$")
cbar3 = fig.colorbar(pcm3, ax=ax3)
cbar3.set_label("SNR gain")

fig.suptitle(
    r"Three-station closure-phase gain for $\nu_{12}=0.3,\ \nu_{23}=0.3\alpha,\ \nu_{31}=0.3\beta$",
    y=1.02,
)

png = OUTDIR / "stas_closure_gain_summary.png"
pdf = OUTDIR / "stas_closure_gain_summary.pdf"
fig.savefig(png, dpi=220, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(png)
print(pdf)
