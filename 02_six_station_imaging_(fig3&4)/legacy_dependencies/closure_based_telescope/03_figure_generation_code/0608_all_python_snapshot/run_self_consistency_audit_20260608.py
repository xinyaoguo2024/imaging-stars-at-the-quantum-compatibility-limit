from __future__ import annotations

from audit_20260608_common import OUT, write_json
from audit_20260608_derivation import compile_derivation_pdf, write_derivation_tex
from audit_20260608_eq19 import run_eq19_checks
from audit_20260608_fig1 import run_fig1_checks
from audit_20260608_fig3 import run_fig3_checks
from audit_20260608_gain_factors import run_gain_factor_audit


def main() -> None:
    eq19 = run_eq19_checks()
    fig1 = run_fig1_checks()
    fig3 = run_fig3_checks()
    gain = run_gain_factor_audit()
    tex = write_derivation_tex(eq19, fig1, fig3, gain)
    derivation = compile_derivation_pdf(tex)
    payload = {
        "eq19": eq19,
        "fig1": fig1,
        "fig3": fig3,
        "gain_factors": gain,
        "derivation": derivation,
    }
    summary = OUT / "self_consistency_audit_summary.json"
    write_json(summary, payload)
    print(f"summary={summary}")
    print(f"eq19_max_rel={eq19['max_rel_projected_minus_schur']:.3e}")
    print(f"fig1_max_rel={fig1['max_rel_error_eq13']:.3e}")
    print(f"near_over_scheduled={fig3['near_over_scheduled_rms_mean_range']}")
    print(f"gain_factor_csv={gain['csv']}")
    print(f"derivation_pdf={derivation.get('pdf')}")


if __name__ == "__main__":
    main()
