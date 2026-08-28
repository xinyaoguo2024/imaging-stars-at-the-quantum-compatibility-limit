# Strict Physical Near With Pair-Combine Taps

This run optimizes independent station-side split variables.  The variables are not tied across stations.

```json
{
  "model": "strict_physical_near_with_paircombine",
  "receiver": "station-side p/alpha/gamma plus pair-combine modules (two stations coherently combined, then beaten with the third station)",
  "objective": "worst-ratio",
  "objective_detail": "direct-score maximizes the direct optimized schedule score on gain vs edge; match-direct penalizes per-loop log(SNR_near/SNR_direct_optimized) deviations, including overshoot; ratio-only uses only -mean(log(SNR_near/SNR_direct_optimized)^2); worst-ratio is dominated by max_l |log(SNR_near,l/SNR_direct,l)|, with small RMS/variance tie-breakers",
  "reproduction_check": {
    "current_near_max_relative_error": 0.0,
    "remote_star_independent_max_relative_error": 0.0
  },
  "elapsed_s": 8.804740190505981,
  "optimized": {
    "score": -0.014185386002986989,
    "best_tag": "checkpoint:checkpoint_worst-ratio.json",
    "objective": "worst-ratio",
    "eval_count": 5,
    "core_cache_size": 4,
    "star_cache_size": 5,
    "alpha_core": [
      0.08966286468242174,
      0.07112631627269807,
      0.12386032688131807
    ],
    "station_budget_total": {
      "Keck II": 0.9999999999999994,
      "Subaru": 0.9999999999999999,
      "Gemini North": 0.9999999999999994,
      "new 5 m r=2km compact": 0.9999999999999991,
      "new 5 m r=4km compact": 1.0000000000000002,
      "new 5 m r=9km compact": 0.9999999999999994
    },
    "station_budget_max_abs_error": 8.881784197001252e-16,
    "gain_vs_edge": {
      "min": 1.2015203879039207,
      "mean": 1.2594747026782305,
      "median": 1.2339028747532206,
      "max": 1.3912313830590968,
      "std": 0.06581366858659018
    },
    "top_paircombine_modules": {
      "S1+S2|S3": {
        "total_q": 0.21059427635595368,
        "fractions": [
          0.06278319205553771,
          0.08622633992735981,
          0.06158474437305617
        ],
        "beta": 0.7853981633974483,
        "delta": 2.441592653589794
      },
      "S1+S2|S5": {
        "total_q": 0.06388041922801233,
        "fractions": [
          0.017110415903294464,
          0.023499419027163643,
          0.02327058429755421
        ],
        "beta": 0.7853981633974483,
        "delta": 2.441592653589794
      }
    }
  },
  "direct_gain_vs_edge": {
    "min": 1.2112182127920084,
    "mean": 1.2657068827332938,
    "median": 1.2348022012752549,
    "max": 1.4027337769266373,
    "std": 0.06450228176532798
  },
  "old_remote_star_independent_gain_vs_edge": {
    "min": 1.0000612560361464,
    "mean": 1.1572738520284234,
    "median": 1.1187292126419792,
    "max": 1.3657509493176099,
    "std": 0.14963130984879344
  },
  "paircombine_strict_near_gain_vs_edge": {
    "min": 1.2015203879039207,
    "mean": 1.2594747026782305,
    "median": 1.2339028747532206,
    "max": 1.3912313830590968,
    "std": 0.06581366858659018
  },
  "paircombine_near_to_direct_snr": {
    "min": 0.9878100076121603,
    "mean": 0.9950259206541114,
    "median": 0.9929051003914171,
    "max": 1.0052512668447333,
    "std": 0.005772067862067776
  },
  "files": {
    "loop_gains_csv": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/27_strict_physical_near_paircombine_20260613/results/paircombine_strict_near_loop_gains_worst_ratio.csv",
    "module_csv": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/27_strict_physical_near_paircombine_20260613/results/paircombine_modules_worst_ratio.csv",
    "summary_json": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/27_strict_physical_near_paircombine_20260613/results/paircombine_strict_near_summary_worst_ratio.json",
    "diagnostic_png": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/27_strict_physical_near_paircombine_20260613/figures/paircombine_strict_near_loop_gain_diagnostic_worst_ratio.png"
  }
}
```

