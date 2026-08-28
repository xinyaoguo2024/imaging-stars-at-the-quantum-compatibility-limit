# Balanced 10-Loop Remote-Star Joint Near Receiver Attempt

This run keeps the six-station case obtained by dropping the original station 1 and relabeling old S2-S7 as S1-S6.

The plotted loop basis is no longer the default root-loop basis.  It is the balanced independent set `{123, 124, 125, 134, 136, 245, 256, 346, 356, 456}`.  Each station appears in five selected loops, so assigning weight 0.2 to every selected direct-closure receiver uses exactly one unit of station-side photon budget at every station.

The near split is re-optimized against this balanced 10-loop direct target.  Its objective keeps every selected loop at or above uniform edge-first and then matches `SNR_near/SNR_direct_balanced` as close to one as possible with a variance penalty.

Current near treats every remote-involved baseline as pairwise edge-first after the compact-core joint receiver.  The new test replaces the three pairwise core-remote beats at each remote station by one local joint receiver on `{core1, core2, core3, remote}`.  This is a physically budget-conserving way to include coherent-sum observables such as `(core1 + core2)` beaten with the remote field.

A `gamma` parameter controls how much of each core-remote directed split enters the remote-star joint receiver; the residual stays in ordinary pairwise edge-first channels.  The scalar scan uses one common gamma for every such directed split, while the independent search allows separate core-to-remote and remote-to-core gamma values.  Both gamma optimizations use the balanced 10-loop direct target and reject candidates with any selected-loop gain below edge-first.

For the reported scalar and independent remote-star columns, the local core-core phases inside each remote-star receiver are treated as nuisance parameters and Schur-complemented out before embedding the Fisher block.  This avoids double-counting compact-core phase information already supplied by the compact-core joint receiver.  The unrestricted full-star result is kept only as a diagnostic in the JSON summary.

## Summary

```json
{
  "case": "six_station_oldS2_to_oldS7_compact_remote3",
  "n_station": 6,
  "n_balanced_loops": 10,
  "loop_set": "balanced_10loop_independent",
  "loop_set_triangles": [
    "S1-S2-S3",
    "S1-S2-S4",
    "S1-S2-S5",
    "S1-S3-S4",
    "S1-S3-S6",
    "S2-S4-S5",
    "S2-S5-S6",
    "S3-S4-S6",
    "S3-S5-S6",
    "S4-S5-S6"
  ],
  "source_start_payload": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/16_six_station_reduced_from7_20260611/results/near_match_direct_split_payload.json",
  "alpha_core": [
    0.14079674136878423,
    0.12175659303653515,
    0.25652618403204075
  ],
  "station_budget_total_minmax": [
    0.9999999999999998,
    1.0
  ],
  "direct_weight_info": {
    "model": "physical_all_triangle_direct_split_station_budget",
    "description": "Each scalar three-mode direct-closure receiver on triangle (a,b,c) consumes the same fraction w from stations a,b,c. With w=1/C(N-1,2), each station's total directed fraction over all triangles is exactly one.",
    "n_triangle_settings": 20,
    "per_triangle_weight": 0.1,
    "station_budget_constraint": "for every station i, sum_{tri contains i} w_tri <= 1",
    "station_weight_sums": {
      "S1": 1.0,
      "S2": 1.0,
      "S3": 1.0,
      "S4": 1.0,
      "S5": 1.0,
      "S6": 1.0
    },
    "max_station_weight_sum": 1.0,
    "total_triangle_weight_sum": 2.0,
    "target": "loop-by-loop matching target for the near split",
    "scalar_fisher_min": 36.07893900155251,
    "scalar_fisher_mean": 2612.256928321971,
    "scalar_fisher_max": 31692.409055153355,
    "scalar_fishers": {
      "S1-S2-S3": 31692.409055153355,
      "S1-S2-S4": 6531.075223271444,
      "S1-S2-S5": 1631.1528273198317,
      "S1-S2-S6": 479.7079228706326,
      "S1-S3-S4": 3566.460925759069,
      "S1-S3-S5": 1425.018069258542,
      "S1-S3-S6": 338.1828920041692,
      "S1-S4-S5": 458.3029643042844,
      "S1-S4-S6": 200.89471378315366,
      "S1-S5-S6": 43.4773342942407,
      "S2-S3-S4": 3100.68878712162,
      "S2-S3-S5": 1131.3849181762184,
      "S2-S3-S6": 295.20068765587,
      "S2-S4-S5": 432.44121432020006,
      "S2-S4-S6": 194.7395036441877,
      "S2-S5-S6": 42.283346188828276,
      "S3-S4-S5": 424.492260299155,
      "S3-S4-S6": 177.8519923508362,
      "S3-S5-S6": 43.294989662223074,
      "S4-S5-S6": 36.07893900155251
    }
  },
  "direct_balanced_10loop_info": {
    "description": "Equal photon budget on ten balanced independent triangle-closure coordinates",
    "per_loop_weight": 0.2,
    "selected_loops": [
      "S1-S2-S3",
      "S1-S2-S4",
      "S1-S2-S5",
      "S1-S3-S4",
      "S1-S3-S6",
      "S2-S4-S5",
      "S2-S5-S6",
      "S3-S4-S6",
      "S3-S5-S6",
      "S4-S5-S6"
    ],
    "selected_loop_rank": 10,
    "station_loop_counts": {
      "Keck II": 5,
      "Subaru": 5,
      "Gemini North": 5,
      "new 5 m r=2km compact": 5,
      "new 5 m r=4km compact": 5,
      "new 5 m r=9km compact": 5
    },
    "station_weight_sums": {
      "Keck II": 1.0,
      "Subaru": 1.0,
      "Gemini North": 1.0,
      "new 5 m r=2km compact": 1.0,
      "new 5 m r=4km compact": 1.0,
      "new 5 m r=9km compact": 1.0
    },
    "max_station_weight_error": 0.0,
    "all_triangle_weights": {
      "S1-S2-S3": 0.2,
      "S1-S2-S4": 0.2,
      "S1-S2-S5": 0.2,
      "S1-S2-S6": 0.0,
      "S1-S3-S4": 0.2,
      "S1-S3-S5": 0.0,
      "S1-S3-S6": 0.2,
      "S1-S4-S5": 0.0,
      "S1-S4-S6": 0.0,
      "S1-S5-S6": 0.0,
      "S2-S3-S4": 0.0,
      "S2-S3-S5": 0.0,
      "S2-S3-S6": 0.0,
      "S2-S4-S5": 0.2,
      "S2-S4-S6": 0.0,
      "S2-S5-S6": 0.2,
      "S3-S4-S5": 0.0,
      "S3-S4-S6": 0.2,
      "S3-S5-S6": 0.2,
      "S4-S5-S6": 0.2
    }
  },
  "direct_balanced_10loop_weights": [
    0.2,
    0.2,
    0.2,
    0.0,
    0.2,
    0.0,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.0,
    0.2,
    0.0,
    0.2,
    0.2,
    0.2
  ],
  "direct_optimized_schedule_info": {
    "objective": "maximize mean log root-loop SNR gain with worst-loop and variance regularization",
    "constraints": "w_tri >= 0 and sum_{tri contains station i} w_tri = 1 for every station",
    "score": 0.29984040038329557,
    "station_weight_sums": {
      "Keck II": 1.0,
      "Subaru": 1.0000000000000002,
      "Gemini North": 0.9999999999999999,
      "new 5 m r=2km compact": 1.0,
      "new 5 m r=4km compact": 1.0000000000000002,
      "new 5 m r=9km compact": 1.0
    },
    "max_station_weight_error": 2.220446049250313e-16,
    "snr_gain_vs_edge": {
      "min": 1.2112182127920084,
      "mean": 1.2657068827332938,
      "median": 1.2348022012752549,
      "max": 1.4027337769266373
    },
    "top_triangle_weights": {
      "S2-S4-S5": 0.21572984095945513,
      "S3-S4-S6": 0.17524674721039313,
      "S1-S3-S6": 0.1676123990741348,
      "S1-S5-S6": 0.15012395655022,
      "S1-S3-S4": 0.14909195088412433,
      "S1-S2-S5": 0.14551303243062574,
      "S1-S2-S3": 0.1417004738791876,
      "S1-S2-S4": 0.13599150283790315
    },
    "all_triangle_weights": {
      "S1-S2-S3": 0.1417004738791876,
      "S1-S2-S4": 0.13599150283790315,
      "S1-S2-S5": 0.14551303243062574,
      "S1-S2-S6": 0.007999415194150913,
      "S1-S3-S4": 0.14909195088412433,
      "S1-S3-S5": 0.011579994512330164,
      "S1-S3-S6": 0.1676123990741348,
      "S1-S4-S5": 0.02246947628159544,
      "S1-S4-S6": 0.06791779835572798,
      "S1-S5-S6": 0.15012395655022,
      "S2-S3-S4": 0.07352191299006763,
      "S2-S3-S5": 0.07000313287922288,
      "S2-S3-S6": 0.041612251127353866,
      "S2-S4-S5": 0.21572984095945513,
      "S2-S4-S6": 0.03930554844695653,
      "S2-S5-S6": 0.12862288925507673,
      "S3-S4-S5": 0.03439868234548806,
      "S3-S4-S6": 0.17524674721039313,
      "S3-S5-S6": 0.1352324550976974,
      "S4-S5-S6": 0.08632653968828861
    }
  },
  "direct_optimized_schedule_weights": [
    0.1417004738791876,
    0.13599150283790315,
    0.14551303243062574,
    0.007999415194150913,
    0.14909195088412433,
    0.011579994512330164,
    0.1676123990741348,
    0.02246947628159544,
    0.06791779835572798,
    0.15012395655022,
    0.07352191299006763,
    0.07000313287922288,
    0.041612251127353866,
    0.21572984095945513,
    0.03930554844695653,
    0.12862288925507673,
    0.03439868234548806,
    0.17524674721039313,
    0.1352324550976974,
    0.08632653968828861
  ],
  "balanced10_near_split_info": {
    "objective": "match_direct_balanced_10loop_independent_set",
    "loop_set": "balanced_10loop_independent",
    "loops": [
      "S1-S2-S3",
      "S1-S2-S4",
      "S1-S2-S5",
      "S1-S3-S4",
      "S1-S3-S6",
      "S2-S4-S5",
      "S2-S5-S6",
      "S3-S4-S6",
      "S3-S5-S6",
      "S4-S5-S6"
    ],
    "alpha": 0.1730265061457867,
    "alpha_core": [
      0.14079674136878423,
      0.12175659303653515,
      0.25652618403204075
    ],
    "near_snr_gain_vs_edge": {
      "min": 1.0001184552907072,
      "mean": 1.0884155555842994,
      "median": 1.042001578790749,
      "max": 1.3657509493176092
    },
    "direct_target_snr_gain_vs_edge": {
      "min": 0.9547422760195636,
      "mean": 1.1625644974626355,
      "median": 1.1268200933530963,
      "max": 1.4908328893349319
    },
    "near_over_direct_snr": {
      "min": 0.8767680858927878,
      "mean": 0.9444704766791603,
      "median": 0.9348367050583675,
      "max": 1.0768808168140593
    },
    "rms_log_near_over_direct_snr": 0.09274420725548607,
    "mean_log_near_over_direct_snr": -0.059677416269917946,
    "var_log_near_over_direct_snr": 0.005040093966795487,
    "n_near_below_edge": 0,
    "n_near_below_direct": 7,
    "core_only_near_over_direct_mean": 0.9160992885841669,
    "one_remote_near_over_direct_mean": 0.8768489896475851,
    "two_remote_near_over_direct_mean": 0.9860821757007592,
    "all_loop_near_snr_gains_vs_edge": {
      "S1-S2-S3": 1.3657509493176092,
      "S1-S2-S4": 1.181304047384807,
      "S1-S2-S5": 1.192557731191296,
      "S1-S3-S4": 1.0558595155346364,
      "S1-S3-S6": 1.0578450001768323,
      "S2-S4-S5": 1.0009684777808607,
      "S2-S5-S6": 1.0001184552907072,
      "S3-S4-S6": 1.0011220387198492,
      "S3-S5-S6": 1.000485698399533,
      "S4-S5-S6": 1.0281436420468617
    },
    "all_loop_direct_snr_gains_vs_edge": {
      "S1-S2-S3": 1.4908328893349319,
      "S1-S2-S4": 1.3473392410056977,
      "S1-S2-S5": 1.3601577532322338,
      "S1-S3-S4": 1.2039383804406019,
      "S1-S3-S6": 1.2064228849444476,
      "S2-S4-S5": 1.0497018062655907,
      "S2-S5-S6": 0.9819833818578994,
      "S3-S4-S6": 1.035493992283949,
      "S3-S5-S6": 0.995032369241438,
      "S4-S5-S6": 0.9547422760195636
    },
    "all_loop_near_over_direct_snr": {
      "S1-S2-S3": 0.9160992885841669,
      "S1-S2-S4": 0.8767680858927878,
      "S1-S2-S5": 0.8767789827006034,
      "S1-S3-S4": 0.8770046147612859,
      "S1-S3-S6": 0.8768442752356634,
      "S2-S4-S5": 0.9535741215325682,
      "S2-S5-S6": 1.018467800746787,
      "S3-S4-S6": 0.9668062259943325,
      "S3-S5-S6": 1.0054805545293488,
      "S4-S5-S6": 1.0768808168140593
    },
    "score": -0.13558755083622206,
    "objective_formula": "maximize min_l log(SNR_near/SNR_direct_balanced)_l with variance and overshoot penalties; candidates with any balanced-loop SNR_near/SNR_edge < 1 are infeasible",
    "match_variance_lambda": 0.2,
    "alpha_bounds": [
      0.02,
      0.8
    ],
    "best_start": "split_coord_outer_2",
    "optimization_counts": {
      "core_recomputes": 174,
      "cached_core_hits": 31,
      "split_evaluations": 8791
    },
    "n_cached_core_blocks": 174
  },
  "balanced10_near_split_payload": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/18_balanced_10loop_independent_set_20260611/results/balanced10_near_split_payload.json",
  "full_star_info": {
    "remote_star_receivers": [
      {
        "remote": "new 5 m r=2km compact",
        "gamma_core_to_remote": "1,1,1",
        "gamma_remote_to_core": "1,1,1",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=2km compact",
        "core_to_remote_available_fractions": "0.41000381,0.3861104,0.37949123",
        "core_to_remote_star_fractions": "0.41000381,0.3861104,0.37949123",
        "remote_to_core_available_fractions": "0.34394156,0.2031972,0.14157886",
        "remote_to_core_joint_available_fraction": 0.6887176241503702,
        "remote_to_core_joint_star_fraction": 0.6887176241503702,
        "total_star_receiver_input_fraction": 1.8643230639615074
      },
      {
        "remote": "new 5 m r=4km compact",
        "gamma_core_to_remote": "1,1,1",
        "gamma_remote_to_core": "1,1,1",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=4km compact",
        "core_to_remote_available_fractions": "0.31801878,0.43693692,0.02",
        "core_to_remote_star_fractions": "0.31801878,0.43693692,0.02",
        "remote_to_core_available_fractions": "0.25512604,0.25512604,0.038861479",
        "remote_to_core_joint_available_fraction": 0.5491135590297128,
        "remote_to_core_joint_star_fraction": 0.5491135590297128,
        "total_star_receiver_input_fraction": 1.3240692575656263
      },
      {
        "remote": "new 5 m r=9km compact",
        "gamma_core_to_remote": "1,1,1",
        "gamma_remote_to_core": "1,1,1",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=9km compact",
        "core_to_remote_available_fractions": "0.13118067,0.055196088,0.34398259",
        "core_to_remote_star_fractions": "0.13118067,0.055196088,0.34398259",
        "remote_to_core_available_fractions": "0.20461897,0.14437412,0.26427487",
        "remote_to_core_joint_available_fraction": 0.613267952939735,
        "remote_to_core_joint_star_fraction": 0.613267952939735,
        "total_star_receiver_input_fraction": 1.1436272961553238
      }
    ],
    "split_row_sums": {
      "Keck II": 0.8592032586312156,
      "Subaru": 0.8782434069634648,
      "Gemini North": 0.7434738159679593,
      "new 5 m r=2km compact": 0.9999999999999999,
      "new 5 m r=4km compact": 1.0,
      "new 5 m r=9km compact": 1.0
    },
    "gamma_model": "gamma fraction of every core-remote directed split enters a remote-star joint receiver; the residual remains pairwise edge-first",
    "core_core_handling": "full",
    "gamma_kind": "by_remote",
    "gamma_core_to_remote": [
      [
        1.0,
        1.0,
        1.0
      ],
      [
        1.0,
        1.0,
        1.0
      ],
      [
        1.0,
        1.0,
        1.0
      ]
    ],
    "gamma_remote_to_core": [
      [
        1.0,
        1.0,
        1.0
      ],
      [
        1.0,
        1.0,
        1.0
      ],
      [
        1.0,
        1.0,
        1.0
      ]
    ]
  },
  "scalar_star_info": {
    "remote_star_receivers": [
      {
        "remote": "new 5 m r=2km compact",
        "gamma_core_to_remote": "0.15,0.15,0.15",
        "gamma_remote_to_core": "0.15,0.15,0.15",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=2km compact",
        "core_to_remote_available_fractions": "0.41000381,0.3861104,0.37949123",
        "core_to_remote_star_fractions": "0.061500572,0.05791656,0.056923684",
        "remote_to_core_available_fractions": "0.34394156,0.2031972,0.14157886",
        "remote_to_core_joint_available_fraction": 0.6887176241503702,
        "remote_to_core_joint_star_fraction": 0.10330764362255555,
        "total_star_receiver_input_fraction": 0.2796484595942262
      },
      {
        "remote": "new 5 m r=4km compact",
        "gamma_core_to_remote": "0.15,0.15,0.15",
        "gamma_remote_to_core": "0.15,0.15,0.15",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=4km compact",
        "core_to_remote_available_fractions": "0.31801878,0.43693692,0.02",
        "core_to_remote_star_fractions": "0.047702817,0.065540538,0.003",
        "remote_to_core_available_fractions": "0.25512604,0.25512604,0.038861479",
        "remote_to_core_joint_available_fraction": 0.5491135590297128,
        "remote_to_core_joint_star_fraction": 0.08236703385445694,
        "total_star_receiver_input_fraction": 0.19861038863484398
      },
      {
        "remote": "new 5 m r=9km compact",
        "gamma_core_to_remote": "0.15,0.15,0.15",
        "gamma_remote_to_core": "0.15,0.15,0.15",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=9km compact",
        "core_to_remote_available_fractions": "0.13118067,0.055196088,0.34398259",
        "core_to_remote_star_fractions": "0.0196771,0.0082794132,0.051597388",
        "remote_to_core_available_fractions": "0.20461897,0.14437412,0.26427487",
        "remote_to_core_joint_available_fraction": 0.613267952939735,
        "remote_to_core_joint_star_fraction": 0.09199019294096027,
        "total_star_receiver_input_fraction": 0.1715440944232986
      }
    ],
    "split_row_sums": {
      "Keck II": 0.8592032586312156,
      "Subaru": 0.8782434069634648,
      "Gemini North": 0.7434738159679593,
      "new 5 m r=2km compact": 0.9999999999999999,
      "new 5 m r=4km compact": 1.0,
      "new 5 m r=9km compact": 1.0
    },
    "gamma_model": "gamma fraction of every core-remote directed split enters a remote-star joint receiver; the residual remains pairwise edge-first",
    "core_core_handling": "nuisance",
    "gamma_kind": "by_remote",
    "gamma_core_to_remote": [
      [
        0.15000000000000002,
        0.15000000000000002,
        0.15000000000000002
      ],
      [
        0.15000000000000002,
        0.15000000000000002,
        0.15000000000000002
      ],
      [
        0.15000000000000002,
        0.15000000000000002,
        0.15000000000000002
      ]
    ],
    "gamma_remote_to_core": [
      [
        0.15000000000000002,
        0.15000000000000002,
        0.15000000000000002
      ],
      [
        0.15000000000000002,
        0.15000000000000002,
        0.15000000000000002
      ],
      [
        0.15000000000000002,
        0.15000000000000002,
        0.15000000000000002
      ]
    ]
  },
  "scalar_gamma_by_remote": [
    0.15000000000000002,
    0.15000000000000002,
    0.15000000000000002
  ],
  "independent_star_info": {
    "remote_star_receivers": [
      {
        "remote": "new 5 m r=2km compact",
        "gamma_core_to_remote": "0.15,0.23,0.15",
        "gamma_remote_to_core": "0.085,0.145,0.1",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=2km compact",
        "core_to_remote_available_fractions": "0.41000381,0.3861104,0.37949123",
        "core_to_remote_star_fractions": "0.061500572,0.088805392,0.056923684",
        "remote_to_core_available_fractions": "0.34394156,0.2031972,0.14157886",
        "remote_to_core_joint_available_fraction": 0.6887176241503702,
        "remote_to_core_joint_star_fraction": 0.07285651273808254,
        "total_star_receiver_input_fraction": 0.28008616086953164
      },
      {
        "remote": "new 5 m r=4km compact",
        "gamma_core_to_remote": "0.15,0.23,0.035",
        "gamma_remote_to_core": "0.41,0,0.26",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=4km compact",
        "core_to_remote_available_fractions": "0.31801878,0.43693692,0.02",
        "core_to_remote_star_fractions": "0.047702817,0.10049549,0.0007",
        "remote_to_core_available_fractions": "0.25512604,0.25512604,0.038861479",
        "remote_to_core_joint_available_fraction": 0.5491135590297128,
        "remote_to_core_joint_star_fraction": 0.1147056609628339,
        "total_star_receiver_input_fraction": 0.263603969121185
      },
      {
        "remote": "new 5 m r=9km compact",
        "gamma_core_to_remote": "0.275,0.07,0.28",
        "gamma_remote_to_core": "0.3,0.215,0.05",
        "subset": "Keck II|Subaru|Gemini North|new 5 m r=9km compact",
        "core_to_remote_available_fractions": "0.13118067,0.055196088,0.34398259",
        "core_to_remote_star_fractions": "0.036074683,0.0038637261,0.096315125",
        "remote_to_core_available_fractions": "0.20461897,0.14437412,0.26427487",
        "remote_to_core_joint_available_fraction": 0.613267952939735,
        "remote_to_core_joint_star_fraction": 0.10563986969965153,
        "total_star_receiver_input_fraction": 0.24189340403917528
      }
    ],
    "split_row_sums": {
      "Keck II": 0.8592032586312156,
      "Subaru": 0.8782434069634648,
      "Gemini North": 0.7434738159679593,
      "new 5 m r=2km compact": 0.9999999999999999,
      "new 5 m r=4km compact": 1.0,
      "new 5 m r=9km compact": 1.0
    },
    "gamma_model": "gamma fraction of every core-remote directed split enters a remote-star joint receiver; the residual remains pairwise edge-first",
    "core_core_handling": "nuisance",
    "gamma_kind": "independent_core_remote_and_remote_core",
    "gamma_core_to_remote": [
      [
        0.15000000000000002,
        0.15000000000000002,
        0.275
      ],
      [
        0.23000000000000004,
        0.23000000000000004,
        0.07000000000000002
      ],
      [
        0.15000000000000002,
        0.03500000000000002,
        0.28
      ]
    ],
    "gamma_remote_to_core": [
      [
        0.08500000000000002,
        0.14500000000000002,
        0.10000000000000002
      ],
      [
        0.41000000000000014,
        0.0,
        0.26000000000000006
      ],
      [
        0.30000000000000004,
        0.21500000000000002,
        0.05000000000000002
      ]
    ],
    "independent_gamma_best_score": -0.0018706374329820622,
    "independent_gamma_vector": [
      0.15000000000000002,
      0.15000000000000002,
      0.275,
      0.23000000000000004,
      0.23000000000000004,
      0.07000000000000002,
      0.15000000000000002,
      0.03500000000000002,
      0.28,
      0.08500000000000002,
      0.14500000000000002,
      0.10000000000000002,
      0.41000000000000014,
      0.0,
      0.26000000000000006,
      0.30000000000000004,
      0.21500000000000002,
      0.05000000000000002
    ]
  },
  "independent_gamma_vector": [
    0.15000000000000002,
    0.15000000000000002,
    0.275,
    0.23000000000000004,
    0.23000000000000004,
    0.07000000000000002,
    0.15000000000000002,
    0.03500000000000002,
    0.28,
    0.08500000000000002,
    0.14500000000000002,
    0.10000000000000002,
    0.41000000000000014,
    0.0,
    0.26000000000000006,
    0.30000000000000004,
    0.21500000000000002,
    0.05000000000000002
  ],
  "snr_gain_vs_edge": {
    "direct_physical": {
      "min": 1.0797700590555976,
      "mean": 1.1802823394500108,
      "median": 1.1695238607799066,
      "max": 1.2991456520665872
    },
    "direct_balanced_10loop": {
      "min": 0.9547422760195636,
      "mean": 1.1625644974626355,
      "median": 1.1268200933530963,
      "max": 1.4908328893349319
    },
    "direct_optimized_schedule": {
      "min": 1.2112182127920084,
      "mean": 1.2657068827332938,
      "median": 1.2348022012752549,
      "max": 1.4027337769266373
    },
    "current_near": {
      "min": 1.0001184552907072,
      "mean": 1.0884155555842994,
      "median": 1.042001578790749,
      "max": 1.3657509493176092
    },
    "remote_star_scalar": {
      "min": 1.0085372571995892,
      "mean": 1.147678881679093,
      "median": 1.088569141139815,
      "max": 1.3657509493176097
    },
    "remote_star_independent": {
      "min": 1.0000612560361464,
      "mean": 1.1572738520284234,
      "median": 1.1187292126419792,
      "max": 1.3657509493176099
    },
    "remote_star_full": {
      "min": 1.0281436420468621,
      "mean": 1.6045727411187418,
      "median": 1.328515037510658,
      "max": 2.9346306527009967
    }
  },
  "snr_ratio_vs_direct": {
    "direct_balanced_10loop": {
      "min": 1.0,
      "mean": 1.0,
      "median": 1.0,
      "max": 1.0
    },
    "direct_optimized_schedule": {
      "min": 0.8561968619002303,
      "mean": 1.1177503595207552,
      "median": 1.0990601225409042,
      "max": 1.4284485660312694
    },
    "current_near": {
      "min": 0.8767680858927878,
      "mean": 0.9444704766791603,
      "median": 0.9348367050583675,
      "max": 1.0768808168140593
    },
    "remote_star_scalar": {
      "min": 0.9160992885841672,
      "mean": 0.9917549699554374,
      "median": 0.986368325894285,
      "max": 1.0768808168140585
    },
    "remote_star_independent": {
      "min": 0.9160992885841674,
      "mean": 0.9993174813132581,
      "median": 0.9997546858958348,
      "max": 1.0768808168140596
    },
    "remote_star_full": {
      "min": 0.9992219925503774,
      "mean": 1.3339789144413863,
      "median": 1.1929450760799885,
      "max": 1.9684504371312537
    }
  },
  "gamma_scan_csv": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/18_balanced_10loop_independent_set_20260611/results/remote_star_gamma_scan.csv",
  "independent_gamma_search_csv": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/18_balanced_10loop_independent_set_20260611/results/remote_star_independent_gamma_search.csv",
  "loop_rows_csv": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/18_balanced_10loop_independent_set_20260611/results/remote_star_joint_loop_gains.csv",
  "figure_png": "/Users/xinyaoguo/Desktop/VLBI/closure_based telescope/18_balanced_10loop_independent_set_20260611/figures/remote_star_joint_loop_gains.png"
}
```

## Interpretation

If the remote-star column improves over current near, the effect is genuine evidence that the edge-first proxy was leaving Fisher information on the table for remote-involved loops.  If it does not, the limiting factor is more likely the station-side budget allocation or the physical split target, not just the lack of coherent addition.

