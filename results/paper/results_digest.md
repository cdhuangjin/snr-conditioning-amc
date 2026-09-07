# Evidence-derived result tables

Accuracy is percent; SD is across five training seeds. These are marginal summaries, not paired-effect intervals.

## 04c

| Condition | Accuracy (%) | SD (pp) |
|---|---:|---:|
| model=M0, condition=agnostic | 73.424 | 0.235 |
| model=M6, condition=oracle | 73.436 | 0.311 |
| model=M6, condition=estimated | 70.561 | 0.420 |

## synthetic

| Condition | Accuracy (%) | SD (pp) |
|---|---:|---:|
| model=M0, channel=A, condition=oracle | 67.400 | 0.301 |
| model=M0, channel=A, condition=K0_frame_estimator | 67.400 | 0.301 |
| model=M0, channel=A, condition=source10a_frozen | 67.400 | 0.301 |
| model=M0, channel=A, condition=K8_pilot | 67.400 | 0.301 |
| model=M0, channel=A, condition=K16_pilot | 67.400 | 0.301 |
| model=M0, channel=A, condition=K32_pilot | 67.400 | 0.301 |
| model=M0, channel=A, condition=K64_pilot | 67.400 | 0.301 |
| model=M0, channel=B, condition=oracle | 47.228 | 1.894 |
| model=M0, channel=B, condition=K0_frame_estimator | 47.228 | 1.894 |
| model=M0, channel=B, condition=source10a_frozen | 47.228 | 1.894 |
| model=M0, channel=B, condition=K8_pilot | 47.228 | 1.894 |
| model=M0, channel=B, condition=K16_pilot | 47.228 | 1.894 |
| model=M0, channel=B, condition=K32_pilot | 47.228 | 1.894 |
| model=M0, channel=B, condition=K64_pilot | 47.228 | 1.894 |
| model=M6, channel=A, condition=oracle | 68.082 | 0.260 |
| model=M6, channel=A, condition=K0_frame_estimator | 64.470 | 0.229 |
| model=M6, channel=A, condition=source10a_frozen | 47.351 | 0.730 |
| model=M6, channel=A, condition=K8_pilot | 65.362 | 0.233 |
| model=M6, channel=A, condition=K16_pilot | 66.504 | 0.304 |
| model=M6, channel=A, condition=K32_pilot | 67.166 | 0.258 |
| model=M6, channel=A, condition=K64_pilot | 67.600 | 0.265 |
| model=M6, channel=B, condition=oracle | 48.066 | 2.671 |
| model=M6, channel=B, condition=K0_frame_estimator | 44.122 | 2.461 |
| model=M6, channel=B, condition=source10a_frozen | 32.713 | 0.716 |
| model=M6, channel=B, condition=K8_pilot | 46.867 | 2.664 |
| model=M6, channel=B, condition=K16_pilot | 47.551 | 2.853 |
| model=M6, channel=B, condition=K32_pilot | 45.521 | 2.593 |
| model=M6, channel=B, condition=K64_pilot | 40.207 | 1.433 |
| model=CLDNN, channel=A, condition=oracle | 63.695 | 3.655 |
| model=CLDNN, channel=A, condition=K0_frame_estimator | 63.695 | 3.655 |
| model=CLDNN, channel=A, condition=source10a_frozen | 63.695 | 3.655 |
| model=CLDNN, channel=A, condition=K8_pilot | 63.695 | 3.655 |
| model=CLDNN, channel=A, condition=K16_pilot | 63.695 | 3.655 |
| model=CLDNN, channel=A, condition=K32_pilot | 63.695 | 3.655 |
| model=CLDNN, channel=A, condition=K64_pilot | 63.695 | 3.655 |
| model=CLDNN, channel=B, condition=oracle | 36.618 | 0.459 |
| model=CLDNN, channel=B, condition=K0_frame_estimator | 36.618 | 0.459 |
| model=CLDNN, channel=B, condition=source10a_frozen | 36.618 | 0.459 |
| model=CLDNN, channel=B, condition=K8_pilot | 36.618 | 0.459 |
| model=CLDNN, channel=B, condition=K16_pilot | 36.618 | 0.459 |
| model=CLDNN, channel=B, condition=K32_pilot | 36.618 | 0.459 |
| model=CLDNN, channel=B, condition=K64_pilot | 36.618 | 0.459 |

## floors

| Condition | Accuracy (%) | SD (pp) |
|---|---:|---:|
| model=M3, condition=oracle, floor_db=None | 66.796 | 0.228 |
| model=M3, condition=oracle, floor_db=-10 | 65.395 | 0.272 |
| model=M3, condition=oracle, floor_db=-8 | 64.199 | 0.294 |
| model=M3, condition=oracle, floor_db=-6 | 62.530 | 0.242 |
| model=M3, condition=estimate, floor_db=None | 56.160 | 0.204 |
| model=M3, condition=estimate, floor_db=-10 | 56.756 | 0.299 |
| model=M3, condition=estimate, floor_db=-8 | 57.279 | 0.151 |
| model=M3, condition=estimate, floor_db=-6 | 57.820 | 0.261 |
| model=M6, condition=oracle, floor_db=None | 66.727 | 0.138 |
| model=M6, condition=oracle, floor_db=-10 | 65.404 | 0.184 |
| model=M6, condition=oracle, floor_db=-8 | 64.218 | 0.211 |
| model=M6, condition=oracle, floor_db=-6 | 62.549 | 0.206 |
| model=M6, condition=estimate, floor_db=None | 53.848 | 0.326 |
| model=M6, condition=estimate, floor_db=-10 | 54.611 | 0.315 |
| model=M6, condition=estimate, floor_db=-8 | 55.341 | 0.360 |
| model=M6, condition=estimate, floor_db=-6 | 55.879 | 0.382 |

## noise

| Condition | Accuracy (%) | SD (pp) |
|---|---:|---:|
| arm=clean, deployment=true | 63.388 | 1.767 |
| arm=clean, deployment=estimated | 55.235 | 0.316 |
| arm=generic, deployment=estimated | 59.844 | 0.404 |
| arm=empirical_snr, deployment=estimated | 61.091 | 0.208 |
| arm=historical_true, deployment=true | 66.727 | 0.138 |
| arm=historical_true, deployment=estimated | 53.848 | 0.326 |

## sidechannel

| Condition | Accuracy (%) | SD (pp) |
|---|---:|---:|
| model=M3, condition=E1 | 66.796 | 0.228 |
| model=M3, condition=E2 | 56.160 | 0.204 |
| model=M3, condition=E3_global | 43.727 | 0.515 |
| model=M3, condition=E3_within_snr | 56.060 | 0.374 |
| model=M3, condition=E3_across_class | 55.642 | 0.851 |
| model=M6, condition=E1 | 66.727 | 0.138 |
| model=M6, condition=E2 | 53.848 | 0.326 |
| model=M6, condition=E3_global | 38.920 | 0.894 |
| model=M6, condition=E3_within_snr | 54.196 | 0.335 |
| model=M6, condition=E3_across_class | 53.897 | 0.335 |
