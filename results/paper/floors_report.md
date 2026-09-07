# Phase 10 tested reliability floors

All 80 matched evaluations use M3/M6, five checkpoint seeds, oracle or frozen supervised SNR estimate and no floor/-10/-8/-6 dB. No floor must reproduce Phase8 E1/E2 before floor results are accepted. Raw estimator errors precede range clipping and flooring; oracle errors are zero by construction and reported separately.

Overall, low (≤-8), mid (-6 through -2), high (≥0), every registered SNR and absolute-error strata [0,2), [2,4), [4,8), [8,∞) are retained. Empty strata have count zero and null metrics. BA averages represented true classes; macro F1 retains all 11 classes. Error-to-dominant share divides wrong predictions sent to the group dominant predicted class by all wrong predictions; it is null if there are no errors.

Paired tables report each floor minus no floor for five matched training seeds on one fixed test set. CI is a two-sided Student-t 95% interval, not independent-dataset uncertainty. No apparently best test floor is selected or called universal/optimal/a phase transition. Figures read saved row/paired JSON. Phase9 cross-domain evaluation remains pending Phase11/12; this run consumes only its completed primary-dataset gate.
