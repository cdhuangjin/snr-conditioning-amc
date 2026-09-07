# Phase 3 Conditioning Mechanism Report

Best overall conditioner: M2 (descriptive ranking on these five seeds, not a population superiority claim).

| Model | Accuracy mean ± SD | Low-SNR accuracy | High-SNR accuracy | Parameters | Batch-1 latency ms | Fixed-batch latency ms |
|---|---:|---:|---:|---:|---:|---:|
| M0 | 0.618195 ± 0.000967 | 0.176857 | 0.907655 | 124043 | 0.937192 | 3.138438 |
| M1 | 0.664741 ± 0.000996 | 0.258039 | 0.916791 | 124363 | 1.215550 | 4.485556 |
| M2 | 0.668227 ± 0.001446 | 0.263104 | 0.919809 | 130443 | 1.219860 | 3.800260 |
| M3 | 0.667964 ± 0.002279 | 0.263234 | 0.919118 | 126763 | 1.274422 | 3.578272 |
| M4 | 0.652982 ± 0.001889 | 0.244078 | 0.914182 | 124263 | 1.046112 | 3.536880 |
| M5 | 0.667514 ± 0.001326 | 0.261481 | 0.918100 | 124555 | 1.277930 | 3.790764 |
| M6 | 0.667273 ± 0.001378 | 0.260883 | 0.918236 | 124299 | 1.253856 | 4.196900 |
| M7 | 0.658318 ± 0.001193 | 0.255468 | 0.908673 | 975452 | 1.090426 | 4.639046 |

Paired comparisons (treatment minus baseline; 95% Student t CI, n=5; secondary p-values are unadjusted):

- M1_minus_M0, overall_accuracy: Δ=0.046545, 95% CI [0.044635879483844156, 0.048455029607064916], dz=30.265275427498583, p=2.8562802314409484e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M1_minus_M0, macro_f1: Δ=0.023862, 95% CI [0.02126691686162022, 0.026457143908384776], dz=11.417082060628374, p=1.3981746153395074e-05, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M1_minus_M0, balanced_accuracy: Δ=0.046545, 95% CI [0.044635879483844024, 0.048455029607065006], dz=30.265275427496775, p=2.8562802314416313e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M2_minus_M0, overall_accuracy: Δ=0.050032, 95% CI [0.04809588119929293, 0.0519677551643434], dz=32.08921982574417, p=2.2605414250904484e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M2_minus_M0, macro_f1: Δ=0.027542, 95% CI [0.021713573638094546, 0.03337101838524359], dz=5.86719956685605, p=0.00019491755195385092, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M2_minus_M0, balanced_accuracy: Δ=0.050032, 95% CI [0.04809588119929292, 0.05196775516434346], dz=32.08921982574358, p=2.260541425090617e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M3_minus_M0, overall_accuracy: Δ=0.049768, 95% CI [0.04784588803058719, 0.05169047560577639], dz=32.1466781084598, p=2.2444333033063261e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M3_minus_M0, macro_f1: Δ=0.029267, 95% CI [0.025209920876154895, 0.03332389084088317], dz=8.957307828673695, p=3.6670625141294016e-05, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M3_minus_M0, balanced_accuracy: Δ=0.049768, 95% CI [0.0478458880305872, 0.05169047560577633], dz=32.14667810846034, p=2.2444333033061734e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M4_minus_M0, overall_accuracy: Δ=0.034786, 95% CI [0.03198162421787408, 0.03759110305485318], dz=15.399995832396508, p=4.2431767738148624e-06, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M4_minus_M0, macro_f1: Δ=0.014576, 95% CI [0.010917934744099838, 0.018234371939930532], dz=4.947403866841461, p=0.000379670699026868, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M4_minus_M0, balanced_accuracy: Δ=0.034786, 95% CI [0.031981624217874025, 0.03759110305485314], dz=15.399995832396451, p=4.243176773814924e-06, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M5_minus_M0, overall_accuracy: Δ=0.049318, 95% CI [0.04670965840233287, 0.05192670523403072], dz=23.47558409805368, p=7.883074244919681e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M5_minus_M0, macro_f1: Δ=0.026111, 95% CI [0.021844506969288958, 0.030378010641748974], dz=7.598616289631845, p=7.035747633275794e-05, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M5_minus_M0, balanced_accuracy: Δ=0.049318, 95% CI [0.04670965840233291, 0.05192670523403058], dz=23.47558409805452, p=7.883074244918561e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M6_minus_M0, overall_accuracy: Δ=0.049077, 95% CI [0.047419912560486444, 0.05073463289405894], dz=36.76779730602819, p=1.311935767918261e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M6_minus_M0, macro_f1: Δ=0.024039, 95% CI [0.021671911461722197, 0.026406311277965985], dz=12.60920086822223, p=9.41515933832193e-06, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M6_minus_M0, balanced_accuracy: Δ=0.049077, 95% CI [0.04741991256048638, 0.050734632894059], dz=36.76779730602693, p=1.3119357679184404e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M7_minus_M0, overall_accuracy: Δ=0.040123, 95% CI [0.03837008565074054, 0.041875368894713924], dz=28.425061541002226, p=3.670195981578955e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M7_minus_M0, macro_f1: Δ=0.018624, 95% CI [0.017094466512662342, 0.020153042877229643], dz=15.12105173111424, p=4.564097624697782e-06, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M7_minus_M0, balanced_accuracy: Δ=0.040123, 95% CI [0.038370085650740675, 0.04187536889471379], dz=28.425061541004407, p=3.670195981577831e-07, directions={'positive': 5, 'zero': 0, 'negative': 0}.
- M5_minus_M3, overall_accuracy: Δ=-0.000450, 95% CI [-0.0044092763226508585, 0.003509276322650869], dz=-0.14112397156902326, p=0.768112848271631, directions={'positive': 2, 'zero': 0, 'negative': 3}.
- M5_minus_M3, macro_f1: Δ=-0.003156, 95% CI [-0.008686865473146859, 0.0023755713671467266], dz=-0.708388828486003, p=0.18836604145759106, directions={'positive': 1, 'zero': 0, 'negative': 4}.
- M5_minus_M3, balanced_accuracy: Δ=-0.000450, 95% CI [-0.004409276322650806, 0.003509276322650772], dz=-0.1411239715690329, p=0.7681128482716157, directions={'positive': 2, 'zero': 0, 'negative': 3}.
- M6_minus_M3, overall_accuracy: Δ=-0.000691, 95% CI [-0.004111461951484146, 0.002729643769665951], dz=-0.2508006685414285, p=0.60485494746189, directions={'positive': 2, 'zero': 0, 'negative': 3}.
- M6_minus_M3, macro_f1: Δ=-0.005228, 95% CI [-0.008964339232475266, -0.0014912497448746178], dz=-1.7372103511849744, p=0.017772979227344474, directions={'positive': 0, 'zero': 0, 'negative': 5}.
- M6_minus_M3, balanced_accuracy: Δ=-0.000691, 95% CI [-0.0041114619514841105, 0.0027296437696659594], dz=-0.2508006685414215, p=0.6048549474618998, directions={'positive': 2, 'zero': 0, 'negative': 3}.
- M7_minus_M3, overall_accuracy: Δ=-0.009645, 95% CI [-0.011986119857895635, -0.007304789233013481], dz=-5.1166707139824235, p=0.0003330126850126063, directions={'positive': 0, 'zero': 0, 'negative': 5}.
- M7_minus_M3, macro_f1: Δ=-0.010643, 95% CI [-0.014011710122404443, -0.007274592204741635], dz=-3.923107117541345, p=0.000931050366707, directions={'positive': 0, 'zero': 0, 'negative': 5}.
- M7_minus_M3, balanced_accuracy: Δ=-0.009645, 95% CI [-0.01198611985789553, -0.00730478923301354], dz=-5.116670713982589, p=0.00033301268501256435, directions={'positive': 0, 'zero': 0, 'negative': 5}.

Low/high-SNR diagnostic (paired deltas vs M0; concentration/entropy diagnose prediction collapse, not correctness):

- M1 − M0, low, accuracy: Δ=0.081182, 95% CI [0.074564, 0.087800].
- M1 − M0, low, balanced_accuracy: Δ=0.081182, 95% CI [0.074564, 0.087800].
- M1 − M0, low, prediction_concentration: Δ=-0.527156, 95% CI [-0.569734, -0.484578].
- M1 − M0, low, normalized_prediction_entropy: Δ=0.415875, 95% CI [0.348071, 0.483679].
- M1 − M0, high, accuracy: Δ=0.009136, 95% CI [0.007170, 0.011102].
- M1 − M0, high, balanced_accuracy: Δ=0.009136, 95% CI [0.007170, 0.011102].
- M1 − M0, high, prediction_concentration: Δ=-0.011718, 95% CI [-0.029649, 0.006212].
- M1 − M0, high, normalized_prediction_entropy: Δ=0.004726, 95% CI [-0.002955, 0.012407].
- M2 − M0, low, accuracy: Δ=0.086247, 95% CI [0.080816, 0.091678].
- M2 − M0, low, balanced_accuracy: Δ=0.086247, 95% CI [0.080816, 0.091678].
- M2 − M0, low, prediction_concentration: Δ=-0.549961, 95% CI [-0.597068, -0.502854].
- M2 − M0, low, normalized_prediction_entropy: Δ=0.431478, 95% CI [0.371632, 0.491323].
- M2 − M0, high, accuracy: Δ=0.012155, 95% CI [0.008995, 0.015314].
- M2 − M0, high, balanced_accuracy: Δ=0.012155, 95% CI [0.008995, 0.015314].
- M2 − M0, high, prediction_concentration: Δ=-0.026545, 95% CI [-0.052052, -0.001039].
- M2 − M0, high, normalized_prediction_entropy: Δ=0.008467, 95% CI [0.000141, 0.016794].
- M3 − M0, low, accuracy: Δ=0.086377, 95% CI [0.079300, 0.093453].
- M3 − M0, low, balanced_accuracy: Δ=0.086377, 95% CI [0.079300, 0.093453].
- M3 − M0, low, prediction_concentration: Δ=-0.518182, 95% CI [-0.563903, -0.472460].
- M3 − M0, low, normalized_prediction_entropy: Δ=0.400628, 95% CI [0.360599, 0.440657].
- M3 − M0, high, accuracy: Δ=0.011464, 95% CI [0.004712, 0.018215].
- M3 − M0, high, balanced_accuracy: Δ=0.011464, 95% CI [0.004712, 0.018215].
- M3 − M0, high, prediction_concentration: Δ=-0.016891, 95% CI [-0.035286, 0.001505].
- M3 − M0, high, normalized_prediction_entropy: Δ=0.006645, 95% CI [-0.001982, 0.015273].
- M4 − M0, low, accuracy: Δ=0.067221, 95% CI [0.059080, 0.075361].
- M4 − M0, low, balanced_accuracy: Δ=0.067221, 95% CI [0.059080, 0.075361].
- M4 − M0, low, prediction_concentration: Δ=-0.445416, 95% CI [-0.530790, -0.360041].
- M4 − M0, low, normalized_prediction_entropy: Δ=0.374977, 95% CI [0.307593, 0.442362].
- M4 − M0, high, accuracy: Δ=0.006527, 95% CI [0.003869, 0.009186].
- M4 − M0, high, balanced_accuracy: Δ=0.006527, 95% CI [0.003869, 0.009186].
- M4 − M0, high, prediction_concentration: Δ=-0.013082, 95% CI [-0.040017, 0.013854].
- M4 − M0, high, normalized_prediction_entropy: Δ=0.003905, 95% CI [-0.005209, 0.013020].
- M5 − M0, low, accuracy: Δ=0.084623, 95% CI [0.074824, 0.094423].
- M5 − M0, low, balanced_accuracy: Δ=0.084623, 95% CI [0.074824, 0.094423].
- M5 − M0, low, prediction_concentration: Δ=-0.525948, 95% CI [-0.608752, -0.443144].
- M5 − M0, low, normalized_prediction_entropy: Δ=0.421976, 95% CI [0.346095, 0.497856].
- M5 − M0, high, accuracy: Δ=0.010445, 95% CI [0.005737, 0.015154].
- M5 − M0, high, balanced_accuracy: Δ=0.010445, 95% CI [0.005737, 0.015154].
- M5 − M0, high, prediction_concentration: Δ=-0.004100, 95% CI [-0.014524, 0.006324].
- M5 − M0, high, normalized_prediction_entropy: Δ=0.001321, 95% CI [-0.004355, 0.006996].
- M6 − M0, low, accuracy: Δ=0.084026, 95% CI [0.076356, 0.091696].
- M6 − M0, low, balanced_accuracy: Δ=0.084026, 95% CI [0.076356, 0.091696].
- M6 − M0, low, prediction_concentration: Δ=-0.513455, 95% CI [-0.577263, -0.449646].
- M6 − M0, low, normalized_prediction_entropy: Δ=0.401370, 95% CI [0.335918, 0.466822].
- M6 − M0, high, accuracy: Δ=0.010582, 95% CI [0.005674, 0.015490].
- M6 − M0, high, balanced_accuracy: Δ=0.010582, 95% CI [0.005674, 0.015490].
- M6 − M0, high, prediction_concentration: Δ=-0.007682, 95% CI [-0.025117, 0.009753].
- M6 − M0, high, normalized_prediction_entropy: Δ=0.002481, 95% CI [-0.005217, 0.010179].
- M7 − M0, low, accuracy: Δ=0.078610, 95% CI [0.075301, 0.081919].
- M7 − M0, low, balanced_accuracy: Δ=0.078610, 95% CI [0.075301, 0.081919].
- M7 − M0, low, prediction_concentration: Δ=-0.569883, 95% CI [-0.635624, -0.504142].
- M7 − M0, low, normalized_prediction_entropy: Δ=0.463250, 95% CI [0.408055, 0.518445].
- M7 − M0, high, accuracy: Δ=0.001018, 95% CI [-0.003372, 0.005408].
- M7 − M0, high, balanced_accuracy: Δ=0.001018, 95% CI [-0.003372, 0.005408].
- M7 − M0, high, prediction_concentration: Δ=-0.032109, 95% CI [-0.053092, -0.011126].
- M7 − M0, high, normalized_prediction_entropy: Δ=0.011113, 95% CI [0.004010, 0.018216].

Capacity and efficiency: M7 is a parameter-rich per-bin control; M7−M3 above measures observed benefit, not guaranteed superiority. Parameter and measured latency costs are reported separately; latency depends on the recorded hardware/batch protocol and is not a universal speed ranking.

Claim allowed: report observed matched-seed changes, confidence intervals, low/high-SNR trade-offs and parameter/runtime costs within this fixed dataset/split protocol.
Claim forbidden: absence of a significant difference does not establish equivalence; no population-best, causal mechanism, unseen-SNR interpolation, or universal efficiency claim follows from these five seeds.


All values in this report are derived from registered Phase 3 artifacts. The preregistered low-SNR region is -20 through -8 dB inclusive.

For embedding concatenation followed by a linear map:

`linear([f(x); e(z)]) = W_f f(x) + W_e e(z) + b`

The real M3 head is Linear → LeakyReLU → Linear: `logits = W2 LeakyReLU(W_f f(x) + W_e e(z) + b1) + b2`. The direct contribution is an additive hidden preactivation shift, not a pure final-logit bias. Although W_f is shared, condition-dependent activation masks can change the local feature-to-logit Jacobian. M4 alone adds a final-logit bias. FiLM and gating introduce explicit feature × condition interaction; M7 permits independent per-bin classifier parameters but does not guarantee higher accuracy.

Geometry is descriptive: PCA is computed independently per seed and only seed 2022 is plotted; unaligned coordinates are never averaged. M7 distances include all head weights and biases, whereas classifier_weight_norm includes weights only. Parameter drift is not a measured decision-boundary distance. Gate sparsity counts |gate| ≤ 1e-6; gate entropy normalizes positive gates across features and is not predictive entropy. M4 class_shift_from_snr_mean subtracts each class's across-SNR mean, not the softmax-invariant common class offset.

## Interpolation diagnostic

不具备严格 unseen/intermediate experimental support。All discrete 2 dB SNR bins are observed during training, so no holdout interpolation score is fabricated. Embedding geometry is descriptive and does not establish continuous generalization.
