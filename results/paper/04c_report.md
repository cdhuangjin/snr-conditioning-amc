# Phase11 cross-dataset evaluation

04C uses the audited unequal-cell fixed split; signals retain original per-frame amplitudes and 128 samples; class labels use the audited dedup mapping.

M0/M6 each use five training seeds, true-SNR training and latest-tie oracle validation selection. M6 was selected on earlier 10a validation only. Dataset-trained Ridge uses training-only scaling/fitting and validation-only alpha selection; the 10a coefficient transfer is reported without refit or calibrated-transfer claims. Raw errors are preserved before classifier range clipping.

Overall, every SNR, low <= -8, mid -6 through -2, high >= 0, unequal-cell accuracy, class-balanced accuracy, concentration, dominance, recall and calibration are retained. Five-seed intervals quantify network-seed variability on one fixed partition, not independent-dataset uncertainty. Independent channel shift remains pending Phase12. Negative results must not be removed or trigger subset enlargement.
