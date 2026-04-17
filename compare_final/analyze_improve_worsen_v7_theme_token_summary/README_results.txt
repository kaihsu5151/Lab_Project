This fixed clean script keeps the clean canonical feature/theme classifier and restores the richer summary analyses.
Metric = R2_ens
Delta definition = after-before
Improved / worsened / stable = 20 / 16 / 21
Clean numeric features = 40
Raw token features = 1952
Theme features = 17

Fairness / robustness additions:
- Each numeric feature now reports non-missing coverage so sparse features do not look artificially strong.
- Raw tokens are tested fairly at the drug level: each drug contributes at most one vote per normalized token, no matter how many times that token appears across rows.
- Theme outputs are kept as a higher-level summary layer on top of the raw-token layer.
- Each theme is counted at most once per drug after token-to-theme mapping, reducing duplicate counting of the same concept.
- Leave-one-drug-out stability is exported to check whether a result is driven by only one drug.
- Threshold robustness is exported across the threshold grid to see whether the same direction survives different improve/worsen cutoffs.

Warning fix:
- inf / -inf and |value| > 1e+150 are coerced to NaN before summary stats and modeling.
- This removes the pandas/numpy overflow and mean-of-empty-slice spam that came from pathological numeric values.

Top numeric associations by |Spearman|:
- kd_best_pchembl: spearman=0.445, pearson=0.444, n=39
- Z_SCORE_abs_mean: spearman=-0.370, pearson=-0.294, n=57
- ic50_best_pchembl: spearman=0.334, pearson=0.269, n=53
- best_standard_value: spearman=-0.277, pearson=-0.059, n=54
- best_pchembl: spearman=0.276, pearson=0.228, n=54
- black_box_warning: spearman=-0.264, pearson=-0.248, n=57
- TCGA_entropy: spearman=0.261, pearson=0.274, n=57
- ec50_best_pchembl: spearman=0.237, pearson=0.308, n=29
- first_approval: spearman=0.221, pearson=0.297, n=19
- Z_SCORE_std: spearman=-0.204, pearson=-0.219, n=57

Top numeric improve-vs-worsen differences by |Cohen d|:
- kd_best_pchembl: cohen_d=1.106, p=0.0127
- ac50_best_pchembl: cohen_d=0.799, p=0.476
- black_box_warning: cohen_d=-0.731, p=0.0408
- Z_SCORE_abs_mean: cohen_d=-0.688, p=0.0248
- ic50_best_pchembl: cohen_d=0.592, p=0.0457
- Z_SCORE_mean: cohen_d=-0.575, p=0.157
- aromatic_rings: cohen_d=0.569, p=0.0999
- act_pchembl_std: cohen_d=0.563, p=0.131
- TCGA_entropy: cohen_d=0.544, p=0.156
- ec50_best_pchembl: cohen_d=0.536, p=0.375

Top raw-token fairness consensus results:
- GLYCOGEN SYNTHASE KINASE-3 BETA: consensus=4.512, support=3, lodo_consistency=1.000, threshold_consistency=1.000
- CYCLIN-G-ASSOCIATED KINASE: consensus=4.498, support=4, lodo_consistency=1.000, threshold_consistency=1.000
- DUAL SPECIFICITY MITOGEN-ACTIVATED PROTEIN KINASE KINASE 5: consensus=4.498, support=4, lodo_consistency=1.000, threshold_consistency=1.000
- EPHRIN TYPE-B RECEPTOR 6: consensus=4.498, support=4, lodo_consistency=1.000, threshold_consistency=1.000
- RECEPTOR TYROSINE-PROTEIN KINASE ERBB-2: consensus=4.498, support=4, lodo_consistency=1.000, threshold_consistency=1.000
- HEPATOCYTE GROWTH FACTOR RECEPTOR: consensus=4.421, support=7, lodo_consistency=1.000, threshold_consistency=1.000
- TYROSINE-PROTEIN KINASE LCK: consensus=4.421, support=7, lodo_consistency=1.000, threshold_consistency=1.000
- VOLTAGE-GATED INWARDLY RECTIFYING POTASSIUM CHANNEL KCNH2: consensus=4.421, support=7, lodo_consistency=1.000, threshold_consistency=1.000
- BILE SALT EXPORT PUMP: consensus=4.366, support=5, lodo_consistency=1.000, threshold_consistency=1.000
- MITOGEN-ACTIVATED PROTEIN KINASE 10: consensus=4.366, support=5, lodo_consistency=1.000, threshold_consistency=1.000

Top theme consensus results:
- ION_CHANNEL_TRANSPORTER: consensus=4.724, support=13, lodo_consistency=1.000, threshold_consistency=1.000
- MAP2K_MEK_ERK_RAF_CASCADE: consensus=4.654, support=11, lodo_consistency=1.000, threshold_consistency=1.000
- MET_HGF_RTK: consensus=4.421, support=7, lodo_consistency=1.000, threshold_consistency=1.000
- SRC_LCK_NONRECEPTOR_TK: consensus=4.421, support=7, lodo_consistency=1.000, threshold_consistency=1.000
- VEGFR_FGFR_PDGFR_KIT_FLT_RTK: consensus=4.386, support=8, lodo_consistency=1.000, threshold_consistency=1.000
- EPH_RECEPTOR_SIGNALING: consensus=4.366, support=5, lodo_consistency=1.000, threshold_consistency=1.000
- JNK_P38_STRESS_MAPK: consensus=4.366, support=5, lodo_consistency=1.000, threshold_consistency=1.000
- CHROMATIN_EPIGENETIC: consensus=4.333, support=8, lodo_consistency=1.000, threshold_consistency=1.000
- RTK_SIGNALING_BROAD: consensus=4.310, support=6, lodo_consistency=1.000, threshold_consistency=1.000
- PROTEOSTASIS: consensus=4.209, support=3, lodo_consistency=1.000, threshold_consistency=1.000

Theme-internal token effect summaries:
- JNK_P38_STRESS_MAPK: theme_log2_or=1.495, token_mean=1.495, token_weighted_mean=1.495, sign_consistency=1.000
- ERBB_EGFR_FAMILY_RTK: theme_log2_or=-0.021, token_mean=1.963, token_weighted_mean=1.417, sign_consistency=0.833
- SER_THR_KINASE_SIGNALING: theme_log2_or=0.073, token_mean=1.307, token_weighted_mean=1.364, sign_consistency=0.833
- MET_HGF_RTK: theme_log2_or=1.041, token_mean=1.041, token_weighted_mean=1.041, sign_consistency=1.000
- MAP2K_MEK_ERK_RAF_CASCADE: theme_log2_or=1.391, token_mean=1.039, token_weighted_mean=1.029, sign_consistency=0.750
- PI3K_AKT_MTOR: theme_log2_or=-0.351, token_mean=1.152, token_weighted_mean=0.997, sign_consistency=1.000
- APOPTOSIS_BCL2: theme_log2_or=0.482, token_mean=0.913, token_weighted_mean=0.913, sign_consistency=0.500
- RTK_SIGNALING_BROAD: theme_log2_or=0.662, token_mean=0.432, token_weighted_mean=0.686, sign_consistency=0.500
- ION_CHANNEL_TRANSPORTER: theme_log2_or=1.198, token_mean=0.499, token_weighted_mean=0.671, sign_consistency=0.667
- PARP_DNA_REPAIR_CHECKPOINT: theme_log2_or=0.453, token_mean=0.664, token_weighted_mean=0.664, sign_consistency=0.600

Top theme enrichments: improve vs worsen:
- MAP2K_MEK_ERK_RAF_CASCADE: odds_ratio=2.89, p=0.156
- ION_CHANNEL_TRANSPORTER: odds_ratio=2.45, p=0.187
- EPH_RECEPTOR_SIGNALING: odds_ratio=3.75, p=0.247
- JNK_P38_STRESS_MAPK: odds_ratio=3.75, p=0.247
- MET_HGF_RTK: odds_ratio=2.33, p=0.306
- SRC_LCK_NONRECEPTOR_TK: odds_ratio=2.33, p=0.306
- RTK_SIGNALING_BROAD: odds_ratio=1.75, p=0.446
- PARP_DNA_REPAIR_CHECKPOINT: odds_ratio=1.44, p=0.486
- VEGFR_FGFR_PDGFR_KIT_FLT_RTK: odds_ratio=1.44, p=0.486
- APOPTOSIS_BCL2: odds_ratio=1.67, p=0.585