function results = analyzeFusionStability(varargin)
% ANALYZEFUSIONSTABILITY Multi-Modal Evidence Fusion Stability Analysis
% Performs repeated stratified 5-fold cross-validation (10 repeats = 50 folds)
% on the 330-image training partition to evaluate:
% 1. Cross-validation performance stability (Mean +/- Std of Sens, Spec, Prec, F1, BalAcc)
% 2. Coefficient stability across all 12 features (% positive/negative, mean, std, range)
% 3. Group-level feature reliability (CNN vs Lesions vs Vessels vs Quality)
% 4. Final model fitting on 330 training samples and single evaluation on untouched 83 validation images.
%
% STRICT SAFEGUARDS:
% - 103-image IDRiD test partition is strictly sequestered and never accessed.
% - 83-image validation split is completely untouched during cross-validation and fitting.
% - Uses fixed deterministic random seeds for all repeats.
% - No unverified clinical claims.
%
% OUTPUT FILES:
% - results/metrics/fusion_stability.mat
% - results/metrics/fusion_stability.csv

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: MULTI-MODAL EVIDENCE FUSION STABILITY ANALYSIS               \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Load Precomputed Development Audit Data
    % -------------------------------------------------------------------------
    audit_file = fullfile(cfg.paths.metrics, 'evidence_channel_audit.mat');
    if exist(audit_file, 'file') == 2
        fprintf('[STEP 1/5] Loading precomputed evidence audit from:\n  %s\n', audit_file);
        loaded = load(audit_file);
        audit_table = loaded.audit_table;
    else
        fprintf('[STEP 1/5] Running evaluateEvidenceChannels to generate audit data...\n');
        audit_res = evaluateEvidenceChannels();
        audit_table = audit_res.audit_table;
    end

    % Extract Training and Validation Subsets
    train_mask = strcmp(audit_table.Split, 'Train') & strcmp(audit_table.Status, 'SUCCESS');
    val_mask   = strcmp(audit_table.Split, 'Validation') & strcmp(audit_table.Status, 'SUCCESS');

    train_tab = audit_table(train_mask, :);
    val_tab   = audit_table(val_mask, :);

    n_train = height(train_tab);
    n_val   = height(val_tab);

    % Strict Assertions
    assert(n_train == 330, 'Training subset must contain exactly 330 images. Found: %d', n_train);
    assert(n_val == 83, 'Validation subset must contain exactly 83 images. Found: %d', n_val);

    y_train = train_tab.True_Referable;
    y_val   = val_tab.True_Referable;

    fprintf('  [PASSED] Partition verified: %d Training images, %d Validation images.\n', n_train, n_val);
    fprintf('  [PASSED] Class Prevalence: Training Ref = %d/330 (%.1f%%) | Val Ref = %d/83 (%.1f%%)\n\n', ...
        sum(y_train == 1), (sum(y_train == 1)/n_train)*100, ...
        sum(y_val == 1), (sum(y_val == 1)/n_val)*100);

    % Define Feature Set
    feature_names = {
        'MA_Count', 'MA_Area', 'HE_Count', 'HE_Area_Fraction', ...
        'EX_Count', 'EX_Area_Fraction', 'DME_Risk', 'NV_Score', ...
        'Vessel_Density', 'Vessel_Tortuosity', 'Quality_Score', 'CNN_P_Referable'
    };
    num_feats = numel(feature_names);

    feature_groups = {
        'Lesion', 'Lesion', 'Lesion', 'Lesion', ...
        'Lesion', 'Lesion', 'Lesion', 'Lesion', ...
        'Vessel', 'Vessel', 'Quality', 'CNN'
    };

    X_train_raw = zeros(n_train, num_feats);
    X_val_raw   = zeros(n_val, num_feats);

    for f = 1:num_feats
        X_train_raw(:, f) = train_tab.(feature_names{f});
        X_val_raw(:, f)   = val_tab.(feature_names{f});
    end

    % -------------------------------------------------------------------------
    % 2. Repeated Stratified 5-Fold Cross-Validation (10 Repeats = 50 Folds)
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   2. REPEATED STRATIFIED 5-FOLD CROSS-VALIDATION (10 REPEATS = 50 FOLDS)\n');
    fprintf('   (Executed STRICTLY within the 330-image training partition)          \n');
    fprintf('========================================================================\n');

    num_repeats = 10;
    num_folds   = 5;
    total_runs  = num_repeats * num_folds;

    % Storage for Fold Metrics
    cv_sens     = zeros(total_runs, 1);
    cv_spec     = zeros(total_runs, 1);
    cv_prec     = zeros(total_runs, 1);
    cv_f1       = zeros(total_runs, 1);
    cv_bal_acc  = zeros(total_runs, 1);
    cv_acc      = zeros(total_runs, 1);
    cv_thresh   = zeros(total_runs, 1);

    % Storage for Coefficients (Intercept + 12 Features)
    cv_coefficients = zeros(total_runs, num_feats + 1);

    run_idx = 0;
    lambda_reg = 0.05; % Regularization parameter

    for rep = 1:num_repeats
        rep_seed = cfg.random_seed + (rep * 100);
        fold_assignments = get_stratified_folds(y_train, num_folds, rep_seed);

        for k = 1:num_folds
            run_idx = run_idx + 1;
            
            val_fold_mask = (fold_assignments == k);
            tr_fold_mask  = ~val_fold_mask;

            X_tr_k = X_train_raw(tr_fold_mask, :);
            y_tr_k = y_train(tr_fold_mask);
            X_val_k = X_train_raw(val_fold_mask, :);
            y_val_k = y_train(val_fold_mask);

            % Standardize using TRAINING FOLD ONLY
            mu_k = mean(X_tr_k, 1);
            sig_k = std(X_tr_k, 0, 1);
            sig_k(sig_k < 1e-6) = 1.0;

            Z_tr_k  = (X_tr_k - mu_k) ./ sig_k;
            Z_val_k = (X_val_k - mu_k) ./ sig_k;

            % Fit Ridge Logistic Model on Training Fold
            n_tr_k = numel(y_tr_k);
            X_aug_tr = [ones(n_tr_k, 1), Z_tr_k];
            R = eye(num_feats + 1); R(1, 1) = 0; % Intercept unregularized

            w_k = (X_aug_tr' * X_aug_tr + lambda_reg * R) \ (X_aug_tr' * (2 * y_tr_k - 1));
            cv_coefficients(run_idx, :) = w_k';

            % Select Decision Threshold on Training Fold Only
            logits_tr_k = X_aug_tr * w_k;
            p_tr_k = 1.0 ./ (1.0 + exp(-logits_tr_k));

            cand_thresh = unique([prctile(p_tr_k, 5:95)'; 0.50]);
            best_tau_k = 0.50;
            best_bal_k = -1;

            for t = 1:numel(cand_thresh)
                th = cand_thresh(t);
                pred_tr_k = p_tr_k >= th;
                tp_tr = sum(pred_tr_k == 1 & y_tr_k == 1);
                fp_tr = sum(pred_tr_k == 1 & y_tr_k == 0);
                fn_tr = sum(pred_tr_k == 0 & y_tr_k == 1);
                tn_tr = sum(pred_tr_k == 0 & y_tr_k == 0);
                bal_tr = (tp_tr/max(1, tp_tr + fn_tr) + tn_tr/max(1, tn_tr + fp_tr)) / 2;

                if bal_tr > best_bal_k
                    best_bal_k = bal_tr;
                    best_tau_k = th;
                end
            end
            cv_thresh(run_idx) = best_tau_k;

            % Evaluate on Held-Out Validation Fold
            n_val_k = numel(y_val_k);
            X_aug_val = [ones(n_val_k, 1), Z_val_k];
            logits_val_k = X_aug_val * w_k;
            p_val_k = 1.0 ./ (1.0 + exp(-logits_val_k));
            pred_val_k = p_val_k >= best_tau_k;

            tp_v = sum(pred_val_k == 1 & y_val_k == 1);
            fp_v = sum(pred_val_k == 1 & y_val_k == 0);
            fn_v = sum(pred_val_k == 0 & y_val_k == 1);
            tn_v = sum(pred_val_k == 0 & y_val_k == 0);

            cv_sens(run_idx)    = tp_v / max(1, tp_v + fn_v);
            cv_spec(run_idx)    = tn_v / max(1, tn_v + fp_v);
            cv_prec(run_idx)    = tp_v / max(1, tp_v + fp_v);
            cv_f1(run_idx)      = (2 * tp_v) / max(1, 2 * tp_v + fp_v + fn_v);
            cv_bal_acc(run_idx) = (cv_sens(run_idx) + cv_spec(run_idx)) / 2;
            cv_acc(run_idx)     = (tp_v + tn_v) / n_val_k;
        end
    end

    % 10-Repeat 5-Fold Cross-Validation Summary
    fprintf('  CROSS-VALIDATION PERFORMANCE ACROSS 50 FOLDS (Mean ± Std):\n');
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  Referable Sensitivity:     %5.2f%% ± %4.2f%%  [Min: %5.2f%%, Max: %5.2f%%]\n', ...
        mean(cv_sens)*100, std(cv_sens)*100, min(cv_sens)*100, max(cv_sens)*100);
    fprintf('  Referable Specificity:     %5.2f%% ± %4.2f%%  [Min: %5.2f%%, Max: %5.2f%%]\n', ...
        mean(cv_spec)*100, std(cv_spec)*100, min(cv_spec)*100, max(cv_spec)*100);
    fprintf('  Referable Precision (PPV): %5.2f%% ± %4.2f%%  [Min: %5.2f%%, Max: %5.2f%%]\n', ...
        mean(cv_prec)*100, std(cv_prec)*100, min(cv_prec)*100, max(cv_prec)*100);
    fprintf('  Referable F1 Score:        %6.4f ± %6.4f   [Min: %6.4f, Max: %6.4f]\n', ...
        mean(cv_f1), std(cv_f1), min(cv_f1), max(cv_f1));
    fprintf('  Balanced Accuracy:         %5.2f%% ± %4.2f%%  [Min: %5.2f%%, Max: %5.2f%%]\n', ...
        mean(cv_bal_acc)*100, std(cv_bal_acc)*100, min(cv_bal_acc)*100, max(cv_bal_acc)*100);
    fprintf('  Overall Accuracy:          %5.2f%% ± %4.2f%%  [Min: %5.2f%%, Max: %5.2f%%]\n', ...
        mean(cv_acc)*100, std(cv_acc)*100, min(cv_acc)*100, max(cv_acc)*100);
    fprintf('  Decision Threshold (tau):  %6.4f ± %6.4f   [Min: %6.4f, Max: %6.4f]\n', ...
        mean(cv_thresh), std(cv_thresh), min(cv_thresh), max(cv_thresh));
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 3. Feature Coefficient Stability Analysis Across 50 Folds
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   3. FEATURE COEFFICIENT STABILITY AUDIT (50 FOLDS)                    \n');
    fprintf('========================================================================\n');
    fprintf('  %-20s %-8s | %8s %8s %8s %8s | %7s %7s | %8s\n', ...
        'Feature', 'Group', 'Mean_Beta', 'Std_Beta', 'Min_Beta', 'Max_Beta', '%_Pos', '%_Neg', 'Stability');
    fprintf('  ---------------------------------------------------------------------------------------------------\n');

    all_feat_labels = [{'Intercept'}; feature_names'];
    all_feat_groups = [{'Baseline'}; feature_groups'];

    coef_means  = mean(cv_coefficients, 1)';
    coef_stds   = std(cv_coefficients, 0, 1)';
    coef_mins   = min(cv_coefficients, [], 1)';
    coef_maxs   = max(cv_coefficients, [], 1)';
    pct_pos     = (sum(cv_coefficients > 0, 1)' / total_runs) * 100;
    pct_neg     = (sum(cv_coefficients < 0, 1)' / total_runs) * 100;
    stab_ratios = abs(coef_means) ./ max(1e-6, coef_stds);

    for f = 1:(num_feats + 1)
        fprintf('  %-20s %-8s | %+8.4f %8.4f %+8.4f %+8.4f | %6.1f%% %6.1f%% | %8.2f\n', ...
            all_feat_labels{f}, all_feat_groups{f}, ...
            coef_means(f), coef_stds(f), coef_mins(f), coef_maxs(f), ...
            pct_pos(f), pct_neg(f), stab_ratios(f));
    end
    fprintf('===================================================================================================\n\n');

    % Group-Level Coefficient Comparison
    fprintf('GROUP-LEVEL FEATURE INFLUENCE AND STABILITY:\n');
    fprintf('----------------------------------------------------------------------\n');
    group_types = {'CNN', 'Lesion', 'Vessel', 'Quality'};
    for g = 1:numel(group_types)
        g_name = group_types{g};
        g_mask = strcmp(all_feat_groups, g_name);
        g_weights = coef_means(g_mask);
        g_abs_sum = sum(abs(g_weights));
        g_mean_stab = mean(stab_ratios(g_mask));
        fprintf('  • %-10s Evidence Channels (N=%2d): Total |Beta| = %6.4f | Mean Stability = %5.2f\n', ...
            g_name, sum(g_mask), g_abs_sum, g_mean_stab);
    end
    fprintf('----------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % 4. Fit Final Model C on Full 330 Training Cohort & Determine Frozen Parameters
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   4. FINAL MODEL C FITTING (330 Training Images Only)                  \n');
    fprintf('========================================================================\n');

    % Compute Final Normalization Parameters
    mu_final = mean(X_train_raw, 1);
    sigma_final = std(X_train_raw, 0, 1);
    sigma_final(sigma_final < 1e-6) = 1.0;

    Z_train_final = (X_train_raw - mu_final) ./ sigma_final;
    X_aug_train_final = [ones(n_train, 1), Z_train_final];

    % Fit Final Ridge Weights
    R_final = eye(num_feats + 1); R_final(1, 1) = 0;
    w_final_all = (X_aug_train_final' * X_aug_train_final + lambda_reg * R_final) \ (X_aug_train_final' * (2 * y_train - 1));

    b_final = w_final_all(1);
    w_final = w_final_all(2:end);

    % Select Final Decision Threshold on 330 Training Samples Only
    logits_train_final = X_aug_train_final * w_final_all;
    p_train_final = 1.0 ./ (1.0 + exp(-logits_train_final));

    cand_thresh_final = unique([prctile(p_train_final, 5:95)'; 0.50]);
    tau_final = 0.50;
    best_bal_final_tr = -1;

    for t = 1:numel(cand_thresh_final)
        th = cand_thresh_final(t);
        pred_tr = p_train_final >= th;
        tp = sum(pred_tr == 1 & y_train == 1);
        fp = sum(pred_tr == 1 & y_train == 0);
        fn = sum(pred_tr == 0 & y_train == 1);
        tn = sum(pred_tr == 0 & y_train == 0);
        bal = (tp/max(1, tp+fn) + tn/max(1, tn+fp)) / 2;

        if bal > best_bal_final_tr
            best_bal_final_tr = bal;
            tau_final = th;
        end
    end

    fprintf('  Final Intercept (b0):            %+8.4f\n', b_final);
    fprintf('  Final Decision Threshold (tau*):  %8.4f  (Optimized on 330 Train)\n', tau_final);
    fprintf('  Final Feature Weights (Beta_1 to Beta_%d):\n', num_feats);
    for f = 1:num_feats
        fprintf('    %-22s: %+8.4f  (Norm: mu=%8.4f, sigma=%8.4f)\n', ...
            feature_names{f}, w_final(f), mu_final(f), sigma_final(f));
    end
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 5. Single Evaluation on Untouched 83-Image Validation Set
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   5. HELD-OUT VALIDATION SET EVALUATION (N = 83 Images)                \n');
    fprintf('========================================================================\n');

    % Standardize Validation set using FROZEN Training Parameters
    Z_val_final = (X_val_raw - mu_final) ./ sigma_final;
    X_aug_val_final = [ones(n_val, 1), Z_val_final];

    logits_val_final = X_aug_val_final * w_final_all;
    p_val_final = 1.0 ./ (1.0 + exp(-logits_val_final));
    pred_val_final = p_val_final >= tau_final;

    tp_C = sum(pred_val_final == 1 & y_val == 1);
    fp_C = sum(pred_val_final == 1 & y_val == 0);
    fn_C = sum(pred_val_final == 0 & y_val == 1);
    tn_C = sum(pred_val_final == 0 & y_val == 0);

    sens_C = tp_C / max(1, tp_C + fn_C);
    spec_C = tn_C / max(1, tn_C + fp_C);
    prec_C = tp_C / max(1, tp_C + fp_C);
    f1_C   = (2 * tp_C) / max(1, 2 * tp_C + fp_C + fn_C);
    bal_C  = (sens_C + spec_C) / 2;
    acc_C  = (tp_C + tn_C) / n_val;

    % Baseline Model A (CNN Only with optimal training threshold)
    cnn_idx = find(strcmp(feature_names, 'CNN_P_Referable'));
    p_cnn_tr = X_train_raw(:, cnn_idx);
    p_cnn_val = X_val_raw(:, cnn_idx);

    cnn_cand_thresh = unique([prctile(p_cnn_tr, 5:95)'; 0.50]);
    tau_A = 0.50;
    best_bal_A = -1;
    for t = 1:numel(cnn_cand_thresh)
        th = cnn_cand_thresh(t);
        pred_tr = p_cnn_tr >= th;
        tp = sum(pred_tr == 1 & y_train == 1);
        fp = sum(pred_tr == 1 & y_train == 0);
        fn = sum(pred_tr == 0 & y_train == 1);
        tn = sum(pred_tr == 0 & y_train == 0);
        bal = (tp/max(1, tp+fn) + tn/max(1, tn+fp)) / 2;
        if bal > best_bal_A
            best_bal_A = bal;
            tau_A = th;
        end
    end

    pred_val_A = p_cnn_val >= tau_A;
    tp_A = sum(pred_val_A == 1 & y_val == 1);
    fp_A = sum(pred_val_A == 1 & y_val == 0);
    fn_A = sum(pred_val_A == 0 & y_val == 1);
    tn_A = sum(pred_val_A == 0 & y_val == 0);

    sens_A = tp_A / max(1, tp_A + fn_A);
    spec_A = tn_A / max(1, tn_A + fp_A);
    prec_A = tp_A / max(1, tp_A + fp_A);
    f1_A   = (2 * tp_A) / max(1, 2 * tp_A + fp_A + fn_A);
    bal_A  = (sens_A + spec_A) / 2;
    acc_A  = (tp_A + tn_A) / n_val;

    % Print Side-by-Side Comparison Table
    fprintf('  Metric                     Model A (CNN Only)    Final Model C (Multi-Modal)   Delta (Abs)\n');
    fprintf('  ------------------------------------------------------------------------------------------\n');
    fprintf('  Referable Sensitivity:     %5.2f%% (%2d / %2d)       %5.2f%% (%2d / %2d)              %+5.2f%%\n', ...
        sens_A*100, tp_A, tp_A+fn_A, sens_C*100, tp_C, tp_C+fn_C, (sens_C - sens_A)*100);
    fprintf('  Referable Specificity:     %5.2f%% (%2d / %2d)       %5.2f%% (%2d / %2d)              %+5.2f%%\n', ...
        spec_A*100, tn_A, tn_A+fp_A, spec_C*100, tn_C, tn_C+fp_C, (spec_C - spec_A)*100);
    fprintf('  Referable Precision (PPV): %5.2f%%                 %5.2f%%                         %+5.2f%%\n', ...
        prec_A*100, prec_C*100, (prec_C - prec_A)*100);
    fprintf('  Referable F1 Score:        %6.4f                  %6.4f                          %+6.4f\n', ...
        f1_A, f1_C, f1_C - f1_A);
    fprintf('  Balanced Accuracy:         %5.2f%%                 %5.2f%%                         %+5.2f%%\n', ...
        bal_A*100, bal_C*100, (bal_C - bal_A)*100);
    fprintf('  Overall Accuracy:          %5.2f%%                 %5.2f%%                         %+5.2f%%\n', ...
        acc_A*100, acc_C*100, (acc_C - acc_A)*100);
    fprintf('  False Positives (FP):      %2d                     %2d                             %+2d\n', ...
        fp_A, fp_C, fp_C - fp_A);
    fprintf('  False Negatives (FN):      %2d                     %2d                             %+2d\n', ...
        fn_A, fn_C, fn_C - fn_A);
    fprintf('  Confusion Matrix [TN,FP;FN,TP]: [%2d,%2d;%2d,%2d]       [%2d,%2d;%2d,%2d]\n', ...
        tn_A, fp_A, fn_A, tp_A, tn_C, fp_C, fn_C, tp_C);
    fprintf('  Decision Threshold (tau):  %6.4f                  %6.4f\n', tau_A, tau_final);
    fprintf('==========================================================================================\n\n');

    % -------------------------------------------------------------------------
    % 6. Save Artifacts
    % -------------------------------------------------------------------------
    coef_table = table(all_feat_labels, all_feat_groups, coef_means, coef_stds, coef_mins, coef_maxs, pct_pos, pct_neg, stab_ratios, ...
        'VariableNames', {'Feature', 'Group', 'Mean_Beta', 'Std_Beta', 'Min_Beta', 'Max_Beta', 'Pct_Positive', 'Pct_Negative', 'Stability_Ratio'});

    cv_summary_table = table(...
        {'Sensitivity'; 'Specificity'; 'Precision'; 'F1_Score'; 'Balanced_Accuracy'; 'Accuracy'; 'Decision_Threshold'}, ...
        [mean(cv_sens); mean(cv_spec); mean(cv_prec); mean(cv_f1); mean(cv_bal_acc); mean(cv_acc); mean(cv_thresh)], ...
        [std(cv_sens); std(cv_spec); std(cv_prec); std(cv_f1); std(cv_bal_acc); std(cv_acc); std(cv_thresh)], ...
        'VariableNames', {'Metric', 'CV_50Fold_Mean', 'CV_50Fold_Std'});

    final_val_comparison = table(...
        {'Sensitivity'; 'Specificity'; 'Precision'; 'F1_Score'; 'Balanced_Accuracy'; 'Accuracy'; 'FP'; 'FN'; 'Decision_Threshold'}, ...
        [sens_A; spec_A; prec_A; f1_A; bal_A; acc_A; fp_A; fn_A; tau_A], ...
        [sens_C; spec_C; prec_C; f1_C; bal_C; acc_C; fp_C; fn_C; tau_final], ...
        [sens_C - sens_A; spec_C - spec_A; prec_C - prec_A; f1_C - f1_A; bal_C - bal_A; acc_C - acc_A; fp_C - fp_A; fn_C - fn_A; tau_final - tau_A], ...
        'VariableNames', {'Metric', 'Model_A_CNN_Only', 'Final_Model_C_MultiModal', 'Delta_Absolute'});

    out_mat = fullfile(cfg.paths.metrics, 'fusion_stability.mat');
    out_csv = fullfile(cfg.paths.metrics, 'fusion_stability.csv');

    save(out_mat, 'coef_table', 'cv_summary_table', 'final_val_comparison', ...
        'w_final', 'b_final', 'mu_final', 'sigma_final', 'tau_final', 'cv_coefficients', 'cfg');
    writetable(final_val_comparison, out_csv);

    fprintf('Stability artifacts saved to:\n  • %s\n  • %s\n\n', out_mat, out_csv);

    % Return Structured Results
    results = struct();
    results.cv_summary = cv_summary_table;
    results.coef_table = coef_table;
    results.val_comparison = final_val_comparison;
    results.final_model = struct(...
        'w', w_final, 'b', b_final, 'mu', mu_final, 'sigma', sigma_final, 'tau', tau_final);
end

% -------------------------------------------------------------------------
% Helper Function: Native Stratified K-Fold Generator
% -------------------------------------------------------------------------
function fold_indices = get_stratified_folds(y, num_folds, seed)
    rng(seed, 'twister');
    n = numel(y);
    fold_indices = zeros(n, 1);
    classes = unique(y);
    for c = 1:numel(classes)
        cls_idx = find(y == classes(c));
        cls_shuffled = cls_idx(randperm(numel(cls_idx)));
        for i = 1:numel(cls_shuffled)
            fold_indices(cls_shuffled(i)) = mod(i - 1, num_folds) + 1;
        end
    end
end
