function results = calibrateEvidenceChannels(varargin)
% CALIBRATEEVIDENCECHANNELS Development-Only Evidence Calibration Experiment
% Evaluates whether calibrated lesion, vessel, and anatomical features provide
% incremental diagnostic value beyond the frozen CNN on the 83-image validation split.
%
% STRICT METHODOLOGICAL SAFEGUARDS:
% 1. Uses ONLY IDRiD training images (413 development cohort: 330 train / 83 val).
% 2. 103-image test partition is strictly sequestered and untouched.
% 3. All thresholds, normalization parameters, and fusion weights are fitted
%    EXCLUSIVELY on the 330-image training subset.
% 4. Evaluates frozen models on the untouched 83-image validation partition.
%
% OUTPUT FILES:
% - results/metrics/evidence_calibration.mat
% - results/metrics/evidence_calibration.csv

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: DEVELOPMENT-ONLY EVIDENCE CALIBRATION EXPERIMENT            \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Load Audit Data & Validate Partition Isolation
    % -------------------------------------------------------------------------
    audit_file = fullfile(cfg.paths.metrics, 'evidence_channel_audit.mat');
    if exist(audit_file, 'file') == 2
        fprintf('[STEP 1/4] Loading precomputed evidence audit from:\n  %s\n', audit_file);
        loaded = load(audit_file);
        audit_table = loaded.audit_table;
    else
        fprintf('[STEP 1/4] Precomputed audit not found. Running evaluateEvidenceChannels...\n');
        audit_res = evaluateEvidenceChannels();
        audit_table = audit_res.audit_table;
    end

    % Partition Subsets
    train_mask = strcmp(audit_table.Split, 'Train') & strcmp(audit_table.Status, 'SUCCESS');
    val_mask   = strcmp(audit_table.Split, 'Validation') & strcmp(audit_table.Status, 'SUCCESS');

    train_tab = audit_table(train_mask, :);
    val_tab   = audit_table(val_mask, :);

    n_train = height(train_tab);
    n_val   = height(val_tab);

    % Strict Partition Assertions
    assert(n_train == 330, 'Training subset must contain exactly 330 images. Found: %d', n_train);
    assert(n_val == 83, 'Validation subset must contain exactly 83 images. Found: %d', n_val);

    y_train = train_tab.True_Referable;
    y_val   = val_tab.True_Referable;

    fprintf('  [PASSED] Partition verified: %d Training images, %d Validation images.\n', n_train, n_val);
    fprintf('  [PASSED] Class Prevalence: Training Ref = %d/330 (%.1f%%) | Val Ref = %d/83 (%.1f%%)\n\n', ...
        sum(y_train == 1), (sum(y_train == 1)/n_train)*100, ...
        sum(y_val == 1), (sum(y_val == 1)/n_val)*100);

    % -------------------------------------------------------------------------
    % 2. Single-Feature Analysis & Training-Only Threshold Calibration
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   2. SINGLE-FEATURE DISCRIMINATION & THRESHOLD CALIBRATION             \n');
    fprintf('   (Thresholds optimized on 330 Train -> Evaluated on 83 Validation)   \n');
    fprintf('========================================================================\n');

    features = {
        'MA_Count', 'MA_Area', 'HE_Count', 'HE_Area_Fraction', ...
        'EX_Count', 'EX_Area_Fraction', 'DME_Risk', 'NV_Score', ...
        'Vessel_Density', 'Vessel_Tortuosity', 'Quality_Score', 'CNN_P_Referable'
    };
    num_feats = numel(features);

    single_feat_results = cell(num_feats, 1);

    fprintf('  %-18s %5s %10s | %6s %6s %6s %6s %7s | %8s\n', ...
        'Feature', 'Dir', 'TrainThresh', 'ValSens', 'ValSpec', 'ValPrec', 'ValF1', 'ValBalAcc', 'Confusion');
    fprintf('  -------------------------------------------------------------------------------------------------\n');

    for f = 1:num_feats
        feat_name = features{f};
        x_tr  = train_tab.(feat_name);
        x_val = val_tab.(feat_name);

        % Training-only Candidate Threshold Search
        % Evaluate candidate percentiles [1st to 99th] + unique values
        cand_thresholds = unique([prctile(x_tr, 1:99)'; min(x_tr); max(x_tr)]);
        
        best_dir = '>=';
        best_thresh = cand_thresholds(1);
        best_tr_bal_acc = -1;
        best_tr_f1 = -1;

        % Test both directions (>= and <=)
        for d = 1:2
            for t = 1:numel(cand_thresholds)
                thresh_val = cand_thresholds(t);
                if d == 1
                    y_pred_tr = x_tr >= thresh_val;
                    dir_str = '>=';
                else
                    y_pred_tr = x_tr <= thresh_val;
                    dir_str = '<=';
                end

                tp = sum(y_pred_tr == 1 & y_train == 1);
                fp = sum(y_pred_tr == 1 & y_train == 0);
                fn = sum(y_pred_tr == 0 & y_train == 1);
                tn = sum(y_pred_tr == 0 & y_train == 0);

                sens = tp / max(1, tp + fn);
                spec = tn / max(1, tn + fp);
                bal_acc = (sens + spec) / 2;
                f1 = (2 * tp) / max(1, 2 * tp + fp + fn);

                % Optimization Criterion: Maximize Training Balanced Accuracy
                if (bal_acc > best_tr_bal_acc) || (abs(bal_acc - best_tr_bal_acc) < 1e-5 && f1 > best_tr_f1)
                    best_tr_bal_acc = bal_acc;
                    best_tr_f1 = f1;
                    best_thresh = thresh_val;
                    best_dir = dir_str;
                end
            end
        end

        % Apply FROZEN optimal training threshold to the 83-image Validation set
        if strcmp(best_dir, '>=')
            y_pred_val = x_val >= best_thresh;
        else
            y_pred_val = x_val <= best_thresh;
        end

        tp_v = sum(y_pred_val == 1 & y_val == 1);
        fp_v = sum(y_pred_val == 1 & y_val == 0);
        fn_v = sum(y_pred_val == 0 & y_val == 1);
        tn_v = sum(y_pred_val == 0 & y_val == 0);

        sens_v = tp_v / max(1, tp_v + fn_v);
        spec_v = tn_v / max(1, tn_v + fp_v);
        prec_v = tp_v / max(1, tp_v + fp_v);
        f1_v   = (2 * tp_v) / max(1, 2 * tp_v + fp_v + fn_v);
        bal_v  = (sens_v + spec_v) / 2;
        acc_v  = (tp_v + tn_v) / n_val;

        single_feat_results{f} = struct(...
            'Feature', feat_name, ...
            'Direction', best_dir, ...
            'Train_Threshold', best_thresh, ...
            'Train_BalAcc', best_tr_bal_acc, ...
            'Val_Sensitivity', sens_v, ...
            'Val_Specificity', spec_v, ...
            'Val_Precision', prec_v, ...
            'Val_F1', f1_v, ...
            'Val_Balanced_Accuracy', bal_v, ...
            'Val_Accuracy', acc_v, ...
            'Val_TP', tp_v, 'Val_FP', fp_v, 'Val_FN', fn_v, 'Val_TN', tn_v);

        if ismember(feat_name, {'HE_Area_Fraction', 'EX_Area_Fraction', 'Vessel_Density', 'CNN_P_Referable'})
            fprintf('  %-18s %5s %10.4f | %5.2f%% %5.2f%% %5.2f%% %6.4f %6.2f%% | [%2d,%2d;%2d,%2d]\n', ...
                feat_name, best_dir, best_thresh, sens_v*100, spec_v*100, prec_v*100, f1_v, bal_v*100, tn_v, fp_v, fn_v, tp_v);
        else
            fprintf('  %-18s %5s %10.2f | %5.2f%% %5.2f%% %5.2f%% %6.4f %6.2f%% | [%2d,%2d;%2d,%2d]\n', ...
                feat_name, best_dir, best_thresh, sens_v*100, spec_v*100, prec_v*100, f1_v, bal_v*100, tn_v, fp_v, fn_v, tp_v);
        end
    end
    fprintf('=================================================================================================\n\n');

    % -------------------------------------------------------------------------
    % 3. Multi-Feature Evidence Modeling: Models A, B, and C
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   3. MULTI-MODAL EVIDENCE COMBINATION MODELS                           \n');
    fprintf('   (Trained on 330 images -> Validated on 83 images)                    \n');
    fprintf('========================================================================\n');

    % Extract Training and Validation Feature Matrices
    X_train_raw = zeros(n_train, num_feats);
    X_val_raw   = zeros(n_val, num_feats);

    for f = 1:num_feats
        X_train_raw(:, f) = train_tab.(features{f});
        X_val_raw(:, f)   = val_tab.(features{f});
    end

    % Normalization Parameters: Derived STRICTLY from 330 Training images
    mu_train    = mean(X_train_raw, 1);
    sigma_train = std(X_train_raw, 0, 1);
    sigma_train(sigma_train < 1e-6) = 1.0; % Prevent zero-division

    Z_train = (X_train_raw - mu_train) ./ sigma_train;
    Z_val   = (X_val_raw - mu_train) ./ sigma_train; % Using training normalization!

    cnn_idx = find(strcmp(features, 'CNN_P_Referable'));

    % -------------------------------------------------------------------------
    % MODEL A: CNN Probability Alone
    % -------------------------------------------------------------------------
    p_cnn_tr  = X_train_raw(:, cnn_idx);
    p_cnn_val = X_val_raw(:, cnn_idx);

    % Find optimal training decision threshold for CNN
    cnn_cand_thresh = unique([prctile(p_cnn_tr, 5:95)'; 0.50]);
    best_tau_A = 0.50;
    best_bal_A_tr = -1;

    for t = 1:numel(cnn_cand_thresh)
        th = cnn_cand_thresh(t);
        pred_tr = p_cnn_tr >= th;
        tp = sum(pred_tr == 1 & y_train == 1);
        fp = sum(pred_tr == 1 & y_train == 0);
        fn = sum(pred_tr == 0 & y_train == 1);
        tn = sum(pred_tr == 0 & y_train == 0);
        bal = (tp/max(1, tp+fn) + tn/max(1, tn+fp)) / 2;
        if bal > best_bal_A_tr
            best_bal_A_tr = bal;
            best_tau_A = th;
        end
    end

    pred_A_val = p_cnn_val >= best_tau_A;
    tp_A = sum(pred_A_val == 1 & y_val == 1);
    fp_A = sum(pred_A_val == 1 & y_val == 0);
    fn_A = sum(pred_A_val == 0 & y_val == 1);
    tn_A = sum(pred_A_val == 0 & y_val == 0);

    sens_A = tp_A / max(1, tp_A + fn_A);
    spec_A = tn_A / max(1, tn_A + fp_A);
    prec_A = tp_A / max(1, tp_A + fp_A);
    f1_A   = (2 * tp_A) / max(1, 2 * tp_A + fp_A + fn_A);
    bal_A  = (sens_A + spec_A) / 2;
    acc_A  = (tp_A + tn_A) / n_val;

    % -------------------------------------------------------------------------
    % MODEL B: CNN + Best Single Lesion Feature (HE Area Fraction)
    % -------------------------------------------------------------------------
    % Identify best non-CNN lesion feature from training balanced accuracy
    lesion_indices = find(~ismember(features, {'CNN_P_Referable', 'Quality_Score', 'Vessel_Density', 'Vessel_Tortuosity'}));
    lesion_train_bal = cellfun(@(s) s.Train_BalAcc, single_feat_results(lesion_indices));
    [~, top_l_idx] = max(lesion_train_bal);
    best_lesion_idx = lesion_indices(top_l_idx);
    best_lesion_name = features{best_lesion_idx};

    % Feature Subset for Model B: [CNN, Best Lesion]
    sub_indices_B = [cnn_idx, best_lesion_idx];
    Z_tr_B  = Z_train(:, sub_indices_B);
    Z_val_B = Z_val(:, sub_indices_B);

    % Fit regularized logistic regression on 330 training samples
    lambda_B = 0.05;
    X_aug_B = [ones(n_train, 1), Z_tr_B];
    % Ridge closed-form for logistic / linear logit
    w_B_raw = (X_aug_B' * X_aug_B + lambda_B * eye(3)) \ (X_aug_B' * (2*y_train - 1));
    
    % Compute logit probabilities via sigmoid
    logits_tr_B = [ones(n_train, 1), Z_tr_B] * w_B_raw;
    p_tr_B = 1.0 ./ (1.0 + exp(-logits_tr_B));

    % Find optimal training decision threshold for Model B
    b_cand_thresh = prctile(p_tr_B, 5:95)';
    best_tau_B = 0.50;
    best_bal_B_tr = -1;

    for t = 1:numel(b_cand_thresh)
        th = b_cand_thresh(t);
        pred_tr = p_tr_B >= th;
        tp = sum(pred_tr == 1 & y_train == 1);
        fp = sum(pred_tr == 1 & y_train == 0);
        fn = sum(pred_tr == 0 & y_train == 1);
        tn = sum(pred_tr == 0 & y_train == 0);
        bal = (tp/max(1, tp+fn) + tn/max(1, tn+fp)) / 2;
        if bal > best_bal_B_tr
            best_bal_B_tr = bal;
            best_tau_B = th;
        end
    end

    logits_val_B = [ones(n_val, 1), Z_val_B] * w_B_raw;
    p_val_B = 1.0 ./ (1.0 + exp(-logits_val_B));
    pred_B_val = p_val_B >= best_tau_B;

    tp_B = sum(pred_B_val == 1 & y_val == 1);
    fp_B = sum(pred_B_val == 1 & y_val == 0);
    fn_B = sum(pred_B_val == 0 & y_val == 1);
    tn_B = sum(pred_B_val == 0 & y_val == 0);

    sens_B = tp_B / max(1, tp_B + fn_B);
    spec_B = tn_B / max(1, tn_B + fp_B);
    prec_B = tp_B / max(1, tp_B + fp_B);
    f1_B   = (2 * tp_B) / max(1, 2 * tp_B + fp_B + fn_B);
    bal_B  = (sens_B + spec_B) / 2;
    acc_B  = (tp_B + tn_B) / n_val;

    % -------------------------------------------------------------------------
    % MODEL C: CNN + Multi-Modal Calibrated Evidence (Regularized Ridge/LASSO)
    % -------------------------------------------------------------------------
    % Train Ridge / Logistic GLM with 5-Fold Cross-Validation on 330 Training Data ONLY
    rng(cfg.random_seed, 'twister');
    cv_folds = 5;
    
    % Native reproducible K-fold partitioning
    shuffled_idx = randperm(n_train)';
    fold_sizes = repmat(floor(n_train / cv_folds), cv_folds, 1);
    fold_sizes(1:mod(n_train, cv_folds)) = fold_sizes(1:mod(n_train, cv_folds)) + 1;
    cv_indices = zeros(n_train, 1);
    curr_pos = 1;
    for k = 1:cv_folds
        cv_indices(shuffled_idx(curr_pos:curr_pos + fold_sizes(k) - 1)) = k;
        curr_pos = curr_pos + fold_sizes(k);
    end

    lambda_grid = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 5.0, 10.0];
    
    cv_scores = zeros(numel(lambda_grid), 1);
    X_aug_C_tr = [ones(n_train, 1), Z_train];

    for l = 1:numel(lambda_grid)
        lam = lambda_grid(l);
        fold_bal = zeros(cv_folds, 1);
        for k = 1:cv_folds
            val_f = (cv_indices == k);
            tr_f  = ~val_f;

            X_k_tr = X_aug_C_tr(tr_f, :);
            y_k_tr = y_train(tr_f);
            X_k_val = X_aug_C_tr(val_f, :);
            y_k_val = y_train(val_f);

            R = eye(num_feats + 1); R(1, 1) = 0; % Don't regularize intercept
            w_k = (X_k_tr' * X_k_tr + lam * R) \ (X_k_tr' * (2*y_k_tr - 1));

            p_k = 1.0 ./ (1.0 + exp(-X_k_val * w_k));
            pred_k = p_k >= 0.50;

            tp_k = sum(pred_k == 1 & y_k_val == 1);
            fp_k = sum(pred_k == 1 & y_k_val == 0);
            fn_k = sum(pred_k == 0 & y_k_val == 1);
            tn_k = sum(pred_k == 0 & y_k_val == 0);
            fold_bal(k) = (tp_k/max(1, tp_k+fn_k) + tn_k/max(1, tn_k+fp_k)) / 2;
        end
        cv_scores(l) = mean(fold_bal);
    end

    [~, best_l_idx] = max(cv_scores);
    best_lambda_C = lambda_grid(best_l_idx);

    % Fit final model C on all 330 training images
    R_final = eye(num_feats + 1); R_final(1, 1) = 0;
    w_C_final = (X_aug_C_tr' * X_aug_C_tr + best_lambda_C * R_final) \ (X_aug_C_tr' * (2*y_train - 1));

    % Determine optimal training threshold for Model C
    logits_tr_C = X_aug_C_tr * w_C_final;
    p_tr_C = 1.0 ./ (1.0 + exp(-logits_tr_C));

    c_cand_thresh = prctile(p_tr_C, 5:95)';
    best_tau_C = 0.50;
    best_bal_C_tr = -1;

    for t = 1:numel(c_cand_thresh)
        th = c_cand_thresh(t);
        pred_tr = p_tr_C >= th;
        tp = sum(pred_tr == 1 & y_train == 1);
        fp = sum(pred_tr == 1 & y_train == 0);
        fn = sum(pred_tr == 0 & y_train == 1);
        tn = sum(pred_tr == 0 & y_train == 0);
        bal = (tp/max(1, tp+fn) + tn/max(1, tn+fp)) / 2;
        if bal > best_bal_C_tr
            best_bal_C_tr = bal;
            best_tau_C = th;
        end
    end

    % Apply Model C to 83-image validation partition
    X_aug_C_val = [ones(n_val, 1), Z_val];
    logits_val_C = X_aug_C_val * w_C_final;
    p_val_C = 1.0 ./ (1.0 + exp(-logits_val_C));
    pred_C_val = p_val_C >= best_tau_C;

    tp_C = sum(pred_C_val == 1 & y_val == 1);
    fp_C = sum(pred_C_val == 1 & y_val == 0);
    fn_C = sum(pred_C_val == 0 & y_val == 1);
    tn_C = sum(pred_C_val == 0 & y_val == 0);

    sens_C = tp_C / max(1, tp_C + fn_C);
    spec_C = tn_C / max(1, tn_C + fp_C);
    prec_C = tp_C / max(1, tp_C + fp_C);
    f1_C   = (2 * tp_C) / max(1, 2 * tp_C + fp_C + fn_C);
    bal_C  = (sens_C + spec_C) / 2;
    acc_C  = (tp_C + tn_C) / n_val;

    % -------------------------------------------------------------------------
    % 4. Model Comparison Table (Validation Set N = 83)
    % -------------------------------------------------------------------------
    fprintf('VALIDATION SET PERFORMANCE COMPARISON (N = 83 Images):\n');
    fprintf('=========================================================================================\n');
    fprintf('  Metric                   Model A (CNN Only)    Model B (CNN + %-6s)   Model C (Multi-Modal)\n', ...
        best_lesion_name(1:min(6, end)));
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('  Referable Sensitivity:   %5.2f%% (%2d/%2d)         %5.2f%% (%2d/%2d)           %5.2f%% (%2d/%2d)\n', ...
        sens_A*100, tp_A, tp_A+fn_A, sens_B*100, tp_B, tp_B+fn_B, sens_C*100, tp_C, tp_C+fn_C);
    fprintf('  Referable Specificity:   %5.2f%% (%2d/%2d)         %5.2f%% (%2d/%2d)           %5.2f%% (%2d/%2d)\n', ...
        spec_A*100, tn_A, tn_A+fp_A, spec_B*100, tn_B, tn_B+fp_B, spec_C*100, tn_C, tn_C+fp_C);
    fprintf('  Referable Precision:     %5.2f%%                  %5.2f%%                    %5.2f%%\n', ...
        prec_A*100, prec_B*100, prec_C*100);
    fprintf('  Referable F1 Score:      %6.4f                   %6.4f                     %6.4f\n', ...
        f1_A, f1_B, f1_C);
    fprintf('  Balanced Accuracy:       %5.2f%%                  %5.2f%%                    %5.2f%%\n', ...
        bal_A*100, bal_B*100, bal_C*100);
    fprintf('  Overall Accuracy:        %5.2f%%                  %5.2f%%                    %5.2f%%\n', ...
        acc_A*100, acc_B*100, acc_C*100);
    fprintf('  False Positives (FP):    %2d                      %2d                        %2d\n', ...
        fp_A, fp_B, fp_C);
    fprintf('  False Negatives (FN):    %2d                      %2d                        %2d\n', ...
        fn_A, fn_B, fn_C);
    fprintf('  Decision Threshold (tau):%6.4f                  %6.4f                     %6.4f\n', ...
        best_tau_A, best_tau_B, best_tau_C);
    fprintf('=========================================================================================\n\n');

    % Learned Coefficients Breakdown
    fprintf('LEARNED MULTI-MODAL MODEL COEFFICIENTS (Model C):\n');
    fprintf('-------------------------------------------------------------------------\n');
    fprintf('  Intercept (b0):           %+8.4f\n', w_C_final(1));
    for f = 1:num_feats
        fprintf('  %-24s  %+8.4f  (Train Mean: %8.4f, Std: %8.4f)\n', ...
            features{f}, w_C_final(f+1), mu_train(f), sigma_train(f));
    end
    fprintf('-------------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % 5. Save Artifacts
    % -------------------------------------------------------------------------
    comparison_table = table(...
        {'Sensitivity'; 'Specificity'; 'Precision'; 'F1_Score'; 'Balanced_Accuracy'; 'Accuracy'; 'FP'; 'FN'; 'Decision_Threshold'}, ...
        [sens_A; spec_A; prec_A; f1_A; bal_A; acc_A; fp_A; fn_A; best_tau_A], ...
        [sens_B; spec_B; prec_B; f1_B; bal_B; acc_B; fp_B; fn_B; best_tau_B], ...
        [sens_C; spec_C; prec_C; f1_C; bal_C; acc_C; fp_C; fn_C; best_tau_C], ...
        'VariableNames', {'Metric', 'Model_A_CNN_Only', 'Model_B_CNN_Plus_Best_Lesion', 'Model_C_MultiModal'});

    single_feat_table = struct2table([single_feat_results{:}]);

    out_mat = fullfile(cfg.paths.metrics, 'evidence_calibration.mat');
    out_csv = fullfile(cfg.paths.metrics, 'evidence_calibration.csv');

    save(out_mat, 'comparison_table', 'single_feat_table', 'w_C_final', 'mu_train', 'sigma_train', ...
        'best_tau_A', 'best_tau_B', 'best_tau_C', 'best_lesion_name', 'cfg');
    writetable(comparison_table, out_csv);

    fprintf('Calibration outputs saved to:\n  • %s\n  • %s\n\n', out_mat, out_csv);

    % Return Structured Results
    results = struct();
    results.comparison_table = comparison_table;
    results.single_feature_table = single_feat_table;
    results.weights_model_c = w_C_final;
    results.mu_train = mu_train;
    results.sigma_train = sigma_train;
    results.best_lesion_feature = best_lesion_name;
end
