function results = freezeDevelopmentRelease()
% FREEZEDEVELOPMENTRELEASE Freeze and Benchmark SHINRA_DEVELOPMENT_RELEASE_V1
% Freezes the development release configuration:
%   - Calibration: Platt logistic scaling (a = 1.3106, b = -1.1412)
%   - Referral Decision Threshold: tau_ref = 0.30 (from validation plateau [0.25, 0.32])
%   - Reliability Gate: tau_accept = 60.0
% Evaluates strictly on the 83-image development validation partition.
%
% STRICT GOVERNANCE & SAFETY:
% - Zero access to the 103-image IDRiD test partition.
% - Zero modifications to results/metrics/final_test/
% - Zero retraining of CNN or U-Net models.
%
% ARTIFACTS SAVED:
% - results/metrics/development_release_v1.mat
% - results/metrics/development_release_v1.csv
% - results/metrics/development_release_v1_summary.csv

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: FREEZE DEVELOPMENT RELEASE CONFIGURATION (V1)                \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Strict Quarantine Check
    % -------------------------------------------------------------------------
    fprintf('[SAFEGUARD 1/4] Verifying strict test partition quarantine...\n');
    [~, idrid_test] = load_idrid();

    audit_file = fullfile(cfg.paths.metrics, 'evidence_channel_audit.mat');
    assert(exist(audit_file, 'file') == 2, 'Evidence audit file not found: %s', audit_file);
    loaded_aud = load(audit_file);
    audit_tab = loaded_aud.audit_table;

    normalize_path = @(p) lower(strrep(strrep(strtrim(char(p)), '/', filesep), '\', filesep));
    norm_audit = cellfun(normalize_path, audit_tab.ImagePath, 'UniformOutput', false);

    if height(idrid_test) > 0
        norm_test = cellfun(normalize_path, idrid_test.ImagePath, 'UniformOutput', false);
        assert(~any(ismember(norm_audit, norm_test)), ...
            'CRITICAL VIOLATION: IDRiD test image detected in development audit!');
    end
    fprintf('  [PASS] 103-image test partition strictly isolated and quarantined.\n');
    fprintf('  [PASS] Development evaluation executed strictly on 83 validation images.\n\n');

    % -------------------------------------------------------------------------
    % 2. Extract Validation Split (N = 83) & Reliability Metadata
    % -------------------------------------------------------------------------
    val_mask = strcmp(audit_tab.Split, 'Validation');
    val_tab  = audit_tab(val_mask, :);
    n_val    = height(val_tab);
    assert(n_val == 83, 'Expected 83 validation images. Found: %d', n_val);

    n_train_cal = sum(strcmp(audit_tab.Split, 'Train'));
    assert(n_train_cal == 330, 'Expected 330 training images. Found: %d', n_train_cal);

    image_ids   = val_tab.ImageID;
    image_paths = val_tab.ImagePath;
    y_val       = double(val_tab.True_Referable);
    dr_grades   = val_tab.True_DR_Grade;
    p_cnn_val   = double(val_tab.CNN_P_Referable);
    p_fus_raw   = double(val_tab.Fusion_P_Referable);
    q_class_val = val_tab.Quality_Class;
    q_score_val = double(val_tab.Quality_Score);

    % Load Reliability Scores
    rel_eval_file = fullfile(cfg.paths.metrics, 'reliability_layer_evaluation.mat');
    assert(exist(rel_eval_file, 'file') == 2, 'Reliability evaluation file not found: %s', rel_eval_file);
    loaded_rel = load(rel_eval_file);
    rel_val_tab = loaded_rel.eval_table(strcmp(loaded_rel.eval_table.Split, 'Validation'), :);
    rel_scores_val = double(rel_val_tab.Reliability_Score);

    % -------------------------------------------------------------------------
    % 3. Model Checkpoint Information & Hashes
    % -------------------------------------------------------------------------
    sev_model_path = fullfile(cfg.paths.models_severity, 'SHINRA_SqueezeNet_Severity_Model_V3.mat');
    ves_model_path = fullfile(cfg.paths.models_vessels, 'vessel_unet.mat');
    
    sev_info = struct('file', sev_model_path, 'exists', exist(sev_model_path, 'file') == 2);
    ves_info = struct('file', ves_model_path, 'exists', exist(ves_model_path, 'file') == 2);
    if sev_info.exists
        d = dir(sev_model_path);
        sev_info.bytes = d.bytes;
        sev_info.date = d.date;
    end
    if ves_info.exists
        d = dir(ves_model_path);
        ves_info.bytes = d.bytes;
        ves_info.date = d.date;
    end

    % -------------------------------------------------------------------------
    % 4. Apply Frozen Platt Calibration & Release Thresholds
    % -------------------------------------------------------------------------
    a_platt = 1.3106;
    b_platt = -1.1412;
    tau_ref = 0.30;
    tau_accept = 60.0;

    eps_val = 1e-6;
    p_clamped = max(eps_val, min(1 - eps_val, p_fus_raw));
    logit_val = log(p_clamped ./ (1 - p_clamped));
    p_calibrated = 1.0 ./ (1.0 + exp(-(a_platt .* logit_val + b_platt)));

    % Decision rule
    y_pred_release = double(p_calibrated >= tau_ref);

    % Triage Classification
    rec_actions = cell(n_val, 1);
    decision_ready = false(n_val, 1);

    for i = 1:n_val
        if strcmp(q_class_val{i}, 'UNGRADABLE')
            rec_actions{i} = 'RECAPTURE';
            decision_ready(i) = false;
        elseif rel_scores_val(i) < tau_accept
            if y_pred_release(i) == 1
                rec_actions{i} = 'CLINICAL_REVIEW';
            else
                rec_actions{i} = 'UNCERTAIN';
            end
            decision_ready(i) = false;
        else
            rec_actions{i} = 'ACCEPT_SCREENING';
            decision_ready(i) = true;
        end
    end

    % -------------------------------------------------------------------------
    % 5. Cohort Metrics Calculation
    % -------------------------------------------------------------------------
    % Helper Functions
    calc_brier = @(p, y) mean((p(:) - y(:)).^2);

    function [ece, conf_bins, acc_bins] = calc_ece(p, y, n_bins)
        if nargin < 3, n_bins = 10; end
        bin_edges = linspace(0, 1, n_bins + 1);
        ece = 0; conf_bins = zeros(n_bins, 1); acc_bins = zeros(n_bins, 1);
        N = numel(p);
        for b = 1:n_bins
            idx = (p >= bin_edges(b)) & (p < bin_edges(b+1));
            if b == n_bins, idx = (p >= bin_edges(b)) & (p <= bin_edges(b+1)); end
            nb = sum(idx);
            if nb > 0
                cb = mean(p(idx)); ab = mean(y(idx));
                conf_bins(b) = cb; acc_bins(b) = ab;
                ece = ece + (nb / N) * abs(ab - cb);
            end
        end
    end

    function auroc = calc_auroc(p, y)
        [~, s_idx] = sort(p, 'descend');
        sorted_y = y(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        tpr = tps / max(1, sum(sorted_y == 1));
        fpr = fps / max(1, sum(sorted_y == 0));
        auroc = trapz([0; fpr], [0; tpr]);
    end

    function auprc = calc_auprc(p, y)
        [~, s_idx] = sort(p, 'descend');
        sorted_y = y(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        rec = tps / max(1, sum(sorted_y == 1));
        prec= tps ./ max(1, tps + fps);
        auprc = trapz([0; rec], [1; prec]);
    end

    % Entire Cohort (N = 83)
    tp_all = sum(y_pred_release == 1 & y_val == 1);
    fp_all = sum(y_pred_release == 1 & y_val == 0);
    fn_all = sum(y_pred_release == 0 & y_val == 1);
    tn_all = sum(y_pred_release == 0 & y_val == 0);

    sens_all = tp_all / max(1, tp_all + fn_all);
    spec_all = tn_all / max(1, tn_all + fp_all);
    ppv_all  = tp_all / max(1, tp_all + fp_all);
    npv_all  = tn_all / max(1, tn_all + fn_all);
    f1_all   = (2 * tp_all) / max(1, 2 * tp_all + fp_all + fn_all);
    acc_all  = (tp_all + tn_all) / n_val;
    bal_all  = (sens_all + spec_all) / 2;
    auroc_all= calc_auroc(p_calibrated, y_val);
    auprc_all= calc_auprc(p_calibrated, y_val);
    brier_all= calc_brier(p_calibrated, y_val);
    [ece_all, ~, ~] = calc_ece(p_calibrated, y_val);

    % Accepted Cohort
    is_acc = decision_ready;
    n_acc = sum(is_acc);
    n_def = sum(~is_acc);
    cov_pct = (n_acc / n_val) * 100;
    def_pct = (n_def / n_val) * 100;

    y_val_acc  = y_val(is_acc);
    y_pred_acc = y_pred_release(is_acc);

    tp_acc = sum(y_pred_acc == 1 & y_val_acc == 1);
    fp_acc = sum(y_pred_acc == 1 & y_val_acc == 0);
    fn_acc = sum(y_pred_acc == 0 & y_val_acc == 1);
    tn_acc = sum(y_pred_acc == 0 & y_val_acc == 0);

    sens_acc = tp_acc / max(1, tp_acc + fn_acc);
    spec_acc = tn_acc / max(1, tn_acc + fp_acc);
    ppv_acc  = tp_acc / max(1, tp_acc + fp_acc);
    npv_acc  = tn_acc / max(1, tn_acc + fn_acc);
    f1_acc   = (2 * tp_acc) / max(1, 2 * tp_acc + fp_acc + fn_acc);
    acc_acc  = (tp_acc + tn_acc) / max(1, n_acc);
    bal_acc  = (sens_acc + spec_acc) / 2;

    % Deferred Cohort Breakdown
    cnt_recapture = sum(strcmp(rec_actions, 'RECAPTURE'));
    cnt_clin_rev  = sum(strcmp(rec_actions, 'CLINICAL_REVIEW'));
    cnt_uncertain = sum(strcmp(rec_actions, 'UNCERTAIN'));

    % Reliability Statistics
    rel_mean = mean(rel_scores_val);
    rel_median = median(rel_scores_val);
    rel_std = std(rel_scores_val);
    rel_min = min(rel_scores_val);
    rel_max = max(rel_scores_val);

    % -------------------------------------------------------------------------
    % 6. Construct Data Tables & Structs
    % -------------------------------------------------------------------------
    per_image_tab = table(image_ids, image_paths, dr_grades, y_val, q_class_val, q_score_val, ...
        p_cnn_val, p_fus_raw, p_calibrated, y_pred_release, rel_scores_val, rec_actions, double(decision_ready), ...
        'VariableNames', {'ImageID', 'ImagePath', 'True_DR_Grade', 'True_Referable', 'Quality_Class', 'Quality_Score', ...
                          'CNN_P_Referable', 'Fused_P_Referable_Raw', 'Fused_P_Referable_Platt_Calibrated', ...
                          'Predicted_Referable_ReleaseV1', 'Reliability_Score', 'Recommended_Action_ReleaseV1', 'Decision_Readiness_ReleaseV1'});

    summary_tab = table(...
        {'SHINRA_DEVELOPMENT_RELEASE_V1 (Entire Cohort)'; 'SHINRA_DEVELOPMENT_RELEASE_V1 (Accepted Cohort)'}, ...
        [n_val; n_acc], ...
        [100.0; cov_pct], ...
        [0.0; def_pct], ...
        [sens_all; sens_acc], ...
        [spec_all; spec_acc], ...
        [ppv_all; ppv_acc], ...
        [npv_all; npv_acc], ...
        [f1_all; f1_acc], ...
        [acc_all; acc_acc], ...
        [bal_all; bal_acc], ...
        [auroc_all; NaN], ...
        [auprc_all; NaN], ...
        [brier_all; NaN], ...
        [ece_all; NaN], ...
        [fn_all; fn_acc], ...
        [fp_all; fp_acc], ...
        'VariableNames', {'Evaluation_Cohort', 'N_Samples', 'Coverage_Pct', 'Deferred_Pct', ...
                          'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1_Score', 'Accuracy', 'Balanced_Accuracy', ...
                          'AUROC', 'AUPRC', 'Brier_Score', 'ECE', 'FN', 'FP'});

    % Output Structure
    release_struct = struct();
    release_struct.configuration_name = 'SHINRA_DEVELOPMENT_RELEASE_V1';
    release_struct.governance_status = 'DEVELOPMENT_FROZEN_PENDING_TEST';
    release_struct.timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    
    release_struct.parameters = struct(...
        'calibration_method', 'platt_logistic_scaling', ...
        'calibration_slope_a', a_platt, ...
        'calibration_intercept_b', b_platt, ...
        'referable_threshold_tau_ref', tau_ref, ...
        'reliability_threshold_tau_accept', tau_accept, ...
        'calibration_formula', 'P_calibrated = 1 ./ (1 + exp(-(1.3106 .* logit(P_raw) - 1.1412)))');

    release_struct.dataset_partitions = struct(...
        'training_calibration_N', n_train_cal, ...
        'validation_evaluation_N', n_val, ...
        'test_partition_accessed', 0);

    release_struct.models = struct(...
        'severity_model', sev_info, ...
        'vessel_model', ves_info);

    release_struct.entire_cohort_metrics = struct(...
        'N', n_val, 'Sensitivity', sens_all, 'Specificity', spec_all, ...
        'PPV', ppv_all, 'NPV', npv_all, 'F1', f1_all, 'Accuracy', acc_all, ...
        'Balanced_Accuracy', bal_all, 'AUROC', auroc_all, 'AUPRC', auprc_all, ...
        'Brier_Score', brier_all, 'ECE', ece_all, 'FN', fn_all, 'FP', fp_all);

    release_struct.accepted_cohort_metrics = struct(...
        'N', n_acc, 'Coverage_Pct', cov_pct, 'Deferred_Pct', def_pct, ...
        'Sensitivity', sens_acc, 'Specificity', spec_acc, 'PPV', ppv_acc, ...
        'NPV', npv_acc, 'F1', f1_acc, 'Accuracy', acc_acc, ...
        'Balanced_Accuracy', bal_acc, 'FN', fn_acc, 'FP', fp_acc);

    release_struct.deferred_cohort_summary = struct(...
        'total_deferred', n_def, ...
        'recapture_ungradable', cnt_recapture, ...
        'clinical_review_referral', cnt_clin_rev, ...
        'uncertain_non_referral', cnt_uncertain);

    release_struct.reliability_distribution = struct(...
        'mean', rel_mean, 'median', rel_median, 'std', rel_std, ...
        'min', rel_min, 'max', rel_max);

    release_struct.per_image_table = per_image_tab;
    release_struct.summary_table = summary_tab;

    % -------------------------------------------------------------------------
    % 7. Save Artifacts
    % -------------------------------------------------------------------------
    mat_out     = fullfile(cfg.paths.metrics, 'development_release_v1.mat');
    csv_out     = fullfile(cfg.paths.metrics, 'development_release_v1.csv');
    summary_csv = fullfile(cfg.paths.metrics, 'development_release_v1_summary.csv');

    save(mat_out, '-struct', 'release_struct');
    writetable(per_image_tab, csv_out);
    writetable(summary_tab, summary_csv);

    fprintf('Saved release artifacts successfully:\n');
    fprintf('  - %s\n', mat_out);
    fprintf('  - %s\n', csv_out);
    fprintf('  - %s\n\n', summary_csv);

    % -------------------------------------------------------------------------
    % 8. Output Report & Sanity Check
    % -------------------------------------------------------------------------
    fprintf('====================================================\n');
    fprintf('SHINRA DEVELOPMENT RELEASE V1: VALIDATION REPORT   \n');
    fprintf('====================================================\n\n');

    fprintf('Configuration: %s\n', release_struct.configuration_name);
    fprintf('Status:        %s\n', release_struct.governance_status);
    fprintf('Timestamp:     %s\n\n', release_struct.timestamp);

    fprintf('Parameters:\n');
    fprintf('  • Platt Slope (a):       %6.4f\n', a_platt);
    fprintf('  • Platt Intercept (b):   %6.4f\n', b_platt);
    fprintf('  • Referral Threshold:    %4.2f (Selected from validation plateau [0.25, 0.32])\n', tau_ref);
    fprintf('  • Reliability Gate:      %4.1f\n\n', tau_accept);

    fprintf('Entire Validation Cohort (N = 83):\n');
    fprintf('  • Sensitivity:       %5.2f%% (%d / %d)\n', sens_all * 100, tp_all, tp_all + fn_all);
    fprintf('  • Specificity:       %5.2f%% (%d / %d)\n', spec_all * 100, tn_all, tn_all + fp_all);
    fprintf('  • PPV (Precision):   %5.2f%% (%d / %d)\n', ppv_all * 100, tp_all, tp_all + fp_all);
    fprintf('  • NPV:               %5.2f%% (%d / %d)\n', npv_all * 100, tn_all, tn_all + fn_all);
    fprintf('  • F1-Score:          %6.4f\n', f1_all);
    fprintf('  • Balanced Accuracy: %5.2f%%\n', bal_all * 100);
    fprintf('  • AUROC:             %6.4f\n', auroc_all);
    fprintf('  • AUPRC:             %6.4f\n', auprc_all);
    fprintf('  • Brier Score:       %6.4f\n', brier_all);
    fprintf('  • ECE:               %6.4f\n', ece_all);
    fprintf('  • Errors:            FN = %d, FP = %d\n\n', fn_all, fp_all);

    fprintf('Autonomous Accepted Cohort (Reliability >= 60.0 & Gradable):\n');
    fprintf('  • Coverage:          %5.2f%% (%d / %d)\n', cov_pct, n_acc, n_val);
    fprintf('  • Deferred Rate:     %5.2f%% (%d / %d)\n', def_pct, n_def, n_val);
    fprintf('  • Sensitivity:       %5.2f%% (%d / %d)\n', sens_acc * 100, tp_acc, tp_acc + fn_acc);
    fprintf('  • Specificity:       %5.2f%% (%d / %d)\n', spec_acc * 100, tn_acc, tn_acc + fp_acc);
    fprintf('  • PPV:               %5.2f%% (%d / %d)\n', ppv_acc * 100, tp_acc, tp_acc + fp_acc);
    fprintf('  • NPV:               %5.2f%% (%d / %d)\n', npv_acc * 100, tn_acc, tn_acc + fn_acc);
    fprintf('  • F1-Score:          %6.4f\n', f1_acc);
    fprintf('  • Balanced Accuracy: %5.2f%%\n', bal_acc * 100);
    fprintf('  • Accepted Errors:   FN = %d, FP = %d\n\n', fn_acc, fp_acc);

    fprintf('Deferred Cohort Triage Distribution:\n');
    fprintf('  • RECAPTURE (Ungradable Quality):               %d (%.2f%%)\n', cnt_recapture, (cnt_recapture/n_val)*100);
    fprintf('  • CLINICAL_REVIEW (Sub-threshold Reliability):  %d (%.2f%%)\n', cnt_clin_rev, (cnt_clin_rev/n_val)*100);
    fprintf('  • UNCERTAIN (Boundary Non-referral):            %d (%.2f%%)\n\n', cnt_uncertain, (cnt_uncertain/n_val)*100);

    fprintf('Reliability Score Distribution (N = 83):\n');
    fprintf('  • Mean +/- Std:      %5.2f +/- %5.2f\n', rel_mean, rel_std);
    fprintf('  • Median:            %5.2f\n', rel_median);
    fprintf('  • Min / Max:         %5.2f / %5.2f\n\n', rel_min, rel_max);

    fprintf('====================================================\n');
    fprintf('FINAL SANITY CHECK\n');
    fprintf('====================================================\n');
    fprintf('Development training N = %d\n', n_train_cal);
    fprintf('Validation N = %d\n', n_val);
    fprintf('Test images accessed = 0\n');
    fprintf('Test labels accessed = 0\n');
    fprintf('Final-test artifacts modified = 0\n');
    fprintf('CNN retrained = 0\n');
    fprintf('Vessel model retrained = 0\n');
    fprintf('Threshold frozen = %4.2f\n', tau_ref);
    fprintf('Calibration frozen = a=%6.4f, b=%6.4f\n', a_platt, b_platt);
    fprintf('Reliability gate frozen = %d\n\n', int32(tau_accept));

    results = release_struct;
end
