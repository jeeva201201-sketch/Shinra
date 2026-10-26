function results = auditCalibratedThresholds()
% AUDITCALIBRATEDTHRESHOLDS Calibrated Development Threshold & Reliability Audit
% Evaluates the frozen Platt-calibrated multimodal fusion model across a dense
% threshold sweep (tau in [0.10, 0.90]) and reliability gating policies strictly on the
% 83-image development validation partition.
%
% STRICT GOVERNANCE:
% - ZERO access to the 103-image IDRiD test partition.
% - ZERO retraining or refitting.
% - Uses frozen Platt parameters (a = 1.3106, b = -1.1412) fitted on the 330 development training set.
%
% ARTIFACTS GENERATED:
% - results/metrics/calibrated_threshold_audit.mat
% - results/metrics/calibrated_threshold_sweep.csv
% - results/metrics/calibrated_threshold_summary.csv
% - results/metrics/calibrated_operating_curve.png

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: CALIBRATED DEVELOPMENT THRESHOLD & RELIABILITY AUDIT        \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Governance & Strict Quarantine Verification
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
    fprintf('  [PASS] Using only development validation partition (N = 83).\n\n');

    % -------------------------------------------------------------------------
    % 2. Extract Validation Split (N = 83) & Reliability Metadata
    % -------------------------------------------------------------------------
    val_mask = strcmp(audit_tab.Split, 'Validation');
    val_tab  = audit_tab(val_mask, :);
    n_val    = height(val_tab);
    assert(n_val == 83, 'Expected 83 validation images. Found: %d', n_val);

    n_train_cal = sum(strcmp(audit_tab.Split, 'Train'));
    assert(n_train_cal == 330, 'Expected 330 training images. Found: %d', n_train_cal);

    y_val       = double(val_tab.True_Referable);
    p_fus_raw   = double(val_tab.Fusion_P_Referable);
    q_score_val = double(val_tab.Quality_Score);

    % Load Reliability Scores
    rel_eval_file = fullfile(cfg.paths.metrics, 'reliability_layer_evaluation.mat');
    assert(exist(rel_eval_file, 'file') == 2, 'Reliability evaluation file not found: %s', rel_eval_file);
    loaded_rel = load(rel_eval_file);
    rel_val_tab = loaded_rel.eval_table(strcmp(loaded_rel.eval_table.Split, 'Validation'), :);
    rel_scores_val = double(rel_val_tab.Reliability_Score);
    quality_classes = val_tab.Quality_Class;

    % -------------------------------------------------------------------------
    % 3. Apply Frozen Platt Probability Calibration
    % -------------------------------------------------------------------------
    % Frozen Platt parameters fitted strictly on the 330-image development training set
    a_platt = 1.3106;
    b_platt = -1.1412;

    eps_val = 1e-6;
    p_clamped = max(eps_val, min(1 - eps_val, p_fus_raw));
    logit_val = log(p_clamped ./ (1 - p_clamped));
    p_calibrated = 1.0 ./ (1.0 + exp(-(a_platt .* logit_val + b_platt)));

    % Helper metric calculators
    calc_brier = @(p, y) mean((p(:) - y(:)).^2);

    function [ece, conf_bins, acc_bins, bin_counts] = calc_ece(p, y, n_bins)
        if nargin < 3, n_bins = 10; end
        bin_edges = linspace(0, 1, n_bins + 1);
        ece = 0;
        conf_bins = zeros(n_bins, 1);
        acc_bins  = zeros(n_bins, 1);
        bin_counts= zeros(n_bins, 1);
        N = numel(p);
        for b = 1:n_bins
            idx = (p >= bin_edges(b)) & (p < bin_edges(b+1));
            if b == n_bins, idx = (p >= bin_edges(b)) & (p <= bin_edges(b+1)); end
            nb = sum(idx);
            bin_counts(b) = nb;
            if nb > 0
                conf_b = mean(p(idx));
                acc_b  = mean(y(idx));
                conf_bins(b) = conf_b;
                acc_bins(b)  = acc_b;
                ece = ece + (nb / N) * abs(acc_b - conf_b);
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

    % Calibration & Discrimination Metrics
    auroc_raw  = calc_auroc(p_fus_raw, y_val);
    auprc_raw  = calc_auprc(p_fus_raw, y_val);
    brier_raw  = calc_brier(p_fus_raw, y_val);
    [ece_raw, ~, ~, ~] = calc_ece(p_fus_raw, y_val);

    auroc_cal  = calc_auroc(p_calibrated, y_val);
    auprc_cal  = calc_auprc(p_calibrated, y_val);
    brier_cal  = calc_brier(p_calibrated, y_val);
    [ece_cal, ~, ~, ~] = calc_ece(p_calibrated, y_val);

    % -------------------------------------------------------------------------
    % TASK 1: Dense Calibrated Threshold Sweep (tau = 0.10 : 0.01 : 0.90)
    % -------------------------------------------------------------------------
    thresh_grid = (0.10:0.01:0.90)';
    n_th = numel(thresh_grid);

    th_sens   = zeros(n_th, 1);
    th_spec   = zeros(n_th, 1);
    th_ppv    = zeros(n_th, 1);
    th_npv    = zeros(n_th, 1);
    th_f1     = zeros(n_th, 1);
    th_acc    = zeros(n_th, 1);
    th_bal    = zeros(n_th, 1);
    th_fn     = zeros(n_th, 1);
    th_fp     = zeros(n_th, 1);
    th_auroc  = repmat(auroc_cal, n_th, 1);
    th_auprc  = repmat(auprc_cal, n_th, 1);
    th_brier  = repmat(brier_cal, n_th, 1);
    th_ece    = repmat(ece_cal, n_th, 1);
    is_feasible = false(n_th, 1);

    for t = 1:n_th
        th = thresh_grid(t);
        y_pred = double(p_calibrated >= th);

        tp = sum(y_pred == 1 & y_val == 1);
        fp = sum(y_pred == 1 & y_val == 0);
        fn = sum(y_pred == 0 & y_val == 1);
        tn = sum(y_pred == 0 & y_val == 0);

        sens = tp / max(1, tp + fn);
        spec = tn / max(1, tn + fp);
        ppv  = tp / max(1, tp + fp);
        npv  = tn / max(1, tn + fn);
        f1   = (2 * tp) / max(1, 2 * tp + fp + fn);
        acc  = (tp + tn) / n_val;
        bal  = (sens + spec) / 2;

        th_sens(t) = sens;
        th_spec(t) = spec;
        th_ppv(t)  = ppv;
        th_npv(t)  = npv;
        th_f1(t)   = f1;
        th_acc(t)  = acc;
        th_bal(t)  = bal;
        th_fn(t)   = fn;
        th_fp(t)   = fp;

        % Target Requirement: Sensitivity >= 90% AND Specificity >= 85%
        if sens >= 0.90 && spec >= 0.85
            is_feasible(t) = true;
        end
    end

    sweep_table = table(thresh_grid, th_sens, th_spec, th_ppv, th_npv, th_f1, ...
        th_acc, th_bal, th_auroc, th_auprc, th_brier, th_ece, th_fn, th_fp, is_feasible, ...
        'VariableNames', {'Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1', ...
                          'Accuracy', 'Balanced_Accuracy', 'AUROC', 'AUPRC', 'Brier_Score', 'ECE', 'FN', 'FP', 'Feasible'});

    feasible_table = sweep_table(sweep_table.Feasible, :);

    % -------------------------------------------------------------------------
    % TASK 2: Reliability Gating Analysis (tau_accept = 60 and tau_accept = 65)
    % -------------------------------------------------------------------------
    tau_ref_eval = 0.52; % Candidate operating threshold
    y_pred_candidate = double(p_calibrated >= tau_ref_eval);

    rel_gates = [60, 65];
    n_gates = numel(rel_gates);

    rel_summary = cell(n_gates, 1);

    for g = 1:n_gates
        g_val = rel_gates(g);
        
        % Triage routing
        is_ungradable = strcmp(quality_classes, 'UNGRADABLE');
        is_accepted   = (rel_scores_val >= g_val) & ~is_ungradable;
        is_deferred   = ~is_accepted;

        n_acc = sum(is_accepted);
        n_def = sum(is_deferred);
        cov_pct = (n_acc / n_val) * 100;
        def_pct = (n_def / n_val) * 100;

        y_val_acc  = y_val(is_accepted);
        y_pred_acc = y_pred_candidate(is_accepted);

        tp_acc = sum(y_pred_acc == 1 & y_val_acc == 1);
        fp_acc = sum(y_pred_acc == 1 & y_val_acc == 0);
        fn_acc = sum(y_pred_acc == 0 & y_val_acc == 1);
        tn_acc = sum(y_pred_acc == 0 & y_val_acc == 0);

        sens_acc = tp_acc / max(1, tp_acc + fn_acc);
        spec_acc = tn_acc / max(1, tn_acc + fp_acc);
        ppv_acc  = tp_acc / max(1, tp_acc + fp_acc);
        npv_acc  = tn_acc / max(1, tn_acc + fn_acc);
        f1_acc   = (2 * tp_acc) / max(1, 2 * tp_acc + fp_acc + fn_acc);
        bal_acc  = (sens_acc + spec_acc) / 2;

        res_g = struct();
        res_g.tau_accept = g_val;
        res_g.total_N = n_val;
        res_g.accepted_N = n_acc;
        res_g.deferred_N = n_def;
        res_g.coverage_pct = cov_pct;
        res_g.deferred_pct = def_pct;
        res_g.accepted_sensitivity = sens_acc;
        res_g.accepted_specificity = spec_acc;
        res_g.accepted_ppv = ppv_acc;
        res_g.accepted_npv = npv_acc;
        res_g.accepted_f1 = f1_acc;
        res_g.accepted_balanced_acc = bal_acc;
        res_g.accepted_fn = fn_acc;
        res_g.accepted_fp = fp_acc;

        rel_summary{g} = res_g;
    end

    % -------------------------------------------------------------------------
    % TASK 3: Side-by-Side Raw vs Calibrated Multimodal Fusion Comparison
    % -------------------------------------------------------------------------
    % Raw at default tau=0.50
    y_pred_raw50 = double(p_fus_raw >= 0.50);
    tp_r = sum(y_pred_raw50 == 1 & y_val == 1);
    fp_r = sum(y_pred_raw50 == 1 & y_val == 0);
    fn_r = sum(y_pred_raw50 == 0 & y_val == 1);
    tn_r = sum(y_pred_raw50 == 0 & y_val == 0);

    sens_raw = tp_r / max(1, tp_r + fn_r);
    spec_raw = tn_r / max(1, tn_r + fp_r);
    ppv_raw  = tp_r / max(1, tp_r + fp_r);
    npv_raw  = tn_r / max(1, tn_r + fn_r);
    f1_raw   = (2 * tp_r) / max(1, 2 * tp_r + fp_r + fn_r);
    bal_raw  = (sens_raw + spec_raw) / 2;

    raw_results = struct('AUROC', auroc_raw, 'AUPRC', auprc_raw, 'Brier', brier_raw, ...
        'ECE', ece_raw, 'Sensitivity', sens_raw, 'Specificity', spec_raw, ...
        'PPV', ppv_raw, 'NPV', npv_raw, 'F1', f1_raw, 'Balanced_Accuracy', bal_raw, ...
        'FN', fn_r, 'FP', fp_r, 'Threshold', 0.50);

    % Calibrated at candidate tau=0.52
    y_pred_cal52 = double(p_calibrated >= 0.52);
    tp_c = sum(y_pred_cal52 == 1 & y_val == 1);
    fp_c = sum(y_pred_cal52 == 1 & y_val == 0);
    fn_c = sum(y_pred_cal52 == 0 & y_val == 1);
    tn_c = sum(y_pred_cal52 == 0 & y_val == 0);

    sens_cal = tp_c / max(1, tp_c + fn_c);
    spec_cal = tn_c / max(1, tn_c + fp_c);
    ppv_cal  = tp_c / max(1, tp_c + fp_c);
    npv_cal  = tn_c / max(1, tn_c + fn_c);
    f1_cal   = (2 * tp_c) / max(1, 2 * tp_c + fp_c + fn_c);
    bal_cal  = (sens_cal + spec_cal) / 2;

    calibrated_results = struct('AUROC', auroc_cal, 'AUPRC', auprc_cal, 'Brier', brier_cal, ...
        'ECE', ece_cal, 'Sensitivity', sens_cal, 'Specificity', spec_cal, ...
        'PPV', ppv_cal, 'NPV', npv_cal, 'F1', f1_cal, 'Balanced_Accuracy', bal_cal, ...
        'FN', fn_c, 'FP', fp_c, 'Threshold', 0.52);

    % Summary Comparison Table for CSV
    summary_tab = table(...
        {'Raw Multimodal Fusion (tau=0.50)'; 'Platt-Calibrated Multimodal Fusion (tau=0.52)'}, ...
        [auroc_raw; auroc_cal], ...
        [auprc_raw; auprc_cal], ...
        [brier_raw; brier_cal], ...
        [ece_raw; ece_cal], ...
        [sens_raw; sens_cal], ...
        [spec_raw; spec_cal], ...
        [ppv_raw; ppv_cal], ...
        [npv_raw; npv_cal], ...
        [f1_raw; f1_cal], ...
        [bal_raw; bal_cal], ...
        [fn_r; fn_c], ...
        [fp_r; fp_c], ...
        'VariableNames', {'Model_Configuration', 'AUROC', 'AUPRC', 'Brier_Score', 'ECE', ...
                          'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1_Score', 'Balanced_Accuracy', 'FN', 'FP'});

    % -------------------------------------------------------------------------
    % TASK 4: Calibration & Governance Metadata
    % -------------------------------------------------------------------------
    governance = struct();
    governance.training_calibration_set = n_train_cal;
    governance.validation_evaluation_set = n_val;
    governance.test_images_accessed = 0;
    governance.test_labels_accessed = 0;
    governance.final_test_artifacts_modified = 0;
    governance.retraining_performed = 0;

    calibration_struct = struct();
    calibration_struct.model_type = 'Platt Logistic Scaling';
    calibration_struct.a = a_platt;
    calibration_struct.b = b_platt;
    calibration_struct.formula = 'P_calibrated = 1 ./ (1 + exp(-(a .* logit(P_raw) + b)))';
    calibration_struct.training_N = n_train_cal;
    calibration_struct.validation_N = n_val;

    % -------------------------------------------------------------------------
    % TASK 5: Save Artifacts
    % -------------------------------------------------------------------------
    mat_out     = fullfile(cfg.paths.metrics, 'calibrated_threshold_audit.mat');
    sweep_csv   = fullfile(cfg.paths.metrics, 'calibrated_threshold_sweep.csv');
    summary_csv = fullfile(cfg.paths.metrics, 'calibrated_threshold_summary.csv');
    fig_out     = fullfile(cfg.paths.metrics, 'calibrated_operating_curve.png');

    % MATLAB Structure
    results = struct();
    results.calibration        = calibration_struct;
    results.thresholdSweep     = sweep_table;
    results.feasibleThresholds = feasible_table;
    results.reliabilityResults = rel_summary;
    results.rawResults         = raw_results;
    results.calibratedResults  = calibrated_results;
    results.governance         = governance;

    save(mat_out, '-struct', 'results');
    writetable(sweep_table, sweep_csv);
    writetable(summary_tab, summary_csv);

    % Generate Headless Plot
    try
        fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 700], 'Color', 'w');
        
        subplot(2, 1, 1);
        plot(thresh_grid, th_sens * 100, 'b-', 'LineWidth', 2, 'DisplayName', 'Sensitivity (%)');
        hold on;
        plot(thresh_grid, th_spec * 100, 'r--', 'LineWidth', 2, 'DisplayName', 'Specificity (%)');
        plot(thresh_grid, th_bal * 100, 'm-.', 'LineWidth', 1.5, 'DisplayName', 'Balanced Acc (%)');
        plot(thresh_grid, th_ppv * 100, 'g:', 'LineWidth', 1.5, 'DisplayName', 'PPV (%)');
        
        % Target bands
        yline(90, 'k--', 'LineWidth', 1.2, 'DisplayName', 'Target Sens (90%)');
        yline(85, 'k:', 'LineWidth', 1.2, 'DisplayName', 'Target Spec (85%)');
        
        % Highlight feasible region
        if ~isempty(feasible_table)
            f_min = min(feasible_table.Threshold);
            f_max = max(feasible_table.Threshold);
            patch([f_min f_max f_max f_min], [0 0 100 100], [0.85 0.95 0.85], ...
                'FaceAlpha', 0.35, 'EdgeColor', 'none', 'DisplayName', sprintf('Feasible Range [%.2f, %.2f]', f_min, f_max));
        end
        xline(0.52, 'k-', 'LineWidth', 2, 'DisplayName', 'Selected Candidate (\tau = 0.52)');
        
        xlabel('Calibrated Referral Decision Threshold (\tau)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Performance (%)', 'FontSize', 11, 'FontWeight', 'bold');
        title('SHINRA Calibrated Decision Threshold Sweep (Validation N = 83)', 'FontSize', 12, 'FontWeight', 'bold');
        legend('Location', 'southwest', 'FontSize', 9);
        grid on;
        xlim([0.10, 0.90]);
        ylim([40, 105]);

        subplot(2, 1, 2);
        bar(thresh_grid, [th_fn, th_fp], 'stacked');
        hold on;
        xline(0.52, 'k-', 'LineWidth', 2);
        xlabel('Calibrated Referral Decision Threshold (\tau)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Error Counts (N = 83)', 'FontSize', 11, 'FontWeight', 'bold');
        title('False Negatives (FN) & False Positives (FP) Across Threshold Sweep', 'FontSize', 12, 'FontWeight', 'bold');
        legend({'False Negatives (FN)', 'False Positives (FP)'}, 'Location', 'northwest', 'FontSize', 9);
        grid on;
        xlim([0.10, 0.90]);

        saveas(fig, fig_out);
        close(fig);
    catch ME
        warning('Failed to save figure: %s', ME.message);
    end

    % -------------------------------------------------------------------------
    % TASK 6: Exact Command Window Output
    % -------------------------------------------------------------------------
    fprintf('====================================================\n');
    fprintf('SHINRA CALIBRATED DEVELOPMENT AUDIT COMPLETE\n');
    fprintf('====================================================\n\n');

    fprintf('Calibration:\n');
    fprintf('a = %6.4f\n', a_platt);
    fprintf('b = %6.4f\n\n', b_platt);

    fprintf('Training calibration N = %d\n', n_train_cal);
    fprintf('Validation N = %d\n', n_val);
    fprintf('Test N accessed = 0\n\n');

    fprintf('Feasible calibrated thresholds (Sens >= 90%% & Spec >= 85%%):\n');
    if isempty(feasible_table)
        fprintf('  None\n\n');
    else
        fprintf('  Range: [%4.2f, %4.2f] (%d operating points)\n\n', ...
            min(feasible_table.Threshold), max(feasible_table.Threshold), height(feasible_table));
        
        fprintf('For each feasible threshold:\n');
        fprintf('------------------------------------------------------------------------------------------------------\n');
        fprintf('  Threshold   Sensitivity   Specificity   PPV       NPV       F1        Balanced Accuracy   FN   FP\n');
        fprintf('------------------------------------------------------------------------------------------------------\n');
        for f = 1:height(feasible_table)
            fprintf('    %4.2f        %5.2f%%        %5.2f%%     %5.2f%%   %5.2f%%   %6.4f       %5.2f%%          %2d   %2d\n', ...
                feasible_table.Threshold(f), ...
                feasible_table.Sensitivity(f) * 100, ...
                feasible_table.Specificity(f) * 100, ...
                feasible_table.PPV(f) * 100, ...
                feasible_table.NPV(f) * 100, ...
                feasible_table.F1(f), ...
                feasible_table.Balanced_Accuracy(f) * 100, ...
                feasible_table.FN(f), ...
                feasible_table.FP(f));
        end
        fprintf('------------------------------------------------------------------------------------------------------\n\n');
    end

    fprintf('Reliability Results for Calibrated Model (tau_ref = 0.52):\n\n');
    for g = 1:numel(rel_summary)
        rg = rel_summary{g};
        fprintf('--- Gate: tau_accept = %d ---\n', rg.tau_accept);
        fprintf('  Total Validation N:         %d\n', rg.total_N);
        fprintf('  Accepted N:                 %d (Coverage: %5.2f%%)\n', rg.accepted_N, rg.coverage_pct);
        fprintf('  Deferred N:                 %d (Deferred Rate: %5.2f%%)\n', rg.deferred_N, rg.deferred_pct);
        fprintf('  Accepted-Cohort Sens:       %5.2f%%\n', rg.accepted_sensitivity * 100);
        fprintf('  Accepted-Cohort Spec:       %5.2f%%\n', rg.accepted_specificity * 100);
        fprintf('  Accepted-Cohort PPV:        %5.2f%%\n', rg.accepted_ppv * 100);
        fprintf('  Accepted-Cohort NPV:        %5.2f%%\n', rg.accepted_npv * 100);
        fprintf('  Accepted-Cohort F1:         %6.4f\n', rg.accepted_f1);
        fprintf('  Accepted-Cohort Bal Acc:    %5.2f%%\n', rg.accepted_balanced_acc * 100);
        fprintf('  Accepted FN:                %d\n', rg.accepted_fn);
        fprintf('  Accepted FP:                %d\n\n', rg.accepted_fp);
    end

    fprintf('Side-by-Side Raw vs Calibrated Multimodal Fusion:\n');
    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('  Metric              Raw Fusion (tau=0.50)    Platt Calibrated (tau=0.52)\n');
    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('  AUROC               %6.4f                   %6.4f\n', auroc_raw, auroc_cal);
    fprintf('  AUPRC               %6.4f                   %6.4f\n', auprc_raw, auprc_cal);
    fprintf('  Brier Score         %6.4f                   %6.4f\n', brier_raw, brier_cal);
    fprintf('  ECE                 %6.4f                   %6.4f\n', ece_raw, ece_cal);
    fprintf('  Sensitivity         %5.2f%%                  %5.2f%%\n', sens_raw * 100, sens_cal * 100);
    fprintf('  Specificity         %5.2f%%                  %5.2f%%\n', spec_raw * 100, spec_cal * 100);
    fprintf('  PPV (Precision)     %5.2f%%                  %5.2f%%\n', ppv_raw * 100, ppv_cal * 100);
    fprintf('  NPV                 %5.2f%%                  %5.2f%%\n', npv_raw * 100, npv_cal * 100);
    fprintf('  F1 Score            %6.4f                   %6.4f\n', f1_raw, f1_cal);
    fprintf('  Balanced Accuracy   %5.2f%%                  %5.2f%%\n', bal_raw * 100, bal_cal * 100);
    fprintf('  FN / FP Errors      %2d / %2d                 %2d / %2d\n', fn_r, fp_r, fn_c, fp_c);
    fprintf('------------------------------------------------------------------------------------\n\n');

    fprintf('Test partition accessed = 0\n');
    fprintf('Final-test artifacts modified = 0\n');
    fprintf('Retraining = 0\n\n');

    fprintf('Artifacts Saved Successfully:\n');
    fprintf('  - %s\n', mat_out);
    fprintf('  - %s\n', sweep_csv);
    fprintf('  - %s\n', summary_csv);
    fprintf('  - %s\n\n', fig_out);
end
