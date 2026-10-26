function results = evaluateDevelopmentCandidates()
% EVALUATEDEVELOPMENTCANDIDATES Evaluates Development Candidate Configurations on Validation (N=83)
% Compares Historical Baseline vs. Development Candidates under Platt calibration and operating points.
%
% STRICT GOVERNANCE & SAFETY SAFEGUARDS:
% 1. The 103-image IDRiD held-out test partition is strictly blocked and quarantined.
% 2. Uses ONLY development validation partition (N = 83) for evaluation.
% 3. Zero test set inference, zero test tuning, zero overwriting of final-test files.
%
% OUTPUT ARTIFACTS:
% - results/metrics/development_optimized_candidate.mat
% - results/metrics/development_optimized_candidate.csv
% - results/metrics/development_optimized_summary.csv
% - results/metrics/development_operating_curve.png

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: DEVELOPMENT CONFIGURATION EVALUATION & CANDIDATE COMPARISON  \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Quarantine Check
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
    % 2. Extract Validation Split (N = 83 Images)
    % -------------------------------------------------------------------------
    val_mask = strcmp(audit_tab.Split, 'Validation');
    val_tab  = audit_tab(val_mask, :);
    n_val    = height(val_tab);
    assert(n_val == 83, 'Expected 83 validation images. Found: %d', n_val);

    image_ids   = val_tab.ImageID;
    y_val       = double(val_tab.True_Referable);
    p_cnn_val   = val_tab.CNN_P_Referable;
    p_fus_val   = val_tab.Fusion_P_Referable;
    q_class_val = val_tab.Quality_Class;
    q_score_val = val_tab.Quality_Score;

    % Load Reliability Metadata
    rel_eval_file = fullfile(cfg.paths.metrics, 'reliability_layer_evaluation.mat');
    loaded_rel = load(rel_eval_file);
    rel_val_tab = loaded_rel.eval_table(strcmp(loaded_rel.eval_table.Split, 'Validation'), :);
    rel_scores_val = rel_val_tab.Reliability_Score;

    % -------------------------------------------------------------------------
    % 3. Apply Platt Calibration (Fitted strictly on 330 Train partition)
    % -------------------------------------------------------------------------
    eps_val = 1e-6;
    slope_platt = cfg.calibration.platt.slope;        % 1.3106
    intercept_platt = cfg.calibration.platt.intercept; % -1.1412

    p_clamped = max(eps_val, min(1 - eps_val, p_fus_val));
    logit_val = log(p_clamped ./ (1 - p_clamped));
    p_platt_val = 1.0 ./ (1.0 + exp(-(slope_platt * logit_val + intercept_platt)));

    % Helper Functions for Discrimination and Calibration
    calc_brier = @(p, y) mean((p(:) - y(:)).^2);
    
    function [ece, conf_b, acc_b] = calc_ece_local(p, y, n_bins)
        if nargin < 3, n_bins = 10; end
        bin_edges = linspace(0, 1, n_bins + 1);
        ece = 0; conf_b = zeros(n_bins, 1); acc_b = zeros(n_bins, 1);
        N = numel(p);
        for b = 1:n_bins
            idx = (p >= bin_edges(b)) & (p < bin_edges(b+1));
            if b == n_bins, idx = (p >= bin_edges(b)) & (p <= bin_edges(b+1)); end
            nb = sum(idx);
            if nb > 0
                cb = mean(p(idx)); ab = mean(y(idx));
                conf_b(b) = cb; acc_b(b) = ab;
                ece = ece + (nb / N) * abs(ab - cb);
            end
        end
    end

    function auroc = calc_auroc_local(p, y)
        [~, s_idx] = sort(p, 'descend');
        sorted_y = y(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        tpr = tps / max(1, sum(sorted_y == 1));
        fpr = fps / max(1, sum(sorted_y == 0));
        auroc = trapz([0; fpr], [0; tpr]);
    end

    function auprc = calc_auprc_local(p, y)
        [~, s_idx] = sort(p, 'descend');
        sorted_y = y(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        rec = tps / max(1, sum(sorted_y == 1));
        prec= tps ./ max(1, tps + fps);
        auprc = trapz([0; rec], [1; prec]);
    end

    % -------------------------------------------------------------------------
    % 4. Evaluate Three Configurations Side-by-Side on Validation (N = 83)
    % -------------------------------------------------------------------------
    % Config A: Historical Baseline (tau_ref = 0.50, tau_rel = 60.0, raw fusion)
    % Config B: Development Candidate 1 (tau_ref = 0.52, tau_rel = 60.0, platt calibrated)
    % Config C: Development Candidate 2 (tau_ref = 0.52, tau_rel = 65.0, platt calibrated)

    configs = {
        'Config A: Historical Baseline (tau=0.50, tau_rel=60)',             0.50, 60.0, p_fus_val;
        'Config B: Dev Candidate 1 (tau=0.52, tau_rel=60, Platt Calib)',    0.52, 60.0, p_platt_val;
        'Config C: Dev Candidate 2 (tau=0.52, tau_rel=65, Platt Calib)',    0.52, 65.0, p_platt_val;
    };

    n_cfg = size(configs, 1);

    % Metric storage matrices
    all_sens = zeros(n_cfg, 1); all_spec = zeros(n_cfg, 1); all_ppv = zeros(n_cfg, 1);
    all_npv  = zeros(n_cfg, 1); all_f1   = zeros(n_cfg, 1); all_acc = zeros(n_cfg, 1);
    all_bal  = zeros(n_cfg, 1); all_fn   = zeros(n_cfg, 1); all_fp  = zeros(n_cfg, 1);
    all_auroc= zeros(n_cfg, 1); all_auprc= zeros(n_cfg, 1); all_brier=zeros(n_cfg, 1); all_ece = zeros(n_cfg, 1);

    acc_sens = zeros(n_cfg, 1); acc_spec = zeros(n_cfg, 1); acc_ppv = zeros(n_cfg, 1);
    acc_npv  = zeros(n_cfg, 1); acc_f1   = zeros(n_cfg, 1); acc_acc = zeros(n_cfg, 1);
    acc_bal  = zeros(n_cfg, 1); acc_fn   = zeros(n_cfg, 1); acc_fp  = zeros(n_cfg, 1);
    acc_cov  = zeros(n_cfg, 1); acc_n    = zeros(n_cfg, 1); def_n   = zeros(n_cfg, 1); def_rate = zeros(n_cfg, 1);

    for c = 1:n_cfg
        th_r   = configs{c, 2};
        th_rel = configs{c, 3};
        p_eval = configs{c, 4};

        % Discrimination & Calibration Metrics
        all_auroc(c) = calc_auroc_local(p_eval, y_val);
        all_auprc(c) = calc_auprc_local(p_eval, y_val);
        all_brier(c) = calc_brier(p_eval, y_val);
        [all_ece(c), ~, ~] = calc_ece_local(p_eval, y_val);

        % All 83 Cohort Metrics
        y_all_pr = double(p_eval >= th_r);
        tp = sum(y_all_pr == 1 & y_val == 1);
        fp = sum(y_all_pr == 1 & y_val == 0);
        fn = sum(y_all_pr == 0 & y_val == 1);
        tn = sum(y_all_pr == 0 & y_val == 0);

        all_sens(c) = tp / max(1, tp + fn);
        all_spec(c) = tn / max(1, tn + fp);
        all_ppv(c)  = tp / max(1, tp + fp);
        all_npv(c)  = tn / max(1, tn + fn);
        all_f1(c)   = (2 * tp) / max(1, 2 * tp + fp + fn);
        all_acc(c)  = (tp + tn) / n_val;
        all_bal(c)  = (all_sens(c) + all_spec(c)) / 2;
        all_fn(c)   = fn;
        all_fp(c)   = fp;

        % Accepted Autonomous Cohort Metrics
        acc_mask = (rel_scores_val >= th_rel) & ~strcmp(q_class_val, 'UNGRADABLE');
        n_a = sum(acc_mask);
        n_d = n_val - n_a;
        acc_n(c)    = n_a;
        def_n(c)    = n_d;
        acc_cov(c)  = (n_a / n_val) * 100;
        def_rate(c) = (n_d / n_val) * 100;

        if n_a > 0
            y_acc_tr = y_val(acc_mask);
            y_acc_pr = double(p_eval(acc_mask) >= th_r);
            tp_a = sum(y_acc_pr == 1 & y_acc_tr == 1);
            fp_a = sum(y_acc_pr == 1 & y_acc_tr == 0);
            fn_a = sum(y_acc_pr == 0 & y_acc_tr == 1);
            tn_a = sum(y_acc_pr == 0 & y_acc_tr == 0);

            acc_sens(c) = tp_a / max(1, tp_a + fn_a);
            acc_spec(c) = tn_a / max(1, tn_a + fp_a);
            acc_ppv(c)  = tp_a / max(1, tp_a + fp_a);
            acc_npv(c)  = tn_a / max(1, tn_a + fn_a);
            acc_f1(c)   = (2 * tp_a) / max(1, 2 * tp_a + fp_a + fn_a);
            acc_acc(c)  = (tp_a + tn_a) / n_a;
            acc_bal(c)  = (acc_sens(c) + acc_spec(c)) / 2;
            acc_fn(c)   = fn_a;
            acc_fp(c)   = fp_a;
        else
            acc_sens(c) = NaN; acc_spec(c) = NaN; acc_ppv(c) = NaN; acc_npv(c) = NaN;
            acc_f1(c) = NaN; acc_acc(c) = NaN; acc_bal(c) = NaN; acc_fn(c) = 0; acc_fp(c) = 0;
        end
    end

    % -------------------------------------------------------------------------
    % 5. Build Master Per-Image Evaluation Table for Candidate
    % -------------------------------------------------------------------------
    cand_actions = cell(n_val, 1);
    cand_readiness = false(n_val, 1);
    for i = 1:n_val
        if strcmp(q_class_val{i}, 'UNGRADABLE')
            cand_actions{i} = 'RECAPTURE';
            cand_readiness(i) = false;
        elseif rel_scores_val(i) < 60.0 && (p_platt_val(i) >= 0.52)
            cand_actions{i} = 'CLINICAL_REVIEW';
            cand_readiness(i) = false;
        elseif rel_scores_val(i) < 60.0
            cand_actions{i} = 'UNCERTAIN';
            cand_readiness(i) = false;
        else
            cand_actions{i} = 'ACCEPT_SCREENING';
            cand_readiness(i) = true;
        end
    end

    per_image_table = table(...
        image_ids, val_tab.ImagePath, val_tab.True_DR_Grade, y_val, ...
        q_class_val, q_score_val, p_cnn_val, p_fus_val, p_platt_val, ...
        double(p_platt_val >= 0.52), rel_scores_val, cand_actions, cand_readiness, ...
        'VariableNames', {...
            'ImageID', 'ImagePath', 'True_DR_Grade', 'True_Referable', ...
            'Quality_Class', 'Quality_Score', 'CNN_P_Referable', 'Fused_P_Referable_Raw', ...
            'Fused_P_Referable_Platt_Calibrated', 'Predicted_Referable_Candidate', ...
            'Reliability_Score', 'Recommended_Action_Candidate', 'Decision_Readiness_Candidate'});

    % -------------------------------------------------------------------------
    % 6. Save MAT & CSV Artifacts
    % -------------------------------------------------------------------------
    metrics_dir = cfg.paths.metrics;
    out_mat = fullfile(metrics_dir, 'development_optimized_candidate.mat');
    out_csv = fullfile(metrics_dir, 'development_optimized_candidate.csv');
    out_sum_csv = fullfile(metrics_dir, 'development_optimized_summary.csv');
    out_fig = fullfile(metrics_dir, 'development_operating_curve.png');

    summary_comparison_table = table(...
        configs(:, 1), ...
        all_sens, all_spec, all_ppv, all_npv, all_f1, all_bal, all_auroc, all_brier, all_ece, all_fn, all_fp, ...
        acc_cov, def_rate, acc_sens, acc_spec, acc_ppv, acc_npv, acc_f1, acc_bal, acc_fn, acc_fp, ...
        'VariableNames', {...
            'Configuration', ...
            'All_Sensitivity', 'All_Specificity', 'All_PPV', 'All_NPV', 'All_F1', 'All_BalAcc', 'All_AUROC', 'All_Brier_Score', 'All_ECE', 'All_FN', 'All_FP', ...
            'Accepted_Coverage_Pct', 'Deferred_Rate_Pct', 'Accepted_Sensitivity', 'Accepted_Specificity', 'Accepted_PPV', 'Accepted_NPV', 'Accepted_F1', 'Accepted_BalAcc', 'Accepted_FN', 'Accepted_FP'});

    save(out_mat, 'per_image_table', 'summary_comparison_table', 'cfg');
    writetable(per_image_table, out_csv);
    writetable(summary_comparison_table, out_sum_csv);

    % Save Operating Curve Figure to results/metrics/
    f_op = figure('Visible', 'off', 'Position', [100, 100, 700, 500], 'Color', 'w');
    th_grid = 0.10:0.01:0.90;
    sens_curve = arrayfun(@(t) sum(p_platt_val >= t & y_val == 1) / sum(y_val == 1) * 100, th_grid);
    spec_curve = arrayfun(@(t) sum(p_platt_val < t & y_val == 0) / sum(y_val == 0) * 100, th_grid);
    bal_curve  = (sens_curve + spec_curve) / 2;
    plot(th_grid, sens_curve, 'b-', 'LineWidth', 2, 'DisplayName', 'Sensitivity (%)'); hold on;
    plot(th_grid, spec_curve, 'r-', 'LineWidth', 2, 'DisplayName', 'Specificity (%)');
    plot(th_grid, bal_curve, 'g--', 'LineWidth', 1.8, 'DisplayName', 'Balanced Accuracy (%)');
    xline(0.52, 'k:', 'LineWidth', 2, 'DisplayName', 'Candidate Threshold (\tau = 0.52)');
    yline(90, 'b:', 'LineWidth', 1.2, 'DisplayName', 'Target Sens (90%)');
    yline(85, 'r:', 'LineWidth', 1.2, 'DisplayName', 'Target Spec (85%)');
    grid on; xlabel('Platt-Calibrated Referral Probability Threshold (\tau)'); ylabel('Validation Metric Value (%)');
    title('SHINRA Development Operating Trade-Off Curve (Platt Calibrated, N=83)', 'FontSize', 12, 'FontWeight', 'bold');
    legend('Location', 'south');
    save_figure_safe(f_op, out_fig);
    close(f_op);

    fprintf('Saved development candidate artifacts to:\n');
    fprintf('  • %s\n', out_mat);
    fprintf('  • %s\n', out_csv);
    fprintf('  • %s\n', out_sum_csv);
    fprintf('  • %s\n\n', out_fig);

    results = struct();
    results.per_image_table = per_image_table;
    results.summary_table = summary_comparison_table;
end
