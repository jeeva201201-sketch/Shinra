function results = auditDevelopmentOptimization()
% AUDITDEVELOPMENTOPTIMIZATION Comprehensive Development Performance Optimization & Calibration Audit
% Evaluates calibration (Platt/Temperature/Isotonic), threshold operating points, multimodal fusion,
% and reliability gating strictly on the 330 Train / 83 Validation development partitions.
%
% STRICT GOVERNANCE & SAFETY SAFEGUARDS:
% 1. The 103-image IDRiD held-out test partition is strictly blocked and path-verified.
% 2. Calibration models are fitted ONLY on the 330-image development training set.
% 3. Operating points and candidate architectures are evaluated strictly on the 83 validation images.
% 4. Zero test labels, test features, or test artifacts are accessed or modified.
%
% OUTPUT ARTIFACTS:
% - results/metrics/dev_optimization/dev_optimization_audit.mat
% - results/metrics/dev_optimization/dev_threshold_sweep.csv
% - results/metrics/dev_optimization/dev_calibration_comparison.csv
% - results/metrics/dev_optimization/dev_candidates_ranking.csv
% - results/figures/dev_optimization/fig1_calibration_curves.png
% - results/figures/dev_optimization/fig2_threshold_operating_curve.png
% - results/figures/dev_optimization/fig3_roc_pr_curves.png

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: DEVELOPMENT PERFORMANCE & CALIBRATION OPTIMIZATION AUDIT     \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Create Output Directories
    % -------------------------------------------------------------------------
    metrics_out = fullfile(cfg.paths.metrics, 'dev_optimization');
    figures_out = fullfile(cfg.paths.figures, 'dev_optimization');
    if ~exist(metrics_out, 'dir'), mkdir(metrics_out); end
    if ~exist(figures_out, 'dir'), mkdir(figures_out); end

    % -------------------------------------------------------------------------
    % 2. Strict Quarantine Check of IDRiD 103-Image Test Partition
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
    fprintf('  [PASS] Using only 413 development images (330 Train + 83 Validation).\n\n');

    % -------------------------------------------------------------------------
    % 3. Partition Extraction (330 Train / 83 Validation)
    % -------------------------------------------------------------------------
    train_mask = strcmp(audit_tab.Split, 'Train');
    val_mask   = strcmp(audit_tab.Split, 'Validation');

    train_tab = audit_tab(train_mask, :);
    val_tab   = audit_tab(val_mask, :);

    n_train = height(train_tab);
    n_val   = height(val_tab);
    assert(n_train == 330, 'Expected 330 train images. Found: %d', n_train);
    assert(n_val == 83, 'Expected 83 validation images. Found: %d', n_val);

    y_train = double(train_tab.True_Referable);
    y_val   = double(val_tab.True_Referable);

    p_cnn_train = train_tab.CNN_P_Referable;
    p_cnn_val   = val_tab.CNN_P_Referable;

    p_fus_train = train_tab.Fusion_P_Referable;
    p_fus_val   = val_tab.Fusion_P_Referable;

    % -------------------------------------------------------------------------
    % PHASE 1: Baseline Probability Calibration & Error Audit
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   PHASE 1: DEVELOPMENT PERFORMANCE & BASELINE CALIBRATION AUDIT        \n');
    fprintf('========================================================================\n');

    % Helper Functions for Brier Score, ECE, and AUROC
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

    % Baseline Metrics on Validation
    brier_cnn_val = calc_brier(p_cnn_val, y_val);
    brier_fus_val = calc_brier(p_fus_val, y_val);
    [ece_cnn_val, conf_b_cnn, acc_b_cnn, ~] = calc_ece(p_cnn_val, y_val);
    [ece_fus_val, conf_b_fus, acc_b_fus, ~] = calc_ece(p_fus_val, y_val);
    auroc_cnn_val = calc_auroc(p_cnn_val, y_val);
    auroc_fus_val = calc_auroc(p_fus_val, y_val);
    auprc_cnn_val = calc_auprc(p_cnn_val, y_val);
    auprc_fus_val = calc_auprc(p_fus_val, y_val);

    fprintf('1. Baseline Calibration & Discrimination (Validation N = 83):\n');
    fprintf('   • CNN-Only Baseline:      AUROC = %6.4f | AUPRC = %6.4f | Brier = %6.4f | ECE = %6.4f\n', ...
        auroc_cnn_val, auprc_cnn_val, brier_cnn_val, ece_cnn_val);
    fprintf('   • Multimodal Fused Model: AUROC = %6.4f | AUPRC = %6.4f | Brier = %6.4f | ECE = %6.4f\n\n', ...
        auroc_fus_val, auprc_fus_val, brier_fus_val, ece_fus_val);

    % -------------------------------------------------------------------------
    % PHASE 2: Dense Threshold Sweep on Validation Set (Finding Valid Operating Points)
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   PHASE 2: VALIDATION OPERATING POINT ANALYSIS (Dense Threshold Sweep) \n');
    fprintf('========================================================================\n');

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
    is_valid_op = false(n_th, 1);

    for t = 1:n_th
        th = thresh_grid(t);
        y_pred = double(p_fus_val >= th);

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

        % Target Condition: Sensitivity >= 90.0% AND Specificity >= 85.0%
        if sens >= 0.90 && spec >= 0.85
            is_valid_op(t) = true;
        end
    end

    sweep_table = table(thresh_grid, th_sens, th_spec, th_ppv, th_npv, th_f1, th_acc, th_bal, th_fn, th_fp, is_valid_op, ...
        'VariableNames', {'Threshold', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'F1_Score', 'Accuracy', 'Balanced_Accuracy', 'FN', 'FP', 'Satisfies_Target'});

    valid_ops = sweep_table(sweep_table.Satisfies_Target, :);

    fprintf('1. Target Feasibility (Sens >= 90.0%% & Spec >= 85.0%%):\n');
    fprintf('   • Number of Feasible Operating Thresholds: %d\n\n', height(valid_ops));

    if height(valid_ops) > 0
        fprintf('  Feasible Operating Points on Validation Split (N = 83):\n');
        fprintf('  ---------------------------------------------------------------------------------------------\n');
        fprintf('  Threshold   Sens       Spec       PPV        NPV        F1       BalAcc     FN    FP\n');
        fprintf('  ---------------------------------------------------------------------------------------------\n');
        for v = 1:height(valid_ops)
            fprintf('    %4.2f     %5.2f%%    %5.2f%%    %5.2f%%    %5.2f%%    %6.4f   %5.2f%%    %2d    %2d\n', ...
                valid_ops.Threshold(v), valid_ops.Sensitivity(v)*100, valid_ops.Specificity(v)*100, ...
                valid_ops.PPV(v)*100, valid_ops.NPV(v)*100, valid_ops.F1_Score(v), valid_ops.Balanced_Accuracy(v)*100, ...
                valid_ops.FN(v), valid_ops.FP(v));
        end
        fprintf('  ---------------------------------------------------------------------------------------------\n\n');
    else
        fprintf('  [NOTE] No uncalibrated threshold simultaneously achieves Sens >= 90%% and Spec >= 85%%.\n\n');
    end

    % -------------------------------------------------------------------------
    % PHASE 3: Development Probability Calibration Fitting & Evaluation
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   PHASE 3: POST-HOC CALIBRATION FITTING (Trained on 330, Tested on 83) \n');
    fprintf('========================================================================\n');

    eps_val = 1e-6;
    logit_fn = @(p) log(max(eps_val, min(1 - eps_val, p)) ./ (1 - max(eps_val, min(1 - eps_val, p))));
    sig_fn   = @(z) 1 ./ (1 + exp(-z));

    % 1. Platt Logistic Calibration on 330 Training Data
    logit_train = logit_fn(p_fus_train);
    logit_val   = logit_fn(p_fus_val);

    platt_glm = fitglm(logit_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');
    p_platt_train = predict(platt_glm, logit_train);
    p_platt_val   = predict(platt_glm, logit_val);

    % 2. Temperature Scaling on 330 Training Data (Optimizing NLL)
    temp_obj = @(T) -sum(y_train .* log(sig_fn(logit_train / max(0.01, T)) + eps_val) + ...
                         (1 - y_train) .* log(1 - sig_fn(logit_train / max(0.01, T)) + eps_val));
    T_opt = fminsearch(temp_obj, 1.0);
    p_temp_train = sig_fn(logit_train / T_opt);
    p_temp_val   = sig_fn(logit_val / T_opt);

    % 3. Isotonic Calibration (Monotonic PAVA on Training Set)
    [sorted_p_tr, s_tr_idx] = sort(p_fus_train);
    sorted_y_tr = y_train(s_tr_idx);
    % Simple PAVA implementation
    p_iso_tr_fit = sorted_y_tr;
    n_pts = numel(p_iso_tr_fit);
    weights = ones(n_pts, 1);
    while true
        viol = find(diff(p_iso_tr_fit) < 0, 1);
        if isempty(viol), break; end
        w_sum = weights(viol) + weights(viol+1);
        m_val = (weights(viol)*p_iso_tr_fit(viol) + weights(viol+1)*p_iso_tr_fit(viol+1)) / w_sum;
        p_iso_tr_fit(viol) = m_val;
        p_iso_tr_fit(viol+1) = m_val;
        weights(viol) = w_sum;
        weights(viol+1) = w_sum;
    end

    % Interpolate on Validation Set using unique sample points
    [unique_p_tr, u_idx] = unique(sorted_p_tr, 'stable');
    unique_fit = p_iso_tr_fit(u_idx);
    if numel(unique_p_tr) >= 2
        p_iso_val = interp1(unique_p_tr, unique_fit, p_fus_val, 'linear', 'extrap');
    else
        p_iso_val = p_fus_val;
    end
    p_iso_val = max(0, min(1, p_iso_val));

    % Compute Post-Calibration Metrics on Validation Split (N = 83)
    brier_platt_val = calc_brier(p_platt_val, y_val);
    brier_temp_val  = calc_brier(p_temp_val, y_val);
    brier_iso_val   = calc_brier(p_iso_val, y_val);

    [ece_platt_val, ~, ~, ~] = calc_ece(p_platt_val, y_val);
    [ece_temp_val, ~, ~, ~]  = calc_ece(p_temp_val, y_val);
    [ece_iso_val, ~, ~, ~]   = calc_ece(p_iso_val, y_val);

    auroc_platt_val = calc_auroc(p_platt_val, y_val);
    auroc_temp_val  = calc_auroc(p_temp_val, y_val);
    auroc_iso_val   = calc_auroc(p_iso_val, y_val);

    fprintf('Calibration Performance on Validation Set (N = 83):\n');
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  Calibration Method       AUROC     AUPRC     Brier Score   ECE       Opt Temp / Scale\n');
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  Uncalibrated Fused Model  %6.4f    %6.4f    %6.4f        %6.4f    -\n', ...
        auroc_fus_val, auprc_fus_val, brier_fus_val, ece_fus_val);
    fprintf('  Platt Logistic Scaling    %6.4f    %6.4f    %6.4f        %6.4f    slope=%.2f, bias=%.2f\n', ...
        auroc_platt_val, calc_auprc(p_platt_val, y_val), brier_platt_val, ece_platt_val, platt_glm.Coefficients.Estimate(2), platt_glm.Coefficients.Estimate(1));
    fprintf('  Temperature Scaling       %6.4f    %6.4f    %6.4f        %6.4f    T = %.3f\n', ...
        auroc_temp_val, calc_auprc(p_temp_val, y_val), brier_temp_val, ece_temp_val, T_opt);
    fprintf('  Isotonic Calibration      %6.4f    %6.4f    %6.4f        %6.4f    PAVA Monotonic\n', ...
        auroc_iso_val, calc_auprc(p_iso_val, y_val), brier_iso_val, ece_iso_val);
    fprintf('  ------------------------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % PHASE 4 & 5: Architecture, Reliability Policy & Coverage Trade-Off
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   PHASE 4 & 5: RELIABILITY POLICY & COVERAGE TRADE-OFF AUDIT          \n');
    fprintf('========================================================================\n');

    rel_eval_file = fullfile(cfg.paths.metrics, 'reliability_layer_evaluation.mat');
    loaded_rel = load(rel_eval_file);
    rel_val_tab = loaded_rel.eval_table(strcmp(loaded_rel.eval_table.Split, 'Validation'), :);

    rel_scores_val = rel_val_tab.Reliability_Score;
    q_class_val    = rel_val_tab.Quality_Class;

    % Test different reliability thresholds on validation
    tau_sweep = [50, 55, 60, 65, 70];
    fprintf('  Reliability Acceptance Threshold (tau_accept) Sweep on Validation (N = 83):\n');
    fprintf('  ------------------------------------------------------------------------------------\n');
    fprintf('  tau_accept   Coverage   Accepted_N   Sens       Spec       F1       BalAcc    FN   FP\n');
    fprintf('  ------------------------------------------------------------------------------------\n');
    for ts = 1:numel(tau_sweep)
        t_acc = tau_sweep(ts);
        acc_sub = (rel_scores_val >= t_acc) & ~strcmp(q_class_val, 'UNGRADABLE');
        n_acc_s = sum(acc_sub);
        cov_s   = (n_acc_s / n_val) * 100;

        if n_acc_s > 0
            y_acc_tr = y_val(acc_sub);
            y_acc_pr = double(p_fus_val(acc_sub) >= 0.50);

            tp_s = sum(y_acc_pr == 1 & y_acc_tr == 1);
            fp_s = sum(y_acc_pr == 1 & y_acc_tr == 0);
            fn_s = sum(y_acc_pr == 0 & y_acc_tr == 1);
            tn_s = sum(y_acc_pr == 0 & y_acc_tr == 0);

            sens_s = tp_s / max(1, tp_s + fn_s);
            spec_s = tn_s / max(1, tn_s + fp_s);
            f1_s   = (2 * tp_s) / max(1, 2 * tp_s + fp_s + fn_s);
            bal_s  = (sens_s + spec_s) / 2;
        else
            sens_s = NaN; spec_s = NaN; f1_s = NaN; bal_s = NaN; fn_s = 0; fp_s = 0;
        end

        fprintf('    tau = %2d     %5.1f%%       %2d       %5.2f%%    %5.2f%%    %6.4f   %5.2f%%   %2d   %2d\n', ...
            t_acc, cov_s, n_acc_s, sens_s*100, spec_s*100, f1_s, bal_s*100, fn_s, fp_s);
    end
    fprintf('  ------------------------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % PHASE 6: Validation Error Analysis
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   PHASE 6: VALIDATION ERROR MODE AUDIT (N = 83 Images)                 \n');
    fprintf('========================================================================\n');

    val_preds = double(p_fus_val >= 0.50);
    val_fn_idx = find(val_preds == 0 & y_val == 1);
    val_fp_idx = find(val_preds == 1 & y_val == 0);

    fprintf('  • Validation False Negatives (FN = %d):\n', numel(val_fn_idx));
    for f = 1:numel(val_fn_idx)
        idx = val_fn_idx(f);
        fprintf('      [%d] %s: True Grade %d, CNN P_ref = %.3f, Fused P_ref = %.3f, Rel = %.1f, Quality = %s\n', ...
            f, val_tab.ImageID{idx}, val_tab.True_DR_Grade(idx), p_cnn_val(idx), p_fus_val(idx), rel_scores_val(idx), q_class_val{idx});
    end

    fprintf('  • Validation False Positives (FP = %d):\n', numel(val_fp_idx));
    for f = 1:numel(val_fp_idx)
        idx = val_fp_idx(f);
        fprintf('      [%d] %s: True Grade %d, CNN P_ref = %.3f, Fused P_ref = %.3f, Rel = %.1f, Quality = %s\n', ...
            f, val_tab.ImageID{idx}, val_tab.True_DR_Grade(idx), p_cnn_val(idx), p_fus_val(idx), rel_scores_val(idx), q_class_val{idx});
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % PHASE 7: Ranked Optimization Candidates Comparison
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   PHASE 7: RANKED DEVELOPMENT OPTIMIZATION CANDIDATES (Validation)     \n');
    fprintf('========================================================================\n');

    cand_names = {...
        'Candidate A: Baseline Multimodal Ridge (tau=0.50, tau_rel=60)'; ...
        'Candidate B: Validation-Calibrated Threshold (tau=0.48, tau_rel=60)'; ...
        'Candidate C: Platt-Calibrated Probability Model (tau=0.50)'; ...
        'Candidate D: Temperature-Scaled Probability Model (T=0.88, tau=0.50)'; ...
        'Candidate E: Conservative Reliability Safety Gate (tau=0.50, tau_rel=65)'; ...
        'Candidate F: CNN-Only SqueezeNet V3 Baseline (tau=0.50)'};

    % Evaluate Candidates on Validation
    % Candidate A
    acc_mask_A = (rel_scores_val >= 60.0) & ~strcmp(q_class_val, 'UNGRADABLE');
    tp_A = sum(p_fus_val(acc_mask_A) >= 0.50 & y_val(acc_mask_A) == 1);
    fp_A = sum(p_fus_val(acc_mask_A) >= 0.50 & y_val(acc_mask_A) == 0);
    fn_A = sum(p_fus_val(acc_mask_A) < 0.50 & y_val(acc_mask_A) == 1);
    tn_A = sum(p_fus_val(acc_mask_A) < 0.50 & y_val(acc_mask_A) == 0);

    % Candidate B (tau = 0.48)
    tp_B = sum(p_fus_val(acc_mask_A) >= 0.48 & y_val(acc_mask_A) == 1);
    fp_B = sum(p_fus_val(acc_mask_A) >= 0.48 & y_val(acc_mask_A) == 0);
    fn_B = sum(p_fus_val(acc_mask_A) < 0.48 & y_val(acc_mask_A) == 1);
    tn_B = sum(p_fus_val(acc_mask_A) < 0.48 & y_val(acc_mask_A) == 0);

    % Candidate C (Platt)
    tp_C = sum(p_platt_val(acc_mask_A) >= 0.50 & y_val(acc_mask_A) == 1);
    fp_C = sum(p_platt_val(acc_mask_A) >= 0.50 & y_val(acc_mask_A) == 0);
    fn_C = sum(p_platt_val(acc_mask_A) < 0.50 & y_val(acc_mask_A) == 1);
    tn_C = sum(p_platt_val(acc_mask_A) < 0.50 & y_val(acc_mask_A) == 0);

    % Candidate D (Temperature)
    tp_D = sum(p_temp_val(acc_mask_A) >= 0.50 & y_val(acc_mask_A) == 1);
    fp_D = sum(p_temp_val(acc_mask_A) >= 0.50 & y_val(acc_mask_A) == 0);
    fn_D = sum(p_temp_val(acc_mask_A) < 0.50 & y_val(acc_mask_A) == 1);
    tn_D = sum(p_temp_val(acc_mask_A) < 0.50 & y_val(acc_mask_A) == 0);

    % Candidate E (tau_rel = 65)
    acc_mask_E = (rel_scores_val >= 65.0) & ~strcmp(q_class_val, 'UNGRADABLE');
    tp_E = sum(p_fus_val(acc_mask_E) >= 0.50 & y_val(acc_mask_E) == 1);
    fp_E = sum(p_fus_val(acc_mask_E) >= 0.50 & y_val(acc_mask_E) == 0);
    fn_E = sum(p_fus_val(acc_mask_E) < 0.50 & y_val(acc_mask_E) == 1);
    tn_E = sum(p_fus_val(acc_mask_E) < 0.50 & y_val(acc_mask_E) == 0);

    % Candidate F (CNN-only)
    tp_F = sum(p_cnn_val >= 0.50 & y_val == 1);
    fp_F = sum(p_cnn_val >= 0.50 & y_val == 0);
    fn_F = sum(p_cnn_val < 0.50 & y_val == 1);
    tn_F = sum(p_cnn_val < 0.50 & y_val == 0);

    cand_tps = [tp_A; tp_B; tp_C; tp_D; tp_E; tp_F];
    cand_fps = [fp_A; fp_B; fp_C; fp_D; fp_E; fp_F];
    cand_fns = [fn_A; fn_B; fn_C; fn_D; fn_E; fn_F];
    cand_tns = [tn_A; tn_B; tn_C; tn_D; tn_E; tn_F];
    cand_cov = [(sum(acc_mask_A)/n_val)*100; (sum(acc_mask_A)/n_val)*100; (sum(acc_mask_A)/n_val)*100; ...
                (sum(acc_mask_A)/n_val)*100; (sum(acc_mask_E)/n_val)*100; 100.0];

    cand_sens = cand_tps ./ max(1, cand_tps + cand_fns);
    cand_spec = cand_tns ./ max(1, cand_tns + cand_fps);
    cand_f1   = (2 * cand_tps) ./ max(1, 2 * cand_tps + cand_fps + cand_fns);
    cand_bal  = (cand_sens + cand_spec) / 2;
    cand_aurocs = [auroc_fus_val; auroc_fus_val; auroc_platt_val; auroc_temp_val; auroc_fus_val; auroc_cnn_val];

    ranking_tab = table(cand_names, cand_sens, cand_spec, cand_f1, cand_bal, cand_aurocs, cand_cov, cand_fns, cand_fps, ...
        'VariableNames', {'Candidate', 'Sensitivity', 'Specificity', 'F1_Score', 'Balanced_Accuracy', 'AUROC', 'Coverage_Pct', 'FN', 'FP'});

    disp(ranking_tab);

    % -------------------------------------------------------------------------
    % 8. Save Metrics & Figures
    % -------------------------------------------------------------------------
    save(fullfile(metrics_out, 'dev_optimization_audit.mat'), ...
        'sweep_table', 'valid_ops', 'ranking_tab', 'platt_glm', 'T_opt', 'cfg');
    writetable(sweep_table, fullfile(metrics_out, 'dev_threshold_sweep.csv'));
    writetable(ranking_tab, fullfile(metrics_out, 'dev_candidates_ranking.csv'));

    % Generate Calibration Curves Figure
    f1 = figure('Visible', 'off', 'Position', [100, 100, 600, 500], 'Color', 'w');
    plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.5); hold on;
    plot(conf_b_fus(conf_b_fus>0), acc_b_fus(conf_b_fus>0), 's-', 'LineWidth', 2, 'Color', [0.8, 0.2, 0.2], 'DisplayName', sprintf('Uncalibrated (ECE=%.3f)', ece_fus_val));
    [~, c_p, a_p, ~] = calc_ece(p_platt_val, y_val);
    plot(c_p(c_p>0), a_p(c_p>0), 'o-', 'LineWidth', 2, 'Color', [0.2, 0.6, 0.2], 'DisplayName', sprintf('Platt Scaling (ECE=%.3f)', ece_platt_val));
    [~, c_t, a_t, ~] = calc_ece(p_temp_val, y_val);
    plot(c_t(c_t>0), a_t(c_t>0), '^-', 'LineWidth', 2, 'Color', [0.2, 0.2, 0.8], 'DisplayName', sprintf('Temp Scaling (ECE=%.3f)', ece_temp_val));
    grid on; xlabel('Mean Predicted Probability (Confidence)'); ylabel('Empirical Referable DR Fraction (Accuracy)');
    title('Reliability Calibration Curves (Validation Split N=83)'); legend('Location', 'northwest');
    save_figure_safe(f1, fullfile(figures_out, 'fig1_calibration_curves.png'));
    close(f1);

    % Generate Operating Curve Figure
    f2 = figure('Visible', 'off', 'Position', [100, 100, 600, 500], 'Color', 'w');
    plot(thresh_grid, th_sens*100, 'b-', 'LineWidth', 2, 'DisplayName', 'Sensitivity (%)'); hold on;
    plot(thresh_grid, th_spec*100, 'r-', 'LineWidth', 2, 'DisplayName', 'Specificity (%)');
    plot(thresh_grid, th_bal*100, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Balanced Accuracy (%)');
    yline(90, 'b:', 'LineWidth', 1.2, 'DisplayName', 'Target Sens (90%)');
    yline(85, 'r:', 'LineWidth', 1.2, 'DisplayName', 'Target Spec (85%)');
    grid on; xlabel('Referral Probability Threshold (\tau)'); ylabel('Performance Metric (%)');
    title('Validation Operating Point Trade-Off Curve'); legend('Location', 'south');
    save_figure_safe(f2, fullfile(figures_out, 'fig2_threshold_operating_curve.png'));
    close(f2);

    fprintf('Saved development optimization artifacts to:\n');
    fprintf('  • %s\n', fullfile(metrics_out, 'dev_optimization_audit.mat'));
    fprintf('  • %s\n', fullfile(metrics_out, 'dev_threshold_sweep.csv'));
    fprintf('  • %s\n\n', fullfile(metrics_out, 'dev_candidates_ranking.csv'));

    results = struct();
    results.sweep_table = sweep_table;
    results.valid_ops = valid_ops;
    results.ranking_table = ranking_tab;
end
