function ablation_table = runAblationStudy(num_eval_samples)
% RUNABLATIONSTUDY Systematic Benchmark of SHINRA Components
% Evaluates progressive contributions of:
% Exp A: Baseline Raw CNN
% Exp B: Baseline + Retinal Preprocessing
% Exp C: CNN + Preprocessing + Quality Gate
% Exp D: Quality Gate + Preprocessing + Lesion Biomarkers
% Exp E: Quality + Preprocessing + Lesions + Evidence Fusion
% Exp F: Full SHINRA System (+ Reliability & Decision Triage)

    cfg = shinra_config();
    fprintf('============================================================\n');
    fprintf('           SHINRA PROGRESSIVE ABLATION BENCHMARK            \n');
    fprintf('============================================================\n');

    [idrid_train_full, ~] = load_idrid();
    [~, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);

    if nargin >= 1 && ~isempty(num_eval_samples) && num_eval_samples < height(val_tab)
        val_tab = val_tab(1:num_eval_samples, :);
    end

    n = height(val_tab);
    fprintf('Evaluating %d validation cases across 6 pipeline configurations...\n\n', n);

    y_true_dr  = val_tab.DR_Grade;
    y_true_ref = val_tab.Referable_DR;

    % Storage for predictions across 6 experiments
    exp_preds = zeros(n, 6);
    exp_probs = zeros(n, 6);

    for i = 1:n
        img_raw = imread(val_tab.ImagePath{i});

        % Exp A: Raw image without preprocessing
        prep_raw = struct('rgb_normalized', im2double(imresize(img_raw, [512, 512])));
        sev_A = predictDRSeverity(prep_raw, []);
        exp_preds(i, 1) = sev_A.p_referable >= 0.50;
        exp_probs(i, 1) = sev_A.p_referable;

        % Exp B: With Retinal Preprocessing
        prep_B = preprocessRetina(img_raw);
        sev_B = predictDRSeverity(prep_B, []);
        exp_preds(i, 2) = sev_B.p_referable >= 0.50;
        exp_probs(i, 2) = sev_B.p_referable;

        % Exp C: With Preprocessing + Quality Gate
        [q_class, q_rep] = qualityClassifier(prep_B.rgb_normalized, prep_B.fov_mask);
        if strcmp(q_class, 'UNGRADABLE')
            % Ungradable flagged for recapture / specialist review
            exp_preds(i, 3) = 1; % Conservative screening referral
            exp_probs(i, 3) = 0.85;
        else
            sev_C = predictDRSeverity(prep_B, []);
            exp_preds(i, 3) = sev_C.p_referable >= 0.50;
            exp_probs(i, 3) = sev_C.p_referable;
        end

        % Exp D: Preprocessing + Quality + Lesion Evidence
        od_res = detectOpticDisc(prep_B.rgb_normalized, prep_B.fov_mask);
        fovea_res = locateFovea(prep_B.rgb_normalized, od_res, prep_B.fov_mask);
        vess_res = segmentVessels(prep_B.rgb_normalized, prep_B.fov_mask, od_res.mask);
        ma_res = detectMicroaneurysms(prep_B.rgb_normalized, prep_B.fov_mask, vess_res.binary_mask, od_res.mask);
        he_res = detectHemorrhages(prep_B.rgb_normalized, prep_B.fov_mask, vess_res.binary_mask, od_res.mask);
        ex_res = detectExudates(prep_B.rgb_normalized, prep_B.fov_mask, od_res.mask, fovea_res);
        nv_res = analyzeNeovascularization(vess_res.binary_mask, prep_B.fov_mask, od_res.mask, vess_res.morphometry);

        lesion_ev = struct(...
            'ma_count', ma_res.count, 'ma_area', ma_res.total_area, ...
            'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, ...
            'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, ...
            'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);

        sev_D = predictDRSeverity(prep_B, lesion_ev);
        exp_preds(i, 4) = sev_D.p_referable >= 0.50;
        exp_probs(i, 4) = sev_D.p_referable;

        % Exp E: Preprocessing + Quality + Lesions + Evidence Fusion
        ev_vec = extractEvidenceVector(sev_D, lesion_ev, vess_res, q_rep);
        fus_res = fuseEvidence(ev_vec, []);
        exp_preds(i, 5) = fus_res.p_referable_fused >= 0.50;
        exp_probs(i, 5) = fus_res.p_referable_fused;

        % Exp F: Full SHINRA (+ Reliability & Safety Decision Gate)
        rel_res = computeReliability(sev_D.confidence, q_rep.score, fus_res.agreement_score);
        dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);
        exp_preds(i, 6) = strcmp(dec_res.final_decision, 'REFERABLE DR') || strcmp(dec_res.final_decision, 'UNCERTAIN / CLINICAL TRIAGE');
        exp_probs(i, 6) = fus_res.p_referable_fused;
    end

    % Calculate metrics for each experiment
    exp_names = {
        'Exp A: Baseline CNN', ...
        'Exp B: CNN + Preprocessing', ...
        'Exp C: CNN + Preproc + Quality Gate', ...
        'Exp D: Quality + Preproc + Lesion Evidence', ...
        'Exp E: Quality + Preproc + Lesions + Fusion', ...
        'Exp F: Full SHINRA System'
    };

    sens_list = zeros(6, 1);
    spec_list = zeros(6, 1);
    f1_list   = zeros(6, 1);
    acc_list  = zeros(6, 1);
    auc_list  = zeros(6, 1);

    for e = 1:6
        m = evaluateClassifier(y_true_ref, exp_preds(:, e), exp_probs(:, e));
        sens_list(e) = m.sensitivity * 100;
        spec_list(e) = m.specificity * 100;
        f1_list(e)   = m.f1_score;
        acc_list(e)  = m.balanced_accuracy * 100;
        auc_list(e)  = m.roc_auc;
    end

    ablation_table = table(exp_names', sens_list, spec_list, acc_list, f1_list, auc_list, ...
        'VariableNames', {'Experiment', 'Sensitivity_Pct', 'Specificity_Pct', 'Balanced_Acc_Pct', 'F1_Score', 'ROC_AUC'});

    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('  SHINRA ABLATION STUDY RESULTS\n');
    fprintf('------------------------------------------------------------------------------------\n');
    disp(ablation_table);

    % Save results
    writetable(ablation_table, fullfile(cfg.paths.metrics, 'ablation_table.csv'));
    save(fullfile(cfg.paths.metrics, 'ablation_results.mat'), 'ablation_table');
    fprintf('Ablation results saved to %s\n\n', fullfile(cfg.paths.metrics, 'ablation_table.csv'));
end
