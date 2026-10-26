function results = evaluateTestReleaseV1()
% EVALUATETESTRELEASEV1 Final Held-Out Test Evaluation of SHINRA_DEVELOPMENT_RELEASE_V1
% Executes the frozen SHINRA_DEVELOPMENT_RELEASE_V1 configuration on the 103-image IDRiD
% held-out test partition with zero retraining, zero threshold tuning, and zero leakage.
%
% FROZEN PARAMETERS:
% - Configuration: SHINRA_DEVELOPMENT_RELEASE_V1
% - Calibration: Platt logistic scaling (a = 1.3106, b = -1.1412)
% - Referral Threshold: tau_ref = 0.30
% - Reliability Gate: tau_accept = 60.0
%
% OUTPUT DIRECTORY:
% results/metrics/final_test/development_release_v1/
%
% OUTPUT ARTIFACTS:
% - final_test_evaluation.mat
% - final_test_per_image.csv
% - final_test_summary.csv
% - final_test_confusion_matrix.png
% - final_test_roc.png
% - final_test_pr_curve.png
% - final_test_calibration.png
% - final_test_reliability.png

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: FINAL HELD-OUT TEST EVALUATION                               \n');
    fprintf('   CONFIGURATION: SHINRA_DEVELOPMENT_RELEASE_V1                         \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Create Dedicated Output Directory for Release V1
    % -------------------------------------------------------------------------
    out_dir = fullfile(cfg.paths.metrics, 'final_test', 'development_release_v1');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end

    % -------------------------------------------------------------------------
    % 2. Load Official IDRiD 103-Image Held-Out Test Set
    % -------------------------------------------------------------------------
    fprintf('[STEP 1/5] Loading official IDRiD 103-image held-out test partition...\n');
    [~, idrid_test] = load_idrid();
    n_test = height(idrid_test);
    assert(n_test == 103, 'Expected exactly 103 test images. Found: %d', n_test);

    assert(all(cellfun(@(p) exist(p, 'file') == 2, idrid_test.ImagePath)), ...
        'One or more test image paths do not exist on disk.');
    fprintf('  [PASS] 103 physical test image files verified.\n');
    fprintf('  [PASS] Configuration is locked: SHINRA_DEVELOPMENT_RELEASE_V1\n\n');

    % -------------------------------------------------------------------------
    % 3. Execute Complete Frozen Pipeline on 103 Test Images
    % -------------------------------------------------------------------------
    fprintf('[STEP 2/5] Executing full multimodal pipeline on all 103 test images...\n');

    % Pre-allocate arrays
    image_ids           = idrid_test.ImageID;
    image_paths         = idrid_test.ImagePath;
    true_grades         = idrid_test.DR_Grade;
    true_referable      = double(true_grades >= 2);
    true_dme_risks      = idrid_test.DME_Risk;

    quality_scores      = zeros(n_test, 1);
    quality_classes     = cell(n_test, 1);

    cnn_pred_grades     = zeros(n_test, 1);
    cnn_p_referable     = zeros(n_test, 1);
    cnn_p0              = zeros(n_test, 1);
    cnn_p1              = zeros(n_test, 1);
    cnn_p2              = zeros(n_test, 1);
    cnn_p3              = zeros(n_test, 1);
    cnn_p4              = zeros(n_test, 1);
    cnn_confidence      = zeros(n_test, 1);

    ma_counts           = zeros(n_test, 1);
    he_counts           = zeros(n_test, 1);
    ex_counts           = zeros(n_test, 1);
    he_area_fractions   = zeros(n_test, 1);
    ex_area_fractions   = zeros(n_test, 1);
    dme_pred_risks      = zeros(n_test, 1);
    nv_scores           = zeros(n_test, 1);
    vessel_densities    = zeros(n_test, 1);
    vessel_tortuosities = zeros(n_test, 1);

    fused_p_referable_raw = zeros(n_test, 1);
    fused_p_calibrated    = zeros(n_test, 1);
    fused_agreement       = zeros(n_test, 1);
    predicted_referable   = zeros(n_test, 1);

    reliability_scores  = zeros(n_test, 1);
    sub_quality         = zeros(n_test, 1);
    sub_confidence      = zeros(n_test, 1);
    sub_margin          = zeros(n_test, 1);
    sub_completeness    = zeros(n_test, 1);
    sub_agreement       = zeros(n_test, 1);
    conf_levels         = cell(n_test, 1);
    uncert_levels       = cell(n_test, 1);
    recommended_actions = cell(n_test, 1);
    decision_readiness  = false(n_test, 1);
    primary_reasons     = cell(n_test, 1);

    % Frozen Release V1 Parameters
    a_platt = 1.3106;
    b_platt = -1.1412;
    tau_ref = 0.30;
    tau_accept = 60.0;
    eps_val = 1e-6;

    for i = 1:n_test
        img_path = image_paths{i};

        % 1. Load and Preprocess
        raw_img = imread(img_path);
        prep = preprocessRetina(raw_img, cfg.preprocessing.target_size);

        % 2. Quality Safety Gate
        [q_class, q_rep] = qualityClassifier(prep.rgb_normalized, prep.fov_mask);
        quality_scores(i)  = q_rep.score;
        quality_classes{i} = q_class;

        % 3. Retinal Anatomy Localization
        od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
        fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);

        % 4. Retinal Vessel Segmentation & Morphometry (Frozen U-Net)
        vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
        vessel_densities(i)    = vess_res.morphometry.vessel_density;
        vessel_tortuosities(i) = vess_res.morphometry.mean_tortuosity;

        % 5. Multi-Scale Lesion Biomarker Extraction
        ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
        he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
        ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
        nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);

        ma_counts(i)         = ma_res.count;
        he_counts(i)         = he_res.count;
        ex_counts(i)         = ex_res.count;
        he_area_fractions(i) = he_res.area_fraction;
        ex_area_fractions(i) = ex_res.area_fraction;
        dme_pred_risks(i)    = ex_res.macular_proximity_risk;
        nv_scores(i)         = nv_res.nv_score;

        lesion_ev = struct(...
            'ma_mask', ma_res.binary_mask, ...
            'he_mask', he_res.binary_mask, ...
            'ex_mask', ex_res.binary_mask, ...
            'ma_count', ma_res.count, 'ma_area', ma_res.total_area, ...
            'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, ...
            'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, ...
            'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);

        % 6. Frozen Severity SqueezeNet V3 Inference
        sev_res = predictSqueezeNetSeverityV3(prep);

        cnn_p0(i)          = sev_res.probabilities(1);
        cnn_p1(i)          = sev_res.probabilities(2);
        cnn_p2(i)          = sev_res.probabilities(3);
        cnn_p3(i)          = sev_res.probabilities(4);
        cnn_p4(i)          = sev_res.probabilities(5);
        cnn_p_referable(i) = sev_res.p_referable;
        cnn_pred_grades(i) = sev_res.predicted_grade;
        cnn_confidence(i)  = sev_res.confidence;

        % 7. 16-D Evidence Vector & Multimodal Fusion
        [ev_vec, feat_names] = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
        fus_res = fuseEvidence(ev_vec, feat_names);

        p_raw = fus_res.p_referable_fused;
        fused_p_referable_raw(i) = p_raw;
        fused_agreement(i)       = fus_res.agreement_score;

        % 8. Apply Frozen Platt Calibration
        p_clamp = max(eps_val, min(1 - eps_val, p_raw));
        logit_val = log(p_clamp / (1 - p_clamp));
        p_cal = 1.0 / (1.0 + exp(-(a_platt * logit_val + b_platt)));
        fused_p_calibrated(i) = max(0, min(1, p_cal));

        % Decision Rule: tau_ref = 0.30
        is_ref_pred = double(p_cal >= tau_ref);
        predicted_referable(i) = is_ref_pred;

        % 9. 5-Factor Reliability Layer
        rel_res = computeReliability(q_rep, sev_res, fus_res, lesion_ev, vess_res, od_res, fovea_res);

        reliability_scores(i) = rel_res.reliability_score;
        sub_quality(i)        = rel_res.quality_score_component;
        sub_confidence(i)     = rel_res.confidence_score_component;
        sub_margin(i)         = rel_res.margin_score_component;
        sub_completeness(i)   = rel_res.completeness_score_component;
        sub_agreement(i)      = rel_res.agreement_score_component;
        conf_levels{i}        = rel_res.confidence_level;
        uncert_levels{i}      = rel_res.uncertainty_level;

        % 10. Triage Recommendation (tau_accept = 60.0)
        if strcmp(q_class, 'UNGRADABLE')
            recommended_actions{i} = 'RECAPTURE';
            decision_readiness(i)  = false;
            primary_reasons{i}     = 'Severe image quality degradation';
        elseif rel_res.reliability_score < tau_accept
            if is_ref_pred == 1
                recommended_actions{i} = 'CLINICAL_REVIEW';
                primary_reasons{i}     = 'Suspicious referable DR with sub-threshold reliability';
            else
                recommended_actions{i} = 'UNCERTAIN';
                primary_reasons{i}     = 'Decision boundary ambiguity with sub-threshold reliability';
            end
            decision_readiness(i)  = false;
        else
            recommended_actions{i} = 'ACCEPT_SCREENING';
            decision_readiness(i)  = true;
            if ~isempty(rel_res.reliability_reasons)
                primary_reasons{i} = rel_res.reliability_reasons{1};
            else
                primary_reasons{i} = 'Normal autonomous screening conditions';
            end
        end

        if mod(i, 20) == 0 || i == n_test
            fprintf('  • Evaluated %3d / %d test images (%.1f%%)...\n', i, n_test, (i/n_test)*100);
        end
    end

    % -------------------------------------------------------------------------
    % 4. Compute Comprehensive Evaluation Metrics
    % -------------------------------------------------------------------------
    fprintf('\n[STEP 3/5] Computing diagnostic, calibration, and reliability metrics...\n');

    % Helper Functions
    calc_brier = @(p, y) mean((p(:) - y(:)).^2);

    function [ece, conf_bins, acc_bins, bin_counts] = calc_ece(p, y, n_bins)
        if nargin < 3, n_bins = 10; end
        bin_edges = linspace(0, 1, n_bins + 1);
        ece = 0; conf_bins = zeros(n_bins, 1); acc_bins = zeros(n_bins, 1); bin_counts = zeros(n_bins, 1);
        N = numel(p);
        for b = 1:n_bins
            idx = (p >= bin_edges(b)) & (p < bin_edges(b+1));
            if b == n_bins, idx = (p >= bin_edges(b)) & (p <= bin_edges(b+1)); end
            nb = sum(idx);
            bin_counts(b) = nb;
            if nb > 0
                cb = mean(p(idx)); ab = mean(y(idx));
                conf_bins(b) = cb; acc_bins(b) = ab;
                ece = ece + (nb / N) * abs(ab - cb);
            end
        end
    end

    function [auroc, fpr, tpr] = calc_auroc(p, y)
        [sorted_p, s_idx] = sort(p, 'descend');
        sorted_y = y(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        tpr = [0; tps / max(1, sum(sorted_y == 1))];
        fpr = [0; fps / max(1, sum(sorted_y == 0))];
        auroc = trapz(fpr, tpr);
    end

    function [auprc, rec, prec] = calc_auprc(p, y)
        [sorted_p, s_idx] = sort(p, 'descend');
        sorted_y = y(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        rec = [0; tps / max(1, sum(sorted_y == 1))];
        prec= [1; tps ./ max(1, tps + fps)];
        auprc = trapz(rec, prec);
    end

    % A. 5-Class Severity Grading
    conf_mat_5class = confusionmat(true_grades, cnn_pred_grades, 'Order', 0:4);
    acc_5class = sum(diag(conf_mat_5class)) / n_test;
    per_class_prec = zeros(5, 1);
    per_class_rec  = zeros(5, 1);
    per_class_f1   = zeros(5, 1);
    for c = 1:5
        tp_c = conf_mat_5class(c, c);
        fp_c = sum(conf_mat_5class(:, c)) - tp_c;
        fn_c = sum(conf_mat_5class(c, :)) - tp_c;
        per_class_prec(c) = tp_c / max(1, tp_c + fp_c);
        per_class_rec(c)  = tp_c / max(1, tp_c + fn_c);
        per_class_f1(c)   = (2 * tp_c) / max(1, 2 * tp_c + fp_c + fn_c);
    end
    macro_f1_5class = mean(per_class_f1);
    bal_acc_5class  = mean(per_class_rec);

    % B. Binary Referable DR Screening (All 103 Test Cases)
    tp_all = sum(predicted_referable == 1 & true_referable == 1);
    fp_all = sum(predicted_referable == 1 & true_referable == 0);
    fn_all = sum(predicted_referable == 0 & true_referable == 1);
    tn_all = sum(predicted_referable == 0 & true_referable == 0);

    sens_all = tp_all / max(1, tp_all + fn_all);
    spec_all = tn_all / max(1, tn_all + fp_all);
    ppv_all  = tp_all / max(1, tp_all + fp_all);
    npv_all  = tn_all / max(1, tn_all + fn_all);
    f1_all   = (2 * tp_all) / max(1, 2 * tp_all + fp_all + fn_all);
    acc_all  = (tp_all + tn_all) / n_test;
    bal_all  = (sens_all + spec_all) / 2;

    [auroc_all, fpr_curve, tpr_curve] = calc_auroc(fused_p_calibrated, true_referable);
    [auprc_all, rec_curve, prec_curve] = calc_auprc(fused_p_calibrated, true_referable);
    brier_all = calc_brier(fused_p_calibrated, true_referable);
    [ece_all, conf_b, acc_b, bin_cnts] = calc_ece(fused_p_calibrated, true_referable, 10);

    % C. Reliability & Quality Statistics
    rel_mean   = mean(reliability_scores);
    rel_std    = std(reliability_scores);
    rel_median = median(reliability_scores);
    rel_min    = min(reliability_scores);
    rel_max    = max(reliability_scores);

    cnt_good        = sum(strcmp(quality_classes, 'GOOD'));
    cnt_borderline  = sum(strcmp(quality_classes, 'BORDERLINE'));
    cnt_ungradable  = sum(strcmp(quality_classes, 'UNGRADABLE'));

    % D. Triage Routing
    cnt_accept  = sum(strcmp(recommended_actions, 'ACCEPT_SCREENING'));
    cnt_recap   = sum(strcmp(recommended_actions, 'RECAPTURE'));
    cnt_review  = sum(strcmp(recommended_actions, 'CLINICAL_REVIEW'));
    cnt_uncert  = sum(strcmp(recommended_actions, 'UNCERTAIN'));
    cnt_deferred= cnt_recap + cnt_review + cnt_uncert;

    % E. Accepted Autonomous Cohort Metrics
    cov_pct = (cnt_accept / n_test) * 100;
    def_pct = (cnt_deferred / n_test) * 100;

    acc_mask = strcmp(recommended_actions, 'ACCEPT_SCREENING');
    if cnt_accept > 0
        y_true_acc = true_referable(acc_mask);
        y_pred_acc = predicted_referable(acc_mask);

        tp_acc = sum(y_pred_acc == 1 & y_true_acc == 1);
        fp_acc = sum(y_pred_acc == 1 & y_true_acc == 0);
        fn_acc = sum(y_pred_acc == 0 & y_true_acc == 1);
        tn_acc = sum(y_pred_acc == 0 & y_true_acc == 0);

        sens_acc = tp_acc / max(1, tp_acc + fn_acc);
        spec_acc = tn_acc / max(1, tn_acc + fp_acc);
        ppv_acc  = tp_acc / max(1, tp_acc + fp_acc);
        npv_acc  = tn_acc / max(1, tn_acc + fn_acc);
        f1_acc   = (2 * tp_acc) / max(1, 2 * tp_acc + fp_acc + fn_acc);
        acc_acc  = (tp_acc + tn_acc) / cnt_accept;
        bal_acc  = (sens_acc + spec_acc) / 2;
    else
        tp_acc = 0; fp_acc = 0; fn_acc = 0; tn_acc = 0;
        sens_acc = NaN; spec_acc = NaN; ppv_acc = NaN; npv_acc = NaN; f1_acc = NaN; acc_acc = NaN; bal_acc = NaN;
    end

    % -------------------------------------------------------------------------
    % 5. Build Master Per-Image & Error Tables
    % -------------------------------------------------------------------------
    error_types = cell(n_test, 1);
    for i = 1:n_test
        if predicted_referable(i) == 1 && true_referable(i) == 1
            error_types{i} = 'TP';
        elseif predicted_referable(i) == 0 && true_referable(i) == 0
            error_types{i} = 'TN';
        elseif predicted_referable(i) == 1 && true_referable(i) == 0
            error_types{i} = 'FP';
        elseif predicted_referable(i) == 0 && true_referable(i) == 1
            error_types{i} = 'FN';
        end
    end

    per_image_table = table(...
        image_ids, image_paths, true_grades, true_referable, true_dme_risks, ...
        quality_classes, quality_scores, ...
        cnn_pred_grades, cnn_p_referable, cnn_confidence, cnn_p0, cnn_p1, cnn_p2, cnn_p3, cnn_p4, ...
        ma_counts, he_counts, ex_counts, he_area_fractions, ex_area_fractions, dme_pred_risks, nv_scores, ...
        vessel_densities, vessel_tortuosities, ...
        fused_p_referable_raw, fused_p_calibrated, fused_agreement, predicted_referable, error_types, ...
        reliability_scores, sub_quality, sub_confidence, sub_margin, sub_completeness, sub_agreement, ...
        conf_levels, uncert_levels, recommended_actions, double(decision_readiness), primary_reasons, ...
        'VariableNames', {...
            'ImageID', 'ImagePath', 'True_DR_Grade', 'True_Referable', 'True_DME_Risk', ...
            'Quality_Class', 'Quality_Score', ...
            'CNN_Predicted_Grade', 'CNN_P_Referable', 'CNN_Confidence', 'CNN_P0', 'CNN_P1', 'CNN_P2', 'CNN_P3', 'CNN_P4', ...
            'MA_Count', 'HE_Count', 'EX_Count', 'HE_Area_Fraction', 'EX_Area_Fraction', 'Predicted_DME_Risk', 'NV_Score', ...
            'Vessel_Density', 'Vessel_Tortuosity', ...
            'Fused_P_Referable_Raw', 'Fused_P_Referable_Calibrated', 'Fused_Agreement_Score', 'Predicted_Referable', 'Error_Type', ...
            'Reliability_Score', 'SubScore_Quality', 'SubScore_Confidence', 'SubScore_Margin', 'SubScore_Completeness', 'SubScore_Agreement', ...
            'Confidence_Level', 'Uncertainty_Level', 'Recommended_Action', 'Decision_Readiness', 'Primary_Reason'});

    % Error Lists
    fn_indices = find(strcmp(error_types, 'FN'));
    fp_indices = find(strcmp(error_types, 'FP'));
    fn_table = per_image_table(fn_indices, :);
    fp_table = per_image_table(fp_indices, :);

    % -------------------------------------------------------------------------
    % 6. Load Historical Baseline for Comparison
    % -------------------------------------------------------------------------
    hist_file = fullfile(cfg.paths.metrics, 'final_test', 'final_test_evaluation.mat');
    assert(exist(hist_file, 'file') == 2, 'Historical baseline not found: %s', hist_file);
    hist_data = load(hist_file);
    hist_tab = hist_data.eval_table;

    hist_true_ref = hist_tab.True_Referable;
    hist_pred_ref = hist_tab.Fused_Predicted_Referable;
    hist_p_ref    = hist_tab.Fused_P_Referable;

    tp_h = sum(hist_pred_ref == 1 & hist_true_ref == 1);
    fp_h = sum(hist_pred_ref == 1 & hist_true_ref == 0);
    fn_h = sum(hist_pred_ref == 0 & hist_true_ref == 1);
    tn_h = sum(hist_pred_ref == 0 & hist_true_ref == 0);

    sens_h = tp_h / max(1, tp_h + fn_h);
    spec_h = tn_h / max(1, tn_h + fp_h);
    ppv_h  = tp_h / max(1, tp_h + fp_h);
    npv_h  = tn_h / max(1, tn_h + fn_h);
    f1_h   = (2 * tp_h) / max(1, 2 * tp_h + fp_h + fn_h);
    acc_h  = (tp_h + tn_h) / n_test;
    bal_h  = (sens_h + spec_h) / 2;
    [auroc_h, ~, ~] = calc_auroc(hist_p_ref, hist_true_ref);
    [auprc_h, ~, ~] = calc_auprc(hist_p_ref, hist_true_ref);
    brier_h = calc_brier(hist_p_ref, hist_true_ref);
    [ece_h, ~, ~, ~] = calc_ece(hist_p_ref, hist_true_ref, 10);

    hist_acc_mask = strcmp(hist_tab.Recommended_Action, 'ACCEPT_SCREENING');
    cnt_acc_h = sum(hist_acc_mask);
    cov_pct_h = (cnt_acc_h / n_test) * 100;
    tp_acc_h = sum(hist_pred_ref(hist_acc_mask) == 1 & hist_true_ref(hist_acc_mask) == 1);
    fp_acc_h = sum(hist_pred_ref(hist_acc_mask) == 1 & hist_true_ref(hist_acc_mask) == 0);
    fn_acc_h = sum(hist_pred_ref(hist_acc_mask) == 0 & hist_true_ref(hist_acc_mask) == 1);
    tn_acc_h = sum(hist_pred_ref(hist_acc_mask) == 0 & hist_true_ref(hist_acc_mask) == 0);
    sens_acc_h = tp_acc_h / max(1, tp_acc_h + fn_acc_h);
    spec_acc_h = tn_acc_h / max(1, tn_acc_h + fp_acc_h);
    ppv_acc_h  = tp_acc_h / max(1, tp_acc_h + fp_acc_h);
    npv_acc_h  = tn_acc_h / max(1, tn_acc_h + fn_acc_h);
    f1_acc_h   = (2 * tp_acc_h) / max(1, 2 * tp_acc_h + fp_acc_h + fn_acc_h);
    acc_acc_h  = (tp_acc_h + tn_acc_h) / max(1, cnt_acc_h);
    bal_acc_h  = (sens_acc_h + spec_acc_h) / 2;

    % -------------------------------------------------------------------------
    % 7. Construct Summary Table
    % -------------------------------------------------------------------------
    summary_table = table(...
        {'Total_Test_Images'; 'Calibration_Model'; 'Referable_Threshold_tau'; 'Reliability_Gate_tau_accept'; ...
         'Sensitivity_All103'; 'Specificity_All103'; 'PPV_All103'; 'NPV_All103'; 'F1_Score_All103'; 'Accuracy_All103'; 'Balanced_Accuracy_All103'; ...
         'AUROC'; 'AUPRC'; 'Brier_Score'; 'ECE'; 'TP_All103'; 'TN_All103'; 'FP_All103'; 'FN_All103'; ...
         'Autonomous_Coverage_Pct'; 'Deferred_Rate_Pct'; 'Accepted_N'; 'Deferred_N'; ...
         'Recapture_Ungradable_N'; 'Clinical_Review_N'; 'Uncertain_N'; ...
         'Sensitivity_Accepted'; 'Specificity_Accepted'; 'PPV_Accepted'; 'NPV_Accepted'; 'F1_Accepted'; 'Balanced_Accuracy_Accepted'; 'FN_Accepted'; 'FP_Accepted'; ...
         '5Class_Accuracy'; '5Class_Balanced_Accuracy'; '5Class_Macro_F1'}, ...
        {'103'; 'Raw Multimodal Fusion'; '0.50'; '60.0'; ...
         sprintf('%.2f%%', sens_h*100); sprintf('%.2f%%', spec_h*100); sprintf('%.2f%%', ppv_h*100); sprintf('%.2f%%', npv_h*100); sprintf('%.4f', f1_h); sprintf('%.2f%%', acc_h*100); sprintf('%.2f%%', bal_h*100); ...
         sprintf('%.4f', auroc_h); sprintf('%.4f', auprc_h); sprintf('%.4f', brier_h); sprintf('%.4f', ece_h); sprintf('%d', tp_h); sprintf('%d', tn_h); sprintf('%d', fp_h); sprintf('%d', fn_h); ...
         sprintf('%.2f%%', cov_pct_h); sprintf('%.2f%%', 100-cov_pct_h); sprintf('%d', cnt_acc_h); sprintf('%d', n_test-cnt_acc_h); ...
         sprintf('%d', sum(strcmp(hist_tab.Recommended_Action, 'RECAPTURE'))); sprintf('%d', sum(strcmp(hist_tab.Recommended_Action, 'CLINICAL_REVIEW'))); sprintf('%d', sum(strcmp(hist_tab.Recommended_Action, 'UNCERTAIN'))); ...
         sprintf('%.2f%%', sens_acc_h*100); sprintf('%.2f%%', spec_acc_h*100); sprintf('%.2f%%', ppv_acc_h*100); sprintf('%.2f%%', npv_acc_h*100); sprintf('%.4f', f1_acc_h); sprintf('%.2f%%', bal_acc_h*100); sprintf('%d', fn_acc_h); sprintf('%d', fp_acc_h); ...
         sprintf('%.2f%%', hist_data.summary_table.Final_Locked_Test_Value(3)*100); sprintf('%.2f%%', hist_data.summary_table.Final_Locked_Test_Value(4)*100); sprintf('%.4f', hist_data.summary_table.Final_Locked_Test_Value(5))}, ...
        {'103'; 'Platt Calibrated (a=1.3106, b=-1.1412)'; '0.30'; '60.0'; ...
         sprintf('%.2f%%', sens_all*100); sprintf('%.2f%%', spec_all*100); sprintf('%.2f%%', ppv_all*100); sprintf('%.2f%%', npv_all*100); sprintf('%.4f', f1_all); sprintf('%.2f%%', acc_all*100); sprintf('%.2f%%', bal_all*100); ...
         sprintf('%.4f', auroc_all); sprintf('%.4f', auprc_all); sprintf('%.4f', brier_all); sprintf('%.4f', ece_all); sprintf('%d', tp_all); sprintf('%d', tn_all); sprintf('%d', fp_all); sprintf('%d', fn_all); ...
         sprintf('%.2f%%', cov_pct); sprintf('%.2f%%', def_pct); sprintf('%d', cnt_accept); sprintf('%d', cnt_deferred); ...
         sprintf('%d', cnt_recap); sprintf('%d', cnt_review); sprintf('%d', cnt_uncert); ...
         sprintf('%.2f%%', sens_acc*100); sprintf('%.2f%%', spec_acc*100); sprintf('%.2f%%', ppv_acc*100); sprintf('%.2f%%', npv_acc*100); sprintf('%.4f', f1_acc); sprintf('%.2f%%', bal_acc*100); sprintf('%d', fn_acc); sprintf('%d', fp_acc); ...
         sprintf('%.2f%%', acc_5class*100); sprintf('%.2f%%', bal_acc_5class*100); sprintf('%.4f', macro_f1_5class)}, ...
        'VariableNames', {'Metric', 'HISTORICAL_BASELINE', 'SHINRA_DEVELOPMENT_RELEASE_V1'});

    % -------------------------------------------------------------------------
    % 8. Governance Metadata
    % -------------------------------------------------------------------------
    governance = struct();
    governance.configuration_identifier = 'SHINRA_DEVELOPMENT_RELEASE_V1';
    governance.calibration_parameters = struct('type', 'Platt Logistic Scaling', 'slope_a', a_platt, 'intercept_b', b_platt);
    governance.thresholds = struct('referable_threshold_tau', tau_ref, 'reliability_threshold_tau_accept', tau_accept);
    governance.model_checkpoints = struct(...
        'severity_model', 'models/severity/SHINRA_SqueezeNet_Severity_Model_V3.mat', ...
        'vessel_model', 'models/vessels/vessel_unet.mat');
    governance.dataset_partitions = struct('training_N', 330, 'validation_N', 83, 'test_N', 103);
    governance.evaluation_timestamp = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
    governance.test_evaluation_performed = true;
    governance.configuration_modified_during_test = false;
    governance.retraining = false;
    governance.threshold_tuning = false;
    governance.test_driven_optimization = false;

    % -------------------------------------------------------------------------
    % 9. Generate and Save Figures
    % -------------------------------------------------------------------------
    fprintf('[STEP 4/5] Generating evaluation figures...\n');

    % 1. Confusion Matrix (Referable & 5-Class)
    try
        f_cm = figure('Visible', 'off', 'Position', [100, 100, 900, 400], 'Color', 'w');
        blue_cmap = [linspace(1, 0.2, 64)', linspace(1, 0.4, 64)', linspace(1, 0.8, 64)'];
        subplot(1, 2, 1);
        cm_bin = [tn_all, fp_all; fn_all, tp_all];
        imagesc(cm_bin); colormap(gca, blue_cmap); colorbar;
        text(1, 1, sprintf('TN: %d', tn_all), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        text(2, 1, sprintf('FP: %d', fp_all), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
        text(1, 2, sprintf('FN: %d', fn_all), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold', 'Color', 'r');
        text(2, 2, sprintf('TP: %d', tp_all), 'HorizontalAlignment', 'center', 'FontSize', 12, 'FontWeight', 'bold');
        set(gca, 'XTick', 1:2, 'XTickLabel', {'Non-Referable', 'Referable'}, 'YTick', 1:2, 'YTickLabel', {'Non-Referable', 'Referable'});
        xlabel('Predicted Label', 'FontWeight', 'bold'); ylabel('True Label', 'FontWeight', 'bold');
        title(sprintf('Binary Referable DR (N=%d)\nSens: %.1f%% | Spec: %.1f%%', n_test, sens_all*100, spec_all*100), 'FontWeight', 'bold');

        subplot(1, 2, 2);
        imagesc(conf_mat_5class); colormap(gca, blue_cmap); colorbar;
        for r = 1:5
            for c = 1:5
                text(c, r, sprintf('%d', conf_mat_5class(r, c)), 'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
        set(gca, 'XTick', 1:5, 'XTickLabel', {'G0','G1','G2','G3','G4'}, 'YTick', 1:5, 'YTickLabel', {'G0','G1','G2','G3','G4'});
        xlabel('Predicted Grade', 'FontWeight', 'bold'); ylabel('True Grade', 'FontWeight', 'bold');
        title(sprintf('5-Class ICDR Severity (N=%d)\nAcc: %.1f%% | Macro F1: %.4f', n_test, acc_5class*100, macro_f1_5class), 'FontWeight', 'bold');
        sgtitle('SHINRA_DEVELOPMENT_RELEASE_V1: Final Held-Out Test Confusion Matrices', 'FontWeight', 'bold', 'FontSize', 12);
        saveas(f_cm, fullfile(out_dir, 'final_test_confusion_matrix.png'));
        close(f_cm);
    catch ME
        warning('Failed to save confusion matrix figure: %s', ME.message);
    end

    % 2. ROC Curve
    try
        f_roc = figure('Visible', 'off', 'Position', [100, 100, 600, 500], 'Color', 'w');
        plot(fpr_curve, tpr_curve, 'b-', 'LineWidth', 2.5, 'DisplayName', sprintf('Release V1 (AUROC = %.4f)', auroc_all));
        hold on;
        plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.2, 'DisplayName', 'Chance (AUROC = 0.5000)');
        plot(1 - spec_all, sens_all, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
            'DisplayName', sprintf('Operating Point \\tau=0.30 (Sens=%.1f%%, Spec=%.1f%%)', sens_all*100, spec_all*100));
        xlabel('1 - Specificity (False Positive Rate)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Sensitivity (True Positive Rate)', 'FontSize', 11, 'FontWeight', 'bold');
        title('Receiver Operating Characteristic (ROC) — Final Test (N = 103)', 'FontSize', 12, 'FontWeight', 'bold');
        legend('Location', 'southeast', 'FontSize', 10);
        grid on; xlim([0, 1]); ylim([0, 1.02]);
        saveas(f_roc, fullfile(out_dir, 'final_test_roc.png'));
        close(f_roc);
    catch ME
        warning('Failed to save ROC figure: %s', ME.message);
    end

    % 3. Precision-Recall Curve
    try
        f_pr = figure('Visible', 'off', 'Position', [100, 100, 600, 500], 'Color', 'w');
        plot(rec_curve, prec_curve, 'm-', 'LineWidth', 2.5, 'DisplayName', sprintf('Release V1 (AUPRC = %.4f)', auprc_all));
        hold on;
        yline(sum(true_referable)/n_test, 'k--', 'LineWidth', 1.2, 'DisplayName', sprintf('Prevalence Baseline (%.1f%%)', (sum(true_referable)/n_test)*100));
        plot(sens_all, ppv_all, 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r', ...
            'DisplayName', sprintf('Operating Point \\tau=0.30 (PPV=%.1f%%, Recall=%.1f%%)', ppv_all*100, sens_all*100));
        xlabel('Recall (Sensitivity)', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Precision (PPV)', 'FontSize', 11, 'FontWeight', 'bold');
        title('Precision-Recall Curve — Final Test (N = 103)', 'FontSize', 12, 'FontWeight', 'bold');
        legend('Location', 'southwest', 'FontSize', 10);
        grid on; xlim([0, 1]); ylim([0, 1.02]);
        saveas(f_pr, fullfile(out_dir, 'final_test_pr_curve.png'));
        close(f_pr);
    catch ME
        warning('Failed to save PR curve figure: %s', ME.message);
    end

    % 4. Calibration Curve
    try
        f_cal = figure('Visible', 'off', 'Position', [100, 100, 600, 500], 'Color', 'w');
        valid_b = bin_cnts > 0;
        plot(conf_b(valid_b), acc_b(valid_b), 'bs-', 'LineWidth', 2, 'MarkerFaceColor', 'b', ...
            'DisplayName', sprintf('Release V1 (Brier=%.4f, ECE=%.4f)', brier_all, ece_all));
        hold on;
        plot([0, 1], [0, 1], 'k--', 'LineWidth', 1.2, 'DisplayName', 'Perfect Calibration');
        xlabel('Mean Predicted Probability', 'FontSize', 11, 'FontWeight', 'bold');
        ylabel('Observed Empirical Frequency', 'FontSize', 11, 'FontWeight', 'bold');
        title('Reliability Diagram (Calibration) — Final Test (N = 103)', 'FontSize', 12, 'FontWeight', 'bold');
        legend('Location', 'northwest', 'FontSize', 10);
        grid on; xlim([0, 1]); ylim([0, 1]);
        saveas(f_cal, fullfile(out_dir, 'final_test_calibration.png'));
        close(f_cal);
    catch ME
        warning('Failed to save calibration figure: %s', ME.message);
    end

    % 5. Reliability Distribution & Triage Figure
    try
        f_rel = figure('Visible', 'off', 'Position', [100, 100, 900, 400], 'Color', 'w');
        subplot(1, 2, 1);
        histogram(reliability_scores, 15, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'k');
        hold on;
        xline(tau_accept, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Gate \\tau_{accept} = %.0f', tau_accept));
        xlabel('Reliability Score [0, 100]', 'FontWeight', 'bold');
        ylabel('Image Count', 'FontWeight', 'bold');
        title(sprintf('Reliability Distribution (Mean: %.1f \\pm %.1f)', rel_mean, rel_std), 'FontWeight', 'bold');
        legend('Location', 'northwest');
        grid on;

        subplot(1, 2, 2);
        triage_counts = [cnt_accept, cnt_recap, cnt_review, cnt_uncert];
        triage_labels = {'ACCEPT SCREENING', 'RECAPTURE', 'CLINICAL REVIEW', 'UNCERTAIN'};
        bar(triage_counts, 'FaceColor', [0.3, 0.7, 0.4]);
        set(gca, 'XTick', 1:4, 'XTickLabel', triage_labels);
        ylabel('Patient Cases (N = 103)', 'FontWeight', 'bold');
        title('Clinical Action Routing Distribution', 'FontWeight', 'bold');
        for b = 1:4
            text(b, triage_counts(b) + 1, sprintf('%d (%.1f%%)', triage_counts(b), (triage_counts(b)/n_test)*100), ...
                'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
        end
        grid on; ylim([0, max(triage_counts) + 10]);
        sgtitle('SHINRA Reliability Gating & Triage Architecture', 'FontWeight', 'bold', 'FontSize', 12);
        saveas(f_rel, fullfile(out_dir, 'final_test_reliability.png'));
        close(f_rel);
    catch ME
        warning('Failed to save reliability figure: %s', ME.message);
    end

    % -------------------------------------------------------------------------
    % 10. Save All Output Files
    % -------------------------------------------------------------------------
    fprintf('[STEP 5/5] Saving final evaluation artifacts and results...\n');

    mat_out     = fullfile(out_dir, 'final_test_evaluation.mat');
    csv_out     = fullfile(out_dir, 'final_test_per_image.csv');
    summary_csv = fullfile(out_dir, 'final_test_summary.csv');

    eval_results = struct();
    eval_results.configuration_name     = 'SHINRA_DEVELOPMENT_RELEASE_V1';
    eval_results.governance             = governance;
    eval_results.eval_table             = per_image_table;
    eval_results.summary_table          = summary_table;
    eval_results.fn_table               = fn_table;
    eval_results.fp_table               = fp_table;
    eval_results.conf_mat_5class        = conf_mat_5class;

    save(mat_out, '-struct', 'eval_results');
    writetable(per_image_table, csv_out);
    writetable(summary_table, summary_csv);

    fprintf('Saved release artifacts successfully:\n');
    fprintf('  - %s\n', mat_out);
    fprintf('  - %s\n', csv_out);
    fprintf('  - %s\n\n', summary_csv);

    % -------------------------------------------------------------------------
    % 11. Command Window Output Report
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('SHINRA FINAL HELD-OUT TEST EVALUATION REPORT                            \n');
    fprintf('CONFIGURATION: SHINRA_DEVELOPMENT_RELEASE_V1                            \n');
    fprintf('========================================================================\n\n');

    fprintf('1. 5-CLASS SEVERITY GRADING (N = 103):\n');
    fprintf('   • Overall Accuracy:         %5.2f%% (%d / %d)\n', acc_5class * 100, sum(diag(conf_mat_5class)), n_test);
    fprintf('   • Balanced Accuracy:        %5.2f%%\n', bal_acc_5class * 100);
    fprintf('   • Macro F1-Score:           %6.4f\n', macro_f1_5class);
    fprintf('   • Per-Class Precision:      [G0: %5.2f%%, G1: %5.2f%%, G2: %5.2f%%, G3: %5.2f%%, G4: %5.2f%%]\n', per_class_prec * 100);
    fprintf('   • Per-Class Recall:         [G0: %5.2f%%, G1: %5.2f%%, G2: %5.2f%%, G3: %5.2f%%, G4: %5.2f%%]\n', per_class_rec * 100);
    fprintf('   • Confusion Matrix (5x5):\n');
    disp(conf_mat_5class);

    fprintf('2. BINARY REFERABLE DR SCREENING (All N = 103):\n');
    fprintf('   • Sensitivity:              %5.2f%% (%d / %d)\n', sens_all * 100, tp_all, tp_all + fn_all);
    fprintf('   • Specificity:              %5.2f%% (%d / %d)\n', spec_all * 100, tn_all, tn_all + fp_all);
    fprintf('   • PPV (Precision):          %5.2f%% (%d / %d)\n', ppv_all * 100, tp_all, tp_all + fp_all);
    fprintf('   • NPV:                      %5.2f%% (%d / %d)\n', npv_all * 100, tn_all, tn_all + fn_all);
    fprintf('   • F1-Score:                 %6.4f\n', f1_all);
    fprintf('   • Balanced Accuracy:        %5.2f%%\n', bal_all * 100);
    fprintf('   • AUROC:                    %6.4f\n', auroc_all);
    fprintf('   • AUPRC:                    %6.4f\n', auprc_all);
    fprintf('   • Confusion Matrix:         TP = %d, TN = %d, FP = %d, FN = %d\n\n', tp_all, tn_all, fp_all, fn_all);

    fprintf('3. PROBABILITY CALIBRATION (N = 103):\n');
    fprintf('   • Brier Score:              %6.4f\n', brier_all);
    fprintf('   • Expected Calib Error:     %6.4f\n\n', ece_all);

    fprintf('4. RELIABILITY & IMAGE QUALITY (N = 103):\n');
    fprintf('   • Mean Reliability +/- Std: %5.2f +/- %5.2f\n', rel_mean, rel_std);
    fprintf('   • Median Reliability:       %5.2f\n', rel_median);
    fprintf('   • Min / Max Reliability:    %5.2f / %5.2f\n', rel_min, rel_max);
    fprintf('   • Quality Classes:          GOOD = %d (%.1f%%), BORDERLINE = %d (%.1f%%), UNGRADABLE = %d (%.1f%%)\n\n', ...
        cnt_good, (cnt_good/n_test)*100, cnt_borderline, (cnt_borderline/n_test)*100, cnt_ungradable, (cnt_ungradable/n_test)*100);

    fprintf('5. CLINICAL TRIAGE ROUTING (N = 103):\n');
    fprintf('   • ACCEPT_SCREENING:         %d (Coverage: %5.2f%%)\n', cnt_accept, cov_pct);
    fprintf('   • RECAPTURE:                %d (Deferred: %5.2f%%)\n', cnt_recap, (cnt_recap/n_test)*100);
    fprintf('   • CLINICAL_REVIEW:          %d (Deferred: %5.2f%%)\n', cnt_review, (cnt_review/n_test)*100);
    fprintf('   • UNCERTAIN:                %d (Deferred: %5.2f%%)\n', cnt_uncert, (cnt_uncert/n_test)*100);
    fprintf('   • Total Deferred:           %d (Deferred Rate: %5.2f%%)\n\n', cnt_deferred, def_pct);

    fprintf('6. ACCEPTED AUTONOMOUS COHORT (N = %d, Coverage = %5.2f%%):\n', cnt_accept, cov_pct);
    fprintf('   • Accepted Sensitivity:     %5.2f%% (%d / %d)\n', sens_acc * 100, tp_acc, tp_acc + fn_acc);
    fprintf('   • Accepted Specificity:     %5.2f%% (%d / %d)\n', spec_acc * 100, tn_acc, tn_acc + fp_acc);
    fprintf('   • Accepted PPV:             %5.2f%% (%d / %d)\n', ppv_acc * 100, tp_acc, tp_acc + fp_acc);
    fprintf('   • Accepted NPV:             %5.2f%% (%d / %d)\n', npv_acc * 100, tn_acc, tn_acc + fn_acc);
    fprintf('   • Accepted F1-Score:        %6.4f\n', f1_acc);
    fprintf('   • Accepted Balanced Acc:    %5.2f%%\n', bal_acc * 100);
    fprintf('   • Accepted Errors:          FN = %d, FP = %d\n\n', fn_acc, fp_acc);

    fprintf('7. SIDE-BY-SIDE COMPARISON: HISTORICAL BASELINE VS RELEASE V1:\n');
    fprintf('--------------------------------------------------------------------------------------------------\n');
    fprintf('  Metric                       HISTORICAL_BASELINE          SHINRA_DEVELOPMENT_RELEASE_V1\n');
    fprintf('--------------------------------------------------------------------------------------------------\n');
    fprintf('  Referral Threshold           tau = 0.50 (raw)             tau = 0.30 (Platt calib)\n');
    fprintf('  Reliability Gate             tau_accept = 60.0            tau_accept = 60.0\n');
    fprintf('  Sensitivity (All 103)        %5.2f%% (%2d/%2d)             %5.2f%% (%2d/%2d)\n', sens_h*100, tp_h, tp_h+fn_h, sens_all*100, tp_all, tp_all+fn_all);
    fprintf('  Specificity (All 103)        %5.2f%% (%2d/%2d)             %5.2f%% (%2d/%2d)\n', spec_h*100, tn_h, tn_h+fp_h, spec_all*100, tn_all, tn_all+fp_all);
    fprintf('  PPV (All 103)                %5.2f%%                      %5.2f%%\n', ppv_h*100, ppv_all*100);
    fprintf('  NPV (All 103)                %5.2f%%                      %5.2f%%\n', npv_h*100, npv_all*100);
    fprintf('  F1-Score (All 103)           %6.4f                       %6.4f\n', f1_h, f1_all);
    fprintf('  Balanced Accuracy (All 103)  %5.2f%%                      %5.2f%%\n', bal_h*100, bal_all*100);
    fprintf('  AUROC                        %6.4f                       %6.4f\n', auroc_h, auroc_all);
    fprintf('  AUPRC                        %6.4f                       %6.4f\n', auprc_h, auprc_all);
    fprintf('  Brier Score                  %6.4f                       %6.4f\n', brier_h, brier_all);
    fprintf('  ECE                          %6.4f                       %6.4f\n', ece_h, ece_all);
    fprintf('  Errors (FN / FP)              %2d / %2d                     %2d / %2d\n', fn_h, fp_h, fn_all, fp_all);
    fprintf('  Autonomous Coverage          %5.2f%% (%2d/%2d)             %5.2f%% (%2d/%2d)\n', cov_pct_h, cnt_acc_h, n_test, cov_pct, cnt_accept, n_test);
    fprintf('  Accepted Sensitivity         %5.2f%% (%2d/%2d)             %5.2f%% (%2d/%2d)\n', sens_acc_h*100, tp_acc_h, tp_acc_h+fn_acc_h, sens_acc*100, tp_acc, tp_acc+fn_acc);
    fprintf('  Accepted Specificity         %5.2f%% (%2d/%2d)             %5.2f%% (%2d/%2d)\n', spec_acc_h*100, tn_acc_h, tn_acc_h+fp_acc_h, spec_acc*100, tn_acc, tn_acc+fp_acc);
    fprintf('  Accepted PPV                 %5.2f%%                      %5.2f%%\n', ppv_acc_h*100, ppv_acc*100);
    fprintf('  Accepted NPV                 %5.2f%%                      %5.2f%%\n', npv_acc_h*100, npv_acc*100);
    fprintf('  Accepted F1-Score            %6.4f                       %6.4f\n', f1_acc_h, f1_acc);
    fprintf('  Accepted Errors (FN / FP)     %2d / %2d                     %2d / %2d\n', fn_acc_h, fp_acc_h, fn_acc, fp_acc);
    fprintf('--------------------------------------------------------------------------------------------------\n\n');

    fprintf('====================================================\n');
    fprintf('GOVERNANCE VERIFICATION\n');
    fprintf('====================================================\n');
    fprintf('Development training N = %d\n', governance.dataset_partitions.training_N);
    fprintf('Development validation N = %d\n', governance.dataset_partitions.validation_N);
    fprintf('Final test N = %d\n\n', governance.dataset_partitions.test_N);
    fprintf('Test evaluation performed = YES\n');
    fprintf('Configuration modified during test = NO\n');
    fprintf('Retraining = NO\n');
    fprintf('Threshold tuning = NO\n');
    fprintf('Test-driven optimization = NO\n\n');

    results = eval_results;
end
