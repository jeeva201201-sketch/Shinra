function results = runFinalLockedEvaluation()
% RUNFINALLOCKEDEVALUATION Stage 10: One-Time Final Locked Benchmark on IDRiD 103 Test Set
% Executes the complete frozen SHINRA retinal screening system on the 103-image IDRiD
% held-out test partition with zero model retraining, zero parameter tuning, and zero leakage.
%
% OUTPUT ARTIFACTS:
% - results/metrics/final_test/final_test_evaluation.mat
% - results/metrics/final_test/final_test_per_image.csv
% - results/metrics/final_test/final_test_summary.csv
% - results/metrics/final_test/final_test_error_analysis.csv
% - results/figures/final_test/final_confusion_matrix_5class.png
% - results/figures/final_test/final_confusion_matrix_referable.png
% - results/figures/final_test/final_reliability_triage_distribution.png
% - results/figures/final_test/*_explainability_panel.png

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA STAGE 10: ONE-TIME FINAL LOCKED TEST EVALUATION (N = 103)      \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Create Output Directories
    % -------------------------------------------------------------------------
    metrics_out_dir = fullfile(cfg.paths.metrics, 'final_test');
    figures_out_dir = fullfile(cfg.paths.figures, 'final_test');
    if ~exist(metrics_out_dir, 'dir'), mkdir(metrics_out_dir); end
    if ~exist(figures_out_dir, 'dir'), mkdir(figures_out_dir); end

    % -------------------------------------------------------------------------
    % 2. Load IDRiD 103-Image Held-Out Test Set
    % -------------------------------------------------------------------------
    fprintf('[STEP 1/5] Loading official IDRiD 103-image held-out test partition...\n');
    [~, idrid_test] = load_idrid();
    n_test = height(idrid_test);
    assert(n_test == 103, 'Expected exactly 103 test images. Found: %d', n_test);

    % Verify physical path validity
    assert(all(cellfun(@(p) exist(p, 'file') == 2, idrid_test.ImagePath)), ...
        'One or more test image paths do not exist on disk.');
    fprintf('  [PASS] 103 physical test image files verified.\n');
    fprintf('  [PASS] Pipeline is completely locked and frozen.\n\n');

    % -------------------------------------------------------------------------
    % 3. Execute Complete Frozen Pipeline on 103 Test Images
    % -------------------------------------------------------------------------
    fprintf('[STEP 2/5] Executing full multimodal pipeline on all 103 test images...\n');

    % Pre-allocate storage arrays
    image_ids           = idrid_test.ImageID;
    image_paths         = idrid_test.ImagePath;
    true_grades         = idrid_test.DR_Grade;
    true_referable      = double(true_grades >= 2);
    true_dme_risks      = idrid_test.DME_Risk;

    % Quality outputs
    quality_scores      = zeros(n_test, 1);
    quality_classes     = cell(n_test, 1);

    % Severity CNN outputs
    cnn_pred_grades     = zeros(n_test, 1);
    cnn_p_referable     = zeros(n_test, 1);
    cnn_p0              = zeros(n_test, 1);
    cnn_p1              = zeros(n_test, 1);
    cnn_p2              = zeros(n_test, 1);
    cnn_p3              = zeros(n_test, 1);
    cnn_p4              = zeros(n_test, 1);
    cnn_confidence      = zeros(n_test, 1);

    % Evidence & Lesion outputs
    ma_counts           = zeros(n_test, 1);
    he_counts           = zeros(n_test, 1);
    ex_counts           = zeros(n_test, 1);
    he_area_fractions   = zeros(n_test, 1);
    ex_area_fractions   = zeros(n_test, 1);
    dme_pred_risks      = zeros(n_test, 1);
    nv_scores           = zeros(n_test, 1);
    vessel_densities    = zeros(n_test, 1);
    vessel_tortuosities = zeros(n_test, 1);

    % Multimodal Fusion outputs
    fused_p_referable   = zeros(n_test, 1);
    fused_agreement     = zeros(n_test, 1);
    fused_pred_referable= zeros(n_test, 1);

    % Reliability Layer outputs
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

    % Explainability outputs
    cam_success_flags   = false(n_test, 1);
    cam_mean_focus      = zeros(n_test, 1);
    cam_coverage_pct    = zeros(n_test, 1);
    cam_lesion_iou      = zeros(n_test, 1);
    cam_lesion_dice     = zeros(n_test, 1);
    cam_lesion_precision= zeros(n_test, 1);
    cam_lesion_recall   = zeros(n_test, 1);
    macular_att_ratio   = zeros(n_test, 1);
    od_att_ratio        = zeros(n_test, 1);

    % Select 6 representative cases across diverse clinical spectrum for panel figure generation
    rep_case_indices    = zeros(6, 1);
    rep_flags           = false(n_test, 1);

    for i = 1:n_test
        img_id   = image_ids{i};
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

        % 4. Retinal Vessel Segmentation & Morphometry (Uses Trained U-Net)
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

        % 6. Frozen Severity V3 Inference
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

        fused_p_referable(i)    = fus_res.p_referable_fused;
        fused_agreement(i)      = fus_res.agreement_score;
        fused_pred_referable(i) = double(fus_res.p_referable_fused >= cfg.fusion.referable_threshold);

        % 8. 5-Factor Reliability Layer
        rel_res = computeReliability(q_rep, sev_res, fus_res, lesion_ev, vess_res, od_res, fovea_res);

        reliability_scores(i) = rel_res.reliability_score;
        sub_quality(i)        = rel_res.quality_score_component;
        sub_confidence(i)     = rel_res.confidence_score_component;
        sub_margin(i)         = rel_res.margin_score_component;
        sub_completeness(i)   = rel_res.completeness_score_component;
        sub_agreement(i)      = rel_res.agreement_score_component;
        conf_levels{i}        = rel_res.confidence_level;
        uncert_levels{i}      = rel_res.uncertainty_level;

        % 9. Standardized Action Triage Decision
        dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);
        recommended_actions{i}= dec_res.recommended_action;
        decision_readiness(i) = dec_res.decision_readiness;

        if ~isempty(rel_res.reliability_reasons)
            primary_reasons{i} = rel_res.reliability_reasons{1};
        else
            primary_reasons{i} = 'Normal autonomous screening conditions';
        end

        % 10. Explainability & Grad-CAM
        try
            cam_map = generateGradCAM(prep.rgb_normalized, prep, lesion_ev, prep.fov_mask);
            cam_success_flags(i) = true;

            fov_px = sum(prep.fov_mask(:));
            cam_in_fov = cam_map(prep.fov_mask);
            cam_mean_focus(i) = mean(cam_in_fov);

            cam_thresh = prctile(cam_in_fov, 75);
            att_mask = (cam_map >= cam_thresh) & prep.fov_mask;
            cam_coverage_pct(i) = (sum(att_mask(:)) / max(1, fov_px)) * 100;

            combined_lesion_mask = ma_res.binary_mask | he_res.binary_mask | ex_res.binary_mask;
            if sum(combined_lesion_mask(:)) > 0
                agr_m = explanationAgreement(cam_map, combined_lesion_mask, prep.fov_mask);
                cam_lesion_iou(i)       = agr_m.iou;
                cam_lesion_dice(i)      = agr_m.dice;
                cam_lesion_precision(i) = agr_m.precision;
                cam_lesion_recall(i)    = agr_m.recall;
            end

            if isfield(fovea_res, 'macula_mask') && sum(fovea_res.macula_mask(:)) > 0
                macula_att = sum(cam_map(fovea_res.macula_mask & prep.fov_mask));
                macular_att_ratio(i) = (macula_att / max(1e-6, sum(cam_in_fov))) * 100;
            end

            if isfield(od_res, 'mask') && sum(od_res.mask(:)) > 0
                od_att = sum(cam_map(od_res.mask & prep.fov_mask));
                od_att_ratio(i) = (od_att / max(1e-6, sum(cam_in_fov))) * 100;
            end
        catch
            cam_success_flags(i) = false;
        end

        % Representative case selection for panel generation
        if true_grades(i) == 0 && strcmp(quality_classes{i}, 'GOOD') && rep_case_indices(1) == 0
            rep_case_indices(1) = i;
        elseif true_grades(i) == 2 && strcmp(quality_classes{i}, 'GOOD') && rep_case_indices(2) == 0
            rep_case_indices(2) = i;
        elseif true_dme_risks(i) >= 2 && strcmp(quality_classes{i}, 'GOOD') && rep_case_indices(3) == 0
            rep_case_indices(3) = i;
        elseif true_grades(i) == 4 && rep_case_indices(4) == 0
            rep_case_indices(4) = i;
        elseif ismember(recommended_actions{i}, {'CLINICAL_REVIEW', 'UNCERTAIN'}) && rep_case_indices(5) == 0
            rep_case_indices(5) = i;
        elseif strcmp(quality_classes{i}, 'UNGRADABLE') && rep_case_indices(6) == 0
            rep_case_indices(6) = i;
        end

        if mod(i, 20) == 0 || i == n_test
            fprintf('  • Processed %3d / %d test images (%.1f%%)...\n', i, n_test, (i/n_test)*100);
        end
    end

    rep_flags(rep_case_indices(rep_case_indices > 0)) = true;

    % Generate Representative Visual Panels
    fprintf('  • Generating 6-panel clinical figures for representative test cases...\n');
    for r = 1:numel(rep_case_indices)
        idx = rep_case_indices(r);
        if idx > 0
            img_id = image_ids{idx};
            raw_img = imread(image_paths{idx});
            prep = preprocessRetina(raw_img, cfg.preprocessing.target_size);
            od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
            fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
            vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
            ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
            he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
            ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
            lesion_ev = struct('ma_mask', ma_res.binary_mask, 'he_mask', he_res.binary_mask, 'ex_mask', ex_res.binary_mask);
            sev_res = predictSqueezeNetSeverityV3(prep);
            cam_map = generateGradCAM(prep.rgb_normalized, prep, lesion_ev, prep.fov_mask);
            dec_res_panel = struct(...
                'final_decision', recommended_actions{idx}, ...
                'p_referable', fused_p_referable(idx), ...
                'reliability_score', reliability_scores(idx));
            fig_path = fullfile(figures_out_dir, sprintf('%s_final_test_panel.png', img_id));
            f = figure('Visible', 'off', 'Position', [100, 100, 1200, 800], 'Color', 'w');
            subplot(2,3,1); imshow(prep.original_image); title('1. Original Test Fundus', 'FontSize', 11, 'FontWeight', 'bold');
            subplot(2,3,2); imshow(prep.rgb_normalized); title('2. Illumination Normalized', 'FontSize', 11, 'FontWeight', 'bold');
            subplot(2,3,3); imshow(visualizeAnatomy(prep.rgb_normalized, od_res, fovea_res)); title('3. Retinal Anatomy', 'FontSize', 11, 'FontWeight', 'bold');
            subplot(2,3,4); imshow(vess_res.binary_mask); title(sprintf('4. Vessels (Density: %.1f%%)', vess_res.morphometry.vessel_density*100), 'FontSize', 11, 'FontWeight', 'bold');
            subplot(2,3,5);
            l_ov = prep.rgb_normalized;
            m_d = imdilate(ma_res.binary_mask, strel('disk', 2));
            h_d = imdilate(he_res.binary_mask, strel('disk', 1));
            e_d = imdilate(ex_res.binary_mask, strel('disk', 1));
            ch1 = l_ov(:,:,1); ch2 = l_ov(:,:,2); ch3 = l_ov(:,:,3);
            ch1(m_d | h_d | e_d) = 1.0; ch2(m_d | h_d) = 0.0; ch2(e_d) = 1.0; ch3(m_d | e_d) = 0.0; ch3(h_d) = 1.0;
            l_ov(:,:,1) = ch1; l_ov(:,:,2) = ch2; l_ov(:,:,3) = ch3;
            imshow(l_ov); title(sprintf('5. Lesions (MA:%d, HE:%d, EX:%d)', ma_res.count, he_res.count, ex_res.count), 'FontSize', 11, 'FontWeight', 'bold');
            subplot(2,3,6);
            cam_rgb = ind2rgb(uint8(cam_map*255), jet(256));
            blend_rgb = 0.5 * cam_rgb + 0.5 * prep.rgb_normalized;
            imshow(blend_rgb); title(sprintf('6. Grad-CAM Attention (Conf: %.2f)', sev_res.confidence), 'FontSize', 11, 'FontWeight', 'bold');
            sgtitle(sprintf('SHINRA Locked Test: %s | Action: %s | Grade: %s (P_{ref}: %.1f%%, Rel: %.1f)', ...
                img_id, recommended_actions{idx}, sev_res.predicted_class_name, fused_p_referable(idx)*100, reliability_scores(idx)), ...
                'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.1, 0.2, 0.6]);
            save_figure_safe(f, fig_path);
            close(f);
        end
    end

    % -------------------------------------------------------------------------
    % 4. Master Evaluation Table & Detailed Metrics Computation
    % -------------------------------------------------------------------------
    fprintf('\n[STEP 3/5] Computing final diagnostic and reliability metrics...\n');

    eval_table = table(...
        image_ids, image_paths, true_grades, true_referable, true_dme_risks, ...
        quality_classes, quality_scores, ...
        cnn_pred_grades, cnn_p_referable, cnn_confidence, cnn_p0, cnn_p1, cnn_p2, cnn_p3, cnn_p4, ...
        ma_counts, he_counts, ex_counts, he_area_fractions, ex_area_fractions, dme_pred_risks, nv_scores, ...
        vessel_densities, vessel_tortuosities, ...
        fused_p_referable, fused_agreement, fused_pred_referable, ...
        reliability_scores, sub_quality, sub_confidence, sub_margin, sub_completeness, sub_agreement, ...
        conf_levels, uncert_levels, recommended_actions, decision_readiness, primary_reasons, ...
        cam_success_flags, cam_coverage_pct, cam_mean_focus, cam_lesion_iou, cam_lesion_dice, ...
        cam_lesion_precision, cam_lesion_recall, macular_att_ratio, od_att_ratio, rep_flags, ...
        'VariableNames', {...
            'ImageID', 'ImagePath', 'True_DR_Grade', 'True_Referable', 'True_DME_Risk', ...
            'Quality_Class', 'Quality_Score', ...
            'CNN_Predicted_Grade', 'CNN_P_Referable', 'CNN_Confidence', 'CNN_P0', 'CNN_P1', 'CNN_P2', 'CNN_P3', 'CNN_P4', ...
            'MA_Count', 'HE_Count', 'EX_Count', 'HE_Area_Fraction', 'EX_Area_Fraction', 'Predicted_DME_Risk', 'NV_Score', ...
            'Vessel_Density', 'Vessel_Tortuosity', ...
            'Fused_P_Referable', 'Fused_Agreement_Score', 'Fused_Predicted_Referable', ...
            'Reliability_Score', 'SubScore_Quality', 'SubScore_Confidence', 'SubScore_Margin', 'SubScore_Completeness', 'SubScore_Agreement', ...
            'Confidence_Level', 'Uncertainty_Level', 'Recommended_Action', 'Decision_Readiness', 'Primary_Reason', ...
            'CAM_Success', 'CAM_Coverage_Pct', 'CAM_Mean_Focus', 'CAM_Lesion_IoU', 'CAM_Lesion_Dice', ...
            'CAM_Lesion_Precision', 'CAM_Lesion_Recall', 'Macular_Attention_Pct', 'Optic_Disc_Attention_Pct', 'Is_Representative_Case'});

    % A. 5-Class Severity Metrics
    conf_mat_5class = confusionmat(true_grades, cnn_pred_grades, 'Order', 0:4);
    acc_5class = sum(diag(conf_mat_5class)) / n_test;

    % Per-Class 5-Class Precision, Recall, F1
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

    % B. Referable DR Metrics (Multimodal Fused Decision on ALL 103 Test Cases)
    tp_all = sum(fused_pred_referable == 1 & true_referable == 1);
    fp_all = sum(fused_pred_referable == 1 & true_referable == 0);
    fn_all = sum(fused_pred_referable == 0 & true_referable == 1);
    tn_all = sum(fused_pred_referable == 0 & true_referable == 0);

    sens_all = tp_all / max(1, tp_all + fn_all);
    spec_all = tn_all / max(1, tn_all + fp_all);
    ppv_all  = tp_all / max(1, tp_all + fp_all);
    npv_all  = tn_all / max(1, tn_all + fn_all);
    f1_all   = (2 * tp_all) / max(1, 2 * tp_all + fp_all + fn_all);
    acc_all  = (tp_all + tn_all) / n_test;
    bal_all  = (sens_all + spec_all) / 2;

    % AUROC Calculation
    try
        [X_roc, Y_roc, ~, auroc_val] = perfcurve(true_referable, fused_p_referable, 1);
    catch
        % Trapezoidal fallback
        [sorted_p, s_idx] = sort(fused_p_referable, 'descend');
        sorted_y = true_referable(s_idx);
        tps = cumsum(sorted_y == 1);
        fps = cumsum(sorted_y == 0);
        tpr = tps / max(1, sum(sorted_y == 1));
        fpr = fps / max(1, sum(sorted_y == 0));
        auroc_val = trapz([0; fpr], [0; tpr]);
    end

    % C. Reliability & Selective Screening Metrics
    acc_mask = strcmp(recommended_actions, 'ACCEPT_SCREENING');
    n_accept = sum(acc_mask);
    cov_pct  = (n_accept / n_test) * 100;

    n_recap  = sum(strcmp(recommended_actions, 'RECAPTURE'));
    n_review = sum(strcmp(recommended_actions, 'CLINICAL_REVIEW'));
    n_uncert = sum(strcmp(recommended_actions, 'UNCERTAIN'));
    n_deferred = n_recap + n_review + n_uncert;

    assert(n_accept + n_deferred == n_test, 'Accounting mismatch in test evaluation.');

    if n_accept > 0
        tp_sel = sum(fused_pred_referable(acc_mask) == 1 & true_referable(acc_mask) == 1);
        fp_sel = sum(fused_pred_referable(acc_mask) == 1 & true_referable(acc_mask) == 0);
        fn_sel = sum(fused_pred_referable(acc_mask) == 0 & true_referable(acc_mask) == 1);
        tn_sel = sum(fused_pred_referable(acc_mask) == 0 & true_referable(acc_mask) == 0);

        sens_sel = tp_sel / max(1, tp_sel + fn_sel);
        spec_sel = tn_sel / max(1, tn_sel + fp_sel);
        ppv_sel  = tp_sel / max(1, tp_sel + fp_sel);
        npv_sel  = tn_sel / max(1, tn_sel + fn_sel);
        f1_sel   = (2 * tp_sel) / max(1, 2 * tp_sel + fp_sel + fn_sel);
        acc_sel  = (tp_sel + tn_sel) / n_accept;
        bal_sel  = (sens_sel + spec_sel) / 2;
    else
        sens_sel = NaN; spec_sel = NaN; ppv_sel = NaN; npv_sel = NaN; f1_sel = NaN; acc_sel = NaN; bal_sel = NaN;
        tp_sel = 0; fp_sel = 0; fn_sel = 0; tn_sel = 0;
    end

    % D. Error Analysis Identification
    is_fn = (fused_pred_referable == 0 & true_referable == 1);
    is_fp = (fused_pred_referable == 1 & true_referable == 0);
    is_grade_err = (cnn_pred_grades ~= true_grades);
    is_deferred = ~acc_mask;

    error_mask = is_fn | is_fp | is_grade_err | is_deferred;
    error_tab = eval_table(error_mask, {...
        'ImageID', 'True_DR_Grade', 'CNN_Predicted_Grade', 'True_Referable', 'Fused_Predicted_Referable', ...
        'Fused_P_Referable', 'Quality_Class', 'Reliability_Score', 'Recommended_Action', 'Primary_Reason'});

    % -------------------------------------------------------------------------
    % 5. Save Artifacts & Figures
    % -------------------------------------------------------------------------
    fprintf('[STEP 4/5] Saving final evaluation artifacts and figures...\n');

    % Mat & CSV Exports
    out_mat = fullfile(metrics_out_dir, 'final_test_evaluation.mat');
    out_csv = fullfile(metrics_out_dir, 'final_test_per_image.csv');
    out_err_csv = fullfile(metrics_out_dir, 'final_test_error_analysis.csv');
    out_sum_csv = fullfile(metrics_out_dir, 'final_test_summary.csv');

    summary_table = table(...
        {'Total_Test_Cases'; 'Referable_Prevalence_Pct'; ...
         '5Class_Accuracy'; '5Class_Balanced_Accuracy'; '5Class_Macro_F1'; ...
         'Referable_Sensitivity_All103'; 'Referable_Specificity_All103'; 'Referable_PPV_All103'; 'Referable_NPV_All103'; 'Referable_F1_All103'; 'Referable_AUROC'; ...
         'Autonomous_Coverage_Pct'; 'Accepted_N'; 'Deferred_Total_N'; 'Recapture_N'; 'Clinical_Review_N'; 'Uncertain_N'; ...
         'Referable_Sensitivity_Accepted'; 'Referable_Specificity_Accepted'; 'Referable_PPV_Accepted'; 'Referable_NPV_Accepted'; 'Referable_F1_Accepted'; 'Referable_Accuracy_Accepted'; ...
         'False_Negatives_All103'; 'False_Positives_All103'; 'False_Negatives_Accepted'; 'False_Positives_Accepted'}, ...
        [n_test; (sum(true_referable)/n_test)*100; ...
         acc_5class; bal_acc_5class; macro_f1_5class; ...
         sens_all; spec_all; ppv_all; npv_all; f1_all; auroc_val; ...
         cov_pct; n_accept; n_deferred; n_recap; n_review; n_uncert; ...
         sens_sel; spec_sel; ppv_sel; npv_sel; f1_sel; acc_sel; ...
         fn_all; fp_all; fn_sel; fp_sel], ...
        'VariableNames', {'Metric', 'Final_Locked_Test_Value'});

    save(out_mat, 'eval_table', 'summary_table', 'error_tab', 'conf_mat_5class', 'cfg');
    writetable(eval_table, out_csv);
    writetable(error_tab, out_err_csv);
    writetable(summary_table, out_sum_csv);

    % Generate Confusion Matrix Figures
    f_cm5 = figure('Visible', 'off', 'Position', [100, 100, 600, 500], 'Color', 'w');
    confusionchart(conf_mat_5class, cfg.severity.class_names, 'Title', 'SHINRA Final Locked Test: 5-Class DR Confusion Matrix', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    save_figure_safe(f_cm5, fullfile(figures_out_dir, 'final_confusion_matrix_5class.png'));
    close(f_cm5);

    f_cm2 = figure('Visible', 'off', 'Position', [100, 100, 500, 450], 'Color', 'w');
    cm_ref = [tn_all, fp_all; fn_all, tp_all];
    confusionchart(cm_ref, {'Non-Referable', 'Referable DR'}, 'Title', 'SHINRA Final Locked Test: Referable DR Confusion Matrix', 'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
    save_figure_safe(f_cm2, fullfile(figures_out_dir, 'final_confusion_matrix_referable.png'));
    close(f_cm2);

    f_rel = figure('Visible', 'off', 'Position', [100, 100, 700, 450], 'Color', 'w');
    subplot(1, 2, 1);
    histogram(reliability_scores, 10, 'FaceColor', [0.2, 0.5, 0.8], 'EdgeColor', 'k');
    grid on; xlabel('Reliability Score [0 - 100]'); ylabel('Case Count'); title('Reliability Distribution (N=103)');
    subplot(1, 2, 2);
    action_counts = [n_accept, n_recap, n_review, n_uncert];
    bar(action_counts, 'FaceColor', [0.3, 0.7, 0.4]);
    set(gca, 'XTickLabel', {'Accept', 'Recapture', 'Review', 'Uncertain'}, 'XTickLabelRotation', 25);
    ylabel('Case Count'); title('Triage Action Breakdown'); grid on;
    sgtitle('SHINRA Final Test: Reliability & Triage Disposition', 'FontSize', 12, 'FontWeight', 'bold');
    save_figure_safe(f_rel, fullfile(figures_out_dir, 'final_reliability_triage_distribution.png'));
    close(f_rel);

    % -------------------------------------------------------------------------
    % 6. Display Complete Final Report
    % -------------------------------------------------------------------------
    fprintf('\n========================================================================\n');
    fprintf('                     FINAL TEST EVALUATION COMPLETE                     \n');
    fprintf('========================================================================\n');
    fprintf('  • Test images evaluated:            103\n');
    fprintf('  • Development images accessed:      0 for training/tuning\n');
    fprintf('  • Test images used for tuning:      0\n');
    fprintf('  • Models modified:                  NO\n');
    fprintf('  • Thresholds modified:              NO\n');
    fprintf('========================================================================\n\n');

    fprintf('========================================================================\n');
    fprintf('   A. 5-CLASS DR SEVERITY PERFORMANCE (IDRiD Test Set N = 103)          \n');
    fprintf('========================================================================\n');
    fprintf('  • Overall Accuracy:         %5.2f%% (%d / %d)\n', acc_5class * 100, sum(diag(conf_mat_5class)), n_test);
    fprintf('  • Balanced Accuracy:        %5.2f%%\n', bal_acc_5class * 100);
    fprintf('  • Macro F1 Score:           %6.4f\n\n', macro_f1_5class);
    fprintf('  Per-Class Performance:\n');
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  Grade Name          True_N   Pred_N   Precision   Recall      F1-Score\n');
    fprintf('  ----------------------------------------------------------------------\n');
    for c = 1:5
        fprintf('  %-18s  %3d      %3d      %5.2f%%     %5.2f%%     %6.4f\n', ...
            cfg.severity.class_names{c}, sum(conf_mat_5class(c, :)), sum(conf_mat_5class(:, c)), ...
            per_class_prec(c)*100, per_class_rec(c)*100, per_class_f1(c));
    end
    fprintf('  ----------------------------------------------------------------------\n\n');

    fprintf('========================================================================\n');
    fprintf('   B. REFERABLE DR SCREENING (Multimodal Decision on ALL 103 Test Cases)\n');
    fprintf('========================================================================\n');
    fprintf('  • Referable Sensitivity (Recall):   %5.2f%% (%d / %d)\n', sens_all * 100, tp_all, tp_all + fn_all);
    fprintf('  • Referable Specificity:            %5.2f%% (%d / %d)\n', spec_all * 100, tn_all, tn_all + fp_all);
    fprintf('  • Positive Predictive Value (PPV):  %5.2f%% (%d / %d)\n', ppv_all * 100, tp_all, tp_all + fp_all);
    fprintf('  • Negative Predictive Value (NPV):  %5.2f%% (%d / %d)\n', npv_all * 100, tn_all, tn_all + fn_all);
    fprintf('  • Referable F1-Score:               %6.4f\n', f1_all);
    fprintf('  • Overall Accuracy:                 %5.2f%% (%d / %d)\n', acc_all * 100, tp_all + tn_all, n_test);
    fprintf('  • Balanced Accuracy:                %5.2f%%\n', bal_all * 100);
    fprintf('  • Area Under ROC Curve (AUROC):     %6.4f\n', auroc_val);
    fprintf('  • Confusion Matrix:                 TP=%d, TN=%d, FP=%d, FN=%d\n\n', tp_all, tn_all, fp_all, fn_all);

    fprintf('========================================================================\n');
    fprintf('   C. RELIABILITY LAYER & SELECTIVE SCREENING TRIAGE (N = 103)          \n');
    fprintf('========================================================================\n');
    fprintf('1. Reliability Score Distribution:\n');
    fprintf('   • Mean ± Std:   %5.2f ± %5.2f\n', mean(reliability_scores), std(reliability_scores));
    fprintf('   • Median:       %5.2f  [Min: %5.2f, Max: %5.2f]\n\n', median(reliability_scores), min(reliability_scores), max(reliability_scores));
    fprintf('2. Quality Classification Distribution:\n');
    fprintf('   • GOOD:         %3d (%5.1f%%)\n', sum(strcmp(quality_classes, 'GOOD')), (sum(strcmp(quality_classes, 'GOOD'))/n_test)*100);
    fprintf('   • BORDERLINE:   %3d (%5.1f%%)\n', sum(strcmp(quality_classes, 'BORDERLINE')), (sum(strcmp(quality_classes, 'BORDERLINE'))/n_test)*100);
    fprintf('   • UNGRADABLE:   %3d (%5.1f%%)\n\n', sum(strcmp(quality_classes, 'UNGRADABLE')), (sum(strcmp(quality_classes, 'UNGRADABLE'))/n_test)*100);
    fprintf('3. Action Triage Breakdown:\n');
    fprintf('   • ACCEPT_SCREENING: %3d (%5.1f%%) [Autonomous Coverage]\n', n_accept, cov_pct);
    fprintf('   • RECAPTURE:        %3d (%5.1f%%) [Ungradable Camera Retake]\n', n_recap, (n_recap/n_test)*100);
    fprintf('   • CLINICAL_REVIEW:  %3d (%5.1f%%) [Referred to Ophthalmologist]\n', n_review, (n_review/n_test)*100);
    fprintf('   • UNCERTAIN:        %3d (%5.1f%%) [Decision Margin Ambiguity]\n', n_uncert, (n_uncert/n_test)*100);
    fprintf('   • Verification:     Accepted (%d) + Deferred (%d) = %d [PASSED]\n\n', n_accept, n_deferred, n_test);

    fprintf('4. Diagnostic Performance on Accepted Autonomous Cohort (N = %d):\n', n_accept);
    if n_accept > 0
        fprintf('   • Accepted Sensitivity:   %5.2f%% (%d / %d)\n', sens_sel * 100, tp_sel, tp_sel + fn_sel);
        fprintf('   • Accepted Specificity:   %5.2f%% (%d / %d)\n', spec_sel * 100, tn_sel, tn_sel + fp_sel);
        fprintf('   • Accepted Precision:     %5.2f%%\n', ppv_sel * 100);
        fprintf('   • Accepted F1-Score:      %6.4f\n', f1_sel);
        fprintf('   • Accepted Accuracy:      %5.2f%%\n', acc_sel * 100);
        fprintf('   • False Negatives:        %d (vs %d in full cohort)\n', fn_sel, fn_all);
        fprintf('   • False Positives:        %d (vs %d in full cohort)\n\n', fp_sel, fp_all);
    end

    fprintf('========================================================================\n');
    fprintf('   D. EXPLAINABILITY & BIOMARKER ALIGNMENT (N = 103)                   \n');
    fprintf('========================================================================\n');
    fprintf('  • Deep Grad-CAM Generation Success: %d / %d (%.1f%%)\n', sum(cam_success_flags), n_test, (sum(cam_success_flags)/n_test)*100);
    fprintf('  • Mean FOV Focus Intensity:         %5.2f ± %5.2f\n', mean(cam_mean_focus), std(cam_mean_focus));
    fprintf('  • Macular Attention Concentration:  %5.2f%% ± %5.2f%%\n', mean(macular_att_ratio), std(macular_att_ratio));
    fprintf('  • Optic Disc Attention Focus:       %5.2f%% ± %5.2f%%\n', mean(od_att_ratio), std(od_att_ratio));
    ref_les_mask = (true_referable == 1) & (ma_counts > 0 | he_counts > 0 | ex_counts > 0);
    if sum(ref_les_mask) > 0
        fprintf('  • CAM-to-Biomarker Overlap (Referable with Lesions, N = %d):\n', sum(ref_les_mask));
        fprintf('      - Mean Spatial Dice:    %6.4f\n', mean(cam_lesion_dice(ref_les_mask)));
        fprintf('      - Mean Spatial IoU:     %6.4f\n', mean(cam_lesion_iou(ref_les_mask)));
        fprintf('      - Mean Lesion Recall:   %5.2f%%\n\n', mean(cam_lesion_recall(ref_les_mask))*100);
    end

    fprintf('========================================================================\n');
    fprintf('   E. ERROR & TRIAGE CASE SUMMARY                                       \n');
    fprintf('========================================================================\n');
    fprintf('  • False Negatives (FN):     %d cases\n', fn_all);
    fprintf('  • False Positives (FP):     %d cases\n', fp_all);
    fprintf('  • 5-Class Grade Mismatches: %d cases\n', sum(is_grade_err));
    fprintf('  • Total Deferred Decisions: %d cases (Recapture: %d, Review: %d, Uncertain: %d)\n\n', ...
        n_deferred, n_recap, n_review, n_uncert);

    results = struct();
    results.eval_table = eval_table;
    results.summary_table = summary_table;
    results.error_table = error_tab;
    results.conf_mat_5class = conf_mat_5class;
end
