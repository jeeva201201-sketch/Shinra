function result = run_single_image(input_img, save_visualizations)
% RUN_SINGLE_IMAGE End-to-End SHINRA Retinal Screening for a Single Fundus Photograph
% Executes the complete closed-loop screening pipeline:
% 1. Image Loading & Validation
% 2. Safety Quality Gate & Defect Analysis
% 3. Adaptive Restorative Enhancement (if Borderline)
% 4. Retinal Anatomy Localization (Optic Disc & Fovea)
% 5. Vascular Architecture & U-Net Segmentation
% 6. Multi-Scale Lesion Quantification (MAs, Hemorrhages, Exudates, DME, NV)
% 7. 5-Level ICDR DR Severity Grading & Expected Score
% 8. Multi-Modal Evidence Fusion
% 9. System Reliability Index & Confidence
% 10. Grad-CAM Visual Attention & Interpretability Alignment
% 11. Final Clinical Disposition & Actionable Guidance
% 12. Automated Diagnostic Report Generation & Multi-Panel Visualization
%
% Usage:
%   result = run_single_image("path/to/retina.jpg");
%   result = run_single_image(img_matrix);

    cfg = shinra_config();
    if nargin < 2 || isempty(save_visualizations)
        save_visualizations = cfg.flags.save_visualizations;
    end

    % 1. Load and identify image
    if ischar(input_img) || isstring(input_img)
        img_path = char(input_img);
        if exist(img_path, 'file') ~= 2
            error('SHINRA Error: Image file not found: %s', img_path);
        end
        [~, img_id, ~] = fileparts(img_path);
        raw_img = imread(img_path);
    else
        raw_img = input_img;
        img_id = sprintf('IMG_%s', char(datetime('now', 'Format', 'yyyyMMdd_HHmmss')));
        img_path = 'in_memory';
    end

    % 2. Retinal Preprocessing & Initial FOV Detection
    prep_res = preprocessRetina(raw_img);

    % 3. Quality Gate Assessment
    [q_class, q_rep] = qualityClassifier(prep_res.rgb_normalized, prep_res.fov_mask);

    % 4. Closed-Loop Enhancement if Quality is Borderline
    if strcmp(q_class, 'BORDERLINE')
        [enhanced_rgb, q_rep_enhanced] = adaptiveEnhance(prep_res.rgb_normalized, prep_res.fov_mask, q_rep);
        prep_res.rgb_normalized = enhanced_rgb;
        q_rep = q_rep_enhanced;
    end

    % 5. Retinal Anatomy Localization
    od_res = detectOpticDisc(prep_res.rgb_normalized, prep_res.fov_mask);
    fovea_res = locateFovea(prep_res.rgb_normalized, od_res, prep_res.fov_mask);

    % 6. Retinal Vessel Segmentation & Morphometry
    vess_res = segmentVessels(prep_res.rgb_normalized, prep_res.fov_mask, od_res.mask);

    % 7. Multi-Scale Lesion Biomarker Extraction
    ma_res = detectMicroaneurysms(prep_res.rgb_normalized, prep_res.fov_mask, vess_res.binary_mask, od_res.mask);
    he_res = detectHemorrhages(prep_res.rgb_normalized, prep_res.fov_mask, vess_res.binary_mask, od_res.mask);
    ex_res = detectExudates(prep_res.rgb_normalized, prep_res.fov_mask, od_res.mask, fovea_res);
    nv_res = analyzeNeovascularization(vess_res.binary_mask, prep_res.fov_mask, od_res.mask, vess_res.morphometry);

    lesion_evidence = struct(...
        'ma_count', ma_res.count, 'ma_area', ma_res.total_area, 'ma_mask', ma_res.binary_mask, ...
        'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, 'he_mask', he_res.binary_mask, ...
        'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, 'ex_mask', ex_res.binary_mask, ...
        'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);

    % 8. 5-Level DR Severity Classification
    sev_res = predictDRSeverity(prep_res, lesion_evidence);

    % 9. Multi-Modal Evidence Fusion
    ev_vec = extractEvidenceVector(sev_res, lesion_evidence, vess_res, q_rep);
    fus_res = fuseEvidence(ev_vec, []);

    % 10. System Reliability & Screening Decision
    rel_res = computeReliability(q_rep, sev_res, fus_res, lesion_evidence, vess_res, od_res, fovea_res);
    dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);

    % 11. Visual Explainability (Grad-CAM)
    cam_map = generateGradCAM(prep_res.rgb_normalized, prep_res, lesion_evidence, prep_res.fov_mask);
    all_lesions_mask = ma_res.binary_mask | he_res.binary_mask | ex_res.binary_mask;
    agree_metrics = explanationAgreement(cam_map, all_lesions_mask, prep_res.fov_mask);

    % 12. Automated Clinical Report Generation
    result = generateClinicalReport(img_id, prep_res, q_rep, od_res, fovea_res, vess_res, ...
        ma_res, he_res, ex_res, nv_res, sev_res, fus_res, rel_res, dec_res);
    result.explainability_agreement = agree_metrics;

    % 13. Save 6-Panel Visualization
    if save_visualizations
        panel_path = generateExplanationPanel(img_id, prep_res, od_res, fovea_res, vess_res, ...
            ma_res, he_res, ex_res, cam_map, dec_res, sev_res);
        result.panel_figure_path = panel_path;
    end

    % Print Summary to Console
    fprintf('\n------------------------------------------------------------\n');
    fprintf('  SHINRA SCREENING SUMMARY: %s\n', img_id);
    fprintf('------------------------------------------------------------\n');
    fprintf('  • Quality:      %s (Score: %.1f/100)\n', q_rep.quality_class, q_rep.score);
    fprintf('  • DR Severity:  Level %d (%s)\n', sev_res.predicted_grade, sev_res.predicted_class_name);
    fprintf('  • Fused P(Ref): %.1f%%\n', fus_res.p_referable_fused * 100);
    fprintf('  • Lesions:      %d MAs, %d Hemorrhages, %d Exudates\n', ma_res.count, he_res.count, ex_res.count);
    fprintf('  • Reliability:  %.1f/100 (%s | Uncertainty: %s)\n', rel_res.reliability_score, rel_res.confidence_level, rel_res.uncertainty_level);
    fprintf('  • RECOMMENDED ACTION: >>> %s <<<\n', dec_res.decision);
    fprintf('  • Decision Readiness: %s\n', iif(dec_res.decision_readiness, 'READY (Accepted)', 'NOT READY (Review/Recapture)'));
    fprintf('  • Urgency:      %s\n', dec_res.urgency);
    fprintf('------------------------------------------------------------\n\n');
end

function out = iif(cond, val_true, val_false)
    if cond
        out = val_true;
    else
        out = val_false;
    end
end
