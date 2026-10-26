function result = runSHINRASingleImage(input_img, varargin)
% RUNSHINRASINGLEIMAGE Master Single-Image Retinal Screening Inference Pipeline
% Executes the complete modular SHINRA screening engine on a single retinal fundus photograph:
% 1. Image Loading & Verification
% 2. Safety Quality Gate & Tri-State Classification
% 3. Retinal Anatomy Localization (Optic Disc & Fovea)
% 4. Deep U-Net Vessel Segmentation & Morphometry
% 5. Multi-Scale Lesion Biomarker Extraction (MAs, Hemorrhages, Exudates, DME, NV)
% 6. 5-Class ICDR DR Severity Inference
% 7. Multi-Modal Evidence Fusion & Platt Calibration
% 8. 5-Factor Reliability Gating & Actionable Triage Decision
% 9. Deep Grad-CAM Visual Interpretability
% 10. Clinical Visualization Overlays & Structured Reporting

    cfg = shinra_config();

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

    % 2. Preprocessing & FOV Boundary Detection
    prep_res = preprocessRetina(raw_img);

    % 3. Quality Gate Assessment
    q_res = assessImageQuality(prep_res.rgb_normalized, prep_res.fov_mask);

    % Closed-loop adaptive enhancement if borderline
    if strcmp(q_res.quality_class, 'BORDERLINE')
        [enhanced_rgb, q_rep_enhanced] = adaptiveEnhance(prep_res.rgb_normalized, prep_res.fov_mask, q_res);
        prep_res.rgb_normalized = enhanced_rgb;
        q_res = assessImageQuality(prep_res.rgb_normalized, prep_res.fov_mask);
    end

    % 4. Retinal Anatomy Localization
    od_res = detectOpticDisc(prep_res.rgb_normalized, prep_res.fov_mask);
    fovea_res = locateFovea(prep_res.rgb_normalized, od_res, prep_res.fov_mask);

    % 5. Retinal Vessel Segmentation & Morphometry (Trained U-Net)
    vess_res = segmentVessels(prep_res.rgb_normalized, prep_res.fov_mask, od_res.mask);

    % 6. Multi-Scale Lesion Biomarker Extraction
    ma_res = detectMicroaneurysms(prep_res.rgb_normalized, prep_res.fov_mask, vess_res.binary_mask, od_res.mask);
    he_res = detectHemorrhages(prep_res.rgb_normalized, prep_res.fov_mask, vess_res.binary_mask, od_res.mask);
    ex_res = detectExudates(prep_res.rgb_normalized, prep_res.fov_mask, od_res.mask, fovea_res);
    nv_res = analyzeNeovascularization(vess_res.binary_mask, prep_res.fov_mask, od_res.mask, vess_res.morphometry);

    lesion_evidence = struct(...
        'ma_count', ma_res.count, 'ma_area', ma_res.total_area, 'ma_mask', ma_res.binary_mask, ...
        'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, 'he_mask', he_res.binary_mask, ...
        'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, 'ex_mask', ex_res.binary_mask, ...
        'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);

    % 7. 5-Class DR Severity Inference
    sev_res = runSeverityInference(prep_res);

    % 8. Referable DR Classification with Platt Probability Calibration
    ref_res = runReferableInference(sev_res, lesion_evidence, vess_res, q_res);

    % Construct fusion struct for reliability layer compatibility
    fus_compat = struct();
    fus_compat.p_referable_fused = ref_res.p_referable_calibrated;
    fus_compat.p_cnn_referable = sev_res.p_referable;
    fus_compat.agreement_score = ref_res.agreement_score;

    % 9. System Reliability & Screening Decision
    rel_res = computeReliabilityAnalysis(q_res, sev_res, fus_compat, lesion_evidence, vess_res, od_res, fovea_res);

    % 10. Visual Explainability (Deep Grad-CAM)
    cam_map = generateGradCAM(prep_res.rgb_normalized, prep_res, lesion_evidence, prep_res.fov_mask);
    all_lesions_mask = ma_res.binary_mask | he_res.binary_mask | ex_res.binary_mask;
    agree_metrics = explanationAgreement(cam_map, all_lesions_mask, prep_res.fov_mask);

    % 11. Generate Interactive UI Overlays
    overlays = generateClinicalOverlays(prep_res, od_res, fovea_res, vess_res, ma_res, he_res, ex_res, cam_map, sev_res, rel_res);

    % 12. Structure Comprehensive Output
    result = struct();
    result.image_id = img_id;
    result.image_path = img_path;
    result.preprocessing = prep_res;
    result.quality = q_res;
    result.anatomy = struct('optic_disc', od_res, 'fovea', fovea_res);
    result.vessels = vess_res;
    result.lesions = struct('microaneurysms', ma_res, 'hemorrhages', he_res, 'exudates', ex_res, 'neovascularization', nv_res);
    result.severity = sev_res;
    result.referable = ref_res;
    result.reliability = rel_res;
    result.explainability = struct('cam_map', cam_map, 'agreement', agree_metrics);
    result.overlays = overlays;
    
    % 13. Formatted Clinical Report
    result.report_text = generateSHINRAReport(result);
end
