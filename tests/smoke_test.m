function test_summary = smoke_test()
% SMOKE_TEST Comprehensive Integration & Component Test Suite for SHINRA
% Verifies that all modules execute without syntax errors, missing paths,
% invalid matrix bounds, or numerical instability (NaN/Inf).
%
% Note: This is an environment and execution test harness, not AI performance evaluation.

    fprintf('============================================================\n');
    fprintf('           SHINRA SYSTEM INTEGRATION SMOKE TEST             \n');
    fprintf('============================================================\n\n');

    start_all = tic;
    tests = {
        '1. Project Setup & Paths',             @test_setup;
        '2. Dataset Discovery & Loading',       @test_datasets;
        '3. Retinal Preprocessing & FOV',       @test_preprocessing_module;
        '4. Quality Gate & Tri-State Triage',   @test_quality_module;
        '5. Retinal Anatomy Localization',      @test_anatomy_module;
        '6. Retinal Vessel Segmentation',       @test_vessels_module;
        '7. Multi-Scale Lesion Extractors',     @test_lesions_module;
        '8. DR Severity Module & API Check',    @test_severity_module;
        '9. Evidence Fusion & Reliability',     @test_fusion_module;
        '10. Visual Explainability & CAM',      @test_explainability_module;
        '11. Clinical Report Structure',        @test_reporting_module;
        '12. 12-Case Reliability Layer Suite',  @test_reliability;
        '13. End-to-End Pipeline Execution',    @test_end_to_end
    };

    num_tests = size(tests, 1);
    results = cell(num_tests, 3);
    passed_count = 0;

    for t = 1:num_tests
        t_name = tests{t, 1};
        t_func = tests{t, 2};
        
        fprintf('[%2d/%2d] Running: %-38s ... ', t, num_tests, t_name);
        t_start = tic;
        try
            t_func();
            t_dur = toc(t_start);
            fprintf('PASSED (%.2f s)\n', t_dur);
            results{t, 1} = t_name;
            results{t, 2} = 'PASSED';
            results{t, 3} = t_dur;
            passed_count = passed_count + 1;
        catch ME
            t_dur = toc(t_start);
            fprintf('FAILED (%.2f s)\n   --> Error: %s\n', t_dur, ME.message);
            results{t, 1} = t_name;
            results{t, 2} = sprintf('FAILED: %s', ME.message);
            results{t, 3} = t_dur;
        end
    end

    total_time = toc(start_all);
    fprintf('\n============================================================\n');
    fprintf('  SMOKE TEST SUMMARY: %d / %d PASSED (Total: %.2f s)\n', passed_count, num_tests, total_time);
    if passed_count == num_tests
        fprintf('  STATUS: ALL SYSTEM TESTS PASSED SUCCESSFULLY!\n');
    else
        fprintf('  STATUS: WARNING - SOME TESTS FAILED.\n');
    end
    fprintf('============================================================\n\n');

    test_summary = cell2table(results, 'VariableNames', {'TestName', 'Status', 'Duration_Sec'});
end

% ---------------- Helper Test Functions ----------------

function test_setup()
    status = setup_SHINRA();
    assert(status.initialized, 'Setup did not initialize properly.');
end

function test_datasets()
    inventory = scan_datasets();
    assert(inventory.idrid.num_train_images > 0, 'IDRiD train images not found.');
    [idrid_tr, idrid_te] = load_idrid();
    assert(height(idrid_tr) > 0, 'IDRiD table empty.');
    [drv_tr, drv_te] = load_drive();
    assert(height(drv_tr) > 0, 'DRIVE train table empty.');
    aptos_tab = load_aptos();
    assert(height(aptos_tab) > 0, 'APTOS table empty.');
end

function test_preprocessing_module()
    synth_img = uint8(repmat(zeros(300, 300), [1, 1, 3]));
    % Draw circular fundus disc
    [xx, yy] = meshgrid(1:300, 1:300);
    disc = ((xx - 150).^2 + (yy - 150).^2) <= (120^2);
    for c = 1:3
        synth_img(:, :, c) = uint8(disc * (120 + 30*c));
    end
    prep = preprocessRetina(synth_img, [256, 256]);
    assert(all(size(prep.rgb_normalized) == [256, 256, 3]), 'Preprocessed size mismatch.');
    assert(~any(isnan(prep.rgb_normalized(:))), 'NaN detected in preprocessed image.');
end

function test_quality_module()
    synth_img = im2double(imread(fullfile(shinra_config().paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg')));
    [fov_m, ~, ~] = detectRetinalFOV(synth_img);
    feats = qualityFeatures(synth_img, fov_m);
    [score, subscores] = qualityScore(feats);
    [q_class, q_rep] = qualityClassifier(synth_img, fov_m);
    assert(score >= 0 && score <= 100, 'Quality score out of bounds.');
    assert(ismember(q_class, {'GOOD', 'BORDERLINE', 'UNGRADABLE'}), 'Invalid quality class.');
end

function test_anatomy_module()
    img_path = fullfile(shinra_config().paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    prep = preprocessRetina(imread(img_path), [512, 512]);
    od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
    fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
    assert(all(od_res.center > 0), 'Invalid OD center.');
    assert(all(fovea_res.center > 0), 'Invalid Fovea center.');
end

function test_vessels_module()
    img_path = fullfile(shinra_config().paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    prep = preprocessRetina(imread(img_path), [512, 512]);
    od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
    vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
    assert(islogical(vess_res.binary_mask), 'Vessel mask must be logical.');
    assert(vess_res.morphometry.vessel_density >= 0, 'Invalid vessel density.');
end

function test_lesions_module()
    img_path = fullfile(shinra_config().paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    prep = preprocessRetina(imread(img_path), [512, 512]);
    od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
    fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
    vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);

    ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
    he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
    ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
    nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);

    assert(ma_res.count >= 0, 'Invalid MA count.');
    assert(he_res.count >= 0, 'Invalid Hemorrhage count.');
    assert(ex_res.count >= 0, 'Invalid Exudate count.');
    assert(nv_res.nv_score >= 0 && nv_res.nv_score <= 1, 'Invalid NV score.');
end

function test_severity_module()
    cfg = shinra_config();
    % Test architecture building without training
    lgraph = buildSeverityCNN([224, 224, 3], 5);
    assert(~isempty(lgraph), 'Failed to build severity CNN layer graph.');
    
    model_file_primary = fullfile(cfg.paths.models_severity, 'SHINRA_DR_Severity_Model.mat');
    model_file_alias   = fullfile(cfg.paths.models_severity, 'severity_cnn.mat');
    if exist(model_file_primary, 'file') == 2 || exist(model_file_alias, 'file') == 2
        img_path = fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
        prep = preprocessRetina(imread(img_path), [512, 512]);
        sev_res = predictDRSeverity(prep);
        assert(numel(sev_res.probabilities) == 5, 'Must have 5-class probabilities.');
        assert(abs(sum(sev_res.probabilities) - 1.0) < 1e-3, 'Probabilities must sum to 1.');
        assert(sev_res.predicted_grade >= 0 && sev_res.predicted_grade <= 4, 'Grade must be in 0-4.');
    else
        fprintf('[Trained model checkpoint not present yet - architecture verified] ');
    end
end

function test_fusion_module()
    sev_res = struct('probabilities', [0.1, 0.1, 0.5, 0.2, 0.1], 'p_referable', 0.8, 'confidence', 0.5);
    lesion_ev = struct('ma_count', 8, 'ma_area', 120, 'he_count', 4, 'he_area_fraction', 0.003, ...
        'ex_count', 12, 'ex_area_fraction', 0.005, 'dme_risk', 1.0, 'nv_score', 0.1);
    vess_res = struct('morphometry', struct('vessel_density', 0.09, 'mean_tortuosity', 1.15));
    q_rep = struct('score', 82.0, 'quality_class', 'GOOD');

    ev_vec = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
    fus_res = fuseEvidence(ev_vec, []);
    rel_res = computeReliability(sev_res.confidence, q_rep.score, fus_res.agreement_score);
    dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);

    assert(fus_res.p_referable_fused >= 0 && fus_res.p_referable_fused <= 1, 'Invalid fused probability.');
    assert(ismember(dec_res.final_decision, {'NOT REFERABLE', 'REFERABLE DR', 'UNCERTAIN / CLINICAL TRIAGE', 'RECAPTURE'}), ...
        'Invalid clinical decision.');
end

function test_explainability_module()
    img_path = fullfile(shinra_config().paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    prep = preprocessRetina(imread(img_path), [512, 512]);
    cam_map = generateGradCAM(prep.rgb_normalized, prep, [], prep.fov_mask);
    assert(all(size(cam_map) == [512, 512]), 'CAM dimension mismatch.');
    agree = explanationAgreement(cam_map, prep.fov_mask, prep.fov_mask);
    assert(agree.iou >= 0 && agree.iou <= 1, 'Invalid IoU.');
end

function test_reporting_module()
    cfg = shinra_config();
    img_id = 'TEST_PATIENT_001';
    prep_res = struct('original_image', zeros(100, 100, 3), 'rgb_normalized', zeros(100, 100, 3));
    q_rep = struct('quality_class', 'GOOD', 'score', 85.0, ...
        'features', struct('tenengrad_sharpness', 12.0, 'laplacian_variance', 45.0, ...
        'mean_intensity', 0.45, 'dark_pixel_ratio', 0.05, 'bright_pixel_ratio', 0.02, ...
        'fov_coverage_ratio', 0.65), 'recapture_guidance', 'Image quality acceptable.');
    od_res = struct('center', [50, 50], 'radius', 15, 'confidence', 0.90);
    fovea_res = struct('center', [80, 50], 'macula_radius', 30);
    vess_res = struct('morphometry', struct('vessel_density', 0.08, 'mean_tortuosity', 1.12, ...
        'num_branch_points', 45, 'branching_density', 3.5));
    ma_res = struct('count', 3, 'total_area', 25);
    he_res = struct('count', 1, 'area_fraction', 0.001);
    ex_res = struct('count', 2, 'area_fraction', 0.002, 'macular_proximity_risk', 0.2);
    nv_res = struct('nv_score', 0.05, 'status', 'EXPLORATORY_BIOMARKER');
    sev_res = struct('predicted_grade', 1, 'predicted_class_name', 'Mild_NPDR', ...
        'expected_grade', 1.1, 'probabilities', [0.3, 0.5, 0.15, 0.03, 0.02]);
    fus_res = struct('p_cnn_referable', 0.20, 'p_referable_fused', 0.22, 'agreement_score', 0.92);

    % 1. Test with full live reliability and decision structs
    rel_live = computeReliability(q_rep, sev_res, fus_res, struct('ma_count', 3, 'he_area_fraction', 0.001, 'ex_area_fraction', 0.002, 'dme_risk', 0.2), vess_res, od_res, fovea_res);
    dec_live = makeScreeningDecision(q_rep, fus_res, rel_live);
    res1 = generateClinicalReport(img_id, prep_res, q_rep, od_res, fovea_res, vess_res, ...
        ma_res, he_res, ex_res, nv_res, sev_res, fus_res, rel_live, dec_live);
    assert(~isempty(res1.report_text), 'Modern report text should not be empty.');

    % 2. Regression Test with legacy / minimal input structs
    rel_legacy = struct('reliability_score', 0.78, 'is_reliable', true);
    dec_legacy = struct('final_decision', 'NOT REFERABLE', 'clinical_disposition', 'Routine follow-up.', 'urgency', 'Routine Annual Follow-up');
    res2 = generateClinicalReport([img_id, '_LEGACY'], prep_res, q_rep, od_res, fovea_res, vess_res, ...
        ma_res, he_res, ex_res, nv_res, sev_res, fus_res, rel_legacy, dec_legacy);
    assert(~isempty(res2.report_text), 'Legacy report text should not be empty.');

    assert(exist(fullfile(cfg.paths.reports, 'TEST_PATIENT_001_clinical_report.txt'), 'file') == 2, ...
        'Report file was not created.');
end

function test_end_to_end()
    cfg = shinra_config();
    model_file_primary = fullfile(cfg.paths.models_severity, 'SHINRA_DR_Severity_Model.mat');
    model_file_alias   = fullfile(cfg.paths.models_severity, 'severity_cnn.mat');
    if exist(model_file_primary, 'file') ~= 2 && exist(model_file_alias, 'file') ~= 2
        fprintf('[MODEL NOT TRAINED YET - end-to-end model inference skipped] ');
        return;
    end
    img_path = fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_002.jpg');
    res = run_single_image(img_path, true);
    assert(isfield(res, 'decision'), 'Result missing decision field.');
    assert(isfield(res, 'panel_figure_path'), 'Result missing panel figure path.');
end
