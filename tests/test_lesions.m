function test_lesions()
% TEST_LESIONS Unit tests for microaneurysm, hemorrhage, exudate, and neovascularization detection
    setup_SHINRA();
    cfg = shinra_config();
    img_path = fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    prep = preprocessRetina(imread(img_path), [512, 512]);
    od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
    fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
    vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
    
    ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
    he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
    ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
    nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);
    
    assert(islogical(ma_res.binary_mask), 'MA mask must be logical.');
    assert(islogical(he_res.binary_mask), 'HE mask must be logical.');
    assert(islogical(ex_res.binary_mask), 'EX mask must be logical.');
    assert(nv_res.nv_score >= 0 && nv_res.nv_score <= 1, 'NV score out of range.');
    fprintf('test_lesions: PASSED\n');
end
