function test_anatomy()
% TEST_ANATOMY Unit tests for optic disc and fovea localization
    setup_SHINRA();
    cfg = shinra_config();
    img_path = fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    prep = preprocessRetina(imread(img_path), [512, 512]);
    
    od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
    assert(od_res.center(1) > 0 && od_res.center(2) > 0, 'Invalid OD coordinates.');
    assert(od_res.radius > 10, 'OD radius is too small.');
    
    fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
    assert(fovea_res.center(1) > 0 && fovea_res.center(2) > 0, 'Invalid Fovea coordinates.');
    
    overlay = visualizeAnatomy(prep.rgb_normalized, od_res, fovea_res);
    assert(all(size(overlay) == size(prep.rgb_normalized)), 'Overlay size mismatch.');
    fprintf('test_anatomy: PASSED\n');
end
