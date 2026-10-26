function test_quality()
% TEST_QUALITY Unit tests for multi-factor quality scoring and tri-state classification
    setup_SHINRA();
    cfg = shinra_config();
    img_path = fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    raw_img = imread(img_path);
    prep = preprocessRetina(raw_img, [512, 512]);
    
    feats = qualityFeatures(prep.rgb_normalized, prep.fov_mask);
    assert(isfield(feats, 'laplacian_variance'), 'Missing laplacian variance.');
    assert(isfield(feats, 'tenengrad_sharpness'), 'Missing tenengrad sharpness.');
    
    [score, subscores] = qualityScore(feats);
    assert(score >= 0 && score <= 100, 'Score must be in [0, 100].');
    
    [q_class, q_rep] = qualityClassifier(prep.rgb_normalized, prep.fov_mask);
    assert(ismember(q_class, {'GOOD', 'BORDERLINE', 'UNGRADABLE'}), 'Invalid classification category.');
    fprintf('test_quality: PASSED\n');
end
