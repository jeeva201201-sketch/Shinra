function test_preprocessing()
% TEST_PREPROCESSING Unit tests for retinal FOV detection and enhancement
    setup_SHINRA();
    cfg = shinra_config();
    img_path = fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    raw_img = imread(img_path);
    
    % Test FOV Detection
    [fov_mask, bbox, cropped] = detectRetinalFOV(raw_img);
    assert(islogical(fov_mask), 'FOV mask must be binary.');
    assert(numel(bbox) == 4, 'Bounding box must have 4 coordinates.');
    assert(bbox(3) > 50 && bbox(4) > 50, 'Detected FOV is unrealistically small.');
    
    % Test Preprocessing Pipeline
    prep = preprocessRetina(raw_img, [512, 512]);
    assert(all(size(prep.rgb_normalized) == [512, 512, 3]), 'Target size mismatch.');
    assert(max(prep.rgb_normalized(:)) <= 1.0 && min(prep.rgb_normalized(:)) >= 0.0, 'Values out of [0, 1].');
    fprintf('test_preprocessing: PASSED\n');
end
