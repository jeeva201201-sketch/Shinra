function baseline_report = evaluateVesselMatchedFilter(varargin)
% EVALUATEVESSELMATCHEDFILTER Rigorous Baseline Evaluation for Non-Deep-Learning Vessel Engine
% Evaluates the multi-directional Gaussian matched filter from segmentVessels.m on DRIVE images
% with manual ground-truth vessel annotations.
%
% Generates:
% 1. Per-image pixel-level evaluation inside the valid retinal FOV (Dice, IoU, Sens, Spec, Prec, TP, FP, FN, TN).
% 2. Aggregate statistics (mean/median Dice, mean IoU, sensitivity, specificity, precision, pooled confusion counts).
% 3. Diagnostic 4-panel visual overlays for representative images saved to results/vessels/.
% 4. Structured metric persistence to results/metrics/drive_matched_filter_baseline.mat and per-image CSV.
%
% Assertions enforced:
% - Exactly 20 DRIVE images and 20 corresponding manual vessel masks.
% - Evaluation uses manual vessel masks, NOT FOV masks.
% - Evaluation restricted strictly within the valid FOV.
% - Matched-filter engine verified (is_deep_model == false).
% - No IDRiD dataset leakage.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('      SHINRA VESSEL SEGMENTATION: MATCHED-FILTER BASELINE EVALUATION    \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Discover DRIVE Dataset and Pair with Manual Vessel Masks
    % -------------------------------------------------------------------------
    drive_root = cfg.paths.drive;
    if exist(drive_root, 'dir') ~= 7
        error('SHINRA:DriveNotFound', 'DRIVE dataset directory not found at: %s', drive_root);
    end

    % Check for manual vessel annotations in test vs training folders
    test_img_dir = fullfile(drive_root, 'test', 'images');
    test_man_dir = fullfile(drive_root, 'test', '1st_manual');
    test_fov_dir = fullfile(drive_root, 'test', 'mask');

    train_img_dir = fullfile(drive_root, 'training', 'images');
    train_man_dir = fullfile(drive_root, 'training', '1st_manual');
    train_fov_dir = fullfile(drive_root, 'training', 'mask');

    % Prioritize test/1st_manual if available; otherwise use training/1st_manual (20 manual masks)
    if exist(test_man_dir, 'dir') && numel(dir(fullfile(test_man_dir, '*.gif'))) >= 20
        cohort_name = 'DRIVE Official Test Set';
        active_img_dir = test_img_dir;
        active_man_dir = test_man_dir;
        active_fov_dir = test_fov_dir;
    elseif exist(train_man_dir, 'dir') && numel(dir(fullfile(train_man_dir, '*.gif'))) >= 20
        cohort_name = 'DRIVE 20-Image Manual Ground-Truth Benchmark Set';
        active_img_dir = train_img_dir;
        active_man_dir = train_man_dir;
        active_fov_dir = train_fov_dir;
    else
        error('SHINRA:ManualMasksNotFound', 'Could not find 20 manual vessel masks in %s or %s', test_man_dir, train_man_dir);
    end

    img_files = dir(fullfile(active_img_dir, '*.tif'));
    if isempty(img_files)
        img_files = dir(fullfile(active_img_dir, '*.png'));
    end

    num_images = numel(img_files);
    assert(num_images == 20, sprintf('Expected exactly 20 DRIVE images, but found %d in %s', num_images, active_img_dir));

    image_paths = cell(num_images, 1);
    manual_mask_paths = cell(num_images, 1);
    fov_mask_paths = cell(num_images, 1);
    image_ids = cell(num_images, 1);

    for i = 1:num_images
        fname = img_files(i).name;
        [~, base_name, ~] = fileparts(fname);
        image_ids{i} = base_name;
        image_paths{i} = fullfile(img_files(i).folder, fname);

        % Extract digit prefix (e.g., '01_test' -> '01', '21_training' -> '21')
        tokens = regexp(base_name, '^(\d+)', 'tokens');
        if ~isempty(tokens)
            id_prefix = tokens{1}{1};
        else
            id_prefix = base_name;
        end

        % Search for corresponding 1st_manual mask
        man_cand1 = fullfile(active_man_dir, [id_prefix, '_manual1.gif']);
        man_cand2 = fullfile(active_man_dir, [id_prefix, '_manual1.tif']);
        man_cand3 = fullfile(active_man_dir, [id_prefix, '_manual1.png']);

        if exist(man_cand1, 'file') == 2
            manual_mask_paths{i} = man_cand1;
        elseif exist(man_cand2, 'file') == 2
            manual_mask_paths{i} = man_cand2;
        elseif exist(man_cand3, 'file') == 2
            manual_mask_paths{i} = man_cand3;
        else
            error('SHINRA:ManualMaskMissing', 'Manual vessel mask not found for image %s in %s', base_name, active_man_dir);
        end

        % Search for corresponding FOV mask
        fov_cand1 = fullfile(active_fov_dir, [id_prefix, '_training_mask.gif']);
        fov_cand2 = fullfile(active_fov_dir, [id_prefix, '_test_mask.gif']);
        fov_cand3 = fullfile(active_fov_dir, [id_prefix, '_mask.gif']);

        if exist(fov_cand1, 'file') == 2
            fov_mask_paths{i} = fov_cand1;
        elseif exist(fov_cand2, 'file') == 2
            fov_mask_paths{i} = fov_cand2;
        elseif exist(fov_cand3, 'file') == 2
            fov_mask_paths{i} = fov_cand3;
        else
            fov_mask_paths{i} = '';
        end
    end

    % -------------------------------------------------------------------------
    % 12. Enforce Data Integrity & Strict Sequestration Assertions
    % -------------------------------------------------------------------------
    assert(numel(image_paths) == 20, 'Assertion Failed: Exactly 20 DRIVE images required.');
    assert(numel(manual_mask_paths) == 20, 'Assertion Failed: Exactly 20 manual vessel masks required.');
    
    % Assert that manual mask paths do NOT equal FOV mask paths (Requirement 11)
    for i = 1:20
        if ~isempty(fov_mask_paths{i})
            assert(~strcmp(manual_mask_paths{i}, fov_mask_paths{i}), ...
                sprintf('CRITICAL ERROR: Manual mask path is identical to FOV mask path for image %s!', image_ids{i}));
        end
        % Assert no IDRiD dataset leakage
        assert(~contains(image_paths{i}, 'IDRid', 'IgnoreCase', true), ...
            'CRITICAL ERROR: IDRiD image detected in DRIVE evaluation pipeline!');
    end

    fprintf('DRIVE Evaluation Cohort: %s\n', cohort_name);
    fprintf('  • Image Directory:       %s (%d images)\n', active_img_dir, num_images);
    fprintf('  • Manual Vessel Masks:   %s (%d masks)\n', active_man_dir, num_images);
    fprintf('  • Retinal FOV Masks:     %s\n\n', active_fov_dir);

    % -------------------------------------------------------------------------
    % 2. Setup Evaluation Storage
    % -------------------------------------------------------------------------
    dice_scores = zeros(num_images, 1);
    iou_scores  = zeros(num_images, 1);
    sens_scores = zeros(num_images, 1);
    spec_scores = zeros(num_images, 1);
    prec_scores = zeros(num_images, 1);
    acc_scores  = zeros(num_images, 1);

    tp_counts = zeros(num_images, 1);
    fp_counts = zeros(num_images, 1);
    fn_counts = zeros(num_images, 1);
    tn_counts = zeros(num_images, 1);
    fov_pixel_counts = zeros(num_images, 1);
    gt_vessel_pixel_counts = zeros(num_images, 1);

    % Diagnostic overlay output directory
    overlay_dir = fullfile(cfg.paths.results, 'vessels');
    if exist(overlay_dir, 'dir') ~= 7
        mkdir(overlay_dir);
    end

    representative_indices = [1, round(num_images/2), num_images]; % At least 3 representative images

    % -------------------------------------------------------------------------
    % 3. Run Matched-Filter Vessel Segmentation & Pixel Evaluation
    % -------------------------------------------------------------------------
    fprintf('Running matched-filter vessel segmentation across %d DRIVE images...\n', num_images);
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('  Idx  Image ID          Dice       IoU      Sensitivity  Specificity  Precision   Accuracy  \n');
    fprintf('-----------------------------------------------------------------------------------------\n');

    for i = 1:num_images
        img_raw = imread(image_paths{i});
        gt_raw  = imread(manual_mask_paths{i});

        % Load or derive FOV mask
        if ~isempty(fov_mask_paths{i}) && exist(fov_mask_paths{i}, 'file') == 2
            fov_raw = imread(fov_mask_paths{i}) > 0;
        else
            [fov_raw, ~, ~] = detectRetinalFOV(img_raw);
        end

        % Spatial dimension verification & alignment
        assert(size(gt_raw, 1) == size(img_raw, 1) && size(gt_raw, 2) == size(img_raw, 2), ...
            sprintf('Dimension mismatch between image %s and its manual mask!', image_ids{i}));
        assert(size(fov_raw, 1) == size(img_raw, 1) && size(fov_raw, 2) == size(img_raw, 2), ...
            sprintf('Dimension mismatch between image %s and its FOV mask!', image_ids{i}));

        % Ground truth binary vessel mask
        gt_vessels = (gt_raw > 0) & fov_raw;

        % Execute Matched-Filter Vessel Segmentation (segmentVessels.m)
        vessel_res = segmentVessels(img_raw, fov_raw);

        % Requirement 13: Verify that segmentVessels used the non-deep-learning matched-filter engine
        assert(isfield(vessel_res, 'is_deep_model'), 'Missing is_deep_model flag in segmentVessels output.');
        assert(vessel_res.is_deep_model == false, ...
            'CRITICAL: segmentVessels used a deep model instead of the baseline matched filter!');

        pred_vessels = (vessel_res.binary_mask > 0) & fov_raw;

        % Dimension check post-prediction
        assert(isequal(size(pred_vessels), size(gt_vessels)), ...
            'Prediction and ground-truth dimension mismatch after segmentation!');

        % Compute Confusion Matrix strictly INSIDE valid retinal FOV
        tp = sum(pred_vessels(:) & gt_vessels(:));
        fp = sum(pred_vessels(:) & ~gt_vessels(:) & fov_raw(:));
        fn = sum(~pred_vessels(:) & gt_vessels(:));
        tn = sum(~pred_vessels(:) & ~gt_vessels(:) & fov_raw(:));

        % Performance Metrics
        dice = (2 * tp) / max(1, 2 * tp + fp + fn);
        iou  = tp / max(1, tp + fp + fn);
        sens = tp / max(1, tp + fn);
        spec = tn / max(1, tn + fp);
        prec = tp / max(1, tp + fp);
        acc  = (tp + tn) / max(1, tp + tn + fp + fn);

        % Record
        dice_scores(i) = dice;
        iou_scores(i)  = iou;
        sens_scores(i) = sens;
        spec_scores(i) = spec;
        prec_scores(i) = prec;
        acc_scores(i)  = acc;

        tp_counts(i) = tp;
        fp_counts(i) = fp;
        fn_counts(i) = fn;
        tn_counts(i) = tn;
        fov_pixel_counts(i) = sum(fov_raw(:));
        gt_vessel_pixel_counts(i) = sum(gt_vessels(:));

        fprintf('  %2d   %-16s %6.4f    %6.4f     %6.2f%%      %6.2f%%     %6.2f%%    %6.2f%%\n', ...
            i, image_ids{i}, dice, iou, sens * 100, spec * 100, prec * 100, acc * 100);

        % ---------------------------------------------------------------------
        % 10. Generate 4-Panel Diagnostic Overlay for Representative Images
        % ---------------------------------------------------------------------
        if ismember(i, representative_indices)
            overlay_file = fullfile(overlay_dir, sprintf('drive_vessel_eval_%s.png', image_ids{i}));
            saveDiagnosticOverlay(img_raw, gt_vessels, pred_vessels, fov_raw, tp, fp, fn, dice, overlay_file);
        end
    end
    fprintf('-----------------------------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % 4. Compute Summary & Aggregate Statistics
    % -------------------------------------------------------------------------
    total_tp = sum(tp_counts);
    total_fp = sum(fp_counts);
    total_fn = sum(fn_counts);
    total_tn = sum(tn_counts);

    pooled_dice = (2 * total_tp) / max(1, 2 * total_tp + total_fp + total_fn);
    pooled_iou  = total_tp / max(1, total_tp + total_fp + total_fn);
    pooled_sens = total_tp / max(1, total_tp + total_fn);
    pooled_spec = total_tn / max(1, total_tn + total_fp);
    pooled_prec = total_tp / max(1, total_tp + total_fp);
    pooled_acc  = (total_tp + total_tn) / max(1, total_tp + total_tn + total_fp + total_fn);

    fprintf('========================================================================\n');
    fprintf('  DRIVE MATCHED-FILTER VESSEL SEGMENTATION: AGGREGATE SUMMARY (N=20)    \n');
    fprintf('========================================================================\n');
    fprintf('  Metric                        Mean ± Std          Median       Pooled\n');
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  • Dice Similarity Coeff:   %6.4f ± %6.4f      %6.4f      %6.4f\n', ...
        mean(dice_scores), std(dice_scores), median(dice_scores), pooled_dice);
    fprintf('  • Jaccard Index (IoU):     %6.4f ± %6.4f      %6.4f      %6.4f\n', ...
        mean(iou_scores), std(iou_scores), median(iou_scores), pooled_iou);
    fprintf('  • Vessel Sensitivity:      %6.2f%% ± %5.2f%%     %6.2f%%     %6.2f%%\n', ...
        mean(sens_scores)*100, std(sens_scores)*100, median(sens_scores)*100, pooled_sens*100);
    fprintf('  • Specificity (Non-Vessel):%6.2f%% ± %5.2f%%     %6.2f%%     %6.2f%%\n', ...
        mean(spec_scores)*100, std(spec_scores)*100, median(spec_scores)*100, pooled_spec*100);
    fprintf('  • Precision (PPV):         %6.2f%% ± %5.2f%%     %6.2f%%     %6.2f%%\n', ...
        mean(prec_scores)*100, std(prec_scores)*100, median(prec_scores)*100, pooled_prec*100);
    fprintf('  • Overall Pixel Accuracy:  %6.2f%% ± %5.2f%%     %6.2f%%     %6.2f%%\n', ...
        mean(acc_scores)*100, std(acc_scores)*100, median(acc_scores)*100, pooled_acc*100);
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  Pooled Confusion Pixels: TP = %d | FP = %d | FN = %d | TN = %d\n', ...
        total_tp, total_fp, total_fn, total_tn);
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 8 & 9. Save Structured Report MAT & CSV
    % -------------------------------------------------------------------------
    % Build Per-Image Table
    per_image_table = table(image_ids, image_paths, manual_mask_paths, fov_mask_paths, ...
        tp_counts, fp_counts, fn_counts, tn_counts, ...
        dice_scores, iou_scores, sens_scores, spec_scores, prec_scores, acc_scores, ...
        'VariableNames', {'ImageID', 'ImagePath', 'ManualMaskPath', 'FOVMaskPath', ...
                          'TP', 'FP', 'FN', 'TN', ...
                          'Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy'});

    % Build Summary Struct
    summary_stats = struct();
    summary_stats.num_images = num_images;
    summary_stats.cohort = cohort_name;
    summary_stats.engine = 'Multi-Directional Gaussian Matched Filter (Non-Deep-Learning Baseline)';
    summary_stats.is_deep_model = false;
    summary_stats.mean_dice = mean(dice_scores);
    summary_stats.median_dice = median(dice_scores);
    summary_stats.std_dice = std(dice_scores);
    summary_stats.mean_iou = mean(iou_scores);
    summary_stats.mean_sensitivity = mean(sens_scores);
    summary_stats.mean_specificity = mean(spec_scores);
    summary_stats.mean_precision = mean(prec_scores);
    summary_stats.mean_accuracy = mean(acc_scores);
    summary_stats.pooled_dice = pooled_dice;
    summary_stats.pooled_iou = pooled_iou;
    summary_stats.pooled_sensitivity = pooled_sens;
    summary_stats.pooled_specificity = pooled_spec;
    summary_stats.pooled_precision = pooled_prec;
    summary_stats.pooled_accuracy = pooled_acc;
    summary_stats.pooled_confusion = struct('TP', total_tp, 'FP', total_fp, 'FN', total_fn, 'TN', total_tn);
    summary_stats.timestamp = char(datetime('now'));

    % Save MAT
    mat_out = fullfile(cfg.paths.metrics, 'drive_matched_filter_baseline.mat');
    save(mat_out, 'summary_stats', 'per_image_table', 'cfg');
    fprintf('Baseline metrics saved to:\n  • %s\n', mat_out);

    % Save CSV
    csv_out = fullfile(cfg.paths.metrics, 'drive_matched_filter_per_image.csv');
    writetable(per_image_table, csv_out);
    fprintf('Per-image table saved to:\n  • %s\n', csv_out);

    fprintf('Representative overlays saved to:\n  • %s\n\n', overlay_dir);

    baseline_report = struct();
    baseline_report.summary = summary_stats;
    baseline_report.per_image = per_image_table;
end

function saveDiagnosticOverlay(img_rgb, gt_mask, pred_mask, fov_mask, tp, fp, fn, dice, out_path)
% SAVEDIAGNOSTICOVERLAY Generates a 4-panel visual comparison overlay
% Panel 1: Original RGB Image
% Panel 2: Ground-Truth Manual Vessel Mask (Green on Black)
% Panel 3: Predicted Matched-Filter Vessel Mask (Cyan on Black)
% Panel 4: RGB Error Visualization (Green = TP, Red = FP, Blue = FN)

    [H, W, ~] = size(img_rgb);
    img_double = im2double(img_rgb);

    % Panel 1: Original Image with FOV boundary
    p1 = img_double;

    % Panel 2: Ground Truth (Green on black)
    p2 = zeros(H, W, 3);
    p2(:, :, 2) = double(gt_mask);

    % Panel 3: Predicted Mask (Cyan on black)
    p3 = zeros(H, W, 3);
    p3(:, :, 2) = double(pred_mask);
    p3(:, :, 3) = double(pred_mask);

    % Panel 4: Color-Coded Error Map
    % TP = Green ([0, 1, 0]), FP = Red ([1, 0, 0]), FN = Yellow/Blue ([1, 1, 0])
    p4 = zeros(H, W, 3);
    
    tp_mask = pred_mask & gt_mask;
    fp_mask = pred_mask & ~gt_mask & fov_mask;
    fn_mask = ~pred_mask & gt_mask;

    % True Positives (Green)
    p4(:, :, 2) = p4(:, :, 2) + double(tp_mask);

    % False Positives (Red)
    p4(:, :, 1) = p4(:, :, 1) + double(fp_mask);

    % False Negatives (Yellow: Red + Green)
    p4(:, :, 1) = p4(:, :, 1) + double(fn_mask);
    p4(:, :, 2) = p4(:, :, 2) + double(fn_mask);

    % Composite 2x2 Montage
    row1 = [p1, p2];
    row2 = [p3, p4];
    composite = [row1; row2];

    % Write to disk
    imwrite(composite, out_path);
end
