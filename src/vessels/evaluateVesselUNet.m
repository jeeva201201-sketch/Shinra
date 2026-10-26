function eval_report = evaluateVesselUNet(varargin)
% EVALUATEVESSELUNET Evaluation for Trained U-Net Retinal Vessel Segmentation
% Evaluates the trained U-Net checkpoint (models/vessels/vessel_unet.mat) across
% the 20-image DRIVE benchmark (16 training + 4 held-out validation images) inside the valid retinal FOV.
%
% Generates:
% 1. Per-image evaluation (Dice, IoU, Sensitivity, Specificity, Precision, Pixel Accuracy, TP, FP, FN, TN).
% 2. Separate metrics for held-out validation subset (N=4) and 20-image benchmark set (N=20).
% 3. Diagnostic 4-panel visual overlays saved to results/vessels/.
% 4. Structured outputs: results/metrics/drive_unet_evaluation.mat and per-image CSV.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('       SHINRA VESSEL SEGMENTATION: U-NET 20-IMAGE BENCHMARK EVALUATION  \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Verify and Load U-Net Checkpoint
    % -------------------------------------------------------------------------
    model_file = fullfile(cfg.paths.models_vessels, 'vessel_unet.mat');
    if exist(model_file, 'file') ~= 2
        error('SHINRA:UNetCheckpointNotFound', ...
            'Trained U-Net model checkpoint not found at: %s\nPlease train the model first by executing: trainVesselSegmentation', ...
            model_file);
    end

    loaded = load(model_file);
    if ~isfield(loaded, 'net')
        error('SHINRA:InvalidCheckpoint', 'Checkpoint does not contain "net" variable in %s', model_file);
    end
    net = loaded.net;
    fprintf('Loaded Trained U-Net Checkpoint: %s\n', model_file);

    % -------------------------------------------------------------------------
    % 2. Defensive Pre-Evaluation Architecture Assertions
    % -------------------------------------------------------------------------
    all_l_classes = cellfun(@class, num2cell(net.Layers), 'UniformOutput', false);
    in_layer = net.Layers(1);
    
    % Search for PixelClassificationLayer
    has_pixel_layer = false;
    pixel_layer = [];
    for i = 1:numel(net.Layers)
        if isa(net.Layers(i), 'nnet.cnn.layer.PixelClassificationLayer')
            has_pixel_layer = true;
            pixel_layer = net.Layers(i);
            break;
        end
    end

    assert(isequal(in_layer.InputSize, [512, 512, 1]), ...
        sprintf('CRITICAL: Network input size must be strictly [512 512 1], found [%s]', num2str(in_layer.InputSize)));
    assert(has_pixel_layer, 'CRITICAL: PixelClassificationLayer must be present in U-Net!');
    assert(~any(contains(all_l_classes, 'ClassificationOutputLayer') | ...
                (contains(all_l_classes, 'ClassificationLayer') & ~contains(all_l_classes, 'Pixel'))), ...
                'CRITICAL: ClassificationOutputLayer must NOT be present in vessel segmentation network!');
    assert(~any(contains(all_l_classes, 'GlobalAveragePooling')), ...
                'CRITICAL: GlobalAveragePooling2DLayer must NOT be present in vessel segmentation network!');

    if isprop(pixel_layer, 'Classes') && ~isempty(pixel_layer.Classes)
        p_classes = cellstr(pixel_layer.Classes);
        assert(ismember('Background', p_classes) && ismember('Vessel', p_classes), ...
            'PixelClassificationLayer must define Background and Vessel classes.');
    end
    fprintf('  • Architecture Pre-Check: PASSED (Authentic 512x512x1 Semantic Segmentation U-Net)\n\n');

    % -------------------------------------------------------------------------
    % 3. Discover and Load DRIVE 20-Image Benchmark Set
    % -------------------------------------------------------------------------
    drive_root = cfg.paths.drive;
    train_img_dir = fullfile(drive_root, 'training', 'images');
    train_man_dir = fullfile(drive_root, 'training', '1st_manual');
    train_fov_dir = fullfile(drive_root, 'training', 'mask');

    img_files = dir(fullfile(train_img_dir, '*.tif'));
    if isempty(img_files)
        img_files = dir(fullfile(train_img_dir, '*.png'));
    end
    num_images = numel(img_files);
    assert(num_images == 20, sprintf('Expected exactly 20 DRIVE images, but found %d.', num_images));

    image_ids = cell(num_images, 1);
    image_paths = cell(num_images, 1);
    manual_mask_paths = cell(num_images, 1);
    fov_mask_paths = cell(num_images, 1);

    for i = 1:num_images
        fname = img_files(i).name;
        [~, base_name, ~] = fileparts(fname);
        image_ids{i} = base_name;
        image_paths{i} = fullfile(img_files(i).folder, fname);

        tokens = regexp(base_name, '^(\d+)', 'tokens');
        if ~isempty(tokens)
            id_prefix = tokens{1}{1};
        else
            id_prefix = base_name;
        end

        man_file = fullfile(train_man_dir, [id_prefix, '_manual1.gif']);
        fov_file = fullfile(train_fov_dir, [id_prefix, '_training_mask.gif']);

        assert(exist(man_file, 'file') == 2, sprintf('Manual mask missing for %s', base_name));
        assert(exist(fov_file, 'file') == 2, sprintf('FOV mask missing for %s', base_name));

        manual_mask_paths{i} = man_file;
        fov_mask_paths{i} = fov_file;
    end

    % -------------------------------------------------------------------------
    % 4. Partition Labeling (16 Train / 4 Held-Out Validation, Seed 42)
    % -------------------------------------------------------------------------
    if isfield(loaded, 'split_info')
        val_ids = loaded.split_info.val_image_ids;
        train_ids = loaded.split_info.train_image_ids;
    else
        rng(cfg.random_seed, 'twister');
        s_idx = randperm(num_images);
        val_ids = image_ids(sort(s_idx(1:4)));
        train_ids = image_ids(sort(s_idx(5:end)));
    end

    is_val_flag = ismember(image_ids, val_ids);

    fprintf('Evaluation Cohort: 20-image DRIVE benchmark (16 training + 4 held-out validation images)\n');
    fprintf('  • Held-Out Validation (N=4): [%s]\n', strjoin(val_ids, ', '));
    fprintf('  • Training Subset (N=16):    [%s]\n\n', strjoin(train_ids, ', '));

    % -------------------------------------------------------------------------
    % 5. Run U-Net Segmentation and Pixel-Level Evaluation
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

    overlay_dir = fullfile(cfg.paths.results, 'vessels');
    if exist(overlay_dir, 'dir') ~= 7
        mkdir(overlay_dir);
    end
    rep_indices = [1, round(num_images/2), num_images];

    fprintf('Evaluating U-Net across 20 DRIVE images (inside valid FOV)...\n');
    fprintf('-------------------------------------------------------------------------------------------------\n');
    fprintf('  Idx  Image ID          Subset      Dice       IoU      Sensitivity  Specificity  Precision   Accuracy  \n');
    fprintf('-------------------------------------------------------------------------------------------------\n');

    target_dim = [512, 512];

    for i = 1:num_images
        img_rgb  = imread(image_paths{i});
        gt_raw   = imread(manual_mask_paths{i}) > 0;
        fov_raw  = imread(fov_mask_paths{i}) > 0;

        [orig_h, orig_w, ~] = size(img_rgb);

        % Preprocessing matching training: CLAHE-enhanced green channel
        g_chan = im2double(img_rgb(:, :, 2));
        g_clahe = adapthisteq(g_chan, 'ClipLimit', 0.015, 'NumTiles', [8, 8]);
        g_clahe(~fov_raw) = 0;

        input_tensor = single(imresize(g_clahe, target_dim, 'bicubic'));

        % U-Net Forward Pass via MATLAB semanticseg API
        pred_cat = semanticseg(input_tensor, net);
        pred_vessels_512 = (pred_cat == 'Vessel');

        % Resize prediction back to original image resolution
        pred_vessels_orig = imresize(pred_vessels_512, [orig_h, orig_w], 'nearest');
        pred_vessels = pred_vessels_orig & fov_raw;
        gt_vessels = gt_raw & fov_raw;

        % Compute Confusion Matrix strictly INSIDE valid retinal FOV
        tp = sum(pred_vessels(:) & gt_vessels(:));
        fp = sum(pred_vessels(:) & ~gt_vessels(:) & fov_raw(:));
        fn = sum(~pred_vessels(:) & gt_vessels(:));
        tn = sum(~pred_vessels(:) & ~gt_vessels(:) & fov_raw(:));

        dice = (2 * tp) / max(1, 2 * tp + fp + fn);
        iou  = tp / max(1, tp + fp + fn);
        sens = tp / max(1, tp + fn);
        spec = tn / max(1, tn + fp);
        prec = tp / max(1, tp + fp);
        acc  = (tp + tn) / max(1, tp + tn + fp + fn);

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

        if is_val_flag(i)
            subset_str = 'VAL (Held)';
        else
            subset_str = 'TRAIN';
        end

        fprintf('  %2d   %-16s  %-9s  %6.4f    %6.4f     %6.2f%%      %6.2f%%     %6.2f%%    %6.2f%%\n', ...
            i, image_ids{i}, subset_str, dice, iou, sens * 100, spec * 100, prec * 100, acc * 100);

        % Diagnostic Visual Overlay
        if ismember(i, rep_indices)
            overlay_file = fullfile(overlay_dir, sprintf('drive_unet_eval_%s.png', image_ids{i}));
            saveDiagnosticOverlay(img_rgb, gt_vessels, pred_vessels, fov_raw, dice, overlay_file);
        end
    end
    fprintf('-------------------------------------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % 6. Aggregate Statistics (Validation Cohort vs Full Benchmark)
    % -------------------------------------------------------------------------
    % A. Held-Out Validation Subset Only (N=4)
    val_mask = is_val_flag;
    val_tp = sum(tp_counts(val_mask));
    val_fp = sum(fp_counts(val_mask));
    val_fn = sum(fn_counts(val_mask));
    val_tn = sum(tn_counts(val_mask));

    val_pooled_dice = (2 * val_tp) / max(1, 2 * val_tp + val_fp + val_fn);
    val_pooled_iou  = val_tp / max(1, val_tp + val_fp + val_fn);
    val_pooled_sens = val_tp / max(1, val_tp + val_fn);
    val_pooled_spec = val_tn / max(1, val_tn + val_fp);
    val_pooled_prec = val_tp / max(1, val_tp + val_fp);
    val_pooled_acc  = (val_tp + val_tn) / max(1, val_tp + val_tn + val_fp + val_fn);

    % B. Full Benchmark Set (N=20)
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
    fprintf('          DRIVE U-NET VESSEL SEGMENTATION PERFORMANCE SUMMARY           \n');
    fprintf('========================================================================\n');
    fprintf('  Metric                   Held-Out Validation (N=4)   20-Image Benchmark (N=20)\n');
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  • Dice Coefficient:           %6.4f ± %6.4f            %6.4f ± %6.4f\n', ...
        mean(dice_scores(val_mask)), std(dice_scores(val_mask)), mean(dice_scores), std(dice_scores));
    fprintf('  • Jaccard Index (IoU):        %6.4f ± %6.4f            %6.4f ± %6.4f\n', ...
        mean(iou_scores(val_mask)), std(iou_scores(val_mask)), mean(iou_scores), std(iou_scores));
    fprintf('  • Vessel Sensitivity:         %5.2f%% ± %5.2f%%           %5.2f%% ± %5.2f%%\n', ...
        mean(sens_scores(val_mask))*100, std(sens_scores(val_mask))*100, mean(sens_scores)*100, std(sens_scores)*100);
    fprintf('  • Specificity (Non-Vessel):   %5.2f%% ± %5.2f%%           %5.2f%% ± %5.2f%%\n', ...
        mean(spec_scores(val_mask))*100, std(spec_scores(val_mask))*100, mean(spec_scores)*100, std(spec_scores)*100);
    fprintf('  • Precision (PPV):            %5.2f%% ± %5.2f%%           %5.2f%% ± %5.2f%%\n', ...
        mean(prec_scores(val_mask))*100, std(prec_scores(val_mask))*100, mean(prec_scores)*100, std(prec_scores)*100);
    fprintf('  • Pixel Accuracy:             %5.2f%% ± %5.2f%%           %5.2f%% ± %5.2f%%\n', ...
        mean(acc_scores(val_mask))*100, std(acc_scores(val_mask))*100, mean(acc_scores)*100, std(acc_scores)*100);
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  Pooled Confusion (N=20): TP = %d | FP = %d | FN = %d | TN = %d\n', ...
        total_tp, total_fp, total_fn, total_tn);
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 7. Save Evaluation Outputs
    % -------------------------------------------------------------------------
    per_image_table = table(image_ids, is_val_flag, image_paths, manual_mask_paths, fov_mask_paths, ...
        tp_counts, fp_counts, fn_counts, tn_counts, ...
        dice_scores, iou_scores, sens_scores, spec_scores, prec_scores, acc_scores, ...
        'VariableNames', {'ImageID', 'IsValidation', 'ImagePath', 'ManualMaskPath', 'FOVMaskPath', ...
                          'TP', 'FP', 'FN', 'TN', ...
                          'Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy'});

    summary_stats = struct();
    summary_stats.model_checkpoint = model_file;
    summary_stats.cohort = '20-image DRIVE benchmark (16 training + 4 held-out validation images)';
    summary_stats.val_subset = struct(...
        'mean_dice', mean(dice_scores(val_mask)), 'std_dice', std(dice_scores(val_mask)), ...
        'mean_iou', mean(iou_scores(val_mask)), 'mean_sensitivity', mean(sens_scores(val_mask)), ...
        'mean_specificity', mean(spec_scores(val_mask)), 'mean_precision', mean(prec_scores(val_mask)), ...
        'mean_accuracy', mean(acc_scores(val_mask)), 'pooled_dice', val_pooled_dice, 'pooled_iou', val_pooled_iou);
    summary_stats.full_benchmark = struct(...
        'mean_dice', mean(dice_scores), 'std_dice', std(dice_scores), ...
        'mean_iou', mean(iou_scores), 'mean_sensitivity', mean(sens_scores), ...
        'mean_specificity', mean(spec_scores), 'mean_precision', mean(prec_scores), ...
        'mean_accuracy', mean(acc_scores), 'pooled_dice', pooled_dice, 'pooled_iou', pooled_iou, ...
        'pooled_sensitivity', pooled_sens, 'pooled_specificity', pooled_spec, ...
        'pooled_precision', pooled_prec, 'pooled_accuracy', pooled_acc, ...
        'pooled_confusion', struct('TP', total_tp, 'FP', total_fp, 'FN', total_fn, 'TN', total_tn));
    summary_stats.timestamp = char(datetime('now'));

    % Save MAT
    mat_out = fullfile(cfg.paths.metrics, 'drive_unet_evaluation.mat');
    save(mat_out, 'summary_stats', 'per_image_table', 'cfg');
    fprintf('U-Net evaluation metrics saved to:\n  • %s\n', mat_out);

    % Save CSV
    csv_out = fullfile(cfg.paths.metrics, 'drive_unet_per_image.csv');
    writetable(per_image_table, csv_out);
    fprintf('Per-image table saved to:\n  • %s\n', csv_out);

    fprintf('Diagnostic overlays saved to:\n  • %s\n\n', overlay_dir);

    eval_report = struct();
    eval_report.summary = summary_stats;
    eval_report.per_image = per_image_table;
end

function saveDiagnosticOverlay(img_rgb, gt_mask, pred_mask, fov_mask, dice, out_path)
    [H, W, ~] = size(img_rgb);
    img_double = im2double(img_rgb);

    p1 = img_double;

    p2 = zeros(H, W, 3);
    p2(:, :, 2) = double(gt_mask);

    p3 = zeros(H, W, 3);
    p3(:, :, 2) = double(pred_mask);
    p3(:, :, 3) = double(pred_mask);

    p4 = zeros(H, W, 3);
    tp_mask = pred_mask & gt_mask;
    fp_mask = pred_mask & ~gt_mask & fov_mask;
    fn_mask = ~pred_mask & gt_mask;

    p4(:, :, 2) = p4(:, :, 2) + double(tp_mask);
    p4(:, :, 1) = p4(:, :, 1) + double(fp_mask);
    p4(:, :, 1) = p4(:, :, 1) + double(fn_mask);
    p4(:, :, 2) = p4(:, :, 2) + double(fn_mask);

    row1 = [p1, p2];
    row2 = [p3, p4];
    composite = [row1; row2];
    imwrite(composite, out_path);
end
