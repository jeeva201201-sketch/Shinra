function metrics = evaluateVesselUNetValidation(net_in)
% EVALUATEVESSELUNETVALIDATION Dedicated Semantic Segmentation Validation Evaluator
% Evaluates the trained U-Net exclusively on the 4 held-out validation images:
%   1. 25_training
%   2. 26_training
%   3. 27_training
%   4. 31_training
%
% Uses the native MATLAB semanticseg API.
% If net_in is provided as an argument, uses it directly; otherwise loads models/vessels/vessel_unet.mat.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('      SHINRA U-NET: HELD-OUT VALIDATION SET EVALUATION (N=4)            \n');
    fprintf('========================================================================\n\n');

    model_file = fullfile(cfg.paths.models_vessels, 'vessel_unet.mat');

    % 1. Resolve Trained Network
    if nargin >= 1 && ~isempty(net_in)
        net = net_in;
        fprintf('Using U-Net passed directly from workspace memory.\n');
    elseif exist(model_file, 'file') == 2
        loaded = load(model_file);
        if ~isfield(loaded, 'net')
            error('SHINRA:InvalidCheckpoint', 'Checkpoint does not contain "net" variable.');
        end
        net = loaded.net;
        fprintf('Loaded U-Net checkpoint from: %s\n', model_file);
    else
        error('SHINRA:ModelNotFound', ...
            'Trained U-Net model not found at:\n  %s\nPlease pass the trained "net" variable or run: trainVesselSegmentation', ...
            model_file);
    end

    % 2. Defensive Check: Confirm Network Compatibility with semanticseg
    % Search for PixelClassificationLayer across all layers in the DAGNetwork
    has_pixel_layer = false;
    pixel_layer = [];
    for i = 1:numel(net.Layers)
        if isa(net.Layers(i), 'nnet.cnn.layer.PixelClassificationLayer')
            has_pixel_layer = true;
            pixel_layer = net.Layers(i);
            break;
        end
    end
    assert(has_pixel_layer, 'Network must contain a PixelClassificationLayer for semanticseg compatibility.');

    if isprop(pixel_layer, 'Classes') && ~isempty(pixel_layer.Classes)
        p_classes = cellstr(pixel_layer.Classes);
        assert(ismember('Background', p_classes) && ismember('Vessel', p_classes), ...
            'PixelClassificationLayer must define Background and Vessel classes.');
        fprintf('Network Output Layer: %s (Classes: [%s])\n', pixel_layer.Name, strjoin(p_classes, ', '));
    else
        fprintf('Network Output Layer: %s\n', pixel_layer.Name);
    end

    % 3. The 4 Held-Out Validation Images (Seed 42)
    val_ids = {'25_training', '26_training', '27_training', '31_training'};
    num_val = numel(val_ids);

    drive_root = cfg.paths.drive;
    img_dir = fullfile(drive_root, 'training', 'images');
    man_dir = fullfile(drive_root, 'training', '1st_manual');
    fov_dir = fullfile(drive_root, 'training', 'mask');

    target_size = [512, 512];

    % -------------------------------------------------------------------------
    % 4. Single-Image Dry-Run Verification (25_training)
    % -------------------------------------------------------------------------
    fprintf('\n[DRY-RUN] Verifying single-image semanticseg inference on %s...\n', val_ids{1});
    test_id = val_ids{1};
    test_prefix = regexp(test_id, '^(\d+)', 'tokens');
    test_prefix = test_prefix{1}{1};
    test_img_path = fullfile(img_dir, [test_id, '.tif']);
    if ~exist(test_img_path, 'file'); test_img_path = fullfile(img_dir, [test_id, '.png']); end
    test_fov_path = fullfile(fov_dir, [test_prefix, '_training_mask.gif']);

    test_img = imread(test_img_path);
    test_fov = imread(test_fov_path) > 0;
    test_g = im2double(test_img(:, :, 2));
    test_g_clahe = adapthisteq(test_g, 'ClipLimit', 0.015, 'NumTiles', [8, 8]);
    test_g_clahe(~test_fov) = 0;
    test_in = single(imresize(test_g_clahe, target_size, 'bicubic'));

    test_pred_cat = semanticseg(test_in, net);
    assert(isequal(size(test_pred_cat), target_size), 'Dry-run prediction must be 512x512.');
    test_cats = categories(test_pred_cat);
    assert(ismember('Background', test_cats) && ismember('Vessel', test_cats), ...
        'Predicted categories must contain Background and Vessel.');
    test_vessel_mask = (test_pred_cat == 'Vessel');
    assert(islogical(test_vessel_mask) && sum(test_vessel_mask(:)) > 0, ...
        'Dry-run must extract a non-empty logical Vessel mask.');
    fprintf('  -> Dry-run PASSED: 512x512 output, categories [%s], %d vessel pixels detected.\n\n', ...
        strjoin(test_cats, ', '), sum(test_vessel_mask(:)));

    % -------------------------------------------------------------------------
    % 5. Evaluate All 4 Held-Out Validation Images (Inside Valid FOV)
    % -------------------------------------------------------------------------
    val_dice = zeros(num_val, 1);
    val_iou  = zeros(num_val, 1);
    val_sens = zeros(num_val, 1);
    val_spec = zeros(num_val, 1);
    val_prec = zeros(num_val, 1);
    val_acc  = zeros(num_val, 1);

    val_tp = zeros(num_val, 1);
    val_fp = zeros(num_val, 1);
    val_fn = zeros(num_val, 1);
    val_tn = zeros(num_val, 1);

    fprintf('Evaluating semantic segmentation across %d held-out validation images...\n', num_val);
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('  Idx  Image ID          Dice       IoU      Sensitivity  Specificity  Precision   Accuracy  \n');
    fprintf('-----------------------------------------------------------------------------------------\n');

    for v = 1:num_val
        id_str = val_ids{v};
        tokens = regexp(id_str, '^(\d+)', 'tokens');
        prefix = tokens{1}{1};

        img_path = fullfile(img_dir, [id_str, '.tif']);
        if ~exist(img_path, 'file')
            img_path = fullfile(img_dir, [id_str, '.png']);
        end
        man_path = fullfile(man_dir, [prefix, '_manual1.gif']);
        fov_path = fullfile(fov_dir, [prefix, '_training_mask.gif']);

        assert(exist(img_path, 'file') == 2, sprintf('Image missing: %s', img_path));
        assert(exist(man_path, 'file') == 2, sprintf('Manual mask missing: %s', man_path));
        assert(exist(fov_path, 'file') == 2, sprintf('FOV mask missing: %s', fov_path));

        img_rgb  = imread(img_path);
        gt_raw   = imread(man_path) > 0;
        fov_raw  = imread(fov_path) > 0;

        % Preprocess: Green channel + CLAHE @ 512x512
        g_chan = im2double(img_rgb(:, :, 2));
        g_clahe = adapthisteq(g_chan, 'ClipLimit', 0.015, 'NumTiles', [8, 8]);
        g_clahe(~fov_raw) = 0;

        val_in = single(imresize(g_clahe, target_size, 'bicubic'));
        gt_resized = imresize(gt_raw, target_size, 'nearest');
        fov_resized = imresize(fov_raw, target_size, 'nearest');

        % Native MATLAB semanticseg inference
        pred_cat = semanticseg(val_in, net);
        pred_vessels = (pred_cat == 'Vessel') & fov_resized;
        gt_vessels   = gt_resized & fov_resized;

        tp = sum(pred_vessels(:) & gt_vessels(:));
        fp = sum(pred_vessels(:) & ~gt_vessels(:) & fov_resized(:));
        fn = sum(~pred_vessels(:) & gt_vessels(:));
        tn = sum(~pred_vessels(:) & ~gt_vessels(:) & fov_resized(:));

        val_tp(v) = tp;
        val_fp(v) = fp;
        val_fn(v) = fn;
        val_tn(v) = tn;

        val_dice(v) = (2 * tp) / max(1, (2 * tp + fp + fn));
        val_iou(v)  = tp / max(1, (tp + fp + fn));
        val_sens(v) = tp / max(1, (tp + fn));
        val_spec(v) = tn / max(1, (tn + fp));
        val_prec(v) = tp / max(1, (tp + fp));
        val_acc(v)  = (tp + tn) / max(1, (tp + tn + fp + fn));

        fprintf('  %2d   %-16s %6.4f    %6.4f     %6.2f%%      %6.2f%%     %6.2f%%    %6.2f%%\n', ...
            v, id_str, val_dice(v), val_iou(v), val_sens(v)*100, val_spec(v)*100, val_prec(v)*100, val_acc(v)*100);
    end
    fprintf('-----------------------------------------------------------------------------------------\n\n');

    % -------------------------------------------------------------------------
    % 6. Aggregate Statistics
    % -------------------------------------------------------------------------
    total_tp = sum(val_tp);
    total_fp = sum(val_fp);
    total_fn = sum(val_fn);
    total_tn = sum(val_tn);

    pooled_dice = (2 * total_tp) / max(1, 2 * total_tp + total_fp + total_fn);
    pooled_iou  = total_tp / max(1, total_tp + total_fp + total_fn);
    pooled_sens = total_tp / max(1, total_tp + total_fn);
    pooled_spec = total_tn / max(1, total_tn + total_fp);
    pooled_prec = total_tp / max(1, total_tp + total_fp);
    pooled_acc  = (total_tp + total_tn) / max(1, total_tp + total_tn + total_fp + total_fn);

    fprintf('========================================================================\n');
    fprintf('  U-NET VALIDATION PERFORMANCE SUMMARY (N=4 HELD-OUT IMAGES)            \n');
    fprintf('========================================================================\n');
    fprintf('  • Mean Dice Coefficient:      %6.4f ± %6.4f  (Median: %6.4f)\n', ...
        mean(val_dice), std(val_dice), median(val_dice));
    fprintf('  • Mean Jaccard Index (IoU):   %6.4f ± %6.4f  (Median: %6.4f)\n', ...
        mean(val_iou), std(val_iou), median(val_iou));
    fprintf('  • Vessel Sensitivity:         %6.2f%% ± %5.2f%% (Pooled: %5.2f%%)\n', ...
        mean(val_sens)*100, std(val_sens)*100, pooled_sens*100);
    fprintf('  • Specificity (Non-Vessel):   %6.2f%% ± %5.2f%% (Pooled: %5.2f%%)\n', ...
        mean(val_spec)*100, std(val_spec)*100, pooled_spec*100);
    fprintf('  • Precision (PPV):            %6.2f%% ± %5.2f%% (Pooled: %5.2f%%)\n', ...
        mean(val_prec)*100, std(val_prec)*100, pooled_prec*100);
    fprintf('  • Overall Pixel Accuracy:     %6.2f%% ± %5.2f%% (Pooled: %5.2f%%)\n', ...
        mean(val_acc)*100, std(val_acc)*100, pooled_acc*100);
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  Pooled Confusion: TP = %d | FP = %d | FN = %d | TN = %d\n', ...
        total_tp, total_fp, total_fn, total_tn);
    fprintf('========================================================================\n\n');

    metrics = struct();
    metrics.val_image_ids = val_ids;
    metrics.val_dice = val_dice;
    metrics.val_iou = val_iou;
    metrics.val_sensitivity = val_sens;
    metrics.val_specificity = val_spec;
    metrics.val_precision = val_prec;
    metrics.val_accuracy = val_acc;
    metrics.mean_dice = mean(val_dice);
    metrics.median_dice = median(val_dice);
    metrics.std_dice = std(val_dice);
    metrics.mean_iou = mean(val_iou);
    metrics.median_iou = median(val_iou);
    metrics.std_iou = std(val_iou);
    metrics.mean_sensitivity = mean(val_sens);
    metrics.mean_specificity = mean(val_spec);
    metrics.mean_precision = mean(val_prec);
    metrics.mean_accuracy = mean(val_acc);
    metrics.pooled_dice = pooled_dice;
    metrics.pooled_iou = pooled_iou;
    metrics.pooled_confusion = struct('TP', total_tp, 'FP', total_fp, 'FN', total_fn, 'TN', total_tn);
    metrics.timestamp = char(datetime('now'));

    % Save validation metrics
    out_file = fullfile(cfg.paths.metrics, 'vessel_unet_validation_metrics.mat');
    save(out_file, 'metrics', 'cfg');
    fprintf('Validation metrics saved to:\n  • %s\n\n', out_file);

    % If model_file does not exist on disk, save checkpoint
    if exist(model_file, 'file') ~= 2
        split_info = struct('train_image_ids', {{'21_training','22_training','23_training','24_training','28_training','29_training','30_training','32_training','33_training','34_training','35_training','36_training','37_training','38_training','39_training','40_training'}}, ...
            'val_image_ids', {val_ids}, 'random_seed', 42, 'train_count', 16, 'val_count', 4, 'timestamp', char(datetime('now')));
        save(model_file, 'net', 'metrics', 'split_info', 'cfg');
        fprintf('Model checkpoint saved to:\n  • %s\n\n', model_file);
    end
end
