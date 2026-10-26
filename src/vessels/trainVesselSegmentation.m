function metrics = trainVesselSegmentation(varargin)
% TRAINVESSELSEGMENTATION Supervised Training for Retinal Vessel Segmentation U-Net on DRIVE
% Partitions the 20 DRIVE training images into 16 training and 4 validation samples (Seed 42).
% Applies multi-orientation data augmentation and trains a MATLAB-native U-Net.
%
% Saves trained model checkpoint to models/vessels/vessel_unet.mat.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('         SHINRA: U-NET RETINAL VESSEL SEGMENTATION TRAINING             \n');
    fprintf('========================================================================\n\n');

    drive_root = cfg.paths.drive;
    train_img_dir = fullfile(drive_root, 'training', 'images');
    train_man_dir = fullfile(drive_root, 'training', '1st_manual');
    train_fov_dir = fullfile(drive_root, 'training', 'mask');

    if exist(train_img_dir, 'dir') ~= 7 || exist(train_man_dir, 'dir') ~= 7
        error('SHINRA:DriveNotFound', 'DRIVE training dataset not found in %s', drive_root);
    end

    img_files = dir(fullfile(train_img_dir, '*.tif'));
    if isempty(img_files)
        img_files = dir(fullfile(train_img_dir, '*.png'));
    end
    num_samples = numel(img_files);
    assert(num_samples == 20, sprintf('Expected exactly 20 DRIVE training images, but found %d.', num_samples));

    image_ids = cell(num_samples, 1);
    image_paths = cell(num_samples, 1);
    manual_mask_paths = cell(num_samples, 1);
    fov_mask_paths = cell(num_samples, 1);

    for i = 1:num_samples
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
    % 3. Fixed Random Seed Partition (16 Train / 4 Validation, Seed 42)
    % -------------------------------------------------------------------------
    rng(cfg.random_seed, 'twister');
    shuffled_idx = randperm(num_samples);
    val_idx = sort(shuffled_idx(1:4));     % Exactly 4 validation images (20%)
    train_idx = sort(shuffled_idx(5:end)); % Exactly 16 training images (80%)

    train_ids = image_ids(train_idx);
    val_ids   = image_ids(val_idx);

    % Assertions on Partition Isolation & Disjointness
    assert(numel(train_idx) == 16, 'Training partition must contain exactly 16 images.');
    assert(numel(val_idx) == 4, 'Validation partition must contain exactly 4 images.');
    assert(isempty(intersect(train_ids, val_ids)), 'Training and Validation partitions must be disjoint!');

    fprintf('Dataset Partition (Seed %d):\n', cfg.random_seed);
    fprintf('  • Training Split:   16 images [%s]\n', strjoin(train_ids, ', '));
    fprintf('  • Validation Split:  4 images [%s]\n\n', strjoin(val_ids, ', '));

    % -------------------------------------------------------------------------
    % 4. Preprocess and Load Arrays
    % -------------------------------------------------------------------------
    target_size = [512, 512];
    X_raw = zeros(target_size(1), target_size(2), 1, num_samples, 'single');
    Y_raw = false(target_size(1), target_size(2), num_samples);
    FOV_raw = false(target_size(1), target_size(2), num_samples);

    fprintf('Loading and preprocessing 20 DRIVE images (enhanced green channel @ 512x512)...\n');
    for i = 1:num_samples
        img_rgb = imread(image_paths{i});
        man_mask = imread(manual_mask_paths{i}) > 0;
        fov_mask = imread(fov_mask_paths{i}) > 0;

        % Extract green channel and apply CLAHE enhancement
        g_chan = im2double(img_rgb(:, :, 2));
        g_clahe = adapthisteq(g_chan, 'ClipLimit', 0.015, 'NumTiles', [8, 8]);
        g_clahe(~fov_mask) = 0;

        X_raw(:, :, 1, i) = single(imresize(g_clahe, target_size, 'bicubic'));
        Y_raw(:, :, i)    = imresize(man_mask, target_size, 'nearest');
        FOV_raw(:, :, i)  = imresize(fov_mask, target_size, 'nearest');
    end

    % Validation Split Data (Raw, Unaugmented)
    X_val = X_raw(:, :, :, val_idx);
    Y_val = Y_raw(:, :, val_idx);
    FOV_val = FOV_raw(:, :, val_idx);

    % Training Split Data with Offline Geometric Augmentations
    X_train_base = X_raw(:, :, :, train_idx);
    Y_train_base = Y_raw(:, :, train_idx);

    fprintf('Applying geometric augmentation (flips, rotations) to 16 training images...\n');
    aug_X = [];
    aug_Y = [];

    for i = 1:16
        base_img = X_train_base(:, :, 1, i);
        base_mask = Y_train_base(:, :, i);

        % 1. Original
        aug_X = cat(4, aug_X, base_img);
        aug_Y = cat(3, aug_Y, base_mask);

        % 2. Horizontal Flip
        aug_X = cat(4, aug_X, fliplr(base_img));
        aug_Y = cat(3, aug_Y, fliplr(base_mask));

        % 3. Vertical Flip
        aug_X = cat(4, aug_X, flipud(base_img));
        aug_Y = cat(3, aug_Y, flipud(base_mask));

        % 4. 180 Rotation (H+V Flip)
        aug_X = cat(4, aug_X, rot90(base_img, 2));
        aug_Y = cat(3, aug_Y, rot90(base_mask, 2));

        % 5. +15 deg rotation
        aug_X = cat(4, aug_X, imrotate(base_img, 15, 'bilinear', 'crop'));
        aug_Y = cat(3, aug_Y, imrotate(base_mask, 15, 'nearest', 'crop'));

        % 6. -15 deg rotation
        aug_X = cat(4, aug_X, imrotate(base_img, -15, 'bilinear', 'crop'));
        aug_Y = cat(3, aug_Y, imrotate(base_mask, -15, 'nearest', 'crop'));
    end

    num_aug_train = size(aug_X, 4);
    fprintf('Training dataset expanded to %d augmented patches.\n', num_aug_train);

    % Categorical Labels for Pixel Classification
    classes = ["Background", "Vessel"];
    labelIDs = [0, 1];
    Y_train_cat = categorical(uint8(aug_Y), labelIDs, classes);
    Y_val_cat   = categorical(uint8(Y_val), labelIDs, classes);

    % -------------------------------------------------------------------------
    % 5. Assemble Combined Datastores for MATLAB R2026a Semantic Segmentation
    % -------------------------------------------------------------------------
    ds_train_x = arrayDatastore(aug_X, 'IterationDimension', 4);
    ds_train_y = arrayDatastore(Y_train_cat, 'IterationDimension', 3);
    ds_train = combine(ds_train_x, ds_train_y);

    ds_val_x = arrayDatastore(X_val, 'IterationDimension', 4);
    ds_val_y = arrayDatastore(Y_val_cat, 'IterationDimension', 3);
    ds_val = combine(ds_val_x, ds_val_y);

    % -------------------------------------------------------------------------
    % 6. Build U-Net Architecture & Enforce Pre-Training Architecture Assertions
    % -------------------------------------------------------------------------
    lgraph = buildVesselUNet([target_size, 1], 2);

    all_l_classes = cellfun(@class, num2cell(lgraph.Layers), 'UniformOutput', false);
    in_layer = lgraph.Layers(1);
    assert(isequal(in_layer.InputSize, [512, 512, 1]), 'Input size must be strictly [512 512 1]!');
    assert(any(contains(all_l_classes, 'PixelClassificationLayer')), 'PixelClassificationLayer must be present in U-Net!');
    assert(~any(contains(all_l_classes, 'ClassificationOutputLayer') | ...
                (contains(all_l_classes, 'ClassificationLayer') & ~contains(all_l_classes, 'Pixel'))), ...
                'ClassificationOutputLayer must NOT be present in vessel segmentation network!');
    assert(~any(contains(all_l_classes, 'GlobalAveragePooling')), ...
                'GlobalAveragePooling2DLayer must NOT be present in vessel segmentation network!');
    fprintf('Architecture Pre-Check: 512x512x1 U-Net with PixelClassificationLayer verified.\n');

    % -------------------------------------------------------------------------
    % 7. Configure Training Options
    % -------------------------------------------------------------------------
    max_epochs = 25;
    mini_batch_size = 4;
    initial_lr = 1e-3;

    opts = trainingOptions('adam', ...
        'InitialLearnRate', initial_lr, ...
        'MaxEpochs', max_epochs, ...
        'MiniBatchSize', mini_batch_size, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', ds_val, ...
        'ValidationFrequency', floor(num_aug_train / mini_batch_size), ...
        'Plots', 'none', ...
        'Verbose', true, ...
        'ExecutionEnvironment', 'auto');

    fprintf('Starting U-Net training (%d epochs, batch size %d, LR %.1e)...\n', ...
        max_epochs, mini_batch_size, initial_lr);

    % -------------------------------------------------------------------------
    % 8. Train U-Net
    % -------------------------------------------------------------------------
    try
        net = trainNetwork(ds_train, lgraph, opts);
    catch ME
        fprintf('\n[ERROR] U-Net training failed: %s\n', ME.message);
        rethrow(ME);
    end

    % -------------------------------------------------------------------------
    % 9. Immediately Save Trained Network Checkpoint
    % -------------------------------------------------------------------------
    split_info = struct();
    split_info.train_image_ids = train_ids;
    split_info.val_image_ids = val_ids;
    split_info.random_seed = cfg.random_seed;
    split_info.train_count = 16;
    split_info.val_count = 4;
    split_info.timestamp = char(datetime('now'));

    model_file = fullfile(cfg.paths.models_vessels, 'vessel_unet.mat');
    if exist(cfg.paths.models_vessels, 'dir') ~= 7
        mkdir(cfg.paths.models_vessels);
    end

    save(model_file, 'net', 'split_info', 'opts', 'cfg');
    save(fullfile(cfg.paths.metrics, 'drive_vessel_split_info.mat'), 'split_info');
    fprintf('\nU-Net trained weights saved to:\n  • %s\n\n', model_file);

    % -------------------------------------------------------------------------
    % 10. Evaluate on Held-Out Validation Split Using semanticseg (N=4, within FOV)
    % -------------------------------------------------------------------------
    fprintf('Evaluating U-Net on held-out validation images (N=4, within FOV)...\n');
    val_dice = zeros(4, 1);
    val_iou  = zeros(4, 1);
    val_sens = zeros(4, 1);
    val_spec = zeros(4, 1);
    val_prec = zeros(4, 1);
    val_acc  = zeros(4, 1);

    val_tp = zeros(4, 1);
    val_fp = zeros(4, 1);
    val_fn = zeros(4, 1);
    val_tn = zeros(4, 1);

    for v = 1:4
        val_in = X_val(:, :, :, v);
        
        % Semantic segmentation inference via MATLAB semanticseg API
        pred_cat = semanticseg(val_in, net);
        pred_vessels = (pred_cat == 'Vessel') & FOV_val(:, :, v);
        gt_vessels   = Y_val(:, :, v) & FOV_val(:, :, v);
        fov_mask     = FOV_val(:, :, v);

        tp = sum(pred_vessels(:) & gt_vessels(:));
        fp = sum(pred_vessels(:) & ~gt_vessels(:) & fov_mask(:));
        fn = sum(~pred_vessels(:) & gt_vessels(:));
        tn = sum(~pred_vessels(:) & ~gt_vessels(:) & fov_mask(:));

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

        fprintf('  Val %d (%-14s): Dice = %6.4f | IoU = %6.4f | Sens = %5.2f%% | Spec = %5.2f%% | Prec = %5.2f%%\n', ...
            v, val_ids{v}, val_dice(v), val_iou(v), val_sens(v)*100, val_spec(v)*100, val_prec(v)*100);
    end

    metrics = struct();
    metrics.val_dice = val_dice;
    metrics.val_iou = val_iou;
    metrics.val_sensitivity = val_sens;
    metrics.val_specificity = val_spec;
    metrics.val_precision = val_prec;
    metrics.val_accuracy = val_acc;
    metrics.val_tp = val_tp;
    metrics.val_fp = val_fp;
    metrics.val_fn = val_fn;
    metrics.val_tn = val_tn;
    metrics.mean_dice = mean(val_dice);
    metrics.mean_iou  = mean(val_iou);
    metrics.mean_sensitivity = mean(val_sens);
    metrics.mean_specificity = mean(val_spec);
    metrics.mean_precision = mean(val_prec);
    metrics.mean_accuracy = mean(val_acc);
    metrics.val_image_ids = val_ids;
    metrics.train_image_ids = train_ids;
    metrics.timestamp = char(datetime('now'));

    fprintf('------------------------------------------------------------------------\n');
    fprintf('  Validation Mean (N=4): Dice = %6.4f | IoU = %6.4f | Sens = %5.2f%% | Spec = %5.2f%%\n', ...
        metrics.mean_dice, metrics.mean_iou, metrics.mean_sensitivity*100, metrics.mean_specificity*100);
    fprintf('========================================================================\n\n');

    % Re-save model checkpoint with validation metrics
    save(model_file, 'net', 'metrics', 'split_info', 'opts', 'cfg');
    save(fullfile(cfg.paths.metrics, 'vessel_unet_validation_metrics.mat'), 'metrics');
    fprintf('Validation metrics saved to:\n  • %s\n\n', fullfile(cfg.paths.metrics, 'vessel_unet_validation_metrics.mat'));
end
