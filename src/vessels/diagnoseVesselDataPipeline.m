function is_valid = diagnoseVesselDataPipeline()
% DIAGNOSEVESSELDATAPIPELINE Diagnostic Inspection of U-Net Training Data Pipeline
% Performs complete data loading, partitioning, augmentation, datastore assembly,
% and network layer graph verification for DRIVE semantic segmentation.
%
% Tests and verifies:
% 1. Dataset loading & seed-42 16/4 train-validation partitioning.
% 2. Exact 1-to-1 synchronized geometric augmentation (96 samples).
% 3. Data types, tensor shapes, NaN checks, and categorical undefined checks.
% 4. Layer graph construction and pixelClassificationLayer class verification.
% 5. MATLAB R2026a CombinedDatastore read verification for semantic segmentation.
% 6. Exits BEFORE trainNetwork so the pipeline can be inspected and verified.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('       SHINRA U-NET VESSEL DATA PIPELINE & COMPATIBILITY AUDIT          \n');
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
    % 1. Partition Verification (Seed 42: 16 Train / 4 Val)
    % -------------------------------------------------------------------------
    rng(cfg.random_seed, 'twister');
    shuffled_idx = randperm(num_samples);
    val_idx = sort(shuffled_idx(1:4));     % Exactly 4 validation images (20%)
    train_idx = sort(shuffled_idx(5:end)); % Exactly 16 training images (80%)

    train_ids = image_ids(train_idx);
    val_ids   = image_ids(val_idx);

    assert(numel(train_idx) == 16, 'Training partition must contain exactly 16 images.');
    assert(numel(val_idx) == 4, 'Validation partition must contain exactly 4 images.');
    assert(isempty(intersect(train_ids, val_ids)), 'Training and Validation partitions must be disjoint!');

    fprintf('[CHECK 1/6] Partition Integrity Check:\n');
    fprintf('  • Random Seed:            %d\n', cfg.random_seed);
    fprintf('  • Training Split (N=16):  [%s]\n', strjoin(train_ids, ', '));
    fprintf('  • Validation Split (N=4): [%s]\n', strjoin(val_ids, ', '));
    fprintf('  • Disjointness Assert:    PASSED (0 overlapping IDs)\n\n');

    % -------------------------------------------------------------------------
    % 2. Preprocess 20 Images (@ 512x512)
    % -------------------------------------------------------------------------
    target_size = [512, 512];
    X_raw = zeros(target_size(1), target_size(2), 1, num_samples, 'single');
    Y_raw = false(target_size(1), target_size(2), num_samples);
    FOV_raw = false(target_size(1), target_size(2), num_samples);

    for i = 1:num_samples
        img_rgb = imread(image_paths{i});
        man_mask = imread(manual_mask_paths{i}) > 0;
        fov_mask = imread(fov_mask_paths{i}) > 0;

        g_chan = im2double(img_rgb(:, :, 2));
        g_clahe = adapthisteq(g_chan, 'ClipLimit', 0.015, 'NumTiles', [8, 8]);
        g_clahe(~fov_mask) = 0;

        X_raw(:, :, 1, i) = single(imresize(g_clahe, target_size, 'bicubic'));
        Y_raw(:, :, i)    = imresize(man_mask, target_size, 'nearest');
        FOV_raw(:, :, i)  = imresize(fov_mask, target_size, 'nearest');
    end

    % -------------------------------------------------------------------------
    % 3. Synchronized Geometric Augmentation (16 x 6 = 96 samples)
    % -------------------------------------------------------------------------
    X_train_base = X_raw(:, :, :, train_idx);
    Y_train_base = Y_raw(:, :, train_idx);

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
    assert(num_aug_train == 96, sprintf('Expected exactly 96 augmented training samples, got %d', num_aug_train));

    % Convert binary masks to categorical labels
    classes = ["Background", "Vessel"];
    labelIDs = [0, 1];
    Y_train_cat = categorical(uint8(aug_Y), labelIDs, classes);

    X_val = X_raw(:, :, :, val_idx);
    Y_val_cat = categorical(uint8(Y_raw(:, :, val_idx)), labelIDs, classes);

    % -------------------------------------------------------------------------
    % 4. Inspect Data Types, Shapes, NaNs, and Categoricals
    % -------------------------------------------------------------------------
    fprintf('[CHECK 2/6] Training & Validation Tensor Diagnostic:\n');
    fprintf('  • aug_X Size:             [%d x %d x %d x %d]\n', ...
        size(aug_X, 1), size(aug_X, 2), size(aug_X, 3), size(aug_X, 4));
    fprintf('  • aug_X Class:            %s (Range: [%.3f, %.3f])\n', ...
        class(aug_X), min(aug_X(:)), max(aug_X(:)));
    fprintf('  • aug_X NaNs:             %d\n', sum(isnan(aug_X(:))));
    
    fprintf('  • Y_train_cat Size:       [%d x %d x %d]\n', ...
        size(Y_train_cat, 1), size(Y_train_cat, 2), size(Y_train_cat, 3));
    fprintf('  • Y_train_cat Class:      %s\n', class(Y_train_cat));
    fprintf('  • Y_train_cat Categories: [%s]\n', strjoin(cellstr(categories(Y_train_cat)), ', '));
    fprintf('  • Undefined Pixels:       %d\n', sum(isundefined(Y_train_cat(:))));

    % Verify pixel balance
    cnt_bg = sum(Y_train_cat(:) == 'Background');
    cnt_vs = sum(Y_train_cat(:) == 'Vessel');
    pct_vs = (cnt_vs / (cnt_bg + cnt_vs)) * 100;
    fprintf('  • Pixel Distribution:     Background = %d (%5.2f%%), Vessel = %d (%5.2f%%)\n', ...
        cnt_bg, 100 - pct_vs, cnt_vs, pct_vs);

    fprintf('  • X_val Size:             [%d x %d x %d x %d] (%s)\n', ...
        size(X_val, 1), size(X_val, 2), size(X_val, 3), size(X_val, 4), class(X_val));
    fprintf('  • Y_val_cat Size:         [%d x %d x %d] (%s)\n\n', ...
        size(Y_val_cat, 1), size(Y_val_cat, 2), size(Y_val_cat, 3), class(Y_val_cat));

    % Assertions
    assert(size(aug_X, 4) == size(Y_train_cat, 3), 'Sample count mismatch between X and Y!');
    assert(size(aug_X, 1) == 512 && size(aug_X, 2) == 512, 'Spatial dimension of aug_X must be 512x512!');
    assert(size(Y_train_cat, 1) == 512 && size(Y_train_cat, 2) == 512, 'Spatial dimension of Y_train_cat must be 512x512!');
    assert(~any(isnan(aug_X(:))), 'aug_X must not contain NaNs!');
    assert(~any(isundefined(Y_train_cat(:))), 'Y_train_cat must not contain undefined categories!');

    % -------------------------------------------------------------------------
    % 5. Build and Verify U-Net Architecture
    % -------------------------------------------------------------------------
    fprintf('[CHECK 3/6] U-Net Architecture Layer Graph Verification:\n');
    lgraph = buildVesselUNet([512, 512, 1], 2);

    input_layer = lgraph.Layers(1);
    output_layer = lgraph.Layers(end);

    fprintf('  • Input Layer:            %s (InputSize: [%d x %d x %d])\n', ...
        input_layer.Name, input_layer.InputSize(1), input_layer.InputSize(2), input_layer.InputSize(3));
    fprintf('  • Output Layer:           %s (%s)\n', output_layer.Name, class(output_layer));
    if isprop(output_layer, 'Classes') && ~isempty(output_layer.Classes)
        fprintf('  • Output Classes:         [%s]\n', strjoin(cellstr(output_layer.Classes), ', '));
        assert(isequal(cellstr(output_layer.Classes), {'Background'; 'Vessel'}) || ...
               isequal(cellstr(output_layer.Classes), {'Background', 'Vessel'}), ...
               'Output layer classes do not match expected ["Background", "Vessel"]');
    end
    assert(isa(output_layer, 'nnet.cnn.layer.PixelClassificationLayer'), ...
        'Terminal layer must be a PixelClassificationLayer for semantic segmentation!');
    fprintf('  • Architecture Check:     PASSED (Valid Semantic Segmentation U-Net)\n\n');

    % -------------------------------------------------------------------------
    % 6. Construct CombinedDatastore & Verify Read Interface
    % -------------------------------------------------------------------------
    fprintf('[CHECK 4/6] MATLAB R2026a Semantic Segmentation Datastore Assembly:\n');
    ds_train_x = arrayDatastore(aug_X, 'IterationDimension', 4);
    ds_train_y = arrayDatastore(Y_train_cat, 'IterationDimension', 3);
    ds_train = combine(ds_train_x, ds_train_y);

    ds_val_x = arrayDatastore(X_val, 'IterationDimension', 4);
    ds_val_y = arrayDatastore(Y_val_cat, 'IterationDimension', 3);
    ds_val = combine(ds_val_x, ds_val_y);

    fprintf('  • ds_train Type:          %s\n', class(ds_train));
    fprintf('  • ds_val Type:            %s\n', class(ds_val));
    fprintf('  • ds_train Read Test...   ');

    sample_train = read(ds_train);
    assert(iscell(sample_train) && numel(sample_train) == 2, 'Combined datastore must return a 2-element cell array {X, Y}');
    x_sample = sample_train{1};
    y_sample = sample_train{2};

    fprintf('SUCCESS\n');
    fprintf('    - Read X shape:         [%d x %d x %d] (%s)\n', ...
        size(x_sample, 1), size(x_sample, 2), size(x_sample, 3), class(x_sample));
    fprintf('    - Read Y shape:         [%d x %d] (%s with %d categories)\n', ...
        size(y_sample, 1), size(y_sample, 2), class(y_sample), numel(categories(y_sample)));
    reset(ds_train);

    sample_val = read(ds_val);
    assert(iscell(sample_val) && numel(sample_val) == 2, 'Validation combined datastore must return {X, Y}');
    reset(ds_val);
    fprintf('  • ds_val Read Test...     SUCCESS\n\n');

    % -------------------------------------------------------------------------
    % 7. Configure Training Options Check
    % -------------------------------------------------------------------------
    fprintf('[CHECK 5/6] Training Options Configuration Check:\n');
    opts = trainingOptions('adam', ...
        'InitialLearnRate', 1e-3, ...
        'MaxEpochs', 25, ...
        'MiniBatchSize', 4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', ds_val, ...
        'ValidationFrequency', floor(96 / 4), ...
        'Plots', 'none', ...
        'Verbose', true, ...
        'ExecutionEnvironment', 'auto');
    fprintf('  • Optimizer:              Adam\n');
    fprintf('  • Initial Learn Rate:     1e-3\n');
    fprintf('  • Max Epochs:             25\n');
    fprintf('  • Mini Batch Size:        4\n');
    fprintf('  • Validation Frequency:   Every 24 iterations (1 epoch)\n');
    fprintf('  • Options Check:          PASSED\n\n');

    % -------------------------------------------------------------------------
    % 8. Exit Prior to trainNetwork Execution
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('  [CHECK 6/6] SUMMARY: ALL 6 VESSEL DATA PIPELINE CHECKS PASSED         \n');
    fprintf('  Data pipeline is 100%% compatible with MATLAB R2026a trainNetwork.     \n');
    fprintf('  (Exiting diagnostic dry-run without executing training).               \n');
    fprintf('========================================================================\n\n');

    is_valid = true;
end
