function metrics = trainSeveritySqueezeNetV3(varargin)
% TRAINSEVERITYSQUEEZENETV3 Supervised Transfer Learning for 5-Class DR Severity (Experiment V3)
% Fine-tunes ImageNet-pretrained SqueezeNet with:
% 1. Frozen conv1 through fire7.
% 2. Fine-tuned fire8 and fire9 (base learn rate = 5e-4).
% 3. Direct logit path (relu_conv10 removed) with 5-class head (learn rate factor = 10 -> 5e-3).
% 4. Standardized ImageNet [0, 255] intensity range at 227x227x3 resolution.
% 5. Preserved seed-42 330/83 stratified split and 545-sample balanced training.
%
% Saves model checkpoint to models/severity/SHINRA_SqueezeNet_Severity_Model_V3.mat.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('  EXPERIMENT V3: SQUEEZENET TRANSFER LEARNING WITH DIRECT LOGIT PATH    \n');
    fprintf('========================================================================\n\n');

    % 1. Verify V3 Architecture Programmatically Prior to Data Preparation
    is_valid_arch = diagnoseSqueezeNetV3();
    if ~is_valid_arch
        error('SHINRA:InvalidArchitecture', 'SqueezeNet V3 architecture verification failed.');
    end

    [idrid_train_full, idrid_test] = load_idrid();
    if height(idrid_train_full) == 0
        error('SHINRA:DataNotFound', 'IDRiD training data not available in %s', cfg.paths.idrid);
    end

    % 2. Stratified 80/20 train/validation split (330 train, 83 validation, Seed 42)
    [train_tab, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);
    
    % Assertions on Data Isolation
    assert(height(train_tab) == 330, 'Training split size mismatch.');
    assert(height(val_tab) == 83, 'Validation split size mismatch.');
    assert(~any(ismember(train_tab.ImagePath, idrid_test.ImagePath)), 'Data Leakage: Test images found in training set!');
    assert(~any(ismember(val_tab.ImagePath, idrid_test.ImagePath)), 'Data Leakage: Test images found in validation set!');

    fprintf('Dataset Partition (Seed %d, Identical to V1 & V2 Baseline):\n', cfg.random_seed);
    fprintf('  • Training Split:   %d labeled images (Used for model training & balance)\n', height(train_tab));
    fprintf('  • Validation Split: %d labeled images (Used strictly for model evaluation)\n', height(val_tab));
    fprintf('  • Held-Out Test:    103 images (Completely isolated & untouched)\n\n');

    % SqueezeNet Native Input Dimension: [227, 227, 3]
    squeezenet_input_size = [227, 227, 3];
    target_size_2d = [227, 227];
    
    % 3. Preprocessing of Training Data (227x227x3, Scaled to [0, 255])
    num_train = height(train_tab);
    X_train_raw = zeros(target_size_2d(1), target_size_2d(2), 3, num_train, 'single');
    Y_train_raw = categorical(train_tab.DR_Grade, 0:4, cfg.severity.class_names);

    fprintf('Loading and preprocessing %d training images (scaled to [0, 255] for ImageNet weights)...\n', num_train);
    for i = 1:num_train
        img_raw = imread(train_tab.ImagePath{i});
        prep = preprocessRetina(img_raw, target_size_2d);
        % Scale [0, 1] normalized RGB to [0, 255] matching pretrained ImageNet filter scales
        X_train_raw(:, :, :, i) = single(prep.rgb_normalized * 255.0);
    end

    % 4. Class-Balanced Resampling on Training Split Only (Identical to V1 & V2)
    rng(cfg.random_seed, 'twister');
    grades_train = train_tab.DR_Grade;
    counts_per_grade = histcounts(grades_train, -0.5:1:4.5);
    max_class_count = max(counts_per_grade);
    
    balanced_indices = [];
    for g = 0:4
        idx_g = find(grades_train == g);
        if ~isempty(idx_g)
            % Resample with replacement up to max_class_count
            resampled_idx = idx_g(randi(numel(idx_g), [max_class_count, 1]));
            balanced_indices = [balanced_indices; resampled_idx]; %#ok<AGROW>
        end
    end
    % Shuffle balanced training set
    balanced_indices = balanced_indices(randperm(numel(balanced_indices)));
    
    X_train = X_train_raw(:, :, :, balanced_indices);
    Y_train = Y_train_raw(balanced_indices);
    fprintf('Class balancing applied to training split: %d balanced samples.\n', numel(Y_train));

    % Verify all 5 classes are present
    for g = 0:4
        c_name = cfg.severity.class_names{g+1};
        assert(sum(Y_train == c_name) > 0, sprintf('Class %s missing in training set!', c_name));
    end

    % 5. Deterministic Preprocessing of Validation Data (No Augmentation, Scaled to [0, 255])
    num_val = height(val_tab);
    X_val = zeros(target_size_2d(1), target_size_2d(2), 3, num_val, 'single');
    Y_val = categorical(val_tab.DR_Grade, 0:4, cfg.severity.class_names);

    fprintf('Loading and preprocessing %d validation images (deterministic [0, 255])...\n', num_val);
    for i = 1:num_val
        img_raw = imread(val_tab.ImagePath{i});
        prep = preprocessRetina(img_raw, target_size_2d);
        X_val(:, :, :, i) = single(prep.rgb_normalized * 255.0);
    end

    % 6. Data Augmentation Pipeline for Training Data Only
    augmenter = imageDataAugmenter(...
        'RandXReflection', true, ...
        'RandYReflection', true, ...
        'RandRotation', [-15, 15], ...
        'RandScale', [0.90, 1.10], ...
        'RandXTranslation', [-5, 5], ...
        'RandYTranslation', [-5, 5]);

    augds_train = augmentedImageDatastore(squeezenet_input_size, X_train, Y_train, ...
        'DataAugmentation', augmenter);

    % 7. Construct SqueezeNet V3 Transfer Learning Architecture
    lgraph = buildSqueezeNetSeverityV3(5);

    % 8. Dimension Consistency Verification Diagnostics
    input_layer = lgraph.Layers(1);
    net_input_dim = input_layer.InputSize;

    fprintf('\n------------------------------------------------------------\n');
    fprintf('  SQUEEZENET V3 DIMENSION & INTENSITY DIAGNOSTIC CHECK\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('  • Pretrained Network Input Size: [%d, %d, %d]\n', ...
        net_input_dim(1), net_input_dim(2), net_input_dim(3));
    fprintf('  • Training Array Dimensions:     [%d, %d, %d, %d] (Range: [%.1f, %.1f])\n', ...
        size(X_train, 1), size(X_train, 2), size(X_train, 3), size(X_train, 4), ...
        min(X_train(:)), max(X_train(:)));
    fprintf('  • Validation Array Dimensions:   [%d, %d, %d, %d] (Range: [%.1f, %.1f])\n', ...
        size(X_val, 1), size(X_val, 2), size(X_val, 3), size(X_val, 4), ...
        min(X_val(:)), max(X_val(:)));
    fprintf('------------------------------------------------------------\n');

    assert(isequal(net_input_dim, squeezenet_input_size), 'Network input dimension mismatch!');
    assert(size(X_train, 1) == 227 && size(X_train, 2) == 227, 'Training array spatial dimension mismatch!');
    assert(size(X_val, 1) == 227 && size(X_val, 2) == 227, 'Validation array spatial dimension mismatch!');
    fprintf('Dimension check PASSED: All tensors standardized to [227, 227, 3].\n\n');

    % 9. Configure Transfer Learning Training Options
    % Base LR = 5e-4 (fine-tunes fire8/fire9 at 5e-4, trains new_conv10 at 5e-3 with factor 10)
    opts = trainingOptions('adam', ...
        'InitialLearnRate', 5e-4, ...
        'MaxEpochs', 20, ...
        'MiniBatchSize', 16, ...
        'L2Regularization', 1e-4, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {X_val, Y_val}, ...
        'ValidationFrequency', max(1, floor(numel(Y_train) / 16)), ...
        'Plots', 'none', ...
        'Verbose', true, ...
        'ExecutionEnvironment', 'auto');

    fprintf('Starting SqueezeNet V3 transfer learning training (%d epochs, batch size 16)...\n', 20);

    % Train Network (No silent fallback)
    try
        net = trainNetwork(augds_train, lgraph, opts);
    catch ME
        fprintf('\n============================================================\n');
        fprintf('  SQUEEZENET V3 TRAINING ERROR DETECTED\n');
        fprintf('  Error Identifier: %s\n', ME.identifier);
        fprintf('  Error Message:    %s\n', ME.message);
        fprintf('============================================================\n\n');
        rethrow(ME);
    end

    % 10. Evaluate Genuine Validation Predictions
    fprintf('\nEvaluating SqueezeNet V3 model on held-out validation images...\n');
    [Y_pred_cat, probs] = classify(net, X_val);
    probs = double(probs);
    
    y_true_numeric = val_tab.DR_Grade;
    y_pred_numeric = double(Y_pred_cat) - 1; % Convert categorical to 0-4

    % 5-Class Accuracy
    acc_5class = mean(y_true_numeric == y_pred_numeric);

    % Per-Class Precision, Recall, F1
    per_class_recalls = zeros(5, 1);
    per_class_precisions = zeros(5, 1);
    per_class_f1 = zeros(5, 1);

    for c = 0:4
        tp_c = sum((y_pred_numeric == c) & (y_true_numeric == c));
        fp_c = sum((y_pred_numeric == c) & (y_true_numeric ~= c));
        fn_c = sum((y_pred_numeric ~= c) & (y_true_numeric == c));

        per_class_recalls(c+1)    = tp_c / max(1, tp_c + fn_c);
        per_class_precisions(c+1) = tp_c / max(1, tp_c + fp_c);
        per_class_f1(c+1)         = (2 * tp_c) / max(1, 2 * tp_c + fp_c + fn_c);
    end

    balanced_acc = mean(per_class_recalls);
    macro_f1     = mean(per_class_f1);

    % Binary Referable DR Interpretation (Non-referable: 0,1 | Referable: 2,3,4)
    ref_true = (y_true_numeric >= 2);
    ref_pred = (y_pred_numeric >= 2);

    tp_ref = sum(ref_pred & ref_true);
    fp_ref = sum(ref_pred & ~ref_true);
    fn_ref = sum(~ref_pred & ref_true);
    tn_ref = sum(~ref_pred & ~ref_true);

    ref_sens = tp_ref / max(1, tp_ref + fn_ref);
    ref_spec = tn_ref / max(1, tn_ref + fp_ref);
    ref_prec = tp_ref / max(1, tp_ref + fp_ref);
    ref_f1   = (2 * tp_ref) / max(1, 2 * tp_ref + fp_ref + fn_ref);

    % Assemble Comprehensive Validation Metrics
    metrics = struct();
    metrics.accuracy_5class = acc_5class;
    metrics.balanced_accuracy = balanced_acc;
    metrics.macro_f1 = macro_f1;
    metrics.per_class_recalls = per_class_recalls;
    metrics.per_class_precisions = per_class_precisions;
    metrics.per_class_f1 = per_class_f1;
    metrics.confusion_matrix_5class = confusionmat(y_true_numeric, y_pred_numeric, 'Order', 0:4);
    
    metrics.referable_sensitivity = ref_sens;
    metrics.referable_specificity = ref_spec;
    metrics.referable_precision = ref_prec;
    metrics.referable_f1 = ref_f1;
    metrics.confusion_matrix_referable = [tn_ref, fp_ref; fn_ref, tp_ref];
    
    metrics.val_probabilities = probs;
    metrics.val_true_labels = y_true_numeric;
    metrics.val_pred_labels = y_pred_numeric;
    metrics.timestamp = char(datetime('now'));

    % Display Validation Report
    fprintf('============================================================\n');
    fprintf('  SQUEEZENET V3 VALIDATION RESULTS (HELD-OUT SET, N=%d)\n', num_val);
    fprintf('------------------------------------------------------------\n');
    fprintf('  • 5-Class Overall Accuracy:   %5.2f%%\n', acc_5class * 100);
    fprintf('  • 5-Class Balanced Accuracy:  %5.2f%%\n', balanced_acc * 100);
    fprintf('  • 5-Class Macro F1 Score:     %6.4f\n', macro_f1);
    fprintf('  ----------------------------------------------------------\n');
    fprintf('  • Referable DR Sensitivity:   %5.2f%%\n', ref_sens * 100);
    fprintf('  • Referable DR Specificity:   %5.2f%%\n', ref_spec * 100);
    fprintf('  • Referable DR Precision:     %5.2f%%\n', ref_prec * 100);
    fprintf('  • Referable DR F1 Score:      %6.4f\n', ref_f1);
    fprintf('============================================================\n\n');

    % Metadata package
    training_info = struct();
    training_info.experiment = 'V3: SqueezeNet Partial Fine-Tuning + Direct Logit Head';
    training_info.backbone = 'squeezenet (ImageNet pretrained)';
    training_info.frozen_layers = 'conv1 through fire7';
    training_info.finetuned_layers = 'fire8, fire9, new_conv10';
    training_info.class_names = cfg.severity.class_names;
    training_info.class_mapping = 0:4;
    training_info.input_size = squeezenet_input_size;
    training_info.input_intensity_range = '[0, 255]';
    training_info.random_seed = cfg.random_seed;
    training_info.train_samples = height(train_tab);
    training_info.val_samples = height(val_tab);
    training_info.training_options = opts;
    training_info.training_date = char(datetime('now'));

    % 11. Save Dedicated V3 Checkpoint (Preserves V1 and V2)
    model_file_v3 = fullfile(cfg.paths.models_severity, 'SHINRA_SqueezeNet_Severity_Model_V3.mat');
    save(model_file_v3, 'net', 'metrics', 'training_info', 'cfg');
    save(fullfile(cfg.paths.metrics, 'squeezenet_severity_V3_validation_metrics.mat'), 'metrics');

    fprintf('SqueezeNet V3 model saved to:\n  • %s\n\n', model_file_v3);
end
