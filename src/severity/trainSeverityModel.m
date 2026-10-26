function metrics = trainSeverityModel(varargin)
% TRAINSEVERITYMODEL Supervised Training of 5-Class Retinal DR Severity Model on IDRiD
% Performs stratified validation splitting, class-balanced resampling,
% training data augmentation, deep CNN training, and validation metric evaluation.
%
% Saves trained model checkpoint to models/severity/SHINRA_DR_Severity_Model.mat.

    cfg = shinra_config();
    fprintf('============================================================\n');
    fprintf('  TRAINING 5-CLASS DR SEVERITY MODEL (IDRiD DATASET)        \n');
    fprintf('============================================================\n');

    [idrid_train_full, ~] = load_idrid();
    if height(idrid_train_full) == 0
        error('SHINRA:DataNotFound', 'IDRiD training data not available in %s', cfg.paths.idrid);
    end

    % 1. Stratified 80/20 train/validation split (330 train, 83 validation)
    [train_tab, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);
    fprintf('Dataset Partition (Seed %d):\n', cfg.random_seed);
    fprintf('  • Training Split:   %d labeled images (Used for model training & balance)\n', height(train_tab));
    fprintf('  • Validation Split: %d labeled images (Used for evaluation & model selection)\n', height(val_tab));
    fprintf('  • Held-Out Test:    103 images (Completely untouched in models/)\n\n');

    % Standardized Input Dimension: [224, 224, 3]
    target_size = [224, 224];
    expected_input_size = [224, 224, 3];
    
    % 2. Deterministic Preprocessing of Training Data (224x224x3)
    num_train = height(train_tab);
    X_train_raw = zeros(target_size(1), target_size(2), 3, num_train, 'single');
    Y_train_raw = categorical(train_tab.DR_Grade, 0:4, cfg.severity.class_names);

    fprintf('Loading and preprocessing %d training images to %dx%dx3...\n', ...
        num_train, target_size(1), target_size(2));
    for i = 1:num_train
        img_raw = imread(train_tab.ImagePath{i});
        prep = preprocessRetina(img_raw, target_size);
        X_train_raw(:, :, :, i) = single(prep.rgb_normalized);
    end

    % 3. Class-Balanced Resampling on Training Split Only
    % Addresses scarcity of Class 1 (Mild NPDR)
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

    % 4. Deterministic Preprocessing of Validation Data (No Augmentation, 224x224x3)
    num_val = height(val_tab);
    X_val = zeros(target_size(1), target_size(2), 3, num_val, 'single');
    Y_val = categorical(val_tab.DR_Grade, 0:4, cfg.severity.class_names);

    fprintf('Loading and preprocessing %d validation images to %dx%dx3 (deterministic)...\n', ...
        num_val, target_size(1), target_size(2));
    for i = 1:num_val
        img_raw = imread(val_tab.ImagePath{i});
        prep = preprocessRetina(img_raw, target_size);
        X_val(:, :, :, i) = single(prep.rgb_normalized);
    end

    % 5. Data Augmentation Pipeline for Training Data Only
    augmenter = imageDataAugmenter(...
        'RandXReflection', true, ...
        'RandYReflection', true, ...
        'RandRotation', [-15, 15], ...
        'RandScale', [0.90, 1.10], ...
        'RandXTranslation', [-5, 5], ...
        'RandYTranslation', [-5, 5]);

    augds_train = augmentedImageDatastore(expected_input_size, X_train, Y_train, ...
        'DataAugmentation', augmenter);

    % 6. Construct Architecture
    lgraph = buildSeverityCNN(expected_input_size, 5);

    % 7. Dimension Consistency Verification Diagnostics
    input_layer = lgraph.Layers(1);
    net_input_dim = input_layer.InputSize;

    fprintf('\n------------------------------------------------------------\n');
    fprintf('  DIMENSION CONSISTENCY DIAGNOSTIC CHECK\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('  • Network Input Layer Size:   [%d, %d, %d]\n', ...
        net_input_dim(1), net_input_dim(2), net_input_dim(3));
    fprintf('  • Training Array Dimensions:  [%d, %d, %d, %d]\n', ...
        size(X_train, 1), size(X_train, 2), size(X_train, 3), size(X_train, 4));
    fprintf('  • Validation Array Dimensions:[%d, %d, %d, %d]\n', ...
        size(X_val, 1), size(X_val, 2), size(X_val, 3), size(X_val, 4));
    fprintf('  • Augmentation Target Size:   [%d, %d, %d]\n', ...
        expected_input_size(1), expected_input_size(2), expected_input_size(3));
    fprintf('------------------------------------------------------------\n');

    if ~isequal(net_input_dim, expected_input_size)
        error('SHINRA:DimensionMismatch', ...
            'Dimension Mismatch! Network input layer is [%d, %d, %d] but pipeline produces [%d, %d, %d].', ...
            net_input_dim(1), net_input_dim(2), net_input_dim(3), ...
            expected_input_size(1), expected_input_size(2), expected_input_size(3));
    end
    fprintf('Dimension check PASSED: All training, validation, and network layers standardized to [%d, %d, %d].\n\n', ...
        expected_input_size(1), expected_input_size(2), expected_input_size(3));

    % 8. Configure Training Options
    opts = trainingOptions('adam', ...
        'InitialLearnRate', cfg.severity.learning_rate, ...
        'MaxEpochs', cfg.severity.max_epochs, ...
        'MiniBatchSize', cfg.severity.mini_batch_size, ...
        'L2Regularization', cfg.severity.l2_regularization, ...
        'Shuffle', 'every-epoch', ...
        'ValidationData', {X_val, Y_val}, ...
        'ValidationFrequency', max(1, floor(numel(Y_train) / cfg.severity.mini_batch_size)), ...
        'Plots', 'none', ...
        'Verbose', cfg.flags.verbose, ...
        'ExecutionEnvironment', 'auto');

    fprintf('Starting genuine model training (%d epochs, batch size %d)...\n', ...
        cfg.severity.max_epochs, cfg.severity.mini_batch_size);

    % Train Network (No silent fallback: if it fails, throw error directly)
    try
        net = trainNetwork(augds_train, lgraph, opts);
    catch ME
        fprintf('\n============================================================\n');
        fprintf('  TRAINING ERROR DETECTED\n');
        fprintf('  Error Identifier: %s\n', ME.identifier);
        fprintf('  Error Message:    %s\n', ME.message);
        fprintf('============================================================\n\n');
        rethrow(ME);
    end

    % 9. Evaluate Genuine Validation Predictions
    fprintf('\nEvaluating model on held-out validation images...\n');
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
    fprintf('  VALIDATION RESULTS (HELD-OUT VALIDATION SET, N=%d)\n', num_val);
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
    training_info.class_names = cfg.severity.class_names;
    training_info.class_mapping = 0:4;
    training_info.input_size = expected_input_size;
    training_info.preprocessing = cfg.preprocessing;
    training_info.random_seed = cfg.random_seed;
    training_info.train_samples = height(train_tab);
    training_info.val_samples = height(val_tab);
    training_info.training_options = opts;
    training_info.training_date = char(datetime('now'));

    % 10. Save Checkpoint and Alias
    model_file_primary = fullfile(cfg.paths.models_severity, 'SHINRA_DR_Severity_Model.mat');
    model_file_alias   = fullfile(cfg.paths.models_severity, 'severity_cnn.mat');

    save(model_file_primary, 'net', 'metrics', 'training_info', 'cfg');
    save(model_file_alias, 'net', 'metrics', 'training_info', 'cfg');
    save(fullfile(cfg.paths.metrics, 'severity_validation_metrics.mat'), 'metrics');

    fprintf('Trained model saved to:\n  • %s\n  • %s\n\n', ...
        model_file_primary, model_file_alias);
end
