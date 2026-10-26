function report = diagnoseSeverityV3Validation()
% DIAGNOSESEVERITYV3VALIDATION Comprehensive Validation Diagnostic for SqueezeNet V3
% Evaluates the trained SqueezeNet V3 checkpoint exclusively on the 83-image validation split.
%
% Generates:
% 1. Full 5x5 confusion matrix (Rows = True Grade 0-4, Cols = Pred Grade 0-4)
% 2. Per-class metrics (support, precision, recall/sensitivity, F1)
% 3. True validation class distribution
% 4. Predicted class distribution
% 5. Binary referable confusion matrix (TN, FP, FN, TP) and binary metrics
% 6. Binary referable prevalence in the validation set
% 7. Distribution of P(DR >= 2) (summary statistics and by true referable status)
% 8. Mean predicted probability matrix by true class (calibration analysis)
% 9. Identification of all off-diagonal grade-to-grade confusions
% 10. Semantic verification of the 5-class ICDR severity grade ordering
% 11. Strict validation partition isolation assertion (no test set leakage)
%
% Checkpoint: models/severity/SHINRA_SqueezeNet_Severity_Model_V3.mat

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('        SHINRA SQUEEZENET V3: VALIDATION SET DIAGNOSTIC AUDIT           \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % Load Checkpoint First to Inspect Actual Model Class Ordering
    % -------------------------------------------------------------------------
    model_file_v3 = fullfile(cfg.paths.models_severity, 'SHINRA_SqueezeNet_Severity_Model_V3.mat');
    if exist(model_file_v3, 'file') ~= 2
        error('SHINRA:ModelNotFound', 'SqueezeNet V3 model checkpoint not found at: %s', model_file_v3);
    end

    loaded = load(model_file_v3);
    if ~isfield(loaded, 'net')
        error('SHINRA:InvalidModel', 'Checkpoint does not contain "net" variable.');
    end
    net = loaded.net;

    % Extract actual model class names from checkpoint
    if isfield(loaded, 'training_info') && isfield(loaded.training_info, 'class_names')
        model_classes = loaded.training_info.class_names;
    elseif isprop(net.Layers(end), 'Classes') && ~isempty(net.Layers(end).Classes)
        model_classes = cellstr(net.Layers(end).Classes);
    else
        model_classes = cfg.severity.class_names;
    end

    % -------------------------------------------------------------------------
    % 10. Verify Semantic Class Label Ordering Specification
    % -------------------------------------------------------------------------
    canonical_icdr = {
        'No DR', ...             % 0
        'Mild NPDR', ...         % 1
        'Moderate NPDR', ...     % 2
        'Severe NPDR', ...       % 3
        'Proliferative DR'       % 4
    };

    fprintf('[VERIFICATION 1/2] 5-Class ICDR Severity Grade Ordering Audit:\n');
    fprintf('------------------------------------------------------------------------\n');
    fprintf('Actual configuration class names (cfg.severity.class_names):\n');
    for c = 0:4
        if c+1 <= numel(cfg.severity.class_names)
            fprintf('  %d -> %s\n', c, cfg.severity.class_names{c+1});
        else
            fprintf('  %d -> [MISSING]\n', c);
        end
    end
    fprintf('\nActual model class names/order:\n');
    for c = 0:4
        if c+1 <= numel(model_classes)
            fprintf('  %d -> %s\n', c, model_classes{c+1});
        else
            fprintf('  %d -> [MISSING]\n', c);
        end
    end
    fprintf('------------------------------------------------------------------------\n');

    % Perform semantic matching verification
    assert(numel(cfg.severity.class_names) == 5, 'Configuration must define exactly 5 classes.');
    assert(numel(model_classes) == 5, 'Model head must define exactly 5 classes.');

    for c = 0:4
        cfg_name = cfg.severity.class_names{c+1};
        mdl_name = model_classes{c+1};

        is_cfg_match = verifySemanticClass(cfg_name, c);
        is_mdl_match = verifySemanticClass(mdl_name, c);

        if ~is_cfg_match
            error('SHINRA:SemanticMismatch', ...
                'Config class at index %d ("%s") does not match ICDR Grade %d (%s)', ...
                c, cfg_name, c, canonical_icdr{c+1});
        end
        if ~is_mdl_match
            error('SHINRA:SemanticMismatch', ...
                'Model class at index %d ("%s") does not match ICDR Grade %d (%s)', ...
                c, mdl_name, c, canonical_icdr{c+1});
        end
    end
    fprintf('Semantic verification PASSED: Both config and model adhere to:\n');
    for c = 0:4
        fprintf('  %d = %s\n', c, canonical_icdr{c+1});
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 11. Verify Validation Partition Isolation & Absence of Test Leakage
    % -------------------------------------------------------------------------
    fprintf('[VERIFICATION 2/2] Checking Validation Partition Isolation:\n');
    [idrid_train_full, idrid_test] = load_idrid();
    [train_tab, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);

    assert(height(val_tab) == 83, 'Validation partition must contain exactly 83 images.');
    assert(height(train_tab) == 330, 'Training partition must contain exactly 330 images.');
    if height(idrid_test) > 0
        assert(~any(ismember(val_tab.ImagePath, idrid_test.ImagePath)), ...
            'CRITICAL ERROR: Data leakage detected! Test images found in validation split!');
        assert(~any(ismember(train_tab.ImagePath, idrid_test.ImagePath)), ...
            'CRITICAL ERROR: Data leakage detected! Test images found in training split!');
        fprintf('  -> 103-image IDRiD test partition strictly sequestered (0 images evaluated).\n');
    end
    fprintf('  -> Validation partition: Exactly 83 images (Seed %d, 80/20 stratified split).\n\n', cfg.random_seed);

    % -------------------------------------------------------------------------
    % Preprocess Validation Images (227x227x3, ImageNet scale [0, 255])
    % -------------------------------------------------------------------------
    num_val = height(val_tab);
    target_size_2d = [227, 227];
    X_val = zeros(target_size_2d(1), target_size_2d(2), 3, num_val, 'single');

    for i = 1:num_val
        img_raw = imread(val_tab.ImagePath{i});
        prep = preprocessRetina(img_raw, target_size_2d);
        X_val(:, :, :, i) = single(prep.rgb_normalized * 255.0);
    end

    % -------------------------------------------------------------------------
    % Run Model Inference
    % -------------------------------------------------------------------------
    [Y_pred_cat, probs] = classify(net, X_val);
    probs = double(probs);
    y_true = val_tab.DR_Grade;
    [~, max_idx] = max(probs, [], 2);
    y_pred = max_idx - 1; % Convert 1-based index (1..5) to 0-4 grade

    % -------------------------------------------------------------------------
    % 1. Full 5x5 Confusion Matrix
    % -------------------------------------------------------------------------
    cm_5class = confusionmat(y_true, y_pred, 'Order', 0:4);

    fprintf('========================================================================\n');
    fprintf('  1. FULL 5x5 CONFUSION MATRIX (Rows = True Grade, Cols = Pred Grade)\n');
    fprintf('========================================================================\n');
    fprintf('               Pred 0    Pred 1    Pred 2    Pred 3    Pred 4    Total\n');
    fprintf('------------------------------------------------------------------------\n');
    for r = 0:4
        fprintf('  True %d (%-4s)   %5d     %5d     %5d     %5d     %5d     %5d\n', ...
            r, canonical_icdr{r+1}(1:min(4, end)), ...
            cm_5class(r+1, 1), cm_5class(r+1, 2), cm_5class(r+1, 3), ...
            cm_5class(r+1, 4), cm_5class(r+1, 5), sum(cm_5class(r+1, :)));
    end
    fprintf('------------------------------------------------------------------------\n');
    fprintf('  Total Pred     %5d     %5d     %5d     %5d     %5d     %5d\n\n', ...
        sum(cm_5class(:, 1)), sum(cm_5class(:, 2)), sum(cm_5class(:, 3)), ...
        sum(cm_5class(:, 4)), sum(cm_5class(:, 5)), sum(cm_5class(:)));

    % -------------------------------------------------------------------------
    % 2. Per-Class Metrics
    % -------------------------------------------------------------------------
    per_class_prec = zeros(5, 1);
    per_class_rec  = zeros(5, 1);
    per_class_f1   = zeros(5, 1);
    per_class_supp = zeros(5, 1);

    for c = 0:4
        tp = cm_5class(c+1, c+1);
        fp = sum(cm_5class(:, c+1)) - tp;
        fn = sum(cm_5class(c+1, :)) - tp;
        supp = sum(cm_5class(c+1, :));

        per_class_supp(c+1) = supp;
        per_class_prec(c+1) = tp / max(1, tp + fp);
        per_class_rec(c+1)  = tp / max(1, tp + fn);
        per_class_f1(c+1)   = (2 * tp) / max(1, 2 * tp + fp + fn);
    end

    acc_5class   = sum(diag(cm_5class)) / sum(cm_5class(:));
    balanced_acc = mean(per_class_rec);
    macro_f1     = mean(per_class_f1);

    fprintf('========================================================================\n');
    fprintf('  2. PER-CLASS PERFORMANCE METRICS\n');
    fprintf('========================================================================\n');
    fprintf('  Grade  Class Name         Support   Precision   Recall/Sens    F1-Score\n');
    fprintf('  ----------------------------------------------------------------------\n');
    for c = 0:4
        fprintf('    %d    %-18s   %4d       %6.2f%%       %6.2f%%      %6.4f\n', ...
            c, canonical_icdr{c+1}, per_class_supp(c+1), ...
            per_class_prec(c+1) * 100, per_class_rec(c+1) * 100, per_class_f1(c+1));
    end
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  Overall Accuracy:    %5.2f%%\n', acc_5class * 100);
    fprintf('  Balanced Accuracy:   %5.2f%%\n', balanced_acc * 100);
    fprintf('  Macro F1-Score:      %6.4f\n\n', macro_f1);

    % -------------------------------------------------------------------------
    % 3 & 4. True vs Predicted Validation Class Distributions
    % -------------------------------------------------------------------------
    pred_counts = sum(cm_5class, 1)';
    fprintf('========================================================================\n');
    fprintf('  3 & 4. CLASS DISTRIBUTIONS (N=%d)\n', num_val);
    fprintf('========================================================================\n');
    fprintf('  Grade  Class Name           True Count (%%)        Pred Count (%%)\n');
    fprintf('  ----------------------------------------------------------------------\n');
    for c = 0:4
        fprintf('    %d    %-18s    %3d (%5.2f%%)           %3d (%5.2f%%)\n', ...
            c, canonical_icdr{c+1}, ...
            per_class_supp(c+1), (per_class_supp(c+1) / num_val) * 100, ...
            pred_counts(c+1), (pred_counts(c+1) / num_val) * 100);
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 5. Binary Referable DR Confusion Matrix & Metrics
    % -------------------------------------------------------------------------
    ref_true = (y_true >= 2);
    ref_pred = (y_pred >= 2);

    tp_ref = sum(ref_pred & ref_true);
    fp_ref = sum(ref_pred & ~ref_true);
    fn_ref = sum(~ref_pred & ref_true);
    tn_ref = sum(~ref_pred & ~ref_true);

    ref_sens = tp_ref / max(1, tp_ref + fn_ref);
    ref_spec = tn_ref / max(1, tn_ref + fp_ref);
    ref_prec = tp_ref / max(1, tp_ref + fp_ref);
    ref_f1   = (2 * tp_ref) / max(1, 2 * tp_ref + fp_ref + fn_ref);
    ref_acc  = (tp_ref + tn_ref) / num_val;

    fprintf('========================================================================\n');
    fprintf('  5. BINARY REFERABLE DR CONFUSION MATRIX (Referable: Grade >= 2)\n');
    fprintf('========================================================================\n');
    fprintf('                          Predicted Non-Ref (0,1)   Predicted Ref (2,3,4)\n');
    fprintf('  Actual Non-Ref (0,1)       TN = %-3d                  FP = %-3d\n', tn_ref, fp_ref);
    fprintf('  Actual Ref (2,3,4)         FN = %-3d                  TP = %-3d\n', fn_ref, tp_ref);
    fprintf('  ----------------------------------------------------------------------\n');
    fprintf('  • Referable Sensitivity (Recall): %5.2f%% (%d/%d)\n', ref_sens * 100, tp_ref, tp_ref + fn_ref);
    fprintf('  • Referable Specificity:          %5.2f%% (%d/%d)\n', ref_spec * 100, tn_ref, tn_ref + fp_ref);
    fprintf('  • Referable Precision (PPV):      %5.2f%% (%d/%d)\n', ref_prec * 100, tp_ref, tp_ref + fp_ref);
    fprintf('  • Referable F1-Score:             %6.4f\n', ref_f1);
    fprintf('  • Referable Binary Accuracy:      %5.2f%%\n\n', ref_acc * 100);

    % -------------------------------------------------------------------------
    % 6. Binary Referable Prevalence
    % -------------------------------------------------------------------------
    num_referable = sum(ref_true);
    num_non_referable = sum(~ref_true);
    prev_referable = num_referable / num_val;
    prev_non_referable = num_non_referable / num_val;

    fprintf('========================================================================\n');
    fprintf('  6. BINARY REFERABLE PREVALENCE IN VALIDATION SET (N=%d)\n', num_val);
    fprintf('========================================================================\n');
    fprintf('  • Referable Cases (Grades 2, 3, 4):     %2d / %2d (%5.2f%%)\n', ...
        num_referable, num_val, prev_referable * 100);
    fprintf('  • Non-Referable Cases (Grades 0, 1):    %2d / %2d (%5.2f%%)\n\n', ...
        num_non_referable, num_val, prev_non_referable * 100);

    % -------------------------------------------------------------------------
    % 7. Distribution of P(DR >= 2)
    % -------------------------------------------------------------------------
    p_ref_all = probs(:, 3) + probs(:, 4) + probs(:, 5);
    p_ref_nonref_cases = p_ref_all(~ref_true);
    p_ref_ref_cases = p_ref_all(ref_true);

    fprintf('========================================================================\n');
    fprintf('  7. DISTRIBUTION OF REFERABLE PROBABILITY P(DR >= 2)\n');
    fprintf('========================================================================\n');
    fprintf('  Cohort                   N    Min     Q25    Median   Mean     Q75     Max     Std\n');
    fprintf('  --------------------------------------------------------------------------------\n');
    fprintf('  All Validation Cases    %2d  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f\n', ...
        num_val, min(p_ref_all), prctile(p_ref_all, 25), median(p_ref_all), ...
        mean(p_ref_all), prctile(p_ref_all, 75), max(p_ref_all), std(p_ref_all));
    fprintf('  True Non-Referable (0,1)%2d  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f\n', ...
        num_non_referable, min(p_ref_nonref_cases), prctile(p_ref_nonref_cases, 25), ...
        median(p_ref_nonref_cases), mean(p_ref_nonref_cases), prctile(p_ref_nonref_cases, 75), ...
        max(p_ref_nonref_cases), std(p_ref_nonref_cases));
    fprintf('  True Referable (2,3,4)  %2d  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f  %6.4f\n\n', ...
        num_referable, min(p_ref_ref_cases), prctile(p_ref_ref_cases, 25), ...
        median(p_ref_ref_cases), mean(p_ref_ref_cases), prctile(p_ref_ref_cases, 75), ...
        max(p_ref_ref_cases), std(p_ref_ref_cases));

    % -------------------------------------------------------------------------
    % 8. Mean Predicted Probability Matrix by True Class
    % -------------------------------------------------------------------------
    mean_prob_matrix = zeros(5, 5);
    for c = 0:4
        idx_c = (y_true == c);
        if any(idx_c)
            mean_prob_matrix(c+1, :) = mean(probs(idx_c, :), 1);
        end
    end

    fprintf('========================================================================\n');
    fprintf('  8. MEAN PREDICTED PROBABILITY BY TRUE CLASS\n');
    fprintf('========================================================================\n');
    fprintf('  True Grade      P(Class 0)  P(Class 1)  P(Class 2)  P(Class 3)  P(Class 4)\n');
    fprintf('  ----------------------------------------------------------------------\n');
    for c = 0:4
        fprintf('  True %d (%-4s)     %6.4f      %6.4f      %6.4f      %6.4f      %6.4f\n', ...
            c, canonical_icdr{c+1}(1:min(4, end)), ...
            mean_prob_matrix(c+1, 1), mean_prob_matrix(c+1, 2), mean_prob_matrix(c+1, 3), ...
            mean_prob_matrix(c+1, 4), mean_prob_matrix(c+1, 5));
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 9. Grade-to-Grade Confusions (Off-Diagonal Frequency Listing)
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('  9. OBSERVED GRADE-TO-GRADE CONFUSIONS (Off-Diagonal from CM)\n');
    fprintf('========================================================================\n');
    
    confusions = [];
    for r = 0:4
        for c = 0:4
            if r ~= c && cm_5class(r+1, c+1) > 0
                entry = struct();
                entry.true_grade = r;
                entry.pred_grade = c;
                entry.count = cm_5class(r+1, c+1);
                entry.pct_of_true_class = (cm_5class(r+1, c+1) / per_class_supp(r+1)) * 100;
                confusions = [confusions; entry]; %#ok<AGROW>
            end
        end
    end

    if ~isempty(confusions)
        % Sort descending by count
        [~, sort_idx] = sort([confusions.count], 'descend');
        sorted_confusions = confusions(sort_idx);

        fprintf('  Rank   True Grade -> Pred Grade                Count   %% of True Class\n');
        fprintf('  ----------------------------------------------------------------------\n');
        for k = 1:numel(sorted_confusions)
            tg = sorted_confusions(k).true_grade;
            pg = sorted_confusions(k).pred_grade;
            fprintf('   %2d    Grade %d (%-4s) -> Grade %d (%-4s)         %2d        %5.2f%%\n', ...
                k, tg, canonical_icdr{tg+1}(1:min(4, end)), ...
                pg, canonical_icdr{pg+1}(1:min(4, end)), ...
                sorted_confusions(k).count, sorted_confusions(k).pct_of_true_class);
        end
    else
        fprintf('  No misclassifications observed on the validation set.\n');
        sorted_confusions = [];
    end
    fprintf('\n========================================================================\n\n');

    % -------------------------------------------------------------------------
    % Return Structured Report
    % -------------------------------------------------------------------------
    report = struct();
    report.confusion_matrix_5class = cm_5class;
    report.per_class_support = per_class_supp;
    report.per_class_precision = per_class_prec;
    report.per_class_recall = per_class_rec;
    report.per_class_f1 = per_class_f1;
    report.overall_accuracy = acc_5class;
    report.balanced_accuracy = balanced_acc;
    report.macro_f1 = macro_f1;
    report.true_distribution = per_class_supp;
    report.pred_distribution = pred_counts;
    report.confusion_matrix_referable = [tn_ref, fp_ref; fn_ref, tp_ref];
    report.binary_metrics = struct(...
        'TN', tn_ref, 'FP', fp_ref, 'FN', fn_ref, 'TP', tp_ref, ...
        'sensitivity', ref_sens, 'specificity', ref_spec, ...
        'precision', ref_prec, 'f1', ref_f1, 'accuracy', ref_acc);
    report.referable_prevalence = struct(...
        'num_referable', num_referable, 'num_non_referable', num_non_referable, ...
        'pct_referable', prev_referable * 100, 'pct_non_referable', prev_non_referable * 100);
    report.p_referable_stats = struct(...
        'all', p_ref_all, ...
        'non_referable_cases', p_ref_nonref_cases, ...
        'referable_cases', p_ref_ref_cases);
    report.mean_probability_matrix = mean_prob_matrix;
    report.confusions = sorted_confusions;
    report.class_names = canonical_icdr;
    report.model_classes = model_classes;
    report.config_classes = cfg.severity.class_names;
    report.val_true_labels = y_true;
    report.val_pred_labels = y_pred;
    report.val_probabilities = probs;
end

function is_match = verifySemanticClass(class_str, expected_grade)
% Helper to semantically match arbitrary string representations to 0..4 ICDR grades
    s = lower(regexprep(char(class_str), '[^a-zA-Z0-9]', ''));
    switch expected_grade
        case 0 % No DR / Normal
            is_match = contains(s, 'nodr') || startsWith(s, 'no') || strcmp(s, 'normal') || strcmp(s, '0');
        case 1 % Mild NPDR
            is_match = contains(s, 'mild') || strcmp(s, '1');
        case 2 % Moderate NPDR
            is_match = contains(s, 'mod') || strcmp(s, '2');
        case 3 % Severe NPDR
            is_match = contains(s, 'sev') || strcmp(s, '3');
        case 4 % Proliferative DR
            is_match = contains(s, 'prolif') || contains(s, 'pdr') || strcmp(s, '4');
        otherwise
            is_match = false;
    end
end
