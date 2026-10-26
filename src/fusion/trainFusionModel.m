function metrics = trainFusionModel(varargin)
% TRAINFUSIONMODEL Supervised Training of Multi-Modal Evidence Fusion Model
% Extracts multi-modal evidence vectors across the training split,
% trains a regularized evidence aggregator, and evaluates on validation split.

    cfg = shinra_config();
    fprintf('============================================================\n');
    fprintf('  TRAINING MULTI-MODAL EVIDENCE FUSION MODEL                \n');
    fprintf('============================================================\n');

    [idrid_train_full, ~] = load_idrid();
    [train_tab, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);

    num_train = height(train_tab);
    num_val   = height(val_tab);

    fprintf('Extracting evidence vectors for %d training cases...\n', num_train);
    X_train = zeros(num_train, 16);
    y_train = train_tab.Referable_DR;

    for i = 1:num_train
        img = imread(train_tab.ImagePath{i});
        prep = preprocessRetina(img);
        [q_class, q_rep] = qualityClassifier(prep.rgb_normalized, prep.fov_mask);
        od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
        fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
        vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
        
        ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
        he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
        ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
        nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);
        
        lesion_ev = struct(...
            'ma_count', ma_res.count, 'ma_area', ma_res.total_area, ...
            'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, ...
            'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, ...
            'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);
            
        sev_res = predictDRSeverity(prep, lesion_ev);
        X_train(i, :) = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
    end

    fprintf('Extracting evidence vectors for %d validation cases...\n', num_val);
    X_val = zeros(num_val, 16);
    y_val = val_tab.Referable_DR;

    for i = 1:num_val
        img = imread(val_tab.ImagePath{i});
        prep = preprocessRetina(img);
        [q_class, q_rep] = qualityClassifier(prep.rgb_normalized, prep.fov_mask);
        od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
        fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
        vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
        
        ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
        he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
        ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
        nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);
        
        lesion_ev = struct(...
            'ma_count', ma_res.count, 'ma_area', ma_res.total_area, ...
            'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, ...
            'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, ...
            'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);
            
        sev_res = predictDRSeverity(prep, lesion_ev);
        X_val(i, :) = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
    end

    % Train Ridge Logistic Regression / Generalized Linear Model
    try
        % Fit logistic regression model
        [B, fit_info] = lassoglm(X_train, y_train, 'binomial', 'Alpha', 0.1, 'CV', 3);
        best_idx = fit_info.IndexMinDeviance;
        weights = B(:, best_idx)';
        bias = fit_info.Intercept(best_idx);
    catch
        % Analytical ridge regression fallback
        lambda = 0.1;
        X_aug = [X_train, ones(num_train, 1)];
        w_all = (X_aug' * X_aug + lambda * eye(17)) \ (X_aug' * y_train);
        weights = w_all(1:16)';
        bias = w_all(17);
    end

    % Evaluate on Validation Set
    val_logits = sum(X_val .* weights, 2) + bias;
    val_probs  = 1.0 ./ (1.0 + exp(-val_logits));
    val_preds  = val_probs >= 0.50;

    tp = sum(val_preds & (y_val == 1));
    fp = sum(val_preds & (y_val == 0));
    fn = sum(~val_preds & (y_val == 1));
    tn = sum(~val_preds & (y_val == 0));

    val_sens = tp / max(1, tp + fn);
    val_spec = tn / max(1, tn + fp);
    val_acc  = (tp + tn) / num_val;

    metrics = struct();
    metrics.weights = weights;
    metrics.bias = bias;
    metrics.val_accuracy = val_acc;
    metrics.val_sensitivity = val_sens;
    metrics.val_specificity = val_spec;
    metrics.timestamp = char(datetime('now'));

    fprintf('Fusion Model Validation: Sens = %.2f%% | Spec = %.2f%% | Acc = %.2f%%\n', ...
        val_sens * 100, val_spec * 100, val_acc * 100);

    model_file = fullfile(cfg.paths.models_fusion, 'fusion_model.mat');
    save(model_file, 'weights', 'bias', 'metrics', 'cfg');
    save(fullfile(cfg.paths.metrics, 'fusion_metrics.mat'), 'metrics');
    fprintf('Fusion model saved to %s\n\n', model_file);
end
