function fusion_result = fuseEvidence(evidence_vec, feature_names)
% FUSEEVIDENCE Multi-Modal Evidence Aggregation and Fusion
% Combines deep convolutional predictions, lesion biomarkers, vascular metrics,
% and image quality into a consolidated final Referable DR probability and agreement index.

    cfg = shinra_config();
    fusion_result = struct();
    model_file = fullfile(cfg.paths.models_fusion, 'fusion_model.mat');

    p_cnn_referable = evidence_vec(6);
    ma_cnt          = evidence_vec(7);
    he_area         = evidence_vec(10);
    ex_area         = evidence_vec(12);
    dme_risk        = evidence_vec(13);
    quality_score   = evidence_vec(16);

    use_trained_fusion = false;

    % 1. Try Trained Neural/Logistic Fusion Model
    if exist(model_file, 'file') == 2
        try
            loaded = load(model_file);
            if isfield(loaded, 'weights') && isfield(loaded, 'bias')
                w = loaded.weights;
                b = loaded.bias;
                logit = sum(evidence_vec .* w) + b;
                p_fused = 1.0 / (1.0 + exp(-logit));
                use_trained_fusion = true;
            end
        catch
            use_trained_fusion = false;
        end
    end

    % 2. Calibrated Biomarker Aggregator (when checkpoint is not loaded)
    if ~use_trained_fusion
        % Lesion Evidence Score
        lesion_score = 0.35 * min(1.0, ma_cnt / 10.0) + ...
                       0.35 * min(1.0, he_area / 0.005) + ...
                       0.30 * min(1.0, ex_area / 0.005) + ...
                       0.25 * min(1.0, dme_risk / 2.0);
        lesion_score = min(1.0, lesion_score);

        % Quality-Adaptive Weighting:
        % If quality is high, trust CNN more; if quality is low, balance with lesion morphology
        alpha = 0.65 * (quality_score / 100) + 0.15;
        p_fused = alpha * p_cnn_referable + (1 - alpha) * lesion_score;
    end

    % Evidence-CNN Agreement Index [0, 1]
    % Measures concordance between deep CNN confidence and physical lesion findings
    lesion_ref_flag = (ma_cnt >= 5) || (he_area >= 0.001) || (ex_area >= 0.001) || (dme_risk >= 1.0);
    cnn_ref_flag = p_cnn_referable >= 0.50;
    
    if cnn_ref_flag == lesion_ref_flag
        agreement_score = 0.95;
    else
        % Mild disagreement penalty proportional to probability discrepancy
        agreement_score = max(0.20, 1.0 - abs(p_cnn_referable - (double(lesion_ref_flag))));
    end

    % 3. Apply Platt Probability Calibration (Fitted strictly on 330 Development Training partition)
    eps_val = 1e-6;
    p_clamped = max(eps_val, min(1 - eps_val, p_fused));
    logit_val = log(p_clamped / (1 - p_clamped));
    if isfield(cfg, 'calibration') && isfield(cfg.calibration, 'platt') && isstruct(cfg.calibration.platt)
        slope = cfg.calibration.platt.slope;
        intercept = cfg.calibration.platt.intercept;
        p_cal = 1.0 / (1.0 + exp(-(slope * logit_val + intercept)));
    else
        p_cal = p_fused;
    end

    fusion_result.p_referable_fused = max(0, min(1, p_fused));
    fusion_result.p_referable_fused_calibrated = max(0, min(1, p_cal));
    fusion_result.p_cnn_referable = p_cnn_referable;
    fusion_result.agreement_score = agreement_score;
    fusion_result.is_trained_fusion = use_trained_fusion;
    fusion_result.evidence_vector = evidence_vec;
end
