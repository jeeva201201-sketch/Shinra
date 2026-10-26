function ref_result = runReferableInference(sev_res, lesion_evidence, vess_res, q_rep)
% RUNREFERABLEINFERENCE Reusable Referable DR Classification & Evidence Fusion
% Combines deep CNN severity probabilities, quantified lesion burdens, vascular density,
% and image quality into calibrated Referable DR probability and binary screening decision.

    cfg = shinra_config();

    % 1. Extract 16-D Evidence Vector
    [ev_vec, feat_names] = extractEvidenceVector(sev_res, lesion_evidence, vess_res, q_rep);

    % 2. Multi-Modal Evidence Fusion
    fus_res = fuseEvidence(ev_vec, feat_names);

    % 3. Apply Frozen Platt Probability Calibration
    % Parameters fitted strictly on 330 development training images
    a_platt = cfg.calibration.platt.slope;        % 1.3106
    b_platt = cfg.calibration.platt.intercept;    % -1.1412
    eps_val = 1e-6;

    p_raw = fus_res.p_referable_fused;
    p_clamped = max(eps_val, min(1 - eps_val, p_raw));
    logit_val = log(p_clamped / (1 - p_clamped));
    p_calibrated = 1.0 / (1.0 + exp(-(a_platt * logit_val + b_platt)));
    p_calibrated = max(0, min(1, p_calibrated));

    % 4. Decision Thresholds
    tau_ref_frozen = 0.30; % Frozen development release threshold (from plateau [0.25, 0.32])
    tau_ref_baseline = 0.50; % Baseline reference threshold

    is_referable_calibrated = (p_calibrated >= tau_ref_frozen);
    is_referable_raw = (p_raw >= tau_ref_baseline);

    % Assemble Output
    ref_result = struct();
    ref_result.p_referable_raw = p_raw;
    ref_result.p_referable_calibrated = p_calibrated;
    ref_result.p_cnn_referable = sev_res.p_referable;
    ref_result.agreement_score = fus_res.agreement_score;
    ref_result.referable_threshold_used = tau_ref_frozen;
    ref_result.is_referable = is_referable_calibrated;
    ref_result.referral_status = iif(is_referable_calibrated, 'REFERABLE DR', 'NON-REFERABLE');
    ref_result.evidence_vector = ev_vec;
    ref_result.feature_names = feat_names;
    ref_result.calibration_params = struct('method', 'Platt Logistic Scaling', 'slope', a_platt, 'intercept', b_platt);
end

function out = iif(cond, val_true, val_false)
    if cond
        out = val_true;
    else
        out = val_false;
    end
end
