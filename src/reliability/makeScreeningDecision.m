function decision_result = makeScreeningDecision(quality_rep, fusion_res, reliability_res, custom_thresholds)
% MAKESCREENINGDECISION Multi-Modal Safety-Gated Screening Decision Engine
% Determines standardized triage action and decision readiness:
%
% STANDARDIZED ACTION TAXONOMY:
% 1. RECAPTURE: Image quality is UNGRADABLE; re-acquisition required.
% 2. CLINICAL_REVIEW: Reliability is below acceptance threshold and referable evidence detected.
% 3. UNCERTAIN: Reliability is below acceptance threshold due to borderline quality or boundary ambiguity.
% 4. ACCEPT_SCREENING: Reliability satisfies safety threshold; autonomous screening accepted.
%
% Note: Engineering decision framework; does not assert clinical diagnosis.

    cfg = shinra_config();

    % Resolve Thresholds
    th_ref = cfg.fusion.referable_threshold;
    th_accept = 60.0;
    if isfield(cfg, 'reliability') && isfield(cfg.reliability, 'threshold_accept')
        th_accept = cfg.reliability.threshold_accept;
    elseif isfield(cfg.fusion, 'reliability_min_threshold')
        th_accept = cfg.fusion.reliability_min_threshold;
    end

    if nargin >= 4 && isstruct(custom_thresholds)
        if isfield(custom_thresholds, 'referable_threshold'), th_ref = custom_thresholds.referable_threshold; end
        if isfield(custom_thresholds, 'threshold_accept'), th_accept = custom_thresholds.threshold_accept; end
        if isfield(custom_thresholds, 'reliability_threshold'), th_accept = custom_thresholds.reliability_threshold; end
    end

    % Extract Quality Status
    q_class = 'GOOD';
    if isstruct(quality_rep) && isfield(quality_rep, 'quality_class')
        q_class = quality_rep.quality_class;
    elseif ischar(quality_rep)
        q_class = quality_rep;
    end

    % Extract Referable Probability & Pred
    p_ref = 0.50;
    if isstruct(fusion_res)
        if isfield(fusion_res, 'p_referable_fused')
            p_ref = fusion_res.p_referable_fused;
        elseif isfield(fusion_res, 'p_cnn_referable')
            p_ref = fusion_res.p_cnn_referable;
        end
    elseif isnumeric(fusion_res) && ~isempty(fusion_res)
        p_ref = fusion_res;
    end
    p_ref = max(0, min(1.0, p_ref));
    predicted_referable = (p_ref >= th_ref);

    % Extract Reliability Score & Factors
    rel_score = 75.0;
    is_rel = true;
    if isstruct(reliability_res)
        if isfield(reliability_res, 'reliability_score')
            rel_score = reliability_res.reliability_score;
        end
        if isfield(reliability_res, 'is_reliable')
            is_rel = reliability_res.is_reliable;
        else
            is_rel = (rel_score >= th_accept);
        end
    elseif isnumeric(reliability_res) && ~isempty(reliability_res)
        rel_score = reliability_res;
        if rel_score <= 1.05 % Legacy float [0, 1] scaling
            rel_score = rel_score * 100.0;
        end
        is_rel = (rel_score >= th_accept);
    end

    % -------------------------------------------------------------------------
    % Determine Standardized Action, Disposition & Readiness
    % -------------------------------------------------------------------------
    if strcmp(q_class, 'UNGRADABLE')
        action = 'RECAPTURE';
        decision_readiness = false;
        decision_reason = 'Optical quality is ungradable; image must be recaptured.';
        clinical_disposition = 'Image ungradable. Retinal camera re-acquisition required.';
        urgency = 'Immediate Re-acquisition';
        final_decision = 'RECAPTURE';

    elseif rel_score < th_accept && predicted_referable
        action = 'CLINICAL_REVIEW';
        decision_readiness = false;
        decision_reason = sprintf('Low screening reliability (%.1f/100) with suspicious referable findings (P_ref = %.1f%%). Secondary clinical review recommended.', ...
            rel_score, p_ref * 100);
        clinical_disposition = 'Sub-threshold reliability with referable findings detected. Expert tele-ophthalmology review recommended.';
        urgency = 'Secondary Tele-Ophthalmology Review';
        final_decision = 'UNCERTAIN / CLINICAL TRIAGE';

    elseif rel_score < th_accept
        action = 'UNCERTAIN';
        decision_readiness = false;
        decision_reason = sprintf('Low screening reliability (%.1f/100) due to quality degradation or boundary ambiguity. Further evaluation required.', rel_score);
        clinical_disposition = 'Screening inconclusive due to borderline image quality or decision ambiguity.';
        urgency = 'Routine Clinical Verification';
        final_decision = 'UNCERTAIN / CLINICAL TRIAGE';

    elseif predicted_referable
        action = 'ACCEPT_SCREENING';
        decision_readiness = true;
        decision_reason = sprintf('High reliability screening (%.1f/100). Referable diabetic retinopathy predicted (P_ref = %.1f%%).', ...
            rel_score, p_ref * 100);
        clinical_disposition = 'Referable diabetic retinopathy detected with high reliability. Specialist referral recommended.';
        urgency = 'Specialist Ophthalmology Referral';
        final_decision = 'REFERABLE DR';

    else
        action = 'ACCEPT_SCREENING';
        decision_readiness = true;
        decision_reason = sprintf('High reliability screening (%.1f/100). No referable retinopathy detected (P_ref = %.1f%%).', ...
            rel_score, p_ref * 100);
        clinical_disposition = 'No evidence of sight-threatening retinopathy. Routine follow-up recommended.';
        urgency = 'Routine Annual Follow-up';
        final_decision = 'NOT REFERABLE';
    end

    % -------------------------------------------------------------------------
    % Assemble Output Structure
    % -------------------------------------------------------------------------
    decision_result = struct();
    decision_result.decision              = action;                % Standardized Action: ACCEPT_SCREENING, RECAPTURE, CLINICAL_REVIEW, UNCERTAIN
    decision_result.recommended_action    = action;
    decision_result.decision_readiness    = decision_readiness;    % Logical boolean
    decision_result.decision_reason       = decision_reason;
    decision_result.predicted_referable   = predicted_referable;   % Logical boolean
    decision_result.p_referable           = p_ref;
    decision_result.reliability_score     = rel_score;
    decision_result.quality_class         = q_class;
    decision_result.urgency               = urgency;
    decision_result.clinical_disposition  = clinical_disposition;
    
    % Backward-Compatible Legacy Field
    decision_result.final_decision        = final_decision;
end
