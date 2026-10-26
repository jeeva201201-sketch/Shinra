function rel_analysis = computeReliabilityAnalysis(q_rep, sev_res, fus_res, lesion_ev, vess_res, od_res, fovea_res)
% COMPUTERELIABILITYANALYSIS Reusable Reliability and Clinical Triage Module
% Evaluates 5-factor multi-modal reliability, determines confidence and uncertainty tiers,
% applies the quality safety gate and decision boundary margin, and produces actionable triage decisions.

    % 1. Compute 5-Factor Reliability Metrics
    rel_res = computeReliability(q_rep, sev_res, fus_res, lesion_ev, vess_res, od_res, fovea_res);

    % 2. Standardized Clinical Triage Decision
    dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);

    % 3. Followup Window & Operational Guidance
    switch dec_res.recommended_action
        case 'ACCEPT_SCREENING'
            if dec_res.predicted_referable
                followup_window = '2 to 4 weeks (Ophthalmology Clinic Referral)';
                action_guidance = 'High-confidence referable DR detected. Schedule clinical examination with ophthalmologist.';
            else
                followup_window = '12 months (Annual Diabetic Retinal Screening)';
                action_guidance = 'No sight-threatening retinopathy detected. Continue regular annual diabetic eye surveillance.';
            end
        case 'RECAPTURE'
            followup_window = 'Immediate / Same-day Visit';
            action_guidance = 'Retinal image ungradable due to optical defects. Recapture fundus photograph following guidance.';
        case 'CLINICAL_REVIEW'
            followup_window = '1 to 2 weeks (Secondary Tele-Ophthalmology Review)';
            action_guidance = 'Sub-threshold reliability with suspicious features. Transmit for specialist tele-ophthalmology triage.';
        case 'UNCERTAIN'
            followup_window = '2 to 4 weeks (Repeat Screening / Clinical Assessment)';
            action_guidance = 'Inconclusive screening due to borderline quality or threshold proximity. Repeat screening advised.';
        otherwise
            followup_window = 'Routine Follow-up';
            action_guidance = 'Clinical review as indicated by healthcare provider.';
    end

    % 4. Assemble Structured Output
    rel_analysis = struct();
    rel_analysis.reliability_score = rel_res.reliability_score;
    rel_analysis.confidence_level = rel_res.confidence_level;
    rel_analysis.confidence_tier = rel_res.confidence_level;
    rel_analysis.uncertainty_level = rel_res.uncertainty_level;
    rel_analysis.passes_reliability_gate = rel_res.is_reliable;
    
    % Sub-Scores [0, 100]
    rel_analysis.subscores = struct(...
        'quality', rel_res.quality_score_component, ...
        'confidence', rel_res.confidence_score_component, ...
        'margin', rel_res.margin_score_component, ...
        'completeness', rel_res.completeness_score_component, ...
        'agreement', rel_res.agreement_score_component, ...
        'multimodal_agreement', rel_res.margin_score_component, ...
        'biomarker_completeness', rel_res.completeness_score_component, ...
        'explainability_agreement', rel_res.agreement_score_component);
        
    rel_analysis.reliability_reasons = rel_res.reliability_reasons;
    
    % Clinical Decision Fields
    rel_analysis.recommended_action = dec_res.recommended_action;
    rel_analysis.action_reason = dec_res.decision_reason;
    rel_analysis.decision_readiness = dec_res.decision_readiness;
    rel_analysis.urgency = dec_res.urgency;
    rel_analysis.followup_window = followup_window;
    rel_analysis.actionable_guidance = action_guidance;
    rel_analysis.clinical_disposition = dec_res.clinical_disposition;
    rel_analysis.final_decision = dec_res.final_decision;
end
