function report_text = generateSHINRAReport(results_struct)
% GENERATESHINRAREPORT Formats Comprehensive Clinical Screening Report as Text/Markdown
% Generates a production-grade diagnostic report summarizing patient/image ID,
% image quality, extracted anatomical & lesion biomarkers, 5-class DR probabilities,
% calibrated Referable DR status, reliability index, and actionable triage recommendations.

    r = results_struct;
    timestamp_str = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

    report_lines = {};
    report_lines{end+1} = '========================================================================';
    report_lines{end+1} = '           SHINRA RETINAL SCREENING SYSTEM — CLINICAL REPORT           ';
    report_lines{end+1} = '   Explainable AI for Diabetic Retinopathy Screening in Rural India    ';
    report_lines{end+1} = '========================================================================';
    report_lines{end+1} = sprintf('Report Generated:   %s', timestamp_str);
    report_lines{end+1} = sprintf('Image / Patient ID: %s', r.image_id);
    report_lines{end+1} = sprintf('System Version:     SHINRA Production Engine V2 (Development Frozen)');
    report_lines{end+1} = '';

    % 1. Safety & Image Quality Triage
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '1. IMAGE QUALITY & SAFETY ASSESSMENT';
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = sprintf('Quality Classification: %s', r.quality.quality_class);
    report_lines{end+1} = sprintf('Quality Composite Score: %.1f / 100', r.quality.score);
    report_lines{end+1} = sprintf('Gradability Status:     %s', iif(r.quality.is_gradable, 'GRADABLE (Proceed with Screening)', 'UNGRADABLE (Recapture Required)'));
    report_lines{end+1} = sprintf('Operational Guidance:   %s', r.quality.recapture_guidance);
    report_lines{end+1} = '';

    % 2. Clinical Evidence & Biomarkers
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '2. EXTRACTED CLINICAL EVIDENCE & BIOMARKERS';
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = sprintf('Optic Disc Localization: Center [%d, %d] px | Radius: %d px (Conf: %.2f)', ...
        round(r.anatomy.optic_disc.center(1)), round(r.anatomy.optic_disc.center(2)), ...
        round(r.anatomy.optic_disc.radius), r.anatomy.optic_disc.confidence);
    report_lines{end+1} = sprintf('Foveal Risk Zone:        Center [%d, %d] px | Macular Radius: %d px', ...
        round(r.anatomy.fovea.center(1)), round(r.anatomy.fovea.center(2)), ...
        round(r.anatomy.fovea.macula_radius));
    report_lines{end+1} = sprintf('Vessel Architecture:     Density: %5.2f%% | Tortuosity Index: %.2f | Branch Points: %d', ...
        r.vessels.morphometry.vessel_density * 100, r.vessels.morphometry.mean_tortuosity, r.vessels.morphometry.num_branch_points);
    report_lines{end+1} = sprintf('Microaneurysms (MAs):    Count: %d | Total Area: %d px', r.lesions.microaneurysms.count, r.lesions.microaneurysms.total_area);
    report_lines{end+1} = sprintf('Hemorrhages (HEs):       Count: %d | Area Fraction: %6.4f%%', r.lesions.hemorrhages.count, r.lesions.hemorrhages.area_fraction * 100);
    report_lines{end+1} = sprintf('Exudates (EXs):          Count: %d | Area Fraction: %6.4f%%', r.lesions.exudates.count, r.lesions.exudates.area_fraction * 100);
    report_lines{end+1} = sprintf('Macular DME Risk Grade:  Level %d / 3', r.lesions.exudates.macular_proximity_risk);
    report_lines{end+1} = sprintf('Neovascularization (NV): Score: %.2f (%s)', r.lesions.neovascularization.nv_score, r.lesions.neovascularization.status);
    report_lines{end+1} = '';

    % 3. DR Severity Classification
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '3. 5-CLASS ICDR DR SEVERITY GRADING';
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = sprintf('Predicted Severity:     Grade %d — %s', r.severity.predicted_grade, r.severity.predicted_class_name);
    report_lines{end+1} = sprintf('Model Confidence:       %5.2f%%', r.severity.confidence * 100);
    report_lines{end+1} = sprintf('Expected Ordinal Score: %.2f', r.severity.expected_grade);
    report_lines{end+1} = 'Class Posterior Distribution:';
    for c = 1:5
        c_name = r.severity.class_names{c};
        report_lines{end+1} = sprintf('  • Grade %d (%-16s): %5.2f%%', c-1, c_name, r.severity.probabilities(c) * 100);
    end
    report_lines{end+1} = '';

    % 4. Referable DR Classification & Calibration
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '4. REFERABLE DR CLASSIFICATION & PROBABILITY CALIBRATION';
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = sprintf('Raw Referable Probability:       %5.2f%%', r.referable.p_referable_raw * 100);
    report_lines{end+1} = sprintf('Platt-Calibrated Probability:    %5.2f%%', r.referable.p_referable_calibrated * 100);
    report_lines{end+1} = sprintf('Referral Threshold Applied:      tau_ref = %.2f (Frozen Development Plateau)', r.referable.referable_threshold_used);
    report_lines{end+1} = sprintf('Referral Screening Decision:     >>> %s <<<', r.referable.referral_status);
    report_lines{end+1} = sprintf('Evidence-CNN Concordance Index:  %5.2f%%', r.referable.agreement_score * 100);
    report_lines{end+1} = '';

    % 5. Reliability Gating & Safety Triage
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '5. RELIABILITY SAFETY LAYER & CLINICAL TRIAGE';
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = sprintf('Composite Reliability Score:     %5.1f / 100 (Tier: %s)', r.reliability.reliability_score, r.reliability.confidence_level);
    report_lines{end+1} = sprintf('Uncertainty Assessment:          %s', r.reliability.uncertainty_level);
    report_lines{end+1} = sprintf('Reliability Sub-Score Breakdown:');
    report_lines{end+1} = sprintf('  • Image Quality (S_q):        %5.1f / 100', r.reliability.subscores.quality);
    report_lines{end+1} = sprintf('  • Model Confidence (S_c):     %5.1f / 100', r.reliability.subscores.confidence);
    report_lines{end+1} = sprintf('  • Boundary Margin (S_m):      %5.1f / 100', r.reliability.subscores.margin);
    report_lines{end+1} = sprintf('  • Completeness (S_comp):      %5.1f / 100', r.reliability.subscores.completeness);
    report_lines{end+1} = sprintf('  • Concordance (S_agree):      %5.1f / 100', r.reliability.subscores.agreement);
    report_lines{end+1} = sprintf('FINAL ACTIONABLE TRIAGE:         >>> %s <<<', r.reliability.recommended_action);
    report_lines{end+1} = sprintf('Decision Readiness:              %s', iif(r.reliability.decision_readiness, 'AUTONOMOUS SCREENING ACCEPTED', 'HUMAN CLINICAL TRIAGE REQUIRED'));
    report_lines{end+1} = sprintf('Clinical Urgency:                %s', r.reliability.urgency);
    report_lines{end+1} = sprintf('Recommended Follow-Up:           %s', r.reliability.followup_window);
    report_lines{end+1} = sprintf('Operational Guidance:            %s', r.reliability.actionable_guidance);
    report_lines{end+1} = '';

    % 6. Clinical Limitations & Regulatory Safety Notices
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '6. CLINICAL LIMITATIONS & SAFETY NOTICES';
    report_lines{end+1} = '------------------------------------------------------------------------';
    report_lines{end+1} = '• RESEARCH / INVESTIGATIONAL USE ONLY — NOT AN INDEPENDENT MEDICAL DEVICE.';
    report_lines{end+1} = '• Designed as an explainable tele-screening aid for rural community clinics.';
    report_lines{end+1} = '• All algorithmic findings require confirmatory review by a licensed ophthalmologist.';
    report_lines{end+1} = '• Neovascularization and macular edema proximity are automated risk indicators';
    report_lines{end+1} = '  derived from 2D fundus images and do not replace OCT or fluorescein angiography.';
    report_lines{end+1} = '========================================================================';

    report_text = strjoin(report_lines, '\n');
end

function out = iif(cond, val_true, val_false)
    if cond
        out = val_true;
    else
        out = val_false;
    end
end
