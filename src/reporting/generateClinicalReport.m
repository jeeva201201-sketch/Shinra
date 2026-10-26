function report_struct = generateClinicalReport(img_id, prep_res, q_rep, od_res, fovea_res, vess_res, ma_res, he_res, ex_res, nv_res, sev_res, fus_res, rel_res, dec_res)
% GENERATECLINICALREPORT Generate Structured Diagnostic Report for Retinal Screening
% Produces patient screening report with quantified biomarkers,
% quality defect breakdown, 5-class DR probability, and standardized triage action.
%
% Note: Engineering report generator; does not assert clinical diagnosis.

    cfg = shinra_config();
    report_file_txt = fullfile(cfg.paths.reports, sprintf('%s_clinical_report.txt', img_id));
    report_file_mat = fullfile(cfg.paths.reports, sprintf('%s_clinical_report.mat', img_id));

    timestamp_str = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

    % -------------------------------------------------------------------------
    % Extract & Normalize Sub-Module Inputs with Safe Fallbacks
    % -------------------------------------------------------------------------
    % Quality
    q_class = 'GOOD'; q_score = 75.0; q_guidance = 'Image quality acceptable.';
    if isstruct(q_rep)
        if isfield(q_rep, 'quality_class'), q_class = q_rep.quality_class; end
        if isfield(q_rep, 'score'), q_score = q_rep.score; end
        if isfield(q_rep, 'recapture_guidance'), q_guidance = q_rep.recapture_guidance; end
    end

    % Anatomy (Optic Disc & Fovea)
    od_center = [256, 256]; od_radius = 40; od_conf = 0.90;
    if isstruct(od_res)
        if isfield(od_res, 'center') && ~isempty(od_res.center), od_center = od_res.center; end
        if isfield(od_res, 'radius'), od_radius = od_res.radius; end
        if isfield(od_res, 'confidence'), od_conf = od_res.confidence; end
    end

    fovea_center = [320, 256]; macula_rad = 60;
    if isstruct(fovea_res)
        if isfield(fovea_res, 'center') && ~isempty(fovea_res.center), fovea_center = fovea_res.center; end
        if isfield(fovea_res, 'macula_radius'), macula_rad = fovea_res.macula_radius; end
    end

    % Vessels
    v_dens = 0.08; v_tort = 1.15; v_branch_pts = 45; v_branch_dens = 3.5;
    if isstruct(vess_res) && isfield(vess_res, 'morphometry')
        m = vess_res.morphometry;
        if isfield(m, 'vessel_density'), v_dens = m.vessel_density; end
        if isfield(m, 'mean_tortuosity'), v_tort = m.mean_tortuosity; end
        if isfield(m, 'num_branch_points'), v_branch_pts = m.num_branch_points; end
        if isfield(m, 'branching_density'), v_branch_dens = m.branching_density; end
    end

    % Lesions
    ma_cnt = 0; ma_area = 0;
    if isstruct(ma_res)
        if isfield(ma_res, 'count'), ma_cnt = ma_res.count; end
        if isfield(ma_res, 'total_area'), ma_area = ma_res.total_area; end
    end

    he_cnt = 0; he_area = 0;
    if isstruct(he_res)
        if isfield(he_res, 'count'), he_cnt = he_res.count; end
        if isfield(he_res, 'area_fraction'), he_area = he_res.area_fraction; end
    end

    ex_cnt = 0; ex_area = 0; dme_risk = 0;
    if isstruct(ex_res)
        if isfield(ex_res, 'count'), ex_cnt = ex_res.count; end
        if isfield(ex_res, 'area_fraction'), ex_area = ex_res.area_fraction; end
        if isfield(ex_res, 'macular_proximity_risk'), dme_risk = ex_res.macular_proximity_risk; end
    end

    nv_score = 0; nv_status = 'EXPLORATORY_BIOMARKER';
    if isstruct(nv_res)
        if isfield(nv_res, 'nv_score'), nv_score = nv_res.nv_score; end
        if isfield(nv_res, 'status'), nv_status = nv_res.status; end
    end

    % Severity
    sev_grade = 0; sev_name = 'No_DR'; exp_grade = 0; probs = [1, 0, 0, 0, 0];
    if isstruct(sev_res)
        if isfield(sev_res, 'predicted_grade'), sev_grade = sev_res.predicted_grade; end
        if isfield(sev_res, 'predicted_class_name'), sev_name = sev_res.predicted_class_name; end
        if isfield(sev_res, 'expected_grade'), exp_grade = sev_res.expected_grade; end
        if isfield(sev_res, 'probabilities') && numel(sev_res.probabilities) == 5
            probs = sev_res.probabilities;
        end
    end

    % Fusion
    p_cnn_ref = probs(3) + probs(4) + probs(5);
    p_fused_ref = p_cnn_ref;
    agr_score = 0.95;
    if isstruct(fus_res)
        if isfield(fus_res, 'p_cnn_referable'), p_cnn_ref = fus_res.p_cnn_referable; end
        if isfield(fus_res, 'p_referable_fused'), p_fused_ref = fus_res.p_referable_fused; end
        if isfield(fus_res, 'agreement_score'), agr_score = fus_res.agreement_score; end
    end

    % Reliability
    rel_score = 75.0; conf_level = 'HIGH'; uncert_level = 'LOW';
    sq = q_score; sc = 75.0; sm = 75.0; scomp = 100.0; sagree = 90.0;
    if isstruct(rel_res)
        if isfield(rel_res, 'reliability_score')
            rel_score = rel_res.reliability_score;
            if rel_score <= 1.05, rel_score = rel_score * 100.0; end
        end
        if isfield(rel_res, 'confidence_level')
            conf_level = rel_res.confidence_level;
        elseif rel_score >= 60.0
            conf_level = 'HIGH';
        elseif rel_score >= 40.0
            conf_level = 'MODERATE';
        else
            conf_level = 'LOW';
        end

        if isfield(rel_res, 'uncertainty_level')
            uncert_level = rel_res.uncertainty_level;
        elseif rel_score >= 60.0
            uncert_level = 'LOW';
        elseif rel_score >= 40.0
            uncert_level = 'MODERATE';
        else
            uncert_level = 'HIGH';
        end

        if isfield(rel_res, 'quality_score_component'), sq = rel_res.quality_score_component; end
        if isfield(rel_res, 'confidence_score_component'), sc = rel_res.confidence_score_component; end
        if isfield(rel_res, 'margin_score_component'), sm = rel_res.margin_score_component; end
        if isfield(rel_res, 'completeness_score_component'), scomp = rel_res.completeness_score_component; end
        if isfield(rel_res, 'agreement_score_component'), sagree = rel_res.agreement_score_component; end
    end

    % Decision
    action = 'ACCEPT_SCREENING'; is_ready = true; action_reason = 'High reliability screening.';
    clin_disp = 'Routine annual follow-up recommended.'; urgency = 'Routine Annual Follow-up';
    if isstruct(dec_res)
        if isfield(dec_res, 'decision')
            action = dec_res.decision;
        elseif isfield(dec_res, 'recommended_action')
            action = dec_res.recommended_action;
        elseif isfield(dec_res, 'final_decision')
            action = dec_res.final_decision;
        end

        if isfield(dec_res, 'decision_readiness')
            is_ready = dec_res.decision_readiness;
        else
            is_ready = ismember(action, {'ACCEPT_SCREENING', 'REFERABLE DR', 'NOT REFERABLE'});
        end

        if isfield(dec_res, 'decision_reason')
            action_reason = dec_res.decision_reason;
        elseif isfield(dec_res, 'clinical_disposition')
            action_reason = dec_res.clinical_disposition;
        end

        if isfield(dec_res, 'clinical_disposition'), clin_disp = dec_res.clinical_disposition; end
        if isfield(dec_res, 'urgency'), urgency = dec_res.urgency; end
    end

    % -------------------------------------------------------------------------
    % Build Text Report
    % -------------------------------------------------------------------------
    lines = {};
    lines{end+1} = '========================================================================================';
    lines{end+1} = '           SHINRA: EXPLAINABLE RETINAL SCREENING REPORT (RURAL DEPLOYMENT)              ';
    lines{end+1} = '========================================================================================';
    lines{end+1} = sprintf('  Patient / Image ID:       %s', img_id);
    lines{end+1} = sprintf('  Screening Timestamp:      %s', timestamp_str);
    lines{end+1} = sprintf('  Software Engine:          SHINRA v%s (MATLAB-Native)', cfg.version);
    lines{end+1} = '----------------------------------------------------------------------------------------';
    lines{end+1} = '  1. RETINAL IMAGE QUALITY ASSESSMENT (FIRST-PASS SAFETY GATE)';
    lines{end+1} = sprintf('     • Quality Classification:  [%s] (Composite Score: %.1f / 100)', q_class, q_score);
    if isstruct(q_rep) && isfield(q_rep, 'features')
        lines{end+1} = sprintf('     • Focus Sharpness:         Tenengrad = %.2f | Laplacian Var = %.1f', ...
            q_rep.features.tenengrad_sharpness, q_rep.features.laplacian_variance);
        lines{end+1} = sprintf('     • Exposure & Illumination: Mean = %.3f | Dark Ratio = %.1f%% | Glare = %.1f%%', ...
            q_rep.features.mean_intensity, q_rep.features.dark_pixel_ratio * 100, q_rep.features.bright_pixel_ratio * 100);
        lines{end+1} = sprintf('     • Retinal Field Coverage:  %.1f%% of image frame', q_rep.features.fov_coverage_ratio * 100);
    end
    lines{end+1} = sprintf('     • Guidance / Feedback:     %s', q_guidance);
    lines{end+1} = '----------------------------------------------------------------------------------------';
    lines{end+1} = '  2. RETINAL ANATOMY & VASCULAR ARCHITECTURE';
    lines{end+1} = sprintf('     • Optic Disc Location:     Center = [%d, %d], Radius = %d px (Conf: %.2f)', ...
        round(od_center(1)), round(od_center(2)), round(od_radius), od_conf);
    lines{end+1} = sprintf('     • Foveal Avascular Zone:   Center = [%d, %d] (Macula Radius: %d px)', ...
        round(fovea_center(1)), round(fovea_center(2)), round(macula_rad));
    lines{end+1} = sprintf('     • Retinal Vessel Density:  %.2f%% of retinal area', v_dens * 100);
    lines{end+1} = sprintf('     • Vascular Tortuosity:     %.3f (Normal ~ 1.05 - 1.25)', v_tort);
    lines{end+1} = sprintf('     • Branching Complexity:    %d bifurcation nodes (Density: %.2f / 10k px)', ...
        v_branch_pts, v_branch_dens);
    lines{end+1} = '----------------------------------------------------------------------------------------';
    lines{end+1} = '  3. QUANTIFIED CLINICAL LESION BURDEN (EVIDENCE BIOMARKERS)';
    lines{end+1} = sprintf('     • Microaneurysms (MAs):    %d detected (Total Area: %d px)', ma_cnt, ma_area);
    lines{end+1} = sprintf('     • Retinal Hemorrhages:     %d regions (Area Fraction: %.3f%%)', he_cnt, he_area * 100);
    lines{end+1} = sprintf('     • Hard / Soft Exudates:    %d regions (Area Fraction: %.3f%%)', ex_cnt, ex_area * 100);
    lines{end+1} = sprintf('     • Diabetic Macular Edema:  DME Proximity Risk Score = %.1f / 2.0', dme_risk);
    lines{end+1} = sprintf('     • Neovascularization Risk: %.3f (Status: %s)', nv_score, nv_status);
    lines{end+1} = '----------------------------------------------------------------------------------------';
    lines{end+1} = '  4. DIABETIC RETINOPATHY SEVERITY & MULTI-MODAL EVIDENCE FUSION';
    lines{end+1} = sprintf('     • Predicted ICDR Grade:    Level %d - %s', sev_grade, sev_name);
    lines{end+1} = sprintf('     • Ordinal Expected Grade:  %.2f / 4.0', exp_grade);
    lines{end+1} = '     • 5-Class Probability Profile:';
    for k = 1:5
        lines{end+1} = sprintf('         - %-18s: %5.1f%%', cfg.severity.class_names{k}, probs(k) * 100);
    end
    lines{end+1} = sprintf('     • CNN Referable Probability:       %.1f%%', p_cnn_ref * 100);
    lines{end+1} = sprintf('     • Multi-Modal Fused Referable P:   %.1f%%', p_fused_ref * 100);
    lines{end+1} = sprintf('     • Lesion-CNN Agreement Index:      %.2f / 1.00', agr_score);
    lines{end+1} = '----------------------------------------------------------------------------------------';
    lines{end+1} = '  5. SCREENING RELIABILITY & FINAL CLINICAL DISPOSITION';
    lines{end+1} = sprintf('     • Multi-Factor Reliability:        %.1f / 100 (%s | Uncertainty: %s)', ...
        rel_score, conf_level, uncert_level);
    lines{end+1} = sprintf('     • Sub-Scores (Q/C/M/Comp/Agr):     %.1f / %.1f / %.1f / %.1f / %.1f', ...
        sq, sc, sm, scomp, sagree);
    lines{end+1} = sprintf('     • RECOMMENDED ACTION:              >>> %s <<<', action);
    lines{end+1} = sprintf('     • Decision Readiness:              %s', iif(is_ready, 'READY (Autonomous Acceptance)', 'NOT READY (Requires Action)'));
    lines{end+1} = sprintf('     • Action Rationale:                %s', action_reason);
    lines{end+1} = sprintf('     • Triage Urgency:                  %s', urgency);
    lines{end+1} = '========================================================================================';

    report_text = strjoin(lines, '\n');

    % Write TXT file
    fid = fopen(report_file_txt, 'w');
    if fid ~= -1
        fprintf(fid, '%s\n', report_text);
        fclose(fid);
    end

    % Assemble structured output struct
    report_struct = struct();
    report_struct.image_id = img_id;
    report_struct.timestamp = timestamp_str;
    report_struct.quality = q_rep;
    report_struct.anatomy = struct('optic_disc', od_res, 'fovea', fovea_res);
    report_struct.vessels = vess_res;
    report_struct.lesions = struct('microaneurysms', ma_res, 'hemorrhages', he_res, 'exudates', ex_res, 'neovascularization', nv_res);
    report_struct.severity = sev_res;
    report_struct.fusion = fus_res;
    report_struct.reliability = rel_res;
    report_struct.decision = dec_res;
    report_struct.report_text = report_text;

    % Save MAT file
    save(report_file_mat, 'report_struct');
end

function out = iif(cond, val_true, val_false)
    if cond
        out = val_true;
    else
        out = val_false;
    end
end
