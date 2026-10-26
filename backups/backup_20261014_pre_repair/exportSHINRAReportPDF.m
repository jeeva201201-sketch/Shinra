function [success, saved_path] = exportSHINRAReportPDF(results_struct, target_pdf_path)
% EXPORTSHINRAREPORTPDF Export Comprehensive Multi-Page Clinical Screening PDF
% Generates a publication-grade, multi-page A4 PDF clinical screening report:
% - Page 1: Executive Clinical Screening Summary & Triage Banner
% - Page 2: Retinal Anatomy & Biomarker Quantitative Inventory
% - Page 3: Model Posterior Distribution & 5-Factor Reliability Breakdown
% - Pages 4-5: Multi-Layer Visual Interpretability Atlas (All 10 Overlays)
%
% Compatible with native MATLAB Desktop and MATLAB Online (no external toolboxes required).

    success = false;
    saved_path = '';

    if nargin < 1 || isempty(results_struct)
        error('exportSHINRAReportPDF: Missing results_struct diagnostic data.');
    end

    res = results_struct;
    img_id = res.image_id;

    % 1. Determine destination path
    if nargin < 2 || isempty(target_pdf_path)
        default_name = sprintf('SHINRA_Report_%s.pdf', img_id);
        reports_dir = fullfile(pwd, 'results', 'reports');
        if ~exist(reports_dir, 'dir')
            mkdir(reports_dir);
        end
        default_target = fullfile(reports_dir, default_name);

        [fname, fpath] = uiputfile('*.pdf', 'Export Clinical Screening Report (PDF)', default_target);
        if isequal(fname, 0) || isequal(fpath, 0)
            % User cancelled
            return;
        end
        target_pdf_path = fullfile(fpath, fname);
    end

    % Ensure target directory exists
    [out_dir, ~, ~] = fileparts(target_pdf_path);
    if ~isempty(out_dir) && ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % Delete existing target file if present
    if exist(target_pdf_path, 'file') == 2
        delete(target_pdf_path);
    end

    % Page specifications (Standard A4 at 150 DPI)
    page_w = 1240;
    page_h = 1754;
    total_pages = 5;

    timestamp_str = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));

    try
        % =====================================================================
        % PAGE 1: EXECUTIVE CLINICAL SUMMARY & TRIAGE BANNER
        % =====================================================================
        fig1 = createPageFigure(page_w, page_h);
        cleanup1 = onCleanup(@() closeFigure(fig1));

        drawPageHeader(fig1, 'EXECUTIVE CLINICAL SCREENING SUMMARY', img_id, timestamp_str, 1, total_pages);

        % Embedded Fundus Image Preview (Top Left)
        ax_thumb = axes('Parent', fig1, 'Units', 'pixels', 'Position', [70, 1140, 360, 360]);
        if isfield(res.overlays, 'original_rgb') && ~isempty(res.overlays.original_rgb)
            imshow(res.overlays.original_rgb, 'Parent', ax_thumb);
        else
            text(ax_thumb, 0.5, 0.5, 'Image Preview Unavailable', 'HorizontalAlignment', 'center');
        end
        title(ax_thumb, 'Original Retinal Fundus Photograph', 'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold');
        box(ax_thumb, 'on');

        % Patient / Image Metadata & System Spec Card (Top Right)
        ax_meta = axes('Parent', fig1, 'Units', 'pixels', 'Position', [460, 1140, 710, 360]);
        axis(ax_meta, 'off');
        text(ax_meta, 0.0, 0.95, 'PATIENT & SCREENING METADATA', 'FontName', 'Helvetica', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        
        meta_lines = {
            sprintf('• Patient / Examination ID:   %s', img_id), ...
            sprintf('• Examination Timestamp:       %s', timestamp_str), ...
            sprintf('• Screening System Engine:     SHINRA Multi-Modal AI Engine V2'), ...
            sprintf('• Deployment Configuration:    Rural Primary Health Center (Offline Mode)'), ...
            sprintf('• Image Resolution:            %d x %d pixels', size(res.preprocessing.rgb_normalized, 2), size(res.preprocessing.rgb_normalized, 1)), ...
            sprintf('• Decision Governance:         Calibrated Safety Gating Active (tau_ref = 0.30)')
        };
        for i = 1:numel(meta_lines)
            text(ax_meta, 0.0, 0.95 - i*0.14, meta_lines{i}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        end

        % Section 1: Image Quality Safety Gate Card
        drawSectionBox(fig1, [70, 930, 1100, 175], '1. IMAGE QUALITY SAFETY GATE', [0.95 0.97 1.0], [0.20 0.40 0.65]);
        ax_q = axes('Parent', fig1, 'Units', 'pixels', 'Position', [90, 940, 1060, 135]);
        axis(ax_q, 'off');
        q_class = res.quality.quality_class;
        q_color = getStatusColor(q_class);
        text(ax_q, 0.0, 0.85, sprintf('Quality Classification:  %s', q_class), 'FontName', 'Helvetica', 'FontSize', 12, 'FontWeight', 'bold', 'Color', q_color, 'Interpreter', 'none');
        text(ax_q, 0.45, 0.85, sprintf('Composite Quality Score:  %.1f / 100', res.quality.quality_score), 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28], 'Interpreter', 'none');
        text(ax_q, 0.0, 0.50, sprintf('Safety Gradability Status: %s', iif(res.quality.is_gradable, 'GRADABLE (Proceed with Autonomous Screening)', 'UNGRADABLE (Recapture Required)')), ...
            'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        text(ax_q, 0.0, 0.18, sprintf('Operational Guidance:      %s', res.quality.recapture_guidance), 'FontName', 'Helvetica', 'FontSize', 10, 'Color', [0.3 0.3 0.3], 'Interpreter', 'none');

        % Section 2: 5-Class Severity & Referable DR Card
        drawSectionBox(fig1, [70, 680, 1100, 225], '2. DIABETIC RETINOPATHY DIAGNOSTIC FINDINGS', [1.0 1.0 1.0], [0.20 0.40 0.65]);
        ax_dr = axes('Parent', fig1, 'Units', 'pixels', 'Position', [90, 690, 1060, 185]);
        axis(ax_dr, 'off');
        
        grade_str = sprintf('Grade %d — %s', res.severity.predicted_grade, strrep(res.severity.predicted_class_name, '_', ' '));
        text(ax_dr, 0.0, 0.88, sprintf('Predicted Severity:  %s', grade_str), 'FontName', 'Helvetica', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.1 0.1 0.1], 'Interpreter', 'none');
        text(ax_dr, 0.60, 0.88, sprintf('Model Confidence:   %.1f%%', res.severity.confidence * 100), 'FontName', 'Helvetica', 'FontSize', 11, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        text(ax_dr, 0.0, 0.62, sprintf('Expected Ordinal Score E[G]: %.2f (Scale: 0.00 - 4.00)', res.severity.expected_grade), 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.3 0.3 0.3], 'Interpreter', 'none');
        
        % Referable DR Line
        ref_status = res.referable.referral_status;
        ref_color = iif(res.referable.is_referable, [0.80 0.15 0.15], [0.10 0.55 0.20]);
        text(ax_dr, 0.0, 0.34, sprintf('Referable DR Decision:  %s', ref_status), 'FontName', 'Helvetica', 'FontSize', 13, 'FontWeight', 'bold', 'Color', ref_color, 'Interpreter', 'none');
        text(ax_dr, 0.0, 0.08, sprintf('Calibrated Referral Probability: %.1f%% (Raw: %.1f%% | Threshold tau_ref = %.2f | Concordance: %.1f%%)', ...
            res.referable.p_referable_calibrated * 100, res.referable.p_referable_raw * 100, res.referable.referable_threshold_used, res.referable.agreement_score * 100), ...
            'FontName', 'Helvetica', 'FontSize', 10, 'Color', [0.35 0.35 0.35], 'Interpreter', 'none');

        % Section 3: Prominent Clinical Triage Banner
        drawSectionBox(fig1, [70, 390, 1100, 265], '3. RELIABILITY SAFETY LAYER & CLINICAL TRIAGE DISPOSITION', [0.97 0.98 0.99], [0.15 0.45 0.25]);
        ax_tr = axes('Parent', fig1, 'Units', 'pixels', 'Position', [90, 400, 1060, 225]);
        axis(ax_tr, 'off');
        
        triage_act = res.reliability.recommended_action;
        t_color = getTriageColor(triage_act);
        
        text(ax_tr, 0.0, 0.88, sprintf('RECOMMENDED ACTION:   >>>  %s  <<<', upper(strrep(triage_act, '_', ' '))), ...
            'FontName', 'Helvetica', 'FontSize', 14, 'FontWeight', 'bold', 'Color', t_color, 'Interpreter', 'none');
        text(ax_tr, 0.0, 0.65, sprintf('Operational Rationale: %s', res.reliability.action_reason), ...
            'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        text(ax_tr, 0.0, 0.45, sprintf('Composite Reliability Score: %.1f / 100 (Tier: %s | Gate tau >= 60.0: %s)', ...
            res.reliability.reliability_score, res.reliability.confidence_level, bool2str(res.reliability.passes_reliability_gate, 'PASSED', 'REVIEW TRIGGERED')), ...
            'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        text(ax_tr, 0.0, 0.25, sprintf('Clinical Urgency:      %s', res.reliability.urgency), ...
            'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        text(ax_tr, 0.0, 0.06, sprintf('Recommended Window:    %s', res.reliability.followup_window), ...
            'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.15 0.35 0.65], 'Interpreter', 'none');

        % Section 4: Summary & Clinical Limitations
        drawSectionBox(fig1, [70, 170, 1100, 195], '4. DIAGNOSTIC SUMMARY & SAFETY LIMITATIONS', [1.0 0.98 0.95], [0.70 0.45 0.10]);
        ax_lim = axes('Parent', fig1, 'Units', 'pixels', 'Position', [90, 180, 1060, 155]);
        axis(ax_lim, 'off');
        summary_text = sprintf('Diagnostic Summary: %s quality fundus photograph. System prediction indicates %s with %.1f%% confidence and %s status (P_cal = %.1f%%). Reliability is %s (%.1f/100).', ...
            res.quality.quality_class, grade_str, res.severity.confidence*100, ref_status, res.referable.p_referable_calibrated*100, res.reliability.confidence_level, res.reliability.reliability_score);
        text(ax_lim, 0.0, 0.85, summary_text, 'FontName', 'Helvetica', 'FontSize', 10, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        
        disclaim_lines = {
            'LIMITATIONS & DISCLAIMER: This screening report is generated by an artificial intelligence decision-support prototype', ...
            'designed for rural clinical triage. It does NOT constitute a definitive ophthalmological diagnosis. All automated findings', ...
            'must be verified by a qualified eye care professional before prescribing therapeutic interventions.'
        };
        for d = 1:numel(disclaim_lines)
            text(ax_lim, 0.0, 0.52 - d*0.18, disclaim_lines{d}, 'FontName', 'Helvetica', 'FontSize', 8.5, 'Color', [0.55 0.25 0.10], 'Interpreter', 'none');
        end

        drawPageFooter(fig1, img_id, timestamp_str, 1, total_pages);
        exportgraphics(fig1, target_pdf_path, 'Resolution', 150);
        clear cleanup1;

        % =====================================================================
        % PAGE 2: CLINICAL EVIDENCE & BIOMARKER INVENTORY
        % =====================================================================
        fig2 = createPageFigure(page_w, page_h);
        cleanup2 = onCleanup(@() closeFigure(fig2));

        drawPageHeader(fig2, 'CLINICAL EVIDENCE & BIOMARKER INVENTORY', img_id, timestamp_str, 2, total_pages);

        % Anatomical Landmarks Table / Box
        drawSectionBox(fig2, [70, 1190, 1100, 310], 'RETINAL ANATOMICAL LANDMARKS', [1.0 1.0 1.0], [0.20 0.40 0.65]);
        ax_anat = axes('Parent', fig2, 'Units', 'pixels', 'Position', [90, 1205, 1060, 260]);
        axis(ax_anat, 'off');
        od_c = res.anatomy.optic_disc.center;
        od_r = res.anatomy.optic_disc.radius;
        od_conf = res.anatomy.optic_disc.confidence;
        fov_c = res.anatomy.fovea.center;
        fov_mr = res.anatomy.fovea.macula_radius;
        
        anat_lines = {
            '• Optic Nerve Head (Optic Disc):', ...
            sprintf('    - Geometric Center Location:   (X = %d, Y = %d) pixels', round(od_c(1)), round(od_c(2))), ...
            sprintf('    - Disc Radius:                 %d pixels', round(od_r)), ...
            sprintf('    - Localization Confidence:     %.2f / 1.00 (%s)', od_conf, iif(od_conf > 0.6, 'Reliable Landmark', 'Uncertain')), ...
            '• Macula & Foveal Risk Zone:', ...
            sprintf('    - Estimated Foveal Center:     (X = %d, Y = %d) pixels', round(fov_c(1)), round(fov_c(2))), ...
            sprintf('    - Macular Threat Radius (1-DD): %d pixels (2.5x Optic Disc Radius)', round(fov_mr)), ...
            sprintf('    - DME Proximity Risk Factor:   Grade %d / 3 (%s)', res.lesions.exudates.macular_proximity_risk, ...
                iif(res.lesions.exudates.macular_proximity_risk > 0, 'Lipid exudates within macula', 'No macular exudates detected'))
        };
        for i = 1:numel(anat_lines)
            text(ax_anat, 0.0, 1.0 - i*0.115, anat_lines{i}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2]);
        end

        % Vascular Network Architecture
        drawSectionBox(fig2, [70, 830, 1100, 330], 'RETINAL VASCULATURE ARCHITECTURE (DEEP U-NET)', [1.0 1.0 1.0], [0.20 0.40 0.65]);
        ax_vess = axes('Parent', fig2, 'Units', 'pixels', 'Position', [90, 845, 1060, 280]);
        axis(ax_vess, 'off');
        v_dens = res.vessels.morphometry.vessel_density * 100;
        v_tort = res.vessels.morphometry.mean_tortuosity;
        v_br = res.vessels.morphometry.num_branch_points;
        
        vess_lines = {
            '• Vessel Segmentation Framework: Deep U-Net Trained Network Architecture', ...
            sprintf('• Total Vessel Caliber Density:   %.2f%% of total retinal field', v_dens), ...
            sprintf('• Mean Vascular Tortuosity Index: %.2f (Reference normal range: 1.05 - 1.25)', v_tort), ...
            sprintf('• Vascular Bifurcations / Branches: %d detected vessel branch points', v_br), ...
            '• Clinical Relevance in Diabetic Eye Disease:', ...
            '    - Vessel density attenuation indicates capillary drop-out and retinal non-perfusion.', ...
            '    - Elevated tortuosity index correlates with vascular remodeling and progression toward proliferative DR.'
        };
        for i = 1:numel(vess_lines)
            text(ax_vess, 0.0, 1.0 - i*0.13, vess_lines{i}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2]);
        end

        % Extracted Lesion Biomarkers Table
        drawSectionBox(fig2, [70, 330, 1100, 470], 'RETINAL LESION BIOMARKER INVENTORY', [1.0 1.0 1.0], [0.20 0.40 0.65]);
        ax_les = axes('Parent', fig2, 'Units', 'pixels', 'Position', [90, 345, 1060, 420]);
        axis(ax_les, 'off');
        
        % Table Header
        text(ax_les, 0.00, 0.95, 'BIOMARKER', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        text(ax_les, 0.28, 0.95, 'QUANTITATIVE MEASUREMENT', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        text(ax_les, 0.62, 0.95, 'CLINICAL SIGNIFICANCE & STATUS', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        line(ax_les, [0, 1], [0.91, 0.91], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);

        table_rows = {
            'Microaneurysms (MAs)', sprintf('%d lesions | %d px area', res.lesions.microaneurysms.count, res.lesions.microaneurysms.total_area), 'Earliest visible sign of NPDR; vascular pouching'; ...
            'Intraretinal Hemorrhages (HEs)', sprintf('%d lesions | %.2f%% area', res.lesions.hemorrhages.count, res.lesions.hemorrhages.area_fraction*100), 'Dot/blot capillary ruptures; severe NPDR marker'; ...
            'Hard Exudates (EXs)', sprintf('%d lesions | %.2f%% area', res.lesions.exudates.count, res.lesions.exudates.area_fraction*100), 'Lipid & lipoprotein extravasation from leaky capillaries'; ...
            'Macular DME Threat Grade', sprintf('Level %d / 3', res.lesions.exudates.macular_proximity_risk), 'Risk of Diabetic Macular Edema threatening central vision'; ...
            'Neovascularization Index (NV)', sprintf('Biomarker Score: %.2f', res.lesions.neovascularization.nv_score), 'Exploratory vessel morphometry (Not clinically certified)'
        };
        for r = 1:size(table_rows, 1)
            y_curr = 0.88 - r * 0.155;
            text(ax_les, 0.00, y_curr, table_rows{r, 1}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28], 'Interpreter', 'none');
            text(ax_les, 0.28, y_curr, table_rows{r, 2}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
            text(ax_les, 0.62, y_curr, table_rows{r, 3}, 'FontName', 'Helvetica', 'FontSize', 10, 'Color', [0.35 0.35 0.35], 'Interpreter', 'none');
            line(ax_les, [0, 1], [y_curr - 0.04, y_curr - 0.04], 'Color', [0.9 0.9 0.9], 'LineWidth', 0.5);
        end

        % Lesion Methodology & Validation Disclosure
        drawSectionBox(fig2, [70, 170, 1100, 130], 'BIOMARKER EXTRACTION DISCLOSURE', [0.97 0.97 0.97], [0.5 0.5 0.5]);
        ax_disc = axes('Parent', fig2, 'Units', 'pixels', 'Position', [90, 180, 1060, 90]);
        axis(ax_disc, 'off');
        text(ax_disc, 0.0, 0.70, '• Microaneurysms, hemorrhages, and exudates are detected using calibrated multi-scale morphological filtering and CLAHE contrast enhancement.', 'FontName', 'Helvetica', 'FontSize', 9.5, 'Color', [0.3 0.3 0.3]);
        text(ax_disc, 0.0, 0.25, '• Neovascularization analysis is an exploratory biomarker derived from vascular branch complexity and does not replace fluorescein angiography.', 'FontName', 'Helvetica', 'FontSize', 9.5, 'Color', [0.3 0.3 0.3]);

        drawPageFooter(fig2, img_id, timestamp_str, 2, total_pages);
        exportgraphics(fig2, target_pdf_path, 'Append', true, 'Resolution', 150);
        clear cleanup2;

        % =====================================================================
        % PAGE 3: MODEL PROBABILITY DISTRIBUTION & RELIABILITY BREAKDOWN
        % =====================================================================
        fig3 = createPageFigure(page_w, page_h);
        cleanup3 = onCleanup(@() closeFigure(fig3));

        drawPageHeader(fig3, 'MODEL OUTPUTS & 5-FACTOR RELIABILITY ANALYSIS', img_id, timestamp_str, 3, total_pages);

        % 5-Class Severity Probability Distribution Chart
        drawSectionBox(fig3, [70, 1070, 1100, 430], '5-CLASS ICDR DR SEVERITY POSTERIOR PROBABILITY DISTRIBUTION', [1.0 1.0 1.0], [0.20 0.40 0.65]);
        ax_bar = axes('Parent', fig3, 'Units', 'pixels', 'Position', [140, 1120, 980, 330]);
        probs = res.severity.probabilities(:)';
        y_pos = 0:4;
        barh(ax_bar, y_pos, probs * 100, 'FaceColor', [0.22 0.45 0.72], 'EdgeColor', 'none', 'BarWidth', 0.55);
        set(ax_bar, 'YDir', 'reverse', 'YTick', 0:4, ...
            'YTickLabel', {'Grade 0: No DR', 'Grade 1: Mild NPDR', 'Grade 2: Moderate NPDR', 'Grade 3: Severe NPDR', 'Grade 4: Proliferative DR'}, ...
            'XLim', [0, 100], 'XTick', 0:20:100, 'FontName', 'Helvetica', 'FontSize', 10, ...
            'Color', [1 1 1], 'XColor', [0.15 0.15 0.15], 'YColor', [0.15 0.15 0.15], 'GridColor', [0.8 0.8 0.8], 'GridAlpha', 0.6);
        xlabel(ax_bar, 'Posterior Probability (%)', 'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15]);
        grid(ax_bar, 'on');
        
        for k = 1:5
            p_val = probs(k) * 100;
            text(ax_bar, p_val + 1.5, k-1, sprintf('%.2f%%', p_val), 'FontName', 'Helvetica', 'FontSize', 9.5, 'FontWeight', 'bold', 'Color', [0.15 0.15 0.15], 'VerticalAlignment', 'middle');
        end

        % Calibration & Threshold Comparison
        drawSectionBox(fig3, [70, 830, 1100, 210], 'PROBABILITY CALIBRATION & THRESHOLD ANALYSIS', [0.97 0.98 1.0], [0.20 0.40 0.65]);
        ax_cal = axes('Parent', fig3, 'Units', 'pixels', 'Position', [90, 845, 1060, 160]);
        axis(ax_cal, 'off');
        cal_lines = {
            sprintf('• Raw Referable DR Probability:      %.2f%% (Sum of Grade 2, 3, 4 posterior probabilities)', res.referable.p_referable_raw * 100), ...
            sprintf('• Platt-Calibrated Probability:       %.2f%% (Fitted on 330-image development partition; a = 1.3106, b = -1.1412)', res.referable.p_referable_calibrated * 100), ...
            sprintf('• Operating Referral Threshold:       tau_ref = %.2f (Frozen development plateau operating point)', res.referable.referable_threshold_used), ...
            sprintf('• Referral Screening Classification:  %s (Calibrated P_ref %s tau_ref)', res.referable.referral_status, iif(res.referable.is_referable, '>=', '<')), ...
            sprintf('• Multimodal Concordance Metric:      %.1f%% agreement between CNN prediction and lesion burden', res.referable.agreement_score * 100)
        };
        for i = 1:numel(cal_lines)
            text(ax_cal, 0.0, 1.0 - i*0.18, cal_lines{i}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
        end

        % 5-Factor Reliability Analysis Table
        drawSectionBox(fig3, [70, 270, 1100, 530], '5-FACTOR ENGINEERING RELIABILITY BREAKDOWN', [1.0 1.0 1.0], [0.20 0.40 0.65]);
        ax_rel = axes('Parent', fig3, 'Units', 'pixels', 'Position', [90, 290, 1060, 470]);
        axis(ax_rel, 'off');
        
        text(ax_rel, 0.00, 0.96, 'RELIABILITY COMPONENT', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        text(ax_rel, 0.32, 0.96, 'WEIGHT', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        text(ax_rel, 0.46, 0.96, 'SCORE [0-100]', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        text(ax_rel, 0.65, 0.96, 'FACTOR ASSESSMENT & RATIONALE', 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        line(ax_rel, [0, 1], [0.93, 0.93], 'Color', [0.7 0.7 0.7], 'LineWidth', 1);

        rel_b = res.reliability.subscores;
        rel_table = {
            '1. Image Quality (S_q)', '25%', sprintf('%.1f', rel_b.quality), sprintf('Classification: %s (Quality Score: %.1f)', res.quality.quality_class, res.quality.quality_score); ...
            '2. Classifier Confidence (S_c)', '25%', sprintf('%.1f', rel_b.confidence), sprintf('Max class posterior margin (Confidence: %.1f%%)', res.severity.confidence*100); ...
            '3. Boundary Margin (S_m)', '20%', sprintf('%.1f', rel_b.margin), sprintf('Separation distance from referral threshold (|P - 0.30| = %.3f)', abs(res.referable.p_referable_calibrated - 0.30)); ...
            '4. Biomarker Completeness (S_comp)', '15%', sprintf('%.1f', rel_b.completeness), 'Integrity across quality, vessels, anatomy, lesions & CNN pipelines'; ...
            '5. Explainability Concordance (S_agree)', '15%', sprintf('%.1f', rel_b.agreement), sprintf('CAM activation and physical lesion overlap (Dice = %.3f)', res.explainability.agreement.dice)
        };
        for r = 1:size(rel_table, 1)
            y_curr = 0.89 - r * 0.13;
            text(ax_rel, 0.00, y_curr, rel_table{r, 1}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28], 'Interpreter', 'none');
            text(ax_rel, 0.32, y_curr, rel_table{r, 2}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'Color', [0.2 0.2 0.2], 'Interpreter', 'none');
            text(ax_rel, 0.46, y_curr, rel_table{r, 3}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.15 0.35 0.65], 'Interpreter', 'none');
            text(ax_rel, 0.65, y_curr, rel_table{r, 4}, 'FontName', 'Helvetica', 'FontSize', 9.5, 'Color', [0.30 0.30 0.30], 'Interpreter', 'none');
            line(ax_rel, [0, 1], [y_curr - 0.035, y_curr - 0.035], 'Color', [0.9 0.9 0.9], 'LineWidth', 0.5);
        end

        % Summary Banner for Reliability
        text(ax_rel, 0.00, 0.08, sprintf('COMPOSITE RELIABILITY: %.1f / 100   |   TIER: %s   |   SAFETY GATE: %s (tau_accept = 60.0)', ...
            res.reliability.reliability_score, res.reliability.confidence_level, bool2str(res.reliability.passes_reliability_gate, 'PASSED', 'SUB-THRESHOLD')), ...
            'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28], 'Interpreter', 'none');

        drawPageFooter(fig3, img_id, timestamp_str, 3, total_pages);
        exportgraphics(fig3, target_pdf_path, 'Append', true, 'Resolution', 150);
        clear cleanup3;

        % =====================================================================
        % PAGE 4: VISUAL INTERPRETABILITY ATLAS — PART 1 (LAYERS 1 - 6)
        % =====================================================================
        fig4 = createPageFigure(page_w, page_h);
        cleanup4 = onCleanup(@() closeFigure(fig4));

        drawPageHeader(fig4, 'VISUAL INTERPRETABILITY ATLAS — PART 1', img_id, timestamp_str, 4, total_pages);

        atlas_p1 = {
            'Original Fundus Photograph', res.overlays.original_rgb, 'True color 45-degree retinal fundus image as acquired from the camera.', 'RGB Fundus Image'; ...
            'Preprocessed & Enhanced', res.overlays.preprocessed_rgb, 'Normalized illumination with shading correction and CLAHE contrast enhancement.', 'Contrast Enhanced'; ...
            'Retinal Anatomy', res.overlays.anatomy, 'Cyan circle: Optic Disc boundary. Amber ring: Macular zone (2.5x OD radius).', 'Cyan: OD | Amber: Macula'; ...
            'Retinal Vessel Architecture', res.overlays.vessels, 'Deep U-Net vessel network segmentation overlaid on normalized fundus.', 'Green: Segmented Vessels'; ...
            'Microaneurysms Detection', res.overlays.microaneurysms, 'Red bounding markers: Detected microaneurysm candidates.', 'Red: MAs (Morphology)'; ...
            'Hemorrhages Detection', res.overlays.hemorrhages, 'Magenta regions: Detected intraretinal hemorrhages.', 'Magenta: Hemorrhages'
        };
        drawAtlasGrid(fig4, atlas_p1, [70, 150, 1100, 1350]);

        drawPageFooter(fig4, img_id, timestamp_str, 4, total_pages);
        exportgraphics(fig4, target_pdf_path, 'Append', true, 'Resolution', 150);
        clear cleanup4;

        % =====================================================================
        % PAGE 5: VISUAL INTERPRETABILITY ATLAS — PART 2 (LAYERS 7 - 10)
        % =====================================================================
        fig5 = createPageFigure(page_w, page_h);
        cleanup5 = onCleanup(@() closeFigure(fig5));

        drawPageHeader(fig5, 'VISUAL INTERPRETABILITY ATLAS — PART 2', img_id, timestamp_str, 5, total_pages);

        atlas_p2 = {
            'Hard Exudates & DME Risk', res.overlays.exudates, 'Yellow regions: Hard lipid exudates. Circles: Foveal proximity risk zones.', 'Yellow: Exudates | Blue: Fovea'; ...
            'Multi-Lesion Composite Map', res.overlays.multi_lesions, 'Combined clinical biomarker overlay: Red (MA), Magenta (HE), Yellow (EX).', 'Red: MA | Magenta: HE | Yellow: EX'; ...
            'Grad-CAM Deep Heatmap', res.overlays.gradcam_heatmap, 'Class activation heatmap from SqueezeNet final convolutional layer.', 'Jet Colormap (Red=High attention)'; ...
            'Grad-CAM Clinical Blend', res.overlays.gradcam_overlay, 'Alpha-blended Grad-CAM attention overlay providing spatial interpretability.', 'Alpha-Blend (0.5 Heatmap + Fundus)'
        };
        drawAtlasGrid(fig5, atlas_p2, [70, 680, 1100, 820]);

        % Final Clinical Notes Card on Page 5
        drawSectionBox(fig5, [70, 170, 1100, 480], 'CLINICAL TRIAGE & TELE-OPHTHALMOLOGY WORKFLOW SUMMARY', [0.98 0.99 1.0], [0.15 0.35 0.65]);
        ax_fin = axes('Parent', fig5, 'Units', 'pixels', 'Position', [90, 185, 1060, 430]);
        axis(ax_fin, 'off');
        
        fin_lines = {
            'PRIMARY HEALTH CENTER PROTOCOL RECOMMENDATION:', ...
            sprintf('• Examination Outcome:          %s (Grade %d)', ref_status, res.severity.predicted_grade), ...
            sprintf('• System Triage Disposition:      %s', upper(strrep(triage_act, '_', ' '))), ...
            sprintf('• Autonomous Decision Readiness:  %s', iif(res.reliability.decision_readiness, 'ACCEPTED (Meets high reliability threshold)', 'TRIAGE REQUIRED (Requires clinician review)')), ...
            sprintf('• Recommended Clinic Action:      %s', res.reliability.actionable_guidance), ...
            sprintf('• Recommended Referral Timeline:  %s', res.reliability.followup_window), ...
            '', ...
            'DOCUMENTATION & AUDIT TRAIL:', ...
            sprintf('• Software Engine Version:        SHINRA Diagnostic Framework Release V2 (Development Frozen)'), ...
            sprintf('• Calibration Checkpoint:         Platt Parameters a = 1.3106, b = -1.1412 (Fitted on IDRiD 330-train set)'), ...
            sprintf('• Operating Threshold:            Referral tau_ref = 0.30 | Acceptance tau_accept = 60.0'), ...
            sprintf('• CAM-Lesion Concordance (Dice):  %.3f (Spatial alignment between neural attention and physical lesions)', res.explainability.agreement.dice), ...
            sprintf('• Report File Generated:          %s', strrep(target_pdf_path, '\', '/'))
        };
        for i = 1:numel(fin_lines)
            if i == 1 || i == 7
                text(ax_fin, 0.0, 1.0 - i*0.075, fin_lines{i}, 'FontName', 'Helvetica', 'FontSize', 11, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28], 'Interpreter', 'none');
            else
                text(ax_fin, 0.0, 1.0 - i*0.075, fin_lines{i}, 'FontName', 'Helvetica', 'FontSize', 10, 'Color', [0.25 0.25 0.25], 'Interpreter', 'none');
            end
        end

        drawPageFooter(fig5, img_id, timestamp_str, 5, total_pages);
        exportgraphics(fig5, target_pdf_path, 'Append', true, 'Resolution', 150);
        clear cleanup5;

        success = true;
        saved_path = target_pdf_path;

    catch ME
        success = false;
        saved_path = '';
        rethrow(ME);
    end
end

% =========================================================================
% HELPER FUNCTIONS
% =========================================================================

function fig = createPageFigure(w, h)
    fig = figure('Visible', 'off', ...
                 'Units', 'pixels', ...
                 'Position', [100, 100, w, h], ...
                 'Color', [1 1 1], ...
                 'PaperType', 'A4', ...
                 'PaperOrientation', 'portrait', ...
                 'InvertHardcopy', 'off');
end

function closeFigure(fig)
    if isvalid(fig)
        close(fig);
    end
end

function drawPageHeader(fig, section_title, img_id, timestamp_str, page_num, total_pages)
    % Header Banner
    ax_h = axes('Parent', fig, 'Units', 'pixels', 'Position', [70, 1610, 1100, 100]);
    axis(ax_h, 'off');
    % Dark Navy header background bar
    rectangle(ax_h, 'Position', [0, 0, 1, 1], 'FaceColor', [0.10 0.18 0.28], 'EdgeColor', 'none');
    text(ax_h, 0.03, 0.68, 'SHINRA | RETINAL FUNDUS SCREENING SYSTEM', 'FontName', 'Helvetica', 'FontSize', 14, 'FontWeight', 'bold', 'Color', [1 1 1], 'Interpreter', 'none');
    text(ax_h, 0.03, 0.28, 'Explainable AI for Diabetic Retinopathy Screening in Rural India', 'FontName', 'Helvetica', 'FontSize', 9.5, 'Color', [0.75 0.85 0.95], 'Interpreter', 'none');
    
    text(ax_h, 0.97, 0.68, sprintf('REPORT ID: %s', img_id), 'FontName', 'Helvetica', 'FontSize', 10, 'FontWeight', 'bold', 'Color', [1 1 1], 'HorizontalAlignment', 'right', 'Interpreter', 'none');
    text(ax_h, 0.97, 0.28, sprintf('DATE: %s', timestamp_str), 'FontName', 'Helvetica', 'FontSize', 9, 'Color', [0.75 0.85 0.95], 'HorizontalAlignment', 'right', 'Interpreter', 'none');

    % Subtitle Bar
    ax_sub = axes('Parent', fig, 'Units', 'pixels', 'Position', [70, 1530, 1100, 60]);
    axis(ax_sub, 'off');
    text(ax_sub, 0.0, 0.50, section_title, 'FontName', 'Helvetica', 'FontSize', 13, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
    line(ax_sub, [0, 1], [0.1, 0.1], 'Color', [0.20 0.40 0.65], 'LineWidth', 2);
end

function drawPageFooter(fig, img_id, timestamp_str, page_num, total_pages)
    ax_f = axes('Parent', fig, 'Units', 'pixels', 'Position', [70, 40, 1100, 50]);
    axis(ax_f, 'off');
    line(ax_f, [0, 1], [0.9, 0.9], 'Color', [0.8 0.8 0.8], 'LineWidth', 1);
    text(ax_f, 0.0, 0.35, 'SHINRA Decision-Support Engine • For Clinical Research Use Only • Confirmed Verification Required', ...
        'FontName', 'Helvetica', 'FontSize', 8.5, 'Color', [0.45 0.45 0.45]);
    text(ax_f, 1.0, 0.35, sprintf('Page %d of %d', page_num, total_pages), ...
        'FontName', 'Helvetica', 'FontSize', 9, 'FontWeight', 'bold', 'Color', [0.3 0.3 0.3], 'HorizontalAlignment', 'right');
end

function drawSectionBox(fig, pos, title_str, bg_color, border_color)
    ax_box = axes('Parent', fig, 'Units', 'pixels', 'Position', pos);
    axis(ax_box, 'off');
    rectangle(ax_box, 'Position', [0, 0, 1, 1], 'FaceColor', bg_color, 'EdgeColor', border_color, 'LineWidth', 1, 'Curvature', [0.01 0.01]);
    
    % Header bar of section box
    h_fraction = 32 / pos(4);
    rectangle(ax_box, 'Position', [0, 1 - h_fraction, 1, h_fraction], 'FaceColor', border_color, 'EdgeColor', 'none');
    text(ax_box, 0.02, 1 - h_fraction/2, title_str, 'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [1 1 1], 'VerticalAlignment', 'middle');
end

function drawAtlasGrid(fig, layers_cell, bbox)
    % Draws 2-column grid of image layers with titles and descriptions
    x0 = bbox(1);
    y0 = bbox(2);
    w_total = bbox(3);
    h_total = bbox(4);

    n_layers = size(layers_cell, 1);
    cols = 2;
    rows = ceil(n_layers / cols);

    card_w = (w_total - 40) / cols;
    card_h = (h_total - (rows-1)*30) / rows;

    for i = 1:n_layers
        r = ceil(i / cols);
        c = mod(i-1, cols) + 1;

        cx = x0 + (c-1)*(card_w + 40);
        cy = y0 + (rows - r)*(card_h + 30);

        % Outer card boundary
        ax_card = axes('Parent', fig, 'Units', 'pixels', 'Position', [cx, cy, card_w, card_h]);
        axis(ax_card, 'off');
        rectangle(ax_card, 'Position', [0, 0, 1, 1], 'FaceColor', [1 1 1], 'EdgeColor', [0.8 0.8 0.8], 'LineWidth', 1);

        % Image Axes inside card
        img_h = card_h - 90;
        img_w = card_w - 20;
        ax_img = axes('Parent', fig, 'Units', 'pixels', 'Position', [cx + 10, cy + 70, img_w, img_h]);
        img_data = layers_cell{i, 2};
        if ~isempty(img_data)
            imshow(img_data, 'Parent', ax_img);
        else
            axis(ax_img, 'off');
            text(ax_img, 0.5, 0.5, 'Layer Unavailable', 'HorizontalAlignment', 'center');
        end
        axis(ax_img, 'image');

        % Title & Description inside card bottom
        ax_txt = axes('Parent', fig, 'Units', 'pixels', 'Position', [cx + 10, cy + 5, card_w - 20, 60]);
        axis(ax_txt, 'off');
        text(ax_txt, 0.0, 0.80, layers_cell{i, 1}, 'FontName', 'Helvetica', 'FontSize', 10.5, 'FontWeight', 'bold', 'Color', [0.10 0.18 0.28]);
        text(ax_txt, 0.0, 0.45, layers_cell{i, 3}, 'FontName', 'Helvetica', 'FontSize', 8.5, 'Color', [0.3 0.3 0.3]);
        text(ax_txt, 0.0, 0.15, sprintf('Legend: %s', layers_cell{i, 4}), 'FontName', 'Helvetica', 'FontSize', 8.5, 'FontWeight', 'bold', 'Color', [0.2 0.45 0.7]);
    end
end

function col = getStatusColor(q_class)
    switch upper(q_class)
        case 'GOOD'
            col = [0.10 0.60 0.25];
        case 'BORDERLINE'
            col = [0.75 0.50 0.05];
        otherwise
            col = [0.80 0.15 0.15];
    end
end

function col = getTriageColor(action)
    switch upper(action)
        case 'ACCEPT_SCREENING'
            col = [0.08 0.55 0.18];
        case 'RECAPTURE'
            col = [0.80 0.15 0.15];
        case 'CLINICAL_REVIEW'
            col = [0.75 0.45 0.05];
        otherwise
            col = [0.4 0.4 0.4];
    end
end

function out = iif(cond, v_true, v_false)
    if cond
        out = v_true;
    else
        out = v_false;
    end
end

function str = bool2str(val, t_str, f_str)
    if val
        str = t_str;
    else
        str = f_str;
    end
end
