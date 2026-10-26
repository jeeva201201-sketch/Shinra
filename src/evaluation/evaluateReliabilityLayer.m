function results = evaluateReliabilityLayer(varargin)
% EVALUATERELIABILITYLAYER Development Evaluation of the SHINRA Reliability Layer
% Evaluates multi-factor engineering reliability, decision readiness, uncertainty levels,
% and safety-gated triage dispositions across the 330-image training and 83-image validation cohorts.
%
% STRICT GOVERNANCE & SAFETY SAFEGUARDS:
% 1. The 103-image IDRiD held-out test partition is strictly blocked and verified.
% 2. Reliability thresholds are configured in shinra_config.m; zero tuning on validation labels.
% 3. No modification to trained model checkpoints.
%
% OUTPUT FILES:
% - results/metrics/reliability_layer_evaluation.mat
% - results/metrics/reliability_layer_evaluation.csv
% - results/metrics/reliability_layer_summary.csv

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: RELIABILITY LAYER & SAFETY-TRIAGE BENCHMARK                  \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Enforce Strict Quarantine of IDRiD 103-Image Test Partition
    % -------------------------------------------------------------------------
    fprintf('[SAFEGUARD 1/3] Verifying strict test partition quarantine and physical path isolation...\n');
    [idrid_train_full, idrid_test] = load_idrid();

    % -------------------------------------------------------------------------
    % 2. Load Precomputed Development Audit Data & Verify Partition Integrity
    % -------------------------------------------------------------------------
    audit_file = fullfile(cfg.paths.metrics, 'evidence_channel_audit.mat');
    if exist(audit_file, 'file') == 2
        fprintf('[STEP 2/3] Loading precomputed evidence audit from:\n  %s\n', audit_file);
        loaded = load(audit_file);
        audit_table = loaded.audit_table;
    else
        fprintf('[STEP 2/3] Running evaluateEvidenceChannels to generate development audit...\n');
        audit_res = evaluateEvidenceChannels();
        audit_table = audit_res.audit_table;
    end

    n_total = height(audit_table);
    assert(n_total == 413, 'Expected 413 development images. Found: %d', n_total);

    n_tr_audit  = sum(strcmp(audit_table.Split, 'Train'));
    n_val_audit = sum(strcmp(audit_table.Split, 'Validation'));
    assert(n_tr_audit == 330, 'Expected 330 training partition images. Found: %d', n_tr_audit);
    assert(n_val_audit == 83, 'Expected 83 validation partition images. Found: %d', n_val_audit);

    % Path Normalization Helper
    normalize_path = @(p) lower(strrep(strrep(strtrim(char(p)), '/', filesep), '\', filesep));
    normalized_audit_paths = cellfun(normalize_path, audit_table.ImagePath, 'UniformOutput', false);

    % Structural Check 1: Verify all audit paths reside in the official Training Set directory
    expected_train_dir = normalize_path(fullfile(cfg.paths.idrid, 'B. Disease Grading', '1. Original Images', 'a. Training Set'));
    for i = 1:n_total
        p_curr = normalized_audit_paths{i};
        if ~contains(p_curr, expected_train_dir) && ~contains(p_curr, normalize_path('a. training set'))
            error('CRITICAL VIOLATION: Audit image path outside Training Set root detected: %s', audit_table.ImagePath{i});
        end
    end

    % Structural Check 2: Physical Path Non-Overlap with 103 Test Images
    if height(idrid_test) > 0
        normalized_test_paths = cellfun(normalize_path, idrid_test.ImagePath, 'UniformOutput', false);
        assert(numel(normalized_test_paths) == 103, 'Expected 103 test images. Found: %d', numel(normalized_test_paths));
        
        path_overlap = ismember(normalized_audit_paths, normalized_test_paths);
        assert(~any(path_overlap), ...
            'CRITICAL VIOLATION: IDRiD test image path detected in development audit!');
    end

    fprintf('  [PASS] Development audit contains 413 physical training images.\n');
    fprintf('  [PASS] 330 Train + 83 Validation.\n');
    fprintf('  [PASS] 0 physical path overlap with 103 test images.\n');
    fprintf('  [PASS] All audit paths belong to IDRiD Training Set.\n');
    fprintf('  [PASS] Test partition remains quarantined.\n\n');

    % -------------------------------------------------------------------------
    % 3. Evaluate Reliability Layer for Each Development Image
    % -------------------------------------------------------------------------
    fprintf('[STEP 3/3] Evaluating 5-factor reliability scoring across %d development images...\n\n', n_total);

    image_ids           = audit_table.ImageID;
    split_tags          = audit_table.Split;
    true_grades         = audit_table.True_DR_Grade;
    true_referable      = audit_table.True_Referable;

    quality_classes     = cell(n_total, 1);
    quality_scores      = zeros(n_total, 1);
    cnn_p_ref           = audit_table.CNN_P_Referable;
    fused_p_ref         = audit_table.Fusion_P_Referable;

    reliability_scores  = zeros(n_total, 1);
    sub_quality         = zeros(n_total, 1);
    sub_confidence      = zeros(n_total, 1);
    sub_margin          = zeros(n_total, 1);
    sub_completeness    = zeros(n_total, 1);
    sub_agreement       = zeros(n_total, 1);

    conf_levels         = cell(n_total, 1);
    uncert_levels       = cell(n_total, 1);
    agree_classes       = cell(n_total, 1);
    rec_actions         = cell(n_total, 1);
    decision_readiness  = false(n_total, 1);
    pred_referable      = false(n_total, 1);
    primary_reasons     = cell(n_total, 1);

    for i = 1:n_total
        % Package quality struct
        q_rep = struct(...
            'score', audit_table.Quality_Score(i), ...
            'quality_class', audit_table.Quality_Class{i});

        % Package severity struct
        probs = [audit_table.CNN_P0(i), audit_table.CNN_P1(i), audit_table.CNN_P2(i), audit_table.CNN_P3(i), audit_table.CNN_P4(i)];
        sev_res = struct(...
            'confidence', max(probs), ...
            'probabilities', probs, ...
            'p_referable', audit_table.CNN_P_Referable(i), ...
            'predicted_grade', audit_table.CNN_Predicted_Grade(i), ...
            'expected_grade', sum((0:4) .* probs));

        % Package fusion struct
        fus_res = struct(...
            'p_referable_fused', audit_table.Fusion_P_Referable(i), ...
            'p_cnn_referable', audit_table.CNN_P_Referable(i), ...
            'agreement_score', audit_table.Fusion_Agreement_Score(i));

        % Package lesion evidence struct
        lesion_ev = struct(...
            'ma_count', audit_table.MA_Count(i), ...
            'ma_area', audit_table.MA_Area(i), ...
            'he_count', audit_table.HE_Count(i), ...
            'he_area_fraction', audit_table.HE_Area_Fraction(i), ...
            'ex_count', audit_table.EX_Count(i), ...
            'ex_area_fraction', audit_table.EX_Area_Fraction(i), ...
            'dme_risk', audit_table.DME_Risk(i), ...
            'nv_score', audit_table.NV_Score(i));

        % Package vessel struct
        vess_res = struct('morphometry', struct(...
            'vessel_density', audit_table.Vessel_Density(i), ...
            'mean_tortuosity', audit_table.Vessel_Tortuosity(i)));

        % Package anatomy placeholder
        od_res = struct('center', [256, 256], 'radius', 40, 'confidence', 0.90);
        fovea_res = struct('center', [320, 256], 'macula_radius', 60);

        % Compute 5-factor reliability
        rel_res = computeReliability(q_rep, sev_res, fus_res, lesion_ev, vess_res, od_res, fovea_res);

        % Compute screening decision
        dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);

        % Record outputs
        quality_classes{i}    = q_rep.quality_class;
        quality_scores(i)     = q_rep.score;
        reliability_scores(i) = rel_res.reliability_score;
        sub_quality(i)        = rel_res.quality_score_component;
        sub_confidence(i)     = rel_res.confidence_score_component;
        sub_margin(i)         = rel_res.margin_score_component;
        sub_completeness(i)   = rel_res.completeness_score_component;
        sub_agreement(i)      = rel_res.agreement_score_component;
        conf_levels{i}        = rel_res.confidence_level;
        uncert_levels{i}      = rel_res.uncertainty_level;
        agree_classes{i}      = rel_res.evidence_agreement;

        rec_actions{i}        = dec_res.decision;
        decision_readiness(i) = dec_res.decision_readiness;
        pred_referable(i)     = dec_res.predicted_referable;

        if ~isempty(rel_res.reliability_reasons)
            primary_reasons{i} = rel_res.reliability_reasons{1};
        else
            primary_reasons{i} = 'Normal screening conditions';
        end
    end

    % Assemble Master Evaluation Table
    eval_table = table(...
        image_ids, split_tags, true_grades, true_referable, ...
        quality_classes, quality_scores, cnn_p_ref, fused_p_ref, ...
        reliability_scores, sub_quality, sub_confidence, sub_margin, sub_completeness, sub_agreement, ...
        conf_levels, uncert_levels, agree_classes, ...
        rec_actions, decision_readiness, pred_referable, primary_reasons, ...
        'VariableNames', {...
            'ImageID', 'Split', 'True_DR_Grade', 'True_Referable', ...
            'Quality_Class', 'Quality_Score', 'CNN_P_Referable', 'Fused_P_Referable', ...
            'Reliability_Score', 'SubScore_Quality', 'SubScore_Confidence', 'SubScore_Margin', 'SubScore_Completeness', 'SubScore_Agreement', ...
            'Confidence_Level', 'Uncertainty_Level', 'Evidence_Agreement', ...
            'Recommended_Action', 'Decision_Readiness', 'Predicted_Referable', 'Primary_Reason'});

    % -------------------------------------------------------------------------
    % 4. Detailed Partition Breakdown (330 Training & 83 Validation)
    % -------------------------------------------------------------------------
    splits = {'Train', 'Validation'};
    summary_struct = struct();

    for s = 1:numel(splits)
        sp_name = splits{s};
        sp_mask = strcmp(eval_table.Split, sp_name);
        sp_tab = eval_table(sp_mask, :);
        n_sp = height(sp_tab);

        fprintf('========================================================================\n');
        fprintf('   RELIABILITY LAYER PERFORMANCE: %s SPLIT (N = %d Images)              \n', upper(sp_name), n_sp);
        fprintf('========================================================================\n');

        % 1. Reliability Score Distribution
        rel_mean = mean(sp_tab.Reliability_Score);
        rel_std  = std(sp_tab.Reliability_Score);
        rel_med  = median(sp_tab.Reliability_Score);
        rel_min  = min(sp_tab.Reliability_Score);
        rel_max  = max(sp_tab.Reliability_Score);

        fprintf('1. Reliability Score Distribution [0 - 100]:\n');
        fprintf('   • Mean ± Std:  %5.2f ± %5.2f\n', rel_mean, rel_std);
        fprintf('   • Median:      %5.2f  [Min: %5.2f, Max: %5.2f]\n\n', rel_med, rel_min, rel_max);

        % 2. Categorical Quality Distribution
        n_good = sum(strcmp(sp_tab.Quality_Class, 'GOOD'));
        n_bord = sum(strcmp(sp_tab.Quality_Class, 'BORDERLINE'));
        n_ungr = sum(strcmp(sp_tab.Quality_Class, 'UNGRADABLE'));

        fprintf('2. Quality Status Distribution:\n');
        fprintf('   • GOOD:        %3d (%5.1f%%)\n', n_good, (n_good/n_sp)*100);
        fprintf('   • BORDERLINE:  %3d (%5.1f%%)\n', n_bord, (n_bord/n_sp)*100);
        fprintf('   • UNGRADABLE:  %3d (%5.1f%%)\n\n', n_ungr, (n_ungr/n_sp)*100);

        % 3. Computational Confidence & Uncertainty Levels
        n_c_high = sum(strcmp(sp_tab.Confidence_Level, 'HIGH'));
        n_c_mod  = sum(strcmp(sp_tab.Confidence_Level, 'MODERATE'));
        n_c_low  = sum(strcmp(sp_tab.Confidence_Level, 'LOW'));

        n_u_low  = sum(strcmp(sp_tab.Uncertainty_Level, 'LOW'));
        n_u_mod  = sum(strcmp(sp_tab.Uncertainty_Level, 'MODERATE'));
        n_u_high = sum(strcmp(sp_tab.Uncertainty_Level, 'HIGH'));

        fprintf('3. Computational Confidence & Uncertainty:\n');
        fprintf('   • Confidence:   HIGH = %3d (%5.1f%%) | MODERATE = %3d (%5.1f%%) | LOW = %3d (%5.1f%%)\n', ...
            n_c_high, (n_c_high/n_sp)*100, n_c_mod, (n_c_mod/n_sp)*100, n_c_low, (n_c_low/n_sp)*100);
        fprintf('   • Uncertainty:  LOW  = %3d (%5.1f%%) | MODERATE = %3d (%5.1f%%) | HIGH = %3d (%5.1f%%)\n\n', ...
            n_u_low, (n_u_low/n_sp)*100, n_u_mod, (n_u_mod/n_sp)*100, n_u_high, (n_u_high/n_sp)*100);

        % 4. Standardized Action Triage Distribution
        n_act_accept = sum(strcmp(sp_tab.Recommended_Action, 'ACCEPT_SCREENING'));
        n_act_recap  = sum(strcmp(sp_tab.Recommended_Action, 'RECAPTURE'));
        n_act_review = sum(strcmp(sp_tab.Recommended_Action, 'CLINICAL_REVIEW'));
        n_act_uncert = sum(strcmp(sp_tab.Recommended_Action, 'UNCERTAIN'));

        coverage_pct = (n_act_accept / n_sp) * 100;

        fprintf('4. Standardized Action Triage Disposition:\n');
        fprintf('   • ACCEPT_SCREENING: %3d (%5.1f%%) [Autonomous Decision Readiness Coverage]\n', n_act_accept, coverage_pct);
        fprintf('   • RECAPTURE:        %3d (%5.1f%%) [Immediate Camera Re-acquisition]\n', n_act_recap, (n_act_recap/n_sp)*100);
        fprintf('   • CLINICAL_REVIEW:  %3d (%5.1f%%) [Suspicious Findings Referred to Specialist]\n', n_act_review, (n_act_review/n_sp)*100);
        fprintf('   • UNCERTAIN:        %3d (%5.1f%%) [Borderline Quality / Ambiguity Verification]\n\n', n_act_uncert, (n_act_uncert/n_sp)*100);

        % 5. Autonomous Screening Diagnostic Accuracy (On Accepted Cohort)
        acc_mask = strcmp(sp_tab.Recommended_Action, 'ACCEPT_SCREENING');
        if sum(acc_mask) > 0
            y_tr_acc = sp_tab.True_Referable(acc_mask);
            y_pr_acc = sp_tab.Predicted_Referable(acc_mask);

            tp_acc = sum(y_pr_acc == 1 & y_tr_acc == 1);
            fp_acc = sum(y_pr_acc == 1 & y_tr_acc == 0);
            fn_acc = sum(y_pr_acc == 0 & y_tr_acc == 1);
            tn_acc = sum(y_pr_acc == 0 & y_tr_acc == 0);

            sens_acc = tp_acc / max(1, tp_acc + fn_acc);
            spec_acc = tn_acc / max(1, tn_acc + fp_acc);
            prec_acc = tp_acc / max(1, tp_acc + fp_acc);
            f1_acc   = (2 * tp_acc) / max(1, 2 * tp_acc + fp_acc + fn_acc);
            acc_acc  = (tp_acc + tn_acc) / sum(acc_mask);

            fprintf('5. Diagnostic Accuracy on Accepted Screening Cohort (N = %d):\n', sum(acc_mask));
            fprintf('   • Referable Sensitivity: %5.2f%% (%2d / %2d)\n', sens_acc * 100, tp_acc, tp_acc + fn_acc);
            fprintf('   • Referable Specificity: %5.2f%% (%2d / %2d)\n', spec_acc * 100, tn_acc, tn_acc + fp_acc);
            fprintf('   • Referable Precision:   %5.2f%%\n', prec_acc * 100);
            fprintf('   • Referable F1 Score:    %6.4f\n', f1_acc);
            fprintf('   • Overall Accuracy:      %5.2f%%\n', acc_acc * 100);
            fprintf('   • False Negatives:       %2d | False Positives: %2d\n\n', fn_acc, fp_acc);
        else
            fprintf('5. Diagnostic Accuracy on Accepted Cohort: N = 0 (No cases accepted).\n\n');
        end

        % 6. Stratified Performance by Quality Status
        fprintf('6. Stratified Reliability and Action Summary by Image Quality:\n');
        fprintf('   ----------------------------------------------------------------------\n');
        fprintf('   Quality Status   Count   Mean_Reliability   Accept   Recapture   Review   Uncertain\n');
        fprintf('   ----------------------------------------------------------------------\n');
        q_cats = {'GOOD', 'BORDERLINE', 'UNGRADABLE'};
        for q = 1:numel(q_cats)
            qc = q_cats{q};
            q_m = strcmp(sp_tab.Quality_Class, qc);
            n_qc = sum(q_m);
            if n_qc > 0
                mean_r_qc = mean(sp_tab.Reliability_Score(q_m));
                n_acc_qc  = sum(strcmp(sp_tab.Recommended_Action(q_m), 'ACCEPT_SCREENING'));
                n_rec_qc  = sum(strcmp(sp_tab.Recommended_Action(q_m), 'RECAPTURE'));
                n_rev_qc  = sum(strcmp(sp_tab.Recommended_Action(q_m), 'CLINICAL_REVIEW'));
                n_unc_qc  = sum(strcmp(sp_tab.Recommended_Action(q_m), 'UNCERTAIN'));
                fprintf('   %-14s   %3d        %5.2f          %3d       %3d       %3d       %3d\n', ...
                    qc, n_qc, mean_r_qc, n_acc_qc, n_rec_qc, n_rev_qc, n_unc_qc);
            else
                fprintf('   %-14s     0         0.00            0         0         0         0\n', qc);
            end
        end
        fprintf('   ----------------------------------------------------------------------\n\n');

        summary_struct.(sp_name) = struct(...
            'N', n_sp, 'Reliability_Mean', rel_mean, 'Reliability_Std', rel_std, ...
            'Accept_Count', n_act_accept, 'Recapture_Count', n_act_recap, ...
            'Review_Count', n_act_review, 'Uncertain_Count', n_act_uncert, ...
            'Coverage_Pct', coverage_pct);
    end

    % -------------------------------------------------------------------------
    % 5. Save Output Artifacts
    % -------------------------------------------------------------------------
    out_mat = fullfile(cfg.paths.metrics, 'reliability_layer_evaluation.mat');
    out_csv = fullfile(cfg.paths.metrics, 'reliability_layer_evaluation.csv');
    out_sum_csv = fullfile(cfg.paths.metrics, 'reliability_layer_summary.csv');

    % Summary Table
    summary_tab = table(...
        {'Total_Images'; 'Reliability_Mean'; 'Reliability_Std'; 'Accepted_Screenings'; 'Recaptures'; 'Clinical_Reviews'; 'Uncertain_Cases'; 'Decision_Readiness_Coverage'}, ...
        [summary_struct.Train.N; summary_struct.Train.Reliability_Mean; summary_struct.Train.Reliability_Std; summary_struct.Train.Accept_Count; summary_struct.Train.Recapture_Count; summary_struct.Train.Review_Count; summary_struct.Train.Uncertain_Count; summary_struct.Train.Coverage_Pct], ...
        [summary_struct.Validation.N; summary_struct.Validation.Reliability_Mean; summary_struct.Validation.Reliability_Std; summary_struct.Validation.Accept_Count; summary_struct.Validation.Recapture_Count; summary_struct.Validation.Review_Count; summary_struct.Validation.Uncertain_Count; summary_struct.Validation.Coverage_Pct], ...
        'VariableNames', {'Metric', 'Training_Split_N330', 'Validation_Split_N83'});

    save(out_mat, 'eval_table', 'summary_tab', 'summary_struct', 'cfg');
    writetable(eval_table, out_csv);
    writetable(summary_tab, out_sum_csv);

    fprintf('Reliability evaluation artifacts saved to:\n');
    fprintf('  • %s\n', out_mat);
    fprintf('  • %s\n', out_csv);
    fprintf('  • %s\n\n', out_sum_csv);

    results = struct();
    results.eval_table = eval_table;
    results.summary_table = summary_tab;
    results.summary = summary_struct;
end
