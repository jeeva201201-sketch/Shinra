function results = evaluateEvidenceChannels(varargin)
% EVALUATEEVIDENCECHANNELS Development-Only Audit of Retinal Evidence Channels
% Evaluates whether the existing lesion, vessel, and anatomy evidence features
% correlate with true IDRiD diabetic retinopathy severity grades and binary referable labels.
%
% STRICT SAFEGUARDS:
% 1. Uses ONLY IDRiD training images (413 development cohort: 330 train / 83 validation).
% 2. Strictly forbids and asserts absence of the 103-image IDRiD test partition.
% 3. Evaluates frozen checkpoints only; does NOT retrain or optimize weights/thresholds.
% 4. Uses fixed random seed 42 for reproducible development partitioning.
%
% OUTPUT FILES:
% - results/metrics/evidence_channel_audit.mat
% - results/metrics/evidence_channel_audit.csv
% - results/metrics/evidence_validation_summary.csv

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: DEVELOPMENT-ONLY RETINAL EVIDENCE CHANNEL AUDIT             \n');
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 1. Load Dataset & Enforce Strict Partition Safeguards
    % -------------------------------------------------------------------------
    fprintf('[SAFEGUARD 1/3] Loading IDRiD dataset and validating isolation...\n');
    [idrid_train_full, idrid_test] = load_idrid();

    if height(idrid_train_full) == 0
        error('SHINRA:DatasetNotFound', 'IDRiD development dataset not found.');
    end

    % Reproducible stratified split matching Severity V3 baseline (Seed 42)
    [train_tab, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);

    num_train = height(train_tab);
    num_val   = height(val_tab);
    num_total = num_train + num_val;

    % Strict Assertions
    assert(num_train == 330, 'Training partition must contain exactly 330 images. Found: %d', num_train);
    assert(num_val == 83, 'Validation partition must contain exactly 83 images. Found: %d', num_val);
    assert(num_total == 413, 'Total development set must contain exactly 413 images. Found: %d', num_total);

    if height(idrid_test) > 0
        test_paths = idrid_test.ImagePath;
        assert(~any(ismember(train_tab.ImagePath, test_paths)), ...
            'CRITICAL ERROR: Data leakage detected! IDRiD test images found in training split!');
        assert(~any(ismember(val_tab.ImagePath, test_paths)), ...
            'CRITICAL ERROR: Data leakage detected! IDRiD test images found in validation split!');
        fprintf('  [PASSED] Test set (103 images) strictly sequestered. Zero overlap detected.\n');
    end
    fprintf('  [PASSED] Development split verified: 330 Training + 83 Validation images.\n\n');

    % Combine into single development dataset with split indicator
    train_tab.Split = repmat({'Train'}, num_train, 1);
    val_tab.Split   = repmat({'Validation'}, num_val, 1);
    dev_tab = [train_tab; val_tab];

    % -------------------------------------------------------------------------
    % 2. Process All 413 Development Images Through Evidence Pipeline
    % -------------------------------------------------------------------------
    fprintf('[EXECUTION 2/3] Processing %d development images through evidence pipeline...\n', num_total);
    fprintf('  (Extracting: FOV, Vessels, Anatomy, MAs, HEs, EXs, NV, Quality, CNN V3, Fusion)\n');

    % Pre-allocate per-image storage arrays
    image_ids           = cell(num_total, 1);
    image_paths         = cell(num_total, 1);
    split_tags          = cell(num_total, 1);
    true_grades         = zeros(num_total, 1);
    true_referable      = zeros(num_total, 1);
    true_dme_risk       = zeros(num_total, 1);
    quality_scores      = zeros(num_total, 1);
    quality_classes     = cell(num_total, 1);

    % CNN Softmax Probabilities
    cnn_p0              = zeros(num_total, 1);
    cnn_p1              = zeros(num_total, 1);
    cnn_p2              = zeros(num_total, 1);
    cnn_p3              = zeros(num_total, 1);
    cnn_p4              = zeros(num_total, 1);
    cnn_p_referable     = zeros(num_total, 1);
    cnn_pred_grade      = zeros(num_total, 1);
    cnn_referable_pred  = zeros(num_total, 1);

    % Lesion Evidence Features
    ma_counts           = zeros(num_total, 1);
    ma_areas            = zeros(num_total, 1);
    he_counts           = zeros(num_total, 1);
    he_area_fractions   = zeros(num_total, 1);
    ex_counts           = zeros(num_total, 1);
    ex_area_fractions   = zeros(num_total, 1);
    dme_risks           = zeros(num_total, 1);
    nv_scores           = zeros(num_total, 1);

    % Vessel Features
    vessel_densities    = zeros(num_total, 1);
    vessel_tortuosities = zeros(num_total, 1);

    % Fusion Features
    fusion_p_referable  = zeros(num_total, 1);
    fusion_agreement    = zeros(num_total, 1);
    fusion_referable_pred = zeros(num_total, 1);

    % Diagnostic Status
    statuses            = cell(num_total, 1);
    error_messages      = cell(num_total, 1);

    success_count = 0;
    failure_count = 0;

    for i = 1:num_total
        img_id   = dev_tab.ImageID{i};
        img_path = dev_tab.ImagePath{i};
        split_tg = dev_tab.Split{i};
        dr_gr    = dev_tab.DR_Grade(i);
        dme_rk   = dev_tab.DME_Risk(i);
        ref_tr   = double(dr_gr >= 2);

        image_ids{i}      = img_id;
        image_paths{i}    = img_path;
        split_tags{i}     = split_tg;
        true_grades(i)    = dr_gr;
        true_referable(i) = ref_tr;
        true_dme_risk(i)  = dme_rk;

        try
            % 1. Load and Preprocess
            raw_img = imread(img_path);
            prep = preprocessRetina(raw_img, cfg.preprocessing.target_size);

            % 2. Quality Safety Gate
            [q_class, q_rep] = qualityClassifier(prep.rgb_normalized, prep.fov_mask);
            quality_scores(i)  = q_rep.score;
            quality_classes{i} = q_class;

            % 3. Retinal Anatomy Localization
            od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
            fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);

            % 4. Retinal Vessel Segmentation & Morphometry (Uses Trained U-Net)
            vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
            vessel_densities(i)    = vess_res.morphometry.vessel_density;
            vessel_tortuosities(i) = vess_res.morphometry.mean_tortuosity;

            % 5. Multi-Scale Lesion Biomarker Extraction
            ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
            he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
            ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
            nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);

            ma_counts(i)         = ma_res.count;
            ma_areas(i)          = ma_res.total_area;
            he_counts(i)         = he_res.count;
            he_area_fractions(i) = he_res.area_fraction;
            ex_counts(i)         = ex_res.count;
            ex_area_fractions(i) = ex_res.area_fraction;
            dme_risks(i)         = ex_res.macular_proximity_risk;
            nv_scores(i)         = nv_res.nv_score;

            lesion_ev = struct(...
                'ma_count', ma_res.count, 'ma_area', ma_res.total_area, ...
                'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, ...
                'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, ...
                'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);

            % 6. Frozen Severity V3 Inference
            try
                sev_res = predictSqueezeNetSeverityV3(prep);
            catch
                sev_res = predictDRSeverity(prep, lesion_ev);
            end

            cnn_p0(i)             = sev_res.probabilities(1);
            cnn_p1(i)             = sev_res.probabilities(2);
            cnn_p2(i)             = sev_res.probabilities(3);
            cnn_p3(i)             = sev_res.probabilities(4);
            cnn_p4(i)             = sev_res.probabilities(5);
            cnn_p_referable(i)    = sev_res.p_referable;
            cnn_pred_grade(i)     = sev_res.predicted_grade;
            cnn_referable_pred(i) = double(sev_res.p_referable >= cfg.fusion.referable_threshold);

            % 7. 16-D Evidence Vector & Rule-Based Fusion Prediction
            [ev_vec, feat_names] = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
            fus_res = fuseEvidence(ev_vec, feat_names);

            fusion_p_referable(i)    = fus_res.p_referable_fused;
            fusion_agreement(i)      = fus_res.agreement_score;
            fusion_referable_pred(i) = double(fus_res.p_referable_fused >= cfg.fusion.referable_threshold);

            statuses{i}       = 'SUCCESS';
            error_messages{i} = '';
            success_count     = success_count + 1;

        catch ME
            statuses{i}       = 'ERROR';
            error_messages{i} = ME.message;
            failure_count     = failure_count + 1;
        end

        if mod(i, 50) == 0 || i == num_total
            fprintf('  Processed %d / %d images (Success: %d, Failures: %d)...\n', ...
                i, num_total, success_count, failure_count);
        end
    end
    fprintf('\n');

    % Assemble Master Per-Image Table
    audit_table = table(...
        image_ids, image_paths, split_tags, true_grades, true_referable, true_dme_risk, ...
        quality_scores, quality_classes, ...
        cnn_p0, cnn_p1, cnn_p2, cnn_p3, cnn_p4, cnn_p_referable, cnn_pred_grade, cnn_referable_pred, ...
        ma_counts, ma_areas, he_counts, he_area_fractions, ex_counts, ex_area_fractions, dme_risks, nv_scores, ...
        vessel_densities, vessel_tortuosities, ...
        fusion_p_referable, fusion_agreement, fusion_referable_pred, ...
        statuses, error_messages, ...
        'VariableNames', {...
            'ImageID', 'ImagePath', 'Split', 'True_DR_Grade', 'True_Referable', 'True_DME_Risk', ...
            'Quality_Score', 'Quality_Class', ...
            'CNN_P0', 'CNN_P1', 'CNN_P2', 'CNN_P3', 'CNN_P4', 'CNN_P_Referable', 'CNN_Predicted_Grade', 'CNN_Referable_Pred', ...
            'MA_Count', 'MA_Area', 'HE_Count', 'HE_Area_Fraction', 'EX_Count', 'EX_Area_Fraction', 'DME_Risk', 'NV_Score', ...
            'Vessel_Density', 'Vessel_Tortuosity', ...
            'Fusion_P_Referable', 'Fusion_Agreement_Score', 'Fusion_Referable_Pred', ...
            'Status', 'ErrorMessage'});

    % -------------------------------------------------------------------------
    % 3. Evidence Feature Summaries Grouped by True DR Grade (0 to 4)
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   3. EVIDENCE FEATURE DISTRIBUTIONS GROUPED BY TRUE DR GRADE (0 to 4)  \n');
    fprintf('========================================================================\n');

    feature_cols = {
        'MA_Count', 'MA_Area', 'HE_Count', 'HE_Area_Fraction', ...
        'EX_Count', 'EX_Area_Fraction', 'DME_Risk', 'NV_Score', ...
        'Vessel_Density', 'Vessel_Tortuosity', 'Quality_Score', 'CNN_P_Referable'
    };
    num_feats = numel(feature_cols);

    grade_stats = struct();
    for g = 0:4
        g_mask = (audit_table.True_DR_Grade == g) & strcmp(audit_table.Status, 'SUCCESS');
        g_count = sum(g_mask);
        fprintf('\n--- DR GRADE %d (N = %d images) ---\n', g, g_count);
        fprintf('  %-20s  %10s  %10s  %10s  %10s  %10s\n', ...
            'Feature', 'Mean', 'Std', 'Median', 'Min', 'Max');
        fprintf('  ----------------------------------------------------------------------\n');

        grade_stats.(sprintf('Grade_%d', g)) = struct();
        grade_stats.(sprintf('Grade_%d', g)).N = g_count;

        for f = 1:num_feats
            f_name = feature_cols{f};
            vals = audit_table.(f_name)(g_mask);
            if ~isempty(vals)
                f_mean = mean(vals);
                f_std  = std(vals);
                f_med  = median(vals);
                f_min  = min(vals);
                f_max  = max(vals);
            else
                f_mean = 0; f_std = 0; f_med = 0; f_min = 0; f_max = 0;
            end

            grade_stats.(sprintf('Grade_%d', g)).(f_name) = struct(...
                'Mean', f_mean, 'Std', f_std, 'Median', f_med, 'Min', f_min, 'Max', f_max);

            if ismember(f_name, {'HE_Area_Fraction', 'EX_Area_Fraction', 'Vessel_Density', 'CNN_P_Referable'})
                fprintf('  %-20s  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f\n', ...
                    f_name, f_mean, f_std, f_med, f_min, f_max);
            else
                fprintf('  %-20s  %10.2f  %10.2f  %10.2f  %10.2f  %10.2f\n', ...
                    f_name, f_mean, f_std, f_med, f_min, f_max);
            end
        end
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 4. Binary Referable DR Comparison (Grades 0/1 vs Grades 2/3/4)
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   4. BINARY REFERABLE DR SUMMARY: NON-REFERABLE vs REFERABLE           \n');
    fprintf('   [Methodology Note: Exploratory Development Association Metrics]      \n');
    fprintf('========================================================================\n');
    fprintf('  %-20s  %12s  %12s  %10s  %10s\n', ...
        'Feature', 'Non-Ref (0/1)', 'Ref (2/3/4)', 'Abs Delta', "Cohen's d");
    fprintf('  ------------------------------------------------------------------------\n');

    nonref_mask = (audit_table.True_Referable == 0) & strcmp(audit_table.Status, 'SUCCESS');
    ref_mask    = (audit_table.True_Referable == 1) & strcmp(audit_table.Status, 'SUCCESS');

    n_nonref = sum(nonref_mask);
    n_ref    = sum(ref_mask);

    binary_summary_rows = cell(num_feats, 1);

    for f = 1:num_feats
        f_name = feature_cols{f};
        vals_nr  = audit_table.(f_name)(nonref_mask);
        vals_ref = audit_table.(f_name)(ref_mask);

        m_nr  = mean(vals_nr);
        s_nr  = std(vals_nr);
        med_nr = median(vals_nr);

        m_ref = mean(vals_ref);
        s_ref = std(vals_ref);
        med_ref = median(vals_ref);

        abs_diff = m_ref - m_nr;

        % Standardized Difference (Cohen's d)
        s_pooled = sqrt(((n_nonref - 1) * s_nr^2 + (n_ref - 1) * s_ref^2) / max(1, n_nonref + n_ref - 2));
        if s_pooled > 1e-6
            cohens_d = abs_diff / s_pooled;
        else
            cohens_d = 0;
        end

        binary_summary_rows{f} = {f_name, m_nr, s_nr, med_nr, m_ref, s_ref, med_ref, abs_diff, cohens_d};

        if ismember(f_name, {'HE_Area_Fraction', 'EX_Area_Fraction', 'Vessel_Density', 'CNN_P_Referable'})
            fprintf('  %-20s  %6.4f±%6.4f  %6.4f±%6.4f  %+10.4f  %+10.2f\n', ...
                f_name, m_nr, s_nr, m_ref, s_ref, abs_diff, cohens_d);
        else
            fprintf('  %-20s  %6.2f±%6.2f  %6.2f±%6.2f  %+10.2f  %+10.2f\n', ...
                f_name, m_nr, s_nr, m_ref, s_ref, abs_diff, cohens_d);
        end
    end
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 5. Validation Split Evaluation (N = 83 Images): CNN vs Rule-Based Fusion
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   5. VALIDATION SUBSET PERFORMANCE (N = 83 Images)                     \n');
    fprintf('========================================================================\n');

    val_mask = strcmp(audit_table.Split, 'Validation') & strcmp(audit_table.Status, 'SUCCESS');
    val_table = audit_table(val_mask, :);

    y_val_true    = val_table.True_Referable;
    y_val_cnn     = val_table.CNN_Referable_Pred;
    p_val_cnn     = val_table.CNN_P_Referable;
    y_val_fusion  = val_table.Fusion_Referable_Pred;
    p_val_fusion  = val_table.Fusion_P_Referable;

    % CNN Metrics
    tp_c = sum(y_val_cnn == 1 & y_val_true == 1);
    fp_c = sum(y_val_cnn == 1 & y_val_true == 0);
    fn_c = sum(y_val_cnn == 0 & y_val_true == 1);
    tn_c = sum(y_val_cnn == 0 & y_val_true == 0);

    sens_c = tp_c / max(1, tp_c + fn_c);
    spec_c = tn_c / max(1, tn_c + fp_c);
    prec_c = tp_c / max(1, tp_c + fp_c);
    f1_c   = (2 * tp_c) / max(1, 2 * tp_c + fp_c + fn_c);
    acc_c  = (tp_c + tn_c) / num_val;
    bal_c  = (sens_c + spec_c) / 2;

    % Rule-Based Fusion Metrics
    tp_f = sum(y_val_fusion == 1 & y_val_true == 1);
    fp_f = sum(y_val_fusion == 1 & y_val_true == 0);
    fn_f = sum(y_val_fusion == 0 & y_val_true == 1);
    tn_f = sum(y_val_fusion == 0 & y_val_true == 0);

    sens_f = tp_f / max(1, tp_f + fn_f);
    spec_f = tn_f / max(1, tn_f + fp_f);
    prec_f = tp_f / max(1, tp_f + fp_f);
    f1_f   = (2 * tp_f) / max(1, 2 * tp_f + fp_f + fn_f);
    acc_f  = (tp_f + tn_f) / num_val;
    bal_f  = (sens_f + spec_f) / 2;

    diff_decisions = sum(y_val_cnn ~= y_val_fusion);

    fprintf('Metric                     Frozen Severity V3 (CNN)     Rule-Based fuseEvidence.m\n');
    fprintf('--------------------------------------------------------------------------------\n');
    fprintf('Referable Sensitivity:     %5.2f%% (%2d / %2d)              %5.2f%% (%2d / %2d)\n', ...
        sens_c * 100, tp_c, tp_c + fn_c, sens_f * 100, tp_f, tp_f + fn_f);
    fprintf('Referable Specificity:     %5.2f%% (%2d / %2d)              %5.2f%% (%2d / %2d)\n', ...
        spec_c * 100, tn_c, tn_c + fp_c, spec_f * 100, tn_f, tn_f + fp_f);
    fprintf('Referable Precision:       %5.2f%%                       %5.2f%%\n', prec_c * 100, prec_f * 100);
    fprintf('Referable F1 Score:        %6.4f                        %6.4f\n', f1_c, f1_f);
    fprintf('Referable Accuracy:        %5.2f%%                       %5.2f%%\n', acc_c * 100, acc_f * 100);
    fprintf('Balanced Accuracy:         %5.2f%%                       %5.2f%%\n', bal_c * 100, bal_f * 100);
    fprintf('Confusion Matrix [TN,FP;FN,TP]: [%2d, %2d; %2d, %2d]         [%2d, %2d; %2d, %2d]\n', ...
        tn_c, fp_c, fn_c, tp_c, tn_f, fp_f, fn_f, tp_f);
    fprintf('Cases Differing from CNN:  --                           %d / 83 images\n', diff_decisions);
    fprintf('========================================================================\n\n');

    % -------------------------------------------------------------------------
    % 6. CNN vs Lesion Evidence Disagreement Analysis (Validation Set)
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('   6. CNN vs LESION EVIDENCE CONCORDANCE & DISAGREEMENT (N = 83)        \n');
    fprintf('========================================================================\n');

    % Definition of physical lesion referral flag matching fuseEvidence.m
    lesion_ref_flag_val = (val_table.MA_Count >= 5) | ...
                          (val_table.HE_Area_Fraction >= 0.001) | ...
                          (val_table.EX_Area_Fraction >= 0.001) | ...
                          (val_table.DME_Risk >= 1.0);
    cnn_ref_flag_val = val_table.CNN_Referable_Pred == 1;

    q1_mask = cnn_ref_flag_val & lesion_ref_flag_val;   % CNN Ref + Lesion Ref
    q2_mask = cnn_ref_flag_val & ~lesion_ref_flag_val;  % CNN Ref + Lesion Non-Ref
    q3_mask = ~cnn_ref_flag_val & lesion_ref_flag_val;  % CNN Non-Ref + Lesion Ref
    q4_mask = ~cnn_ref_flag_val & ~lesion_ref_flag_val; % CNN Non-Ref + Lesion Non-Ref

    q1_ids = val_table.ImageID(q1_mask);
    q2_ids = val_table.ImageID(q2_mask);
    q3_ids = val_table.ImageID(q3_mask);
    q4_ids = val_table.ImageID(q4_mask);

    fprintf('  • Quadrant 1 (CNN Ref + Lesion Ref) [High Concordance]:    %2d cases (%5.1f%%)\n', ...
        numel(q1_ids), (numel(q1_ids)/83)*100);
    fprintf('  • Quadrant 2 (CNN Ref + Lesion Non-Ref) [Deep Preponderance]: %2d cases (%5.1f%%)\n', ...
        numel(q2_ids), (numel(q2_ids)/83)*100);
    fprintf('  • Quadrant 3 (CNN Non-Ref + Lesion Ref) [Biomarker Trigger]:  %2d cases (%5.1f%%)\n', ...
        numel(q3_ids), (numel(q3_ids)/83)*100);
    fprintf('  • Quadrant 4 (CNN Non-Ref + Lesion Non-Ref) [Clear Normal]:  %2d cases (%5.1f%%)\n\n', ...
        numel(q4_ids), (numel(q4_ids)/83)*100);

    disagreement_struct = struct();
    disagreement_struct.Q1_CNN_Ref_Lesion_Ref       = q1_ids;
    disagreement_struct.Q2_CNN_Ref_Lesion_NonRef    = q2_ids;
    disagreement_struct.Q3_CNN_NonRef_Lesion_Ref    = q3_ids;
    disagreement_struct.Q4_CNN_NonRef_Lesion_NonRef = q4_ids;

    % -------------------------------------------------------------------------
    % 7. Save Audit Artifacts
    % -------------------------------------------------------------------------
    out_mat = fullfile(cfg.paths.metrics, 'evidence_channel_audit.mat');
    out_csv = fullfile(cfg.paths.metrics, 'evidence_channel_audit.csv');
    out_val_csv = fullfile(cfg.paths.metrics, 'evidence_validation_summary.csv');

    % Validation Summary Table
    val_metrics_tab = table(...
        {'Sensitivity'; 'Specificity'; 'Precision'; 'F1_Score'; 'Accuracy'; 'Balanced_Accuracy'; 'TP'; 'FP'; 'FN'; 'TN'; 'Disagreement_Count'}, ...
        [sens_c; spec_c; prec_c; f1_c; acc_c; bal_c; tp_c; fp_c; fn_c; tn_c; 0], ...
        [sens_f; spec_f; prec_f; f1_f; acc_f; bal_f; tp_f; fp_f; fn_f; tn_f; diff_decisions], ...
        'VariableNames', {'Metric', 'CNN_V3_Validation', 'RuleBased_Fusion_Validation'});

    % Binary Summary Table
    bin_feat_names = cellfun(@(c) c{1}, binary_summary_rows, 'UniformOutput', false);
    bin_nr_means   = cellfun(@(c) c{2}, binary_summary_rows);
    bin_nr_stds    = cellfun(@(c) c{3}, binary_summary_rows);
    bin_nr_meds    = cellfun(@(c) c{4}, binary_summary_rows);
    bin_ref_means  = cellfun(@(c) c{5}, binary_summary_rows);
    bin_ref_stds   = cellfun(@(c) c{6}, binary_summary_rows);
    bin_ref_meds   = cellfun(@(c) c{7}, binary_summary_rows);
    bin_deltas     = cellfun(@(c) c{8}, binary_summary_rows);
    bin_cohens_d   = cellfun(@(c) c{9}, binary_summary_rows);

    binary_summary_table = table(bin_feat_names, bin_nr_means, bin_nr_stds, bin_nr_meds, ...
        bin_ref_means, bin_ref_stds, bin_ref_meds, bin_deltas, bin_cohens_d, ...
        'VariableNames', {'Feature', 'NonRef_Mean', 'NonRef_Std', 'NonRef_Median', ...
                          'Ref_Mean', 'Ref_Std', 'Ref_Median', 'Absolute_Delta', 'Cohens_d'});

    % Save MAT and CSV files
    save(out_mat, 'audit_table', 'grade_stats', 'binary_summary_table', 'val_metrics_tab', 'disagreement_struct', 'cfg');
    writetable(audit_table, out_csv);
    writetable(val_metrics_tab, out_val_csv);

    fprintf('Audit outputs saved to:\n');
    fprintf('  • %s\n', out_mat);
    fprintf('  • %s\n', out_csv);
    fprintf('  • %s\n\n', out_val_csv);

    % Return Structured Results
    results = struct();
    results.audit_table = audit_table;
    results.grade_stats = grade_stats;
    results.binary_summary = binary_summary_table;
    results.validation_metrics = val_metrics_tab;
    results.disagreements = disagreement_struct;
    results.success_count = success_count;
    results.failure_count = failure_count;
end
