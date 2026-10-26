function comparison_report = compareVesselSegmentation(varargin)
% COMPAREVESSELSEGMENTATION Comparative Evaluation: Matched-Filter vs Deep U-Net
% Loads the evaluation results of:
% A) Non-Deep-Learning Matched-Filter Baseline (results/metrics/drive_matched_filter_baseline.mat)
% B) Trained Deep U-Net (results/metrics/drive_unet_evaluation.mat)
%
% Compares all 6 quantitative metrics:
% 1. Dice Similarity Coefficient
% 2. Jaccard Index (IoU)
% 3. Vessel Sensitivity (Recall)
% 4. Specificity (Non-Vessel discrimination)
% 5. Precision (PPV)
% 6. Overall Pixel Accuracy
%
% Outputs structured side-by-side comparison matrix and saves to results/metrics/.

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('   SHINRA: VESSEL SEGMENTATION METHOD COMPARISON (DRIVE BENCHMARK)      \n');
    fprintf('========================================================================\n\n');

    fprintf('========================================================================\n');
    fprintf('  [METHODOLOGICAL NOTICE / DATA-LEAKAGE DISCLOSURE]\n');
    fprintf('  Both methods are evaluated on the same 20 DRIVE images; however,\n');
    fprintf('  the U-Net was trained on 16 of these images, so this is not an\n');
    fprintf('  independent test comparison.\n');
    fprintf('  The 4-image held-out validation subset is reported separately below.\n');
    fprintf('========================================================================\n\n');

    baseline_file = fullfile(cfg.paths.metrics, 'drive_matched_filter_baseline.mat');
    unet_file     = fullfile(cfg.paths.metrics, 'drive_unet_evaluation.mat');

    if exist(baseline_file, 'file') ~= 2
        error('SHINRA:BaselineNotFound', 'Baseline file not found at: %s\nPlease run evaluateVesselMatchedFilter first.', baseline_file);
    end
    if exist(unet_file, 'file') ~= 2
        error('SHINRA:UNetEvalNotFound', 'U-Net evaluation file not found at: %s\nPlease run evaluateVesselUNet first.', unet_file);
    end

    loaded_base = load(baseline_file);
    loaded_unet = load(unet_file);

    base_table = loaded_base.per_image_table;
    unet_table = loaded_unet.per_image_table;

    assert(height(base_table) == 20, 'Baseline table must contain exactly 20 images.');
    assert(height(unet_table) == 20, 'U-Net table must contain exactly 20 images.');

    % Extract Metrics (20 Images)
    metrics_list = {'Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy'};
    num_metrics = numel(metrics_list);

    base_means = zeros(num_metrics, 1);
    base_stds  = zeros(num_metrics, 1);
    unet_means = zeros(num_metrics, 1);
    unet_stds  = zeros(num_metrics, 1);
    delta_abs  = zeros(num_metrics, 1);
    delta_pct  = zeros(num_metrics, 1);

    for m = 1:num_metrics
        m_name = metrics_list{m};
        val_base = base_table.(m_name);
        val_unet = unet_table.(m_name);

        base_means(m) = mean(val_base);
        base_stds(m)  = std(val_base);
        unet_means(m) = mean(val_unet);
        unet_stds(m)  = std(val_unet);

        delta_abs(m)  = unet_means(m) - base_means(m);
        delta_pct(m)  = (delta_abs(m) / max(1e-6, base_means(m))) * 100;
    end

    % -------------------------------------------------------------------------
    % Print Comprehensive 20-Image Comparison Table
    % -------------------------------------------------------------------------
    fprintf('1. 20-IMAGE DRIVE BENCHMARK COMPARISON (16 Training + 4 Held-Out Validation):\n');
    fprintf('=========================================================================================\n');
    fprintf('  Metric               Matched Filter Baseline      Trained Deep U-Net       Delta (Abs / %%)    \n');
    fprintf('-----------------------------------------------------------------------------------------\n');
    for m = 1:num_metrics
        m_name = metrics_list{m};
        if ismember(m_name, {'Sensitivity', 'Specificity', 'Precision', 'Accuracy'})
            fprintf('  %-18s  %5.2f%% ± %5.2f%%          %5.2f%% ± %5.2f%%         %+5.2f%% (%+5.1f%%)\n', ...
                m_name, base_means(m)*100, base_stds(m)*100, unet_means(m)*100, unet_stds(m)*100, ...
                delta_abs(m)*100, delta_pct(m));
        else
            fprintf('  %-18s  %6.4f ± %6.4f          %6.4f ± %6.4f         %+6.4f (%+5.1f%%)\n', ...
                m_name, base_means(m), base_stds(m), unet_means(m), unet_stds(m), ...
                delta_abs(m), delta_pct(m));
        end
    end
    fprintf('=========================================================================================\n\n');

    % -------------------------------------------------------------------------
    % 2. Held-Out Validation Subset Only Comparison (N=4, Disjoint from Training)
    % -------------------------------------------------------------------------
    if ismember('IsValidation', unet_table.Properties.VariableNames)
        val_mask = unet_table.IsValidation;
        fprintf('2. HELD-OUT VALIDATION SUBSET COMPARISON (N=4, Disjoint from Training):\n');
        fprintf('=========================================================================================\n');
        fprintf('  Metric               Matched Filter (Val)       U-Net (Val)             Delta (Abs)\n');
        fprintf('-----------------------------------------------------------------------------------------\n');
        for m = 1:num_metrics
            m_name = metrics_list{m};
            vb = mean(base_table.(m_name)(val_mask));
            vu = mean(unet_table.(m_name)(val_mask));
            dv = vu - vb;
            if ismember(m_name, {'Sensitivity', 'Specificity', 'Precision', 'Accuracy'})
                fprintf('  %-18s  %5.2f%%                     %5.2f%%                  %+5.2f%%\n', ...
                    m_name, vb*100, vu*100, dv*100);
            else
                fprintf('  %-18s  %6.4f                     %6.4f                  %+6.4f\n', ...
                    m_name, vb, vu, dv);
            end
        end
        fprintf('=========================================================================================\n\n');
    end

    % -------------------------------------------------------------------------
    % Summary Findings
    % -------------------------------------------------------------------------
    dice_improved = delta_abs(1) > 0;
    iou_improved  = delta_abs(2) > 0;

    fprintf('Summary Metrics Analysis:\n');
    fprintf('  • 20-Image Benchmark Dice:   Matched Filter = %6.4f -> U-Net = %6.4f (Delta = %+6.4f)\n', ...
        base_means(1), unet_means(1), delta_abs(1));
    fprintf('  • 20-Image Benchmark IoU:    Matched Filter = %6.4f -> U-Net = %6.4f (Delta = %+6.4f)\n', ...
        base_means(2), unet_means(2), delta_abs(2));
    fprintf('  • 20-Image Sensitivity:      Matched Filter = %5.2f%% -> U-Net = %5.2f%% (Delta = %+5.2f%%)\n', ...
        base_means(3)*100, unet_means(3)*100, delta_abs(3)*100);
    fprintf('  • 20-Image Specificity:      Matched Filter = %5.2f%% -> U-Net = %5.2f%% (Delta = %+5.2f%%)\n', ...
        base_means(4)*100, unet_means(4)*100, delta_abs(4)*100);
    fprintf('  • 20-Image Precision:        Matched Filter = %5.2f%% -> U-Net = %5.2f%% (Delta = %+5.2f%%)\n', ...
        base_means(5)*100, unet_means(5)*100, delta_abs(5)*100);
    fprintf('  • 20-Image Pixel Accuracy:   Matched Filter = %5.2f%% -> U-Net = %5.2f%% (Delta = %+5.2f%%)\n\n', ...
        base_means(6)*100, unet_means(6)*100, delta_abs(6)*100);

    % -------------------------------------------------------------------------
    % Save Comparison Artifacts
    % -------------------------------------------------------------------------
    comparison_table = table(metrics_list', base_means, base_stds, unet_means, unet_stds, delta_abs, delta_pct, ...
        'VariableNames', {'Metric', 'Baseline_Mean', 'Baseline_Std', 'UNet_Mean', 'UNet_Std', 'Delta_Absolute', 'Delta_Percent'});

    out_mat = fullfile(cfg.paths.metrics, 'drive_vessel_method_comparison.mat');
    out_csv = fullfile(cfg.paths.metrics, 'drive_vessel_method_comparison.csv');

    save(out_mat, 'comparison_table', 'base_table', 'unet_table', 'cfg');
    writetable(comparison_table, out_csv);

    fprintf('Comparison results saved to:\n  • %s\n  • %s\n\n', out_mat, out_csv);

    comparison_report = struct();
    comparison_report.table = comparison_table;
    comparison_report.dice_improved = dice_improved;
    comparison_report.iou_improved = iou_improved;
end
