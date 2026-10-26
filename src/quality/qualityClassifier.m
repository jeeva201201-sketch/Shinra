function [quality_class, quality_report] = qualityClassifier(img, fov_mask, custom_thresholds)
% QUALITYCLASSIFIER Tri-State Retinal Quality Gate (GOOD / BORDERLINE / UNGRADABLE)
% Evaluates image quality features against calibrated decision thresholds.
% Returns categorical quality grade and comprehensive diagnostic defect analysis.

    cfg = shinra_config();
    if nargin < 3 || isempty(custom_thresholds)
        thresh_good = cfg.quality.threshold_good;
        thresh_borderline = cfg.quality.threshold_borderline;
    else
        thresh_good = custom_thresholds.threshold_good;
        thresh_borderline = custom_thresholds.threshold_borderline;
    end

    if nargin < 2 || isempty(fov_mask)
        [fov_mask, ~, ~] = detectRetinalFOV(img);
    end

    % Extract quantitative features
    feats = qualityFeatures(img, fov_mask);
    [score, subscores] = qualityScore(feats);

    % Diagnostic Defect Flags
    defects = struct();
    defects.is_blurry = feats.laplacian_variance < cfg.quality.laplacian_blur_min;
    defects.is_underexposed = feats.dark_pixel_ratio > cfg.quality.underexposure_max || feats.mean_intensity < 0.12;
    defects.is_overexposed = feats.bright_pixel_ratio > cfg.quality.overexposure_max;
    defects.is_incomplete_fov = feats.fov_coverage_ratio < cfg.quality.fov_coverage_min;
    defects.is_low_contrast = feats.rms_contrast < 0.15;
    defects.is_noisy = feats.noise_estimate > 0.08;

    % Critical failure conditions (Immediate UNGRADABLE regardless of composite score)
    critical_defect = (defects.is_incomplete_fov && feats.fov_coverage_ratio < 0.25) || ...
                      (defects.is_underexposed && feats.mean_intensity < 0.06) || ...
                      (feats.laplacian_variance < 5.0 && defects.is_blurry);

    % Categorical Decision
    if critical_defect || score < thresh_borderline
        quality_class = 'UNGRADABLE';
    elseif score < thresh_good
        quality_class = 'BORDERLINE';
    else
        quality_class = 'GOOD';
    end

    % Build structured diagnostic report
    quality_report = struct();
    quality_report.quality_class = quality_class;
    quality_report.score = score;
    quality_report.subscores = subscores;
    quality_report.features = feats;
    quality_report.defects = defects;
    quality_report.recapture_guidance = recaptureGuidance(quality_class, defects, score);
end
