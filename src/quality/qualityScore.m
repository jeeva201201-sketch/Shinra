function [score, subscores] = qualityScore(feats)
% QUALITYSCORE Compute Normalized Multi-Factor Retinal Quality Index [0 - 100]
% Combines sharpness, illumination balance, exposure fidelity, contrast,
% and retinal field coverage into an interpretable composite quality score.

    cfg = shinra_config();
    w = cfg.quality.weights;
    subscores = struct();

    % 1. Sharpness Subscore [0 - 100]
    % Sigmoidal transformation of Laplacian variance & Tenengrad
    lap_norm = min(1.0, feats.laplacian_variance / 50.0);
    ten_norm = min(1.0, feats.tenengrad_sharpness / 15.0);
    subscores.sharpness = 0.6 * lap_norm + 0.4 * ten_norm;
    subscores.sharpness = 100 * (1 / (1 + exp(-6 * (subscores.sharpness - 0.4))));

    % 2. Illumination Subscore [0 - 100]
    % Optimal mean intensity is around 0.35 - 0.65
    illum_dev = abs(feats.mean_intensity - 0.45);
    subscores.illumination = 100 * max(0, 1.0 - (illum_dev / 0.35)^2);

    % 3. Exposure Subscore [0 - 100]
    % Penalize severe dark or saturated glare pixels
    dark_penalty = min(1.0, feats.dark_pixel_ratio / 0.40);
    bright_penalty = min(1.0, feats.bright_pixel_ratio / 0.15);
    subscores.exposure = 100 * max(0, 1.0 - 0.6 * dark_penalty - 0.4 * bright_penalty);

    % 4. Contrast Subscore [0 - 100]
    % RMS contrast typically 0.25 - 0.65
    contrast_norm = min(1.0, feats.rms_contrast / 0.50);
    subscores.contrast = 100 * contrast_norm;

    % 5. Retinal Field Coverage Subscore [0 - 100]
    % Adequate retinal coverage is >= 0.50 of total frame
    fov_norm = min(1.0, feats.fov_coverage_ratio / 0.65);
    subscores.fov_coverage = 100 * fov_norm;

    % Weighted Composite Quality Score
    score = w.sharpness    * subscores.sharpness + ...
            w.illumination  * subscores.illumination + ...
            w.exposure      * subscores.exposure + ...
            w.contrast      * subscores.contrast + ...
            w.fov_coverage  * subscores.fov_coverage;

    score = max(0, min(100, score));
end
