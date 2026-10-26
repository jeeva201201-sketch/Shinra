function feats = qualityFeatures(img, fov_mask)
% QUALITYFEATURES Extract Quantitative Retinal Image Quality Features
% Computes mathematical biomarkers of image quality including:
% - Focus/sharpness: Laplacian variance, Tenengrad gradient energy
% - Exposure & Illumination: Mean intensity, standard deviation, dark/bright saturation
% - Retinal Field Coverage: Mask area ratio, circularity
% - Contrast: RMS contrast and Michelson contrast within FOV
% - Noise/Artifacts: High-frequency residual variance

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if size(img, 3) == 3
        gray = rgb2gray(img);
        g_chan = img(:, :, 2);
    else
        gray = img;
        g_chan = img;
    end

    if nargin < 2 || isempty(fov_mask)
        fov_mask = gray > 0.05;
    end

    feats = struct();
    retinal_pixels = gray(fov_mask);
    num_retinal = numel(retinal_pixels);
    total_pixels = numel(gray);

    if num_retinal == 0
        % Completely black or empty image
        feats.laplacian_variance = 0;
        feats.tenengrad_sharpness = 0;
        feats.mean_intensity = 0;
        feats.std_intensity = 0;
        feats.dark_pixel_ratio = 1.0;
        feats.bright_pixel_ratio = 0;
        feats.fov_coverage_ratio = 0;
        feats.rms_contrast = 0;
        feats.entropy_val = 0;
        feats.noise_estimate = 1.0;
        return;
    end

    % 1. Sharpness / Focus Metrics
    % A. Laplacian Variance (measure of high frequency edge focus)
    lap_filter = [0, 1, 0; 1, -4, 1; 0, 1, 0];
    lap_response = imfilter(g_chan, lap_filter, 'replicate');
    lap_in_fov = lap_response(fov_mask);
    feats.laplacian_variance = var(lap_in_fov) * 1e4; % Scaled for numerical readability

    % B. Tenengrad Gradient Energy (Sobel gradient magnitude squared)
    [gx, gy] = imgradientxy(g_chan, 'Sobel');
    grad_mag = sqrt(gx.^2 + gy.^2);
    grad_in_fov = grad_mag(fov_mask);
    feats.tenengrad_sharpness = mean(grad_in_fov.^2) * 1e3;

    % 2. Exposure & Illumination Metrics
    feats.mean_intensity = mean(retinal_pixels);
    feats.std_intensity = std(retinal_pixels);
    
    % Underexposure (fraction of extremely dark retinal pixels < 0.08)
    feats.dark_pixel_ratio = sum(retinal_pixels < 0.08) / num_retinal;
    
    % Overexposure / Glare saturation (fraction of bright pixels > 0.92)
    feats.bright_pixel_ratio = sum(retinal_pixels > 0.92) / num_retinal;

    % 3. Retinal Field Coverage
    feats.fov_coverage_ratio = num_retinal / total_pixels;

    % 4. Contrast Metrics
    % RMS Contrast within FOV
    feats.rms_contrast = feats.std_intensity / (feats.mean_intensity + 1e-6);
    
    % Retinal Entropy
    feats.entropy_val = entropy(gray(fov_mask));

    % 5. High-Frequency Noise / Artifact Estimation
    % Difference between original and median filtered image
    med_diff = abs(g_chan - medfilt2(g_chan, [3, 3]));
    feats.noise_estimate = mean(med_diff(fov_mask));
end
