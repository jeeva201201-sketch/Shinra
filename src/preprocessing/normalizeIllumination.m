function norm_img = normalizeIllumination(img, fov_mask, sigma)
% NORMALIZEILLUMINATION Retinal Illumination Correction and Shading Compensation
% Estimates low-frequency non-uniform background illumination via large Gaussian
% filtering and normalizes local intensities within the retinal FOV.

    if nargin < 3 || isempty(sigma)
        sigma = 30;
    end

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if nargin < 2 || isempty(fov_mask)
        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end
        fov_mask = gray > 0.05;
    end

    norm_img = zeros(size(img));
    num_channels = size(img, 3);

    for c = 1:num_channels
        ch = img(:, :, c);
        
        % Background estimation via Gaussian smoothing
        % Replace background outside FOV with mean inside FOV to avoid boundary bleed
        mean_val = mean(ch(fov_mask));
        if isnan(mean_val)
            mean_val = 0.5;
        end
        ch_filled = ch;
        ch_filled(~fov_mask) = mean_val;
        
        % Gaussian low-pass filter
        bg_illum = imgaussfilt(ch_filled, sigma);
        
        % Shading compensation: channel / background * target_mean
        bg_illum(bg_illum < 1e-4) = 1e-4;
        comp_ch = (ch ./ bg_illum) * mean_val;
        
        % Constrain to [0, 1] within FOV, zero outside
        comp_ch(~fov_mask) = 0;
        comp_ch = max(0, min(1, comp_ch));
        
        norm_img(:, :, c) = comp_ch;
    end
end
