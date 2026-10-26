function enhanced_green = enhanceRetina(img, fov_mask, clip_limit, num_tiles)
% ENHANCERETINA Green Channel Contrast Enhancement for Retinal Lesions & Vessels
% Extracts the green channel, applies Contrast-Limited Adaptive Histogram Equalization
% (CLAHE), and applies median filtering within the retinal FOV.

    if nargin < 3 || isempty(clip_limit)
        clip_limit = 0.015;
    end
    if nargin < 4 || isempty(num_tiles)
        num_tiles = [8, 8];
    end

    if ~isa(img, 'double')
        img = im2double(img);
    end

    % Extract green channel (highest vascular/lesion contrast)
    if size(img, 3) == 3
        g_channel = img(:, :, 2);
    else
        g_channel = img;
    end

    if nargin < 2 || isempty(fov_mask)
        fov_mask = g_channel > 0.05;
    end

    % Apply CLAHE
    clahe_g = adapthisteq(g_channel, ...
        'ClipLimit', clip_limit, ...
        'NumTiles', num_tiles, ...
        'Distribution', 'uniform');

    % Denoise with 3x3 median filter
    denoised_g = medfilt2(clahe_g, [3, 3]);

    % Zero background outside FOV
    denoised_g(~fov_mask) = 0;
    enhanced_green = max(0, min(1, denoised_g));
end
