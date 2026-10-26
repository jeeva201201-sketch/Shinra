function prep_result = preprocessRetina(input_img, target_size)
% PREPROCESSRETINA Comprehensive Retinal Fundus Preprocessing Pipeline
% Executes:
% 1. Image loading / validation
% 2. Retinal FOV boundary detection and black-border removal
% 3. Illumination compensation and shading correction
% 4. Multi-channel CLAHE and Green-channel enhancement
% 5. Standardized spatial resizing (default 512x512)
%
% Returns a structured result containing processed RGB, enhanced green channel,
% FOV mask, bounding box, and preprocessing metadata.

    cfg = shinra_config();
    if nargin < 2 || isempty(target_size)
        target_size = cfg.preprocessing.target_size;
    end

    % 1. Load image if string path is passed
    if ischar(input_img) || isstring(input_img)
        img_path = char(input_img);
        if exist(img_path, 'file') ~= 2
            error('Retinal image file not found: %s', img_path);
        end
        [orig_img, ~, ~] = imread(img_path);
    else
        orig_img = input_img;
        img_path = 'in_memory_image';
    end

    % Ensure 3-channel RGB double
    if size(orig_img, 3) == 1
        orig_rgb = repmat(orig_img, [1, 1, 3]);
    else
        orig_rgb = orig_img(:, :, 1:3);
    end
    orig_double = im2double(orig_rgb);

    % 2. Retinal FOV Detection & Cropping
    [raw_fov_mask, bbox, cropped_rgb] = detectRetinalFOV(orig_double, cfg.preprocessing.crop_margin);

    % Crop FOV mask to match cropped RGB
    x1 = bbox(1); y1 = bbox(2);
    x2 = x1 + bbox(3) - 1; y2 = y1 + bbox(4) - 1;
    cropped_fov_mask = raw_fov_mask(y1:y2, x1:x2);

    % 3. Standardized Resizing
    resized_rgb = imresize(cropped_rgb, target_size, 'bicubic');
    resized_fov_mask = imresize(cropped_fov_mask, target_size, 'nearest') > 0.5;

    % 4. Illumination Normalization
    norm_rgb = normalizeIllumination(resized_rgb, resized_fov_mask, cfg.preprocessing.gauss_sigma_illum);

    % 5. Green Channel CLAHE & Enhancement
    enhanced_green = enhanceRetina(norm_rgb, resized_fov_mask, ...
        cfg.preprocessing.clahe_clip_limit, cfg.preprocessing.clahe_num_tiles);

    % Build structured output
    prep_result = struct();
    prep_result.original_image = orig_rgb;
    prep_result.image_path = img_path;
    prep_result.original_size = size(orig_rgb);
    prep_result.bbox = bbox;
    prep_result.fov_mask = resized_fov_mask;
    prep_result.rgb_normalized = norm_rgb;
    prep_result.enhanced_green = enhanced_green;
    prep_result.target_size = target_size;
end
