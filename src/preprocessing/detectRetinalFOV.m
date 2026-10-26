function [fov_mask, bbox, cropped_img] = detectRetinalFOV(img, margin)
% DETECTRETINALFOV Automatic Retinal Field-of-View (FOV) and Border Detection
% Detects the circular/elliptical retinal fundus boundary, removes black borders,
% and returns the binary FOV mask, bounding box [x, y, width, height], and cropped image.

    if nargin < 2 || isempty(margin)
        margin = 0.02; % 2% safety margin around detected mask
    end

    % Convert to grayscale double [0, 1]
    if size(img, 3) == 3
        gray = rgb2gray(img);
    else
        gray = img;
    end
    if ~isa(gray, 'double')
        gray = im2double(gray);
    end

    % Threshold background to detect retinal tissue
    % Fundus background is typically near zero (dark black)
    level = graythresh(gray);
    % Safeguard threshold level
    thresh = max(0.04, min(level * 0.5, 0.15));
    binary = gray > thresh;

    % Fill holes and remove small background noise
    binary = imfill(binary, 'holes');
    binary = bwareaopen(binary, 500);

    % Find largest connected component (the retinal disc)
    cc = bwconncomp(binary);
    if cc.NumObjects > 0
        num_pixels = cellfun(@numel, cc.PixelIdxList);
        [~, max_idx] = max(num_pixels);
        fov_mask = false(size(binary));
        fov_mask(cc.PixelIdxList{max_idx}) = true;
        
        % Smooth boundary with morphological closing
        se = strel('disk', 15);
        fov_mask = imclose(fov_mask, se);
        fov_mask = imfill(fov_mask, 'holes');
    else
        % Fallback: entire image is FOV
        fov_mask = true(size(gray));
    end

    % Compute bounding box of the FOV mask
    props = regionprops(fov_mask, 'BoundingBox');
    if ~isempty(props)
        raw_box = props(1).BoundingBox;
        x = max(1, floor(raw_box(1) - margin * raw_box(3)));
        y = max(1, floor(raw_box(2) - margin * raw_box(4)));
        w = min(size(img, 2) - x, ceil(raw_box(3) * (1 + 2 * margin)));
        h = min(size(img, 1) - y, ceil(raw_box(4) * (1 + 2 * margin)));
        bbox = [x, y, w, h];
    else
        bbox = [1, 1, size(img, 2), size(img, 1)];
    end

    % Crop image and mask to bounding box
    x1 = bbox(1);
    y1 = bbox(2);
    x2 = x1 + bbox(3) - 1;
    y2 = y1 + bbox(4) - 1;
    
    if size(img, 3) == 3
        cropped_img = img(y1:y2, x1:x2, :);
    else
        cropped_img = img(y1:y2, x1:x2);
    end
end
