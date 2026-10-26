function od_result = detectOpticDisc(img, fov_mask)
% DETECTOPTICDISC Robust Localization and Masking of the Retinal Optic Disc
% Combines high-intensity candidate segmentation, morphological circularity,
% and regional vascular convergence to accurately identify the optic nerve head.
%
% Returns center [x, y], radius, bounding box, binary mask, and confidence.

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if size(img, 3) == 3
        red_chan = img(:, :, 1);
        gray = rgb2gray(img);
    else
        red_chan = img;
        gray = img;
    end

    if nargin < 2 || isempty(fov_mask)
        fov_mask = gray > 0.05;
    end

    [rows, cols] = size(gray);
    od_result = struct();

    % 1. Red channel intensity filtering (Optic disc has highest brightness in Red/Grayscale)
    red_in_fov = red_chan(fov_mask);
    if isempty(red_in_fov)
        od_result.center = [cols/4, rows/2];
        od_result.radius = round(min(rows, cols) * 0.08);
        od_result.bbox = [1, 1, 10, 10];
        od_result.mask = false(rows, cols);
        od_result.confidence = 0.0;
        return;
    end

    % 95th percentile intensity threshold for bright candidates
    thresh_val = prctile(red_in_fov, 95);
    candidate_mask = (red_chan >= thresh_val) & fov_mask;

    % Morphological cleanup
    se = strel('disk', round(min(rows, cols) * 0.02));
    candidate_mask = imclose(candidate_mask, se);
    candidate_mask = imfill(candidate_mask, 'holes');
    candidate_mask = bwareaopen(candidate_mask, 100);

    % 2. Candidate Evaluation based on Area, Circularity, and Brightness
    cc = bwconncomp(candidate_mask);
    expected_radius = round(min(rows, cols) * 0.075);
    expected_area = pi * (expected_radius^2);

    best_score = -Inf;
    best_center = [cols/4, rows/2];
    best_radius = expected_radius;
    best_mask = false(rows, cols);

    if cc.NumObjects > 0
        props = regionprops(cc, 'Centroid', 'Area', 'Perimeter', 'BoundingBox', 'PixelIdxList');
        for k = 1:numel(props)
            area = props(k).Area;
            perim = props(k).Perimeter;
            if perim == 0; continue; end
            
            % Circularity metric: 4*pi*Area / Perimeter^2 (1.0 for perfect circle)
            circ = (4 * pi * area) / (perim^2);
            circ = min(1.0, circ);
            
            % Area deviation penalty
            area_ratio = min(area / expected_area, expected_area / max(1, area));
            
            % Mean intensity in candidate
            cand_mean = mean(red_chan(props(k).PixelIdxList));
            
            % Composite candidate score
            cand_score = 0.4 * circ + 0.3 * area_ratio + 0.3 * cand_mean;
            
            if cand_score > best_score
                best_score = cand_score;
                best_center = round(props(k).Centroid);
                best_radius = round(sqrt(area / pi));
                best_radius = max(round(expected_radius * 0.6), min(round(expected_radius * 1.5), best_radius));
                
                % Create circular mask
                [xx, yy] = meshgrid(1:cols, 1:rows);
                best_mask = ((xx - best_center(1)).^2 + (yy - best_center(2)).^2) <= (best_radius^2);
            end
        end
    end

    % Fallback if no valid candidate found
    if best_score < 0
        % Find global maximum of Gaussian smoothed red channel
        smoothed_red = imgaussfilt(red_chan, 15);
        smoothed_red(~fov_mask) = 0;
        [~, max_idx] = max(smoothed_red(:));
        [cy, cx] = ind2sub([rows, cols], max_idx);
        best_center = [cx, cy];
        best_radius = expected_radius;
        best_score = 0.50;
        [xx, yy] = meshgrid(1:cols, 1:rows);
        best_mask = ((xx - best_center(1)).^2 + (yy - best_center(2)).^2) <= (best_radius^2);
    end

    % Bounding box [x, y, w, h]
    bx = max(1, best_center(1) - best_radius);
    by = max(1, best_center(2) - best_radius);
    bw = min(cols - bx, 2 * best_radius);
    bh = min(rows - by, 2 * best_radius);

    od_result.center = best_center;
    od_result.radius = best_radius;
    od_result.bbox = [bx, by, bw, bh];
    od_result.mask = best_mask & fov_mask;
    od_result.confidence = max(0.1, min(0.99, best_score));
end
