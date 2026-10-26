function he_result = detectHemorrhages(img, fov_mask, vessel_mask, od_mask)
% DETECTHEMORRHAGES Segmentation and Area Quantification of Retinal Hemorrhages
% Detects blot, dot, and flame-shaped intraretinal hemorrhages using multi-scale
% morphological Black-Hat filtering, large-vessel suppression, and regional contrast analysis.
%
% Returns binary hemorrhage mask, region count, total area fraction, and confidence.

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if size(img, 3) == 3
        green_chan = img(:, :, 2);
    else
        green_chan = img;
    end

    [rows, cols] = size(green_chan);
    if nargin < 2 || isempty(fov_mask)
        fov_mask = green_chan > 0.05;
    end
    if nargin < 3 || isempty(vessel_mask)
        vessel_mask = false(rows, cols);
    end
    if nargin < 4 || isempty(od_mask)
        od_mask = false(rows, cols);
    end

    he_result = struct();

    % Exclude large vessel trunks and optic disc
    thick_vessels = imopen(vessel_mask, strel('disk', 2));
    exclusion_zone = imdilate(thick_vessels, strel('disk', 3)) | imdilate(od_mask, strel('disk', 8));
    valid_roi = fov_mask & ~exclusion_zone;

    % Multi-Scale Black-Hat Transform on Green Channel
    scales = [6, 10, 15, 22];
    multi_scale_he = zeros(rows, cols);

    for s = 1:numel(scales)
        se = strel('disk', scales(s));
        bh = imbothat(green_chan, se);
        multi_scale_he = max(multi_scale_he, bh);
    end

    multi_scale_he(~valid_roi) = 0;

    % Regional Candidate Isolation
    he_in_roi = multi_scale_he(valid_roi);
    if isempty(he_in_roi) || max(he_in_roi) == 0
        he_result.binary_mask = false(rows, cols);
        he_result.count = 0;
        he_result.total_area = 0;
        he_result.area_fraction = 0;
        he_result.confidence = 0.0;
        return;
    end

    % Threshold at 96th percentile
    he_thresh = prctile(he_in_roi, 96);
    he_candidates = (multi_scale_he >= he_thresh) & valid_roi;

    % Filter candidates (larger than microaneurysms, area >= 80 pixels)
    cc = bwconncomp(he_candidates);
    final_he_mask = false(rows, cols);
    valid_count = 0;

    if cc.NumObjects > 0
        props = regionprops(cc, 'Area', 'PixelIdxList', 'Eccentricity');
        for k = 1:numel(props)
            area = props(k).Area;
            if area >= 60 && area <= 8000
                valid_count = valid_count + 1;
                final_he_mask(props(k).PixelIdxList) = true;
            end
        end
    end

    % Smooth lesion boundaries
    final_he_mask = imclose(final_he_mask, strel('disk', 2));

    retinal_area = max(1, sum(fov_mask(:)));
    total_area = sum(final_he_mask(:));

    he_result.binary_mask = final_he_mask;
    he_result.count = valid_count;
    he_result.total_area = total_area;
    he_result.area_fraction = total_area / retinal_area;
    he_result.confidence = min(1.0, 0.4 + 0.1 * min(6, valid_count));
end
