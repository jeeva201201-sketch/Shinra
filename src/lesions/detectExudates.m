function ex_result = detectExudates(img, fov_mask, od_mask, fovea_result)
% DETECTEXUDATES Segmentation of Bright Retinal Exudates with Strict Optic Disc Suppression
% Isolates hard/soft lipid exudates using multi-scale morphological Top-Hat filtering,
% dynamic thresholding, and anatomical optic disc masking to prevent false positives.
%
% Returns binary exudate mask, region count, total area, DME proximity index, and confidence.

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if size(img, 3) == 3
        gray_chan = rgb2gray(img);
        green_chan = img(:, :, 2);
    else
        gray_chan = img;
        green_chan = img;
    end

    [rows, cols] = size(gray_chan);
    if nargin < 2 || isempty(fov_mask)
        fov_mask = gray_chan > 0.05;
    end
    if nargin < 3 || isempty(od_mask)
        od_mask = false(rows, cols);
    end

    ex_result = struct();

    % STRICT OPTIC DISC SUPPRESSION: Dilate OD mask with safety margin
    od_dilated = imdilate(od_mask, strel('disk', 12));
    valid_roi = fov_mask & ~od_dilated;

    % Multi-Scale Top-Hat Transform on Grayscale/Green (Bright Lesion Enhancement)
    scales = [3, 7, 12, 18];
    multi_scale_ex = zeros(rows, cols);

    for s = 1:numel(scales)
        se = strel('disk', scales(s));
        th = imtophat(green_chan, se);
        multi_scale_ex = max(multi_scale_ex, th);
    end

    multi_scale_ex(~valid_roi) = 0;

    % Dynamic Thresholding
    ex_in_roi = multi_scale_ex(valid_roi);
    if isempty(ex_in_roi) || max(ex_in_roi) == 0
        ex_result.binary_mask = false(rows, cols);
        ex_result.count = 0;
        ex_result.total_area = 0;
        ex_result.area_fraction = 0;
        ex_result.macular_proximity_risk = 0;
        ex_result.confidence = 0.0;
        return;
    end

    % 97th percentile threshold for bright candidate spots
    ex_thresh = prctile(ex_in_roi, 97);
    ex_candidates = (multi_scale_ex >= ex_thresh) & valid_roi;

    % Intensity check: Hard exudates have high absolute brightness
    mean_bg = mean(green_chan(valid_roi));
    intensity_gate = green_chan > (mean_bg + 0.08);
    ex_candidates = ex_candidates & intensity_gate;

    % Connected Components Filtering
    cc = bwconncomp(ex_candidates);
    final_ex_mask = false(rows, cols);
    valid_count = 0;

    if cc.NumObjects > 0
        props = regionprops(cc, 'Area', 'PixelIdxList');
        for k = 1:numel(props)
            area = props(k).Area;
            if area >= 10 && area <= 6000
                valid_count = valid_count + 1;
                final_ex_mask(props(k).PixelIdxList) = true;
            end
        end
    end

    retinal_area = max(1, sum(fov_mask(:)));
    total_area = sum(final_ex_mask(:));

    % Macular Proximity Risk (DME Assessment)
    % Higher risk if hard exudates are within 1-2 disc diameters of the fovea
    macular_risk = 0.0;
    if nargin >= 4 && isfield(fovea_result, 'center') && valid_count > 0
        fx = fovea_result.center(1);
        fy = fovea_result.center(2);
        rad = fovea_result.macula_radius;
        
        [ey, ex] = find(final_ex_mask);
        dist_to_fovea = sqrt((ex - fx).^2 + (ey - fy).^2);
        min_dist = min(dist_to_fovea);
        
        if min_dist <= rad * 0.5
            macular_risk = 2.0; % High DME Risk: Exudates within 500um of fovea center
        elseif min_dist <= rad
            macular_risk = 1.0; % Moderate DME Risk: Exudates within 1 disc diameter
        else
            macular_risk = 0.2; % Mild/Peripheral Exudates
        end
    end

    ex_result.binary_mask = final_ex_mask;
    ex_result.count = valid_count;
    ex_result.total_area = total_area;
    ex_result.area_fraction = total_area / retinal_area;
    ex_result.macular_proximity_risk = macular_risk;
    ex_result.confidence = min(1.0, 0.45 + 0.1 * min(5, valid_count));
end
