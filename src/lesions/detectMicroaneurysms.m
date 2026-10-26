function ma_result = detectMicroaneurysms(img, fov_mask, vessel_mask, od_mask)
% DETECTMICROANEURYSMS Multi-Scale Retinal Microaneurysm (MA) Candidate Detection
% Implements multi-scale mathematical morphology (Top-Hat / Black-Hat filtering)
% with strict vascular and optic-disc exclusion and geometric shape filtering.
%
% Returns binary MA mask, MA count, total area, coordinates [x, y], and confidence.

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

    ma_result = struct();

    % Dilate vessel mask and optic disc mask to avoid edge false alarms
    vessel_exclusion = imdilate(vessel_mask, strel('disk', 2));
    od_exclusion     = imdilate(od_mask, strel('disk', 5));
    valid_roi = fov_mask & ~vessel_exclusion & ~od_exclusion;

    % Invert green channel (dark MAs become bright peaks)
    inv_g = 1.0 - green_chan;
    inv_g(~fov_mask) = 0;

    % Multi-Scale Circular Top-Hat Transformation
    % Radii corresponding to small focal microaneurysms (2, 3, 4, 5 pixels)
    scales = [2, 3, 4, 5];
    multi_scale_response = zeros(rows, cols);

    for s = 1:numel(scales)
        se = strel('disk', scales(s));
        top_hat = imtophat(inv_g, se);
        multi_scale_response = max(multi_scale_response, top_hat);
    end

    % Zero outside valid ROI
    multi_scale_response(~valid_roi) = 0;

    % Adaptive Candidate Thresholding
    resp_in_roi = multi_scale_response(valid_roi);
    if isempty(resp_in_roi) || max(resp_in_roi) == 0
        ma_result.binary_mask = false(rows, cols);
        ma_result.count = 0;
        ma_result.total_area = 0;
        ma_result.coordinates = [];
        ma_result.confidence = 0.0;
        return;
    end

    % 97th percentile threshold for candidate isolation
    cand_thresh = prctile(resp_in_roi, 97);
    cand_binary = (multi_scale_response >= cand_thresh) & valid_roi;

    % Morphological Size & Shape Filtering
    % Microaneurysms: Area between 4 and 90 pixels, circularity >= 0.50
    cc = bwconncomp(cand_binary);
    final_ma_mask = false(rows, cols);
    centroids = [];
    valid_count = 0;

    if cc.NumObjects > 0
        props = regionprops(cc, 'Area', 'Perimeter', 'Centroid', 'PixelIdxList', 'Eccentricity');
        for k = 1:numel(props)
            area = props(k).Area;
            perim = props(k).Perimeter;
            ecc = props(k).Eccentricity;

            if area >= 3 && area <= 90 && perim > 0
                circ = (4 * pi * area) / (perim^2);
                % Round, compact shape criterion
                if circ >= 0.45 && ecc <= 0.85
                    valid_count = valid_count + 1;
                    final_ma_mask(props(k).PixelIdxList) = true;
                    centroids = [centroids; round(props(k).Centroid)];
                end
            end
        end
    end

    ma_result.binary_mask = final_ma_mask;
    ma_result.count = valid_count;
    ma_result.total_area = sum(final_ma_mask(:));
    ma_result.coordinates = centroids;
    ma_result.confidence = min(1.0, 0.5 + 0.05 * valid_count);
end
