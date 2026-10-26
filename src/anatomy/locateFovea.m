function fovea_result = locateFovea(img, od_result, fov_mask)
% LOCATEFOVEA Localization of the Retinal Fovea and Macular Region
% Leverages anatomical spatial constraints from the Optic Disc (OD)
% and localized dark pigmentary intensity minima to estimate the Foveal Avascular Zone (FAZ).
%
% Returns fovea center [x, y], macular risk zone mask, and confidence.

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if size(img, 3) == 3
        green_chan = img(:, :, 2);
    else
        green_chan = img;
    end

    [rows, cols] = size(green_chan);
    if nargin < 3 || isempty(fov_mask)
        fov_mask = green_chan > 0.05;
    end

    od_center = od_result.center;
    od_radius = od_result.radius;
    disc_diameter = 2 * od_radius;

    % Retinal Geometry:
    % Determine if OD is located on left or right half of the retina
    if od_center(1) < (cols / 2)
        % OD is on Left (Nasal in Right Eye) -> Fovea is Temporal (Right of OD)
        temporal_sign = +1;
    else
        % OD is on Right (Nasal in Left Eye) -> Fovea is Temporal (Left of OD)
        temporal_sign = -1;
    end

    % Anatomical distance: ~2.5 Disc Diameters temporal, slightly vertical offset
    expected_dist_x = temporal_sign * 2.5 * disc_diameter;
    expected_dist_y = 0.2 * disc_diameter; % Slight inferior/superior tilt

    fovea_prior_x = round(od_center(1) + expected_dist_x);
    fovea_prior_y = round(od_center(2) + expected_dist_y);

    % Constrain search window around anatomical prior
    search_half_w = round(1.0 * disc_diameter);
    search_half_h = round(0.8 * disc_diameter);

    min_x = max(1, fovea_prior_x - search_half_w);
    max_x = min(cols, fovea_prior_x + search_half_w);
    min_y = max(1, fovea_prior_y - search_half_h);
    max_y = min(rows, fovea_prior_y + search_half_h);

    % Smooth green channel (fovea is a smooth dark region)
    smoothed_g = imgaussfilt(green_chan, 10);
    search_roi = smoothed_g(min_y:max_y, min_x:max_x);
    roi_fov = fov_mask(min_y:max_y, min_x:max_x);

    search_roi(~roi_fov) = 1.0; % High value outside FOV so minimum is inside

    if ~isempty(search_roi) && any(roi_fov(:))
        [~, min_idx] = min(search_roi(:));
        [ry, rx] = ind2sub(size(search_roi), min_idx);
        fovea_x = min_x + rx - 1;
        fovea_y = min_y + ry - 1;
        confidence = 0.85;
    else
        fovea_x = max(1, min(cols, fovea_prior_x));
        fovea_y = max(1, min(rows, fovea_prior_y));
        confidence = 0.50;
    end

    % Define Macula region (circular radius ~1.5 DD around fovea)
    macula_radius = round(1.5 * disc_diameter);
    [xx, yy] = meshgrid(1:cols, 1:rows);
    macula_mask = ((xx - fovea_x).^2 + (yy - fovea_y).^2) <= (macula_radius^2);

    fovea_result = struct();
    fovea_result.center = [fovea_x, fovea_y];
    fovea_result.macula_radius = macula_radius;
    fovea_result.macula_mask = macula_mask & fov_mask;
    fovea_result.confidence = confidence;
end
