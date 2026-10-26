function vessel_result = segmentVessels(img, fov_mask, od_mask)
% SEGMENTVESSELS Multi-Scale Retinal Vessel Segmentation and Probability Mapping
% Uses trained Deep U-Net when available, paired with a robust multi-directional
% Gaussian matched filter & morphological enhancement engine.
%
% Returns binary vessel mask, continuous vessel probability map, and vascular morphometry.

    cfg = shinra_config();
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
    if nargin < 3 || isempty(od_mask)
        od_mask = false(rows, cols);
    end

    model_path = fullfile(cfg.paths.models_vessels, 'vessel_unet.mat');
    use_deep_model = false;

    if exist(model_path, 'file') == 2
        try
            loaded = load(model_path);
            if isfield(loaded, 'net')
                net = loaded.net;
                % Deep U-Net inference
                input_tensor = imresize(green_chan, [512, 512]);
                pred_map = predict(net, reshape(input_tensor, [512, 512, 1]));
                if size(pred_map, 3) >= 2
                    prob_map = imresize(pred_map(:, :, 2), [rows, cols], 'bicubic');
                else
                    prob_map = imresize(pred_map, [rows, cols], 'bicubic');
                end
                use_deep_model = true;
            end
        catch
            use_deep_model = false;
        end
    end

    if ~use_deep_model
        % Multi-Directional Gaussian Matched Filter Engine
        % Inverted green channel (vessels are bright on inverted background)
        inv_green = 1.0 - green_chan;
        inv_green(~fov_mask) = 0;
        
        % Background shading subtraction with morphological opening
        se_bg = strel('disk', 15);
        bg_shade = imopen(inv_green, se_bg);
        vessel_enhanced = max(0, inv_green - bg_shade);
        
        % Multi-angle line filtering (12 directions: 0 to 165 degrees, step 15)
        angles = 0:15:165;
        num_angles = numel(angles);
        matched_responses = zeros(rows, cols, num_angles);
        
        line_len = 11;
        for a = 1:num_angles
            se_line = strel('line', line_len, angles(a));
            matched_responses(:, :, a) = imtophat(vessel_enhanced, se_line);
        end
        
        % Maximum response across orientations
        max_response = max(matched_responses, [], 3);
        
        % Normalize probability map [0, 1] within FOV
        resp_fov = max_response(fov_mask);
        if ~isempty(resp_fov) && max(resp_fov) > min(resp_fov)
            p_min = prctile(resp_fov, 2);
            p_max = prctile(resp_fov, 98);
            prob_map = (max_response - p_min) / (p_max - p_min + 1e-6);
            prob_map = max(0, min(1, prob_map));
        else
            prob_map = max_response;
        end
        prob_map(~fov_mask) = 0;
    end

    % Binary Vessel Segmentation
    thresh = cfg.vessels.binarization_threshold;
    binary_vessels = (prob_map >= thresh) & fov_mask;
    
    % Post-processing: remove isolated noise pixels and fill small vessel gaps
    binary_vessels = bwareaopen(binary_vessels, 30);
    binary_vessels = imclose(binary_vessels, strel('disk', 1));

    % Compute Vascular Morphometry
    morpho = vesselMorphometry(binary_vessels, fov_mask, od_mask);

    % Build structured output
    vessel_result = struct();
    vessel_result.binary_mask = binary_vessels;
    vessel_result.probability_map = prob_map;
    vessel_result.morphometry = morpho;
    vessel_result.is_deep_model = use_deep_model;
end
