function cam_map = generateGradCAM(img, prep_result, lesion_evidence, fov_mask)
% GENERATEGRADCAM Gradient-Weighted Class Activation Mapping for Retinal DR
% Computes visual attention heatmaps highlighting discriminative pathological zones.
% Uses Deep Learning gradCAM when trained network is available, with biomarker density mapping.
%
% Note: Grad-CAM represents model visual focus and is an interpretability aid.

    cfg = shinra_config();
    if ~isa(img, 'double')
        img = im2double(img);
    end

    [rows, cols, ~] = size(img);
    if nargin < 4 || isempty(fov_mask)
        fov_mask = true(rows, cols);
    end

    model_file_v3 = fullfile(cfg.paths.models_severity, 'SHINRA_SqueezeNet_Severity_Model_V3.mat');
    model_file_legacy = fullfile(cfg.paths.models_severity, 'severity_cnn.mat');
    use_deep_cam = false;

    if exist(model_file_v3, 'file') == 2
        model_file = model_file_v3;
    elseif exist(model_file_legacy, 'file') == 2
        model_file = model_file_legacy;
    else
        model_file = '';
    end

    if ~isempty(model_file)
        try
            loaded = load(model_file);
            if isfield(loaded, 'net')
                net = loaded.net;
                
                % SqueezeNet input resolution
                target_size = [227, 227];
                if isstruct(prep_result) && isfield(prep_result, 'rgb_normalized')
                    norm_img = prep_result.rgb_normalized;
                else
                    norm_img = img;
                end
                
                if max(norm_img(:)) <= 1.05
                    input_tensor = single(imresize(norm_img, target_size) * 255.0);
                else
                    input_tensor = single(imresize(norm_img, target_size));
                end
                
                % Check if gradCAM function exists
                if exist('gradCAM', 'file') == 2
                    % Determine target class
                    if nargin >= 3 && isstruct(lesion_evidence) && isfield(lesion_evidence, 'target_class') && ~isempty(lesion_evidence.target_class)
                        target_class = lesion_evidence.target_class;
                    else
                        % Default to predicted class
                        [pred_cat, ~] = classify(net, input_tensor);
                        target_class = pred_cat;
                    end
                    cam_raw = gradCAM(net, input_tensor, target_class);
                    cam_map = imresize(cam_raw, [rows, cols], 'bicubic');
                    use_deep_cam = true;
                end
            end
        catch
            use_deep_cam = false;
        end
    end

    if ~use_deep_cam
        % Biomarker Lesion Density Map as Interpretable Visual Attention
        combined_lesions = zeros(rows, cols);
        if nargin >= 3 && ~isempty(lesion_evidence)
            if isfield(lesion_evidence, 'ma_mask')
                combined_lesions = combined_lesions + 0.3 * double(lesion_evidence.ma_mask);
            end
            if isfield(lesion_evidence, 'he_mask')
                combined_lesions = combined_lesions + 0.4 * double(lesion_evidence.he_mask);
            end
            if isfield(lesion_evidence, 'ex_mask')
                combined_lesions = combined_lesions + 0.5 * double(lesion_evidence.ex_mask);
            end
        end
        
        % Gaussian diffusion kernel to generate continuous heatmap
        cam_map = imgaussfilt(combined_lesions, 25);
    end

    % Normalize to [0, 1] within FOV
    cam_in_fov = cam_map(fov_mask);
    if ~isempty(cam_in_fov) && max(cam_in_fov) > min(cam_in_fov)
        cam_map = (cam_map - min(cam_in_fov)) / (max(cam_in_fov) - min(cam_in_fov) + 1e-6);
    else
        cam_map = zeros(rows, cols);
    end

    cam_map(~fov_mask) = 0;
    cam_map = max(0, min(1, cam_map));
end
