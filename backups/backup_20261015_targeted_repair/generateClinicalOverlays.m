function overlays = generateClinicalOverlays(prep_res, od_res, fovea_res, vess_res, ma_res, he_res, ex_res, cam_map, sev_res, dec_res)
% GENERATECLINICALOVERLAYS Reusable Clinical Overlay & Visualization Generator
% Creates standardized RGB overlay images for interactive UI inspection:
% 1. Retinal Anatomy (OD & Fovea)
% 2. Retinal Vessel Architecture
% 3. Individual Biomarkers (MAs, Hemorrhages, Exudates)
% 4. Multi-Lesion Composite
% 5. Grad-CAM Deep Activation Heatmap & Alpha-Blended Overlay

    base_rgb = prep_res.rgb_normalized;
    if max(base_rgb(:)) > 1.05
        base_rgb = base_rgb / 255.0;
    end
    [rows, cols, ~] = size(base_rgb);

    % 1. Retinal Anatomy Overlay (Cyan: Optic Disc, Amber: Macula)
    anatomy_img = visualizeAnatomy(base_rgb, od_res, fovea_res);

    % 2. Retinal Vessel Overlay (Green vessels over darkened fundus)
    vess_mask = vess_res.binary_mask;
    vessel_img = base_rgb * 0.7;
    for c = 1:3
        ch = vessel_img(:, :, c);
        if c == 2
            ch(vess_mask) = 1.0; % Green channel bright
        else
            ch(vess_mask) = ch(vess_mask) * 0.3;
        end
        vessel_img(:, :, c) = ch;
    end

    % 3. Microaneurysms Overlay (Red [1, 0, 0])
    ma_dil = imdilate(ma_res.binary_mask, strel('disk', 2));
    ma_img = base_rgb;
    ch1 = ma_img(:,:,1); ch2 = ma_img(:,:,2); ch3 = ma_img(:,:,3);
    ch1(ma_dil) = 1.0; ch2(ma_dil) = 0.0; ch3(ma_dil) = 0.0;
    ma_img(:,:,1) = ch1; ma_img(:,:,2) = ch2; ma_img(:,:,3) = ch3;

    % 4. Hemorrhages Overlay (Magenta [1, 0, 1])
    he_dil = imdilate(he_res.binary_mask, strel('disk', 1));
    he_img = base_rgb;
    ch1 = he_img(:,:,1); ch2 = he_img(:,:,2); ch3 = he_img(:,:,3);
    ch1(he_dil) = 1.0; ch2(he_dil) = 0.0; ch3(he_dil) = 1.0;
    he_img(:,:,1) = ch1; he_img(:,:,2) = ch2; he_img(:,:,3) = ch3;

    % 5. Exudates Overlay (Yellow [1, 1, 0])
    ex_dil = imdilate(ex_res.binary_mask, strel('disk', 1));
    ex_img = base_rgb;
    ch1 = ex_img(:,:,1); ch2 = ex_img(:,:,2); ch3 = ex_img(:,:,3);
    ch1(ex_dil) = 1.0; ch2(ex_dil) = 1.0; ch3(ex_dil) = 0.0;
    ex_img(:,:,1) = ch1; ex_img(:,:,2) = ch2; ex_img(:,:,3) = ch3;

    % 6. Multi-Lesion Composite Overlay (Red: MA, Magenta: HE, Yellow: EX)
    multi_lesion_img = base_rgb;
    ch1 = multi_lesion_img(:,:,1); ch2 = multi_lesion_img(:,:,2); ch3 = multi_lesion_img(:,:,3);
    % Exudates first
    ch1(ex_dil) = 1.0; ch2(ex_dil) = 1.0; ch3(ex_dil) = 0.0;
    % Hemorrhages second
    ch1(he_dil) = 1.0; ch2(he_dil) = 0.0; ch3(he_dil) = 1.0;
    % MAs third
    ch1(ma_dil) = 1.0; ch2(ma_dil) = 0.0; ch3(ma_dil) = 0.0;
    multi_lesion_img(:,:,1) = ch1; multi_lesion_img(:,:,2) = ch2; multi_lesion_img(:,:,3) = ch3;

    % 7. Grad-CAM Heatmap & Blend
    if isempty(cam_map) || max(cam_map(:)) == 0
        cam_map = zeros(rows, cols);
    end
    cam_norm = (cam_map - min(cam_map(:))) / (max(cam_map(:)) - min(cam_map(:)) + 1e-6);
    cam_norm(~prep_res.fov_mask) = 0;
    
    cam_gray = uint8(cam_norm * 255);
    cam_rgb = ind2rgb(cam_gray, jet(256));
    cam_rgb(repmat(~prep_res.fov_mask, [1, 1, 3])) = 0;

    alpha_val = 0.50;
    gradcam_blend = alpha_val * cam_rgb + (1 - alpha_val) * base_rgb;

    % Assemble Structured Overlays
    overlays = struct();
    overlays.original_rgb = prep_res.original_image;
    overlays.preprocessed_rgb = base_rgb;
    overlays.anatomy = anatomy_img;
    overlays.vessels = vessel_img;
    overlays.vessel_mask = uint8(vess_mask * 255);
    overlays.microaneurysms = ma_img;
    overlays.hemorrhages = he_img;
    overlays.exudates = ex_img;
    overlays.multi_lesions = multi_lesion_img;
    overlays.gradcam_heatmap = cam_rgb;
    overlays.gradcam_overlay = gradcam_blend;
end
