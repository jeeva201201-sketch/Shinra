function overlays = generateClinicalOverlays(prep_res, od_res, fovea_res, vess_res, ma_res, he_res, ex_res, cam_map, sev_res, dec_res)
% GENERATECLINICALOVERLAYS Reusable Clinical Overlay & Visualization Generator
% Creates standardized RGB overlay images for interactive UI inspection and PDF reporting:
% 1. Original Retinal Fundus Photograph
% 2. Preprocessed & CLAHE Contrast Enhanced
% 3. Retinal Anatomy (OD & Fovea)
% 4. Retinal Vessel Architecture (Deep U-Net)
% 5. Microaneurysms Detection Overlay (High-Contrast Red with Target Rings)
% 6. Hemorrhages Detection Overlay (Magenta with Delineation Outlines)
% 7. Exudates & DME Risk Map (Yellow with Foveal Risk Zones)
% 8. Multi-Lesion Clinical Composite (MA, HE, EX)
% 9. Grad-CAM Deep Activation Heatmap (Jet Colormap)
% 10. Grad-CAM Clinical Explainability Blend (Activation-Weighted Alpha Blend)

    base_rgb = prep_res.rgb_normalized;
    if max(base_rgb(:)) > 1.05
        base_rgb = base_rgb / 255.0;
    end
    [rows, cols, ~] = size(base_rgb);

    % 1. Original RGB image (standardized uint8)
    if isfield(prep_res, 'original_image') && ~isempty(prep_res.original_image)
        orig_img = prep_res.original_image;
        if isa(orig_img, 'double')
            orig_rgb = uint8(round(min(max(orig_img, 0), 1) * 255));
        else
            orig_rgb = orig_img;
        end
    else
        orig_rgb = uint8(round(base_rgb * 255));
    end

    % 2. Preprocessed & CLAHE Contrast Enhanced (Green Channel Priority)
    clahe_rgb = base_rgb;
    if isfield(prep_res, 'enhanced_green') && ~isempty(prep_res.enhanced_green)
        eg = prep_res.enhanced_green;
        if max(eg(:)) > 1.05
            eg = eg / 255.0;
        end
        clahe_rgb(:, :, 2) = eg;
    end

    % 3. Retinal Anatomy Overlay (Cyan: Optic Disc, Amber: Macula)
    anatomy_img = visualizeAnatomy(base_rgb, od_res, fovea_res);

    % 4. Retinal Vessel Overlay (Green vessels over darkened fundus)
    vess_mask = vess_res.binary_mask;
    vessel_img = base_rgb * 0.70;
    for c = 1:3
        ch = vessel_img(:, :, c);
        if c == 2
            ch(vess_mask) = 1.0; % Green channel bright
        else
            ch(vess_mask) = ch(vess_mask) * 0.25;
        end
        vessel_img(:, :, c) = ch;
    end

    % 5. Microaneurysms Detection Overlay (Red with High-Contrast Target Markers)
    ma_mask = ma_res.binary_mask;
    ma_dot = imdilate(ma_mask, strel('disk', 2));
    ma_ring = imdilate(ma_mask, strel('disk', 4)) & ~ma_dot;
    ma_img = base_rgb;
    ch1 = ma_img(:,:,1); ch2 = ma_img(:,:,2); ch3 = ma_img(:,:,3);
    % Yellow outer target ring for clinical visibility against red retina
    ch1(ma_ring) = 1.0; ch2(ma_ring) = 1.0; ch3(ma_ring) = 0.0;
    % Pure red candidate core
    ch1(ma_dot) = 1.0; ch2(ma_dot) = 0.0; ch3(ma_dot) = 0.0;
    ma_img(:,:,1) = ch1; ma_img(:,:,2) = ch2; ma_img(:,:,3) = ch3;

    % 6. Hemorrhages Overlay (Magenta [1, 0, 1] with Cyan Demarcation Outline)
    he_mask = he_res.binary_mask;
    he_dil = imdilate(he_mask, strel('disk', 1));
    he_border = imdilate(he_dil, strel('disk', 2)) & ~he_dil;
    he_img = base_rgb;
    ch1 = he_img(:,:,1); ch2 = he_img(:,:,2); ch3 = he_img(:,:,3);
    % Cyan outer perimeter border
    ch1(he_border) = 0.1; ch2(he_border) = 0.9; ch3(he_border) = 1.0;
    % Magenta hemorrhage fill
    ch1(he_dil) = 1.0; ch2(he_dil) = 0.0; ch3(he_dil) = 1.0;
    he_img(:,:,1) = ch1; he_img(:,:,2) = ch2; he_img(:,:,3) = ch3;

    % 7. Exudates Overlay (Yellow [1, 1, 0] with Foveal Proximity Zone)
    ex_dil = imdilate(ex_res.binary_mask, strel('disk', 1));
    ex_img = base_rgb;
    ch1 = ex_img(:,:,1); ch2 = ex_img(:,:,2); ch3 = ex_img(:,:,3);
    ch1(ex_dil) = 1.0; ch2(ex_dil) = 1.0; ch3(ex_dil) = 0.0;
    ex_img(:,:,1) = ch1; ex_img(:,:,2) = ch2; ex_img(:,:,3) = ch3;

    % 8. Multi-Lesion Composite Overlay (Red: MA, Magenta: HE, Yellow: EX)
    multi_lesion_img = base_rgb;
    ch1 = multi_lesion_img(:,:,1); ch2 = multi_lesion_img(:,:,2); ch3 = multi_lesion_img(:,:,3);
    % Exudates first (Yellow)
    ch1(ex_dil) = 1.0; ch2(ex_dil) = 1.0; ch3(ex_dil) = 0.0;
    % Hemorrhages second (Magenta)
    ch1(he_dil) = 1.0; ch2(he_dil) = 0.0; ch3(he_dil) = 1.0;
    % Microaneurysms third (Red with yellow ring)
    ch1(ma_ring) = 1.0; ch2(ma_ring) = 1.0; ch3(ma_ring) = 0.0;
    ch1(ma_dot) = 1.0; ch2(ma_dot) = 0.0; ch3(ma_dot) = 0.0;
    multi_lesion_img(:,:,1) = ch1; multi_lesion_img(:,:,2) = ch2; multi_lesion_img(:,:,3) = ch3;

    % 9. Grad-CAM Heatmap (Jet Colormap within FOV)
    if isempty(cam_map) || max(cam_map(:)) == 0
        cam_map = zeros(rows, cols);
    end
    cam_norm = (cam_map - min(cam_map(:))) / (max(cam_map(:)) - min(cam_map(:)) + 1e-6);
    cam_norm(~prep_res.fov_mask) = 0;
    
    cam_gray = uint8(round(cam_norm * 255));
    cam_rgb = ind2rgb(cam_gray, jet(256));
    cam_rgb(repmat(~prep_res.fov_mask, [1, 1, 3])) = 0;

    % 10. Grad-CAM Clinical Explainability Blend
    % Activation-weighted alpha blending: preserves clear natural fundus where CAM is low
    alpha_map = zeros(rows, cols);
    act_mask = cam_norm > 0.08 & prep_res.fov_mask;
    alpha_map(act_mask) = 0.65 * (cam_norm(act_mask) - 0.08) / 0.92;
    alpha_3d = repmat(alpha_map, [1, 1, 3]);
    gradcam_blend = (1 - alpha_3d) .* base_rgb + alpha_3d .* cam_rgb;

    % Assemble Structured Overlays
    overlays = struct();
    overlays.original_rgb = orig_rgb;
    overlays.preprocessed_rgb = clahe_rgb;
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
