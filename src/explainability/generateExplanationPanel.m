function fig_path = generateExplanationPanel(img_id, prep_res, od_res, fovea_res, vess_res, ma_res, he_res, ex_res, cam_map, decision_res, sev_res)
% GENERATEEXPLANATIONPANEL Generate 6-Panel Clinical Explainability Figure
% Visualizes full diagnostic pipeline evidence:
% 1. Original Retinal Image
% 2. Preprocessed & Illumination Corrected
% 3. Retinal Anatomy (Optic Disc & Fovea)
% 4. Vessel Segmentation Map
% 5. Multi-Lesion Map (MAs, Hemorrhages, Exudates)
% 6. Grad-CAM Visual Attention & Decision Banner

    cfg = shinra_config();
    fig_path = fullfile(cfg.paths.figures, 'explainability', sprintf('%s_explainability_panel.png', img_id));

    % Create invisible figure for headless rendering
    f = figure('Visible', 'off', 'Position', [100, 100, 1200, 800], 'Color', 'w');

    % 1. Original Image
    subplot(2, 3, 1);
    imshow(prep_res.original_image);
    title('1. Original Fundus Image', 'FontSize', 11, 'FontWeight', 'bold');

    % 2. Preprocessed RGB
    subplot(2, 3, 2);
    imshow(prep_res.rgb_normalized);
    title('2. Preprocessed & Illumination Norm', 'FontSize', 11, 'FontWeight', 'bold');

    % 3. Retinal Anatomy Overlay
    subplot(2, 3, 3);
    anatomy_overlay = visualizeAnatomy(prep_res.rgb_normalized, od_res, fovea_res);
    imshow(anatomy_overlay);
    title('3. Anatomy (Cyan: OD, Amber: Macula)', 'FontSize', 11, 'FontWeight', 'bold');

    % 4. Vessel Segmentation
    subplot(2, 3, 4);
    imshow(vess_res.binary_mask);
    title(sprintf('4. Retinal Vessels (Density: %.1f%%)', vess_res.morphometry.vessel_density * 100), ...
        'FontSize', 11, 'FontWeight', 'bold');

    % 5. Multi-Lesion Map
    subplot(2, 3, 5);
    lesion_overlay = prep_res.rgb_normalized;
    % Microaneurysms in Red [1, 0, 0]
    ma_dil = imdilate(ma_res.binary_mask, strel('disk', 2));
    % Hemorrhages in Magenta [1, 0, 1]
    he_dil = imdilate(he_res.binary_mask, strel('disk', 1));
    % Exudates in Yellow [1, 1, 0]
    ex_dil = imdilate(ex_res.binary_mask, strel('disk', 1));

    for c = 1:3
        ch = lesion_overlay(:, :, c);
        if c == 1
            ch(ma_dil | he_dil | ex_dil) = 1.0;
        elseif c == 2
            ch(ma_dil | he_dil) = 0.0;
            ch(ex_dil) = 1.0;
        else
            ch(ma_dil) = 0.0;
            ch(he_dil) = 1.0;
            ch(ex_dil) = 0.0;
        end
        lesion_overlay(:, :, c) = ch;
    end
    imshow(lesion_overlay);
    title(sprintf('5. Lesions (MA:%d, HE:%d, EX:%d)', ma_res.count, he_res.count, ex_res.count), ...
        'FontSize', 11, 'FontWeight', 'bold');

    % 6. Grad-CAM Attention Heatmap
    subplot(2, 3, 6);
    % Convert CAM to JET colormap
    cam_gray = uint8(cam_map * 255);
    cam_rgb = ind2rgb(cam_gray, jet(256));
    % Alpha blend with preprocessed image
    alpha = 0.5;
    blend_rgb = alpha * cam_rgb + (1 - alpha) * prep_res.rgb_normalized;
    imshow(blend_rgb);
    title(sprintf('6. Grad-CAM Focus (Conf: %.2f)', sev_res.confidence), ...
        'FontSize', 11, 'FontWeight', 'bold');

    % Master Title Banner
    sgtitle(sprintf('SHINRA Clinical Decision: %s | Grade: %s (P_{ref}: %.1f%%, Reliability: %.2f)', ...
        decision_res.final_decision, sev_res.predicted_class_name, ...
        decision_res.p_referable * 100, decision_res.reliability_score), ...
        'FontSize', 14, 'FontWeight', 'bold', 'Color', [0.1, 0.2, 0.6]);

    % Save figure
    save_figure_safe(f, fig_path);
    close(f);
end
