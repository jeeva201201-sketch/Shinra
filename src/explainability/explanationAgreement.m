function metrics = explanationAgreement(cam_map, lesion_mask, fov_mask)
% EXPLANATIONAGREEMENT Quantitative Evaluation of Model Visual Attention vs Lesion Ground Truth
% Calculates spatial intersection, Dice overlap, and IoU between high-attention
% Grad-CAM regions (top 20% intensity) and anatomical lesion locations.
%
% Note: Quantitative metric of interpretability alignment.

    if nargin < 3 || isempty(fov_mask)
        fov_mask = true(size(cam_map));
    end

    % High visual attention threshold: Top 25% of CAM intensity
    cam_in_fov = cam_map(fov_mask);
    if isempty(cam_in_fov) || max(cam_in_fov) == 0
        cam_attention_mask = false(size(cam_map));
    else
        thresh = prctile(cam_in_fov, 75);
        cam_attention_mask = (cam_map >= thresh) & fov_mask;
    end

    lesion_binary = lesion_mask & fov_mask;

    tp = sum(cam_attention_mask(:) & lesion_binary(:));
    fp = sum(cam_attention_mask(:) & ~lesion_binary(:));
    fn = sum(~cam_attention_mask(:) & lesion_binary(:));

    iou  = tp / max(1, (tp + fp + fn));
    dice = (2 * tp) / max(1, (2 * tp + fp + fn));
    precision = tp / max(1, (tp + fp));
    recall    = tp / max(1, (tp + fn));

    metrics = struct();
    metrics.iou = iou;
    metrics.dice = dice;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.cam_area = sum(cam_attention_mask(:));
    metrics.lesion_area = sum(lesion_binary(:));
end
