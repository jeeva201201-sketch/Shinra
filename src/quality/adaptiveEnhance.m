function [enhanced_img, new_quality_report] = adaptiveEnhance(img, fov_mask, current_report)
% ADAPTIVEENHANCE Closed-Loop Retinal Enhancement for Borderline Quality Images
% Applies targeted restorative processing tailored to specific detected defects:
% - Unsharp masking for blur
% - Dynamic range stretching & adaptive CLAHE for underexposure / low contrast
% - Multi-scale illumination compensation
%
% Re-assesses post-enhancement quality score and returns updated diagnostic report.

    if ~isa(img, 'double')
        img = im2double(img);
    end

    if nargin < 2 || isempty(fov_mask)
        [fov_mask, ~, ~] = detectRetinalFOV(img);
    end

    enhanced_img = img;
    defects = current_report.defects;

    % 1. Targeted Illumination Correction
    if defects.is_underexposed || current_report.score < 60
        % Dynamic range expansion on green/RGB
        enhanced_img = normalizeIllumination(enhanced_img, fov_mask, 20);
    end

    % 2. Targeted Sharpness Restoration (Unsharp Masking)
    if defects.is_blurry
        for c = 1:size(enhanced_img, 3)
            ch = enhanced_img(:, :, c);
            blurred = imgaussfilt(ch, 1.2);
            sharp = ch + 0.6 * (ch - blurred);
            sharp(~fov_mask) = 0;
            enhanced_img(:, :, c) = max(0, min(1, sharp));
        end
    end

    % 3. Targeted CLAHE for Low Contrast
    if defects.is_low_contrast || current_report.score < 65
        for c = 1:size(enhanced_img, 3)
            ch = enhanced_img(:, :, c);
            clahe_ch = adapthisteq(ch, 'ClipLimit', 0.025, 'NumTiles', [10, 10]);
            clahe_ch(~fov_mask) = 0;
            enhanced_img(:, :, c) = clahe_ch;
        end
    end

    % 4. Re-assess Quality Post-Enhancement
    [new_class, new_quality_report] = qualityClassifier(enhanced_img, fov_mask);
    new_quality_report.is_enhanced = true;
    new_quality_report.original_score = current_report.score;
    new_quality_report.improvement = new_quality_report.score - current_report.score;
end
