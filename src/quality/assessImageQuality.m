function q_result = assessImageQuality(img, fov_mask)
% ASSESSIMAGEQUALITY Reusable Image Quality Assessment Module for Single Image
% Evaluates multi-factor image quality features, computes composite quality score [0, 100],
% determines tri-state classification (GOOD / BORDERLINE / UNGRADABLE), and provides guidance.

    if ~isa(img, 'double')
        img = im2double(img);
    end
    
    if nargin < 2 || isempty(fov_mask)
        if size(img, 3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end
        fov_mask = gray > 0.05;
    end

    % 1. Tri-State Quality Classification & Diagnostic Report
    [q_class, q_rep] = qualityClassifier(img, fov_mask);
    
    % 2. Assemble Structured Output
    q_result = q_rep;
    q_result.quality_score = q_rep.score;
    q_result.quality_class = q_class;
    q_result.is_gradable = ~strcmp(q_class, 'UNGRADABLE');
    
    % Defect indicators
    if isfield(q_rep, 'defects')
        q_result.has_severe_artifacts = (q_rep.defects.is_blurry || ...
                                         q_rep.defects.is_underexposed || ...
                                         q_rep.defects.is_overexposed || ...
                                         q_rep.defects.is_incomplete_fov);
    else
        q_result.has_severe_artifacts = false;
    end
end
