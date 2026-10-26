function guidance = recaptureGuidance(quality_class, defects, score)
% RECAPTUREGUIDANCE Generate Actionable Clinical Guidance for Rural Health Workers
% Translates technical image defects into plain-language operational instructions
% for community health personnel (ASHA / ANM / Vision Technicians).

    if strcmp(quality_class, 'GOOD')
        guidance = 'Image quality is clinically acceptable for automated AI screening.';
        return;
    end

    reasons = {};
    actions = {};

    if defects.is_blurry
        reasons{end+1} = 'Excessive optical blur / motion artifact';
        actions{end+1} = 'Stabilize handheld camera against patient forehead and ask patient to blink then hold gaze fixed.';
    end

    if defects.is_underexposed
        reasons{end+1} = 'Severe underexposure / dark fundus';
        actions{end+1} = 'Increase camera illumination/flash level or ensure adequate pupil dilation/dark room environment.';
    end

    if defects.is_overexposed
        reasons{end+1} = 'Severe overexposure / corneal glare reflection';
        actions{end+1} = 'Decrease camera flash power and adjust optical angle slightly to avoid corneal light reflex.';
    end

    if defects.is_incomplete_fov
        reasons{end+1} = 'Incomplete retinal field / severe boundary clipping';
        actions{end+1} = 'Re-center the camera objective on the pupil and ensure the target fixation LED is followed.';
    end

    if defects.is_low_contrast
        reasons{end+1} = 'Low contrast across retinal vascular bed';
        actions{end+1} = 'Check lens cleanliness and verify optical focus dial.';
    end

    if isempty(reasons)
        if strcmp(quality_class, 'BORDERLINE')
            guidance = sprintf('Borderline image quality (Score: %.1f/100). Adaptive contrast enhancement will be attempted.', score);
        else
            guidance = sprintf('Image quality insufficient for diagnostic reliability (Score: %.1f/100). Please recapture image.', score);
        end
    else
        reason_str = strjoin(reasons, '; ');
        action_str = strjoin(actions, ' ');
        if strcmp(quality_class, 'UNGRADABLE')
            guidance = sprintf('RECAPTURE REQUIRED (%s). Action: %s', reason_str, action_str);
        else
            guidance = sprintf('BORDERLINE QUALITY (%s). Action: Enhancing image; if AI reliability is low, %s', reason_str, action_str);
        end
    end
end
