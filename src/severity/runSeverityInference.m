function sev_result = runSeverityInference(prep_result, model_version)
% RUNSEVERITYINFERENCE Reusable 5-Class DR Severity Inference Module
% Runs inference using the trained SqueezeNet severity model and extracts
% class probabilities [P0, P1, P2, P3, P4], predicted ICDR grade (0-4),
% confidence score, and expected ordinal score.

    cfg = shinra_config();
    if nargin < 2 || isempty(model_version)
        model_version = 'v3'; % Default to frozen validated V3
    end

    switch lower(model_version)
        case 'v3'
            sev_result = predictSqueezeNetSeverityV3(prep_result);
        case 'v4'
            v4_path = fullfile(cfg.paths.models_severity, 'SHINRA_SqueezeNet_Severity_Model_V4_CANDIDATE.mat');
            if exist(v4_path, 'file') == 2
                loaded = load(v4_path);
                net = loaded.net_severity;
                target_size_2d = [227, 227];
                if isstruct(prep_result) && isfield(prep_result, 'rgb_normalized')
                    input_img = prep_result.rgb_normalized;
                elseif isnumeric(prep_result)
                    input_img = prep_result;
                else
                    error('Invalid input provided to runSeverityInference.');
                end
                if max(input_img(:)) <= 1.05
                    input_tensor = single(imresize(input_img, target_size_2d) * 255.0);
                else
                    input_tensor = single(imresize(input_img, target_size_2d));
                end
                [pred_cat, raw_probs] = classify(net, input_tensor);
                probs = double(raw_probs);
                predicted_grade = double(pred_cat) - 1;
                [max_prob, ~] = max(probs);
                
                sev_result = struct();
                sev_result.probabilities = probs;
                sev_result.class_names = cfg.severity.class_names;
                sev_result.predicted_grade = predicted_grade;
                sev_result.predicted_class_name = cfg.severity.class_names{predicted_grade + 1};
                sev_result.p_referable = probs(3) + probs(4) + probs(5);
                sev_result.expected_grade = sum((0:4) .* probs);
                sev_result.confidence = max_prob;
                sev_result.model_source = v4_path;
            else
                % Fallback to V3 if V4 checkpoint is not found
                sev_result = predictSqueezeNetSeverityV3(prep_result);
            end
        otherwise
            sev_result = predictSqueezeNetSeverityV3(prep_result);
    end
end
