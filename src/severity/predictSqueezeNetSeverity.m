function severity_result = predictSqueezeNetSeverity(prep_result, varargin)
% PREDICTSQUEEZENETSEVERITY Predict 5-Class DR Severity Using SqueezeNet V2 Checkpoint
% Loads the trained SqueezeNet transfer-learning model (SHINRA_SqueezeNet_Severity_Model.mat)
% and extracts 5-class posterior probabilities and Referable DR probability P(DR >= 2).
%
% Throws explicit error if the SqueezeNet V2 model checkpoint has not been trained.

    cfg = shinra_config();
    model_file_v2 = fullfile(cfg.paths.models_severity, 'SHINRA_SqueezeNet_Severity_Model.mat');

    if exist(model_file_v2, 'file') ~= 2
        error('SHINRA:SqueezeNetModelNotFound', ...
            'Trained SqueezeNet V2 model not found at:\n  %s\nPlease train the model first by executing: trainSeveritySqueezeNet', ...
            model_file_v2);
    end

    % Load V2 Model Checkpoint
    loaded = load(model_file_v2);
    if ~isfield(loaded, 'net')
        error('Invalid SqueezeNet model file: missing "net" variable in %s', model_file_v2);
    end
    net = loaded.net;

    % SqueezeNet requires [227, 227] input resolution
    target_size_2d = [227, 227];
    if isstruct(prep_result) && isfield(prep_result, 'rgb_normalized')
        input_img = prep_result.rgb_normalized;
    elseif isnumeric(prep_result)
        input_img = prep_result;
    else
        error('Invalid input provided to predictSqueezeNetSeverity.');
    end

    % Standardized Preprocessing & Tensor Formatting (227x227x3)
    input_tensor = single(imresize(input_img, target_size_2d));

    % 1. Model Inference (Softmax Probabilities)
    [pred_cat, raw_probs] = classify(net, input_tensor);
    probs = double(raw_probs);

    % 2. Predicted Grade (Argmax: 0 to 4)
    predicted_grade = double(pred_cat) - 1;
    [max_prob, ~] = max(probs);

    % 3. Referable DR Probability: P(DR >= 2) = P(Class 2) + P(Class 3) + P(Class 4)
    p_referable = probs(3) + probs(4) + probs(5);

    % 4. Ordinal Expected Score: sum(k * P(DR = k))
    expected_grade = sum((0:4) .* probs);

    % Assemble Structured Result
    severity_result = struct();
    severity_result.probabilities = probs;
    severity_result.class_names = cfg.severity.class_names;
    severity_result.predicted_grade = predicted_grade;
    severity_result.predicted_class_name = cfg.severity.class_names{predicted_grade + 1};
    severity_result.p_referable = p_referable;
    severity_result.expected_grade = expected_grade;
    severity_result.confidence = max_prob;
    severity_result.model_source = model_file_v2;
    if isfield(loaded, 'training_info')
        severity_result.training_info = loaded.training_info;
    end
end
