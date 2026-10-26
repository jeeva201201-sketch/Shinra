function severity_result = predictDRSeverity(prep_result, varargin)
% PREDICTDRSEVERITY Predict 5-Class Diabetic Retinopathy Severity Grade Using Trained Model
% Loads the trained SHINRA severity model checkpoint and extracts true posterior
% probabilities [P0, P1, P2, P3, P4], predicted ICDR grade, and referable DR probability P(DR >= 2).
%
% If no trained model checkpoint exists, raises an explicit error.

    cfg = shinra_config();
    model_file_primary = fullfile(cfg.paths.models_severity, 'SHINRA_DR_Severity_Model.mat');
    model_file_alias   = fullfile(cfg.paths.models_severity, 'severity_cnn.mat');

    if exist(model_file_primary, 'file') == 2
        model_path = model_file_primary;
    elseif exist(model_file_alias, 'file') == 2
        model_path = model_file_alias;
    else
        error('SHINRA:ModelNotFound', ...
            'Trained DR severity model not found at:\n  %s\nPlease train the model first by executing: trainSeverityModel', ...
            model_file_primary);
    end

    % Load Model Checkpoint
    loaded = load(model_path);
    if ~isfield(loaded, 'net')
        error('Invalid severity model file: missing "net" variable in %s', model_path);
    end
    net = loaded.net;

    target_size = cfg.severity.input_size(1:2);
    if isstruct(prep_result) && isfield(prep_result, 'rgb_normalized')
        input_img = prep_result.rgb_normalized;
    elseif isnumeric(prep_result)
        input_img = prep_result;
    else
        error('Invalid input provided to predictDRSeverity.');
    end

    % Standardized Preprocessing & Tensor Formatting
    input_tensor = single(imresize(input_img, target_size));

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
    severity_result.model_source = model_path;
    if isfield(loaded, 'training_info')
        severity_result.training_info = loaded.training_info;
    end
end
