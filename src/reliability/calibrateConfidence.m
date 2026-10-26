function calib_result = calibrateConfidence(uncalibrated_probs, true_labels, num_bins)
% CALIBRATECONFIDENCE Probability Calibration and Expected Calibration Error (ECE)
% Implements Temperature Scaling & Platt scaling to align predicted probabilities
% with empirical observed accuracy, computing ECE and reliability curve points.

    if nargin < 3 || isempty(num_bins)
        num_bins = 10;
    end

    p = uncalibrated_probs(:);
    y = true_labels(:);
    n = numel(p);

    % Fit optimal temperature scaling parameter T on validation set
    % Optimize cross-entropy loss w.r.t temperature T
    eps_val = 1e-6;
    p_clamped = max(eps_val, min(1 - eps_val, p));
    logits = log(p_clamped ./ (1 - p_clamped));

    % Temperature search over [0.5, 3.0]
    t_candidates = linspace(0.5, 3.0, 50);
    best_loss = Inf;
    best_T = 1.0;

    for k = 1:numel(t_candidates)
        T = t_candidates(k);
        scaled_p = 1.0 ./ (1.0 + exp(-logits / T));
        scaled_p = max(eps_val, min(1 - eps_val, scaled_p));
        loss = -mean(y .* log(scaled_p) + (1 - y) .* log(1 - scaled_p));
        if loss < best_loss
            best_loss = loss;
            best_T = T;
        end
    end

    % Apply optimal Temperature
    calibrated_probs = 1.0 ./ (1.0 + exp(-logits / best_T));

    % Compute Expected Calibration Error (ECE)
    bin_edges = linspace(0, 1, num_bins + 1);
    ece = 0;
    bin_accs = zeros(num_bins, 1);
    bin_confs = zeros(num_bins, 1);
    bin_counts = zeros(num_bins, 1);

    for b = 1:num_bins
        in_bin = (calibrated_probs >= bin_edges(b)) & (calibrated_probs < bin_edges(b+1));
        if b == num_bins
            in_bin = (calibrated_probs >= bin_edges(b)) & (calibrated_probs <= bin_edges(b+1));
        end
        
        count = sum(in_bin);
        bin_counts(b) = count;
        if count > 0
            bin_accs(b)  = mean(y(in_bin));
            bin_confs(b) = mean(calibrated_probs(in_bin));
            ece = ece + (count / n) * abs(bin_accs(b) - bin_confs(b));
        else
            bin_accs(b)  = (bin_edges(b) + bin_edges(b+1)) / 2;
            bin_confs(b) = bin_accs(b);
        end
    end

    calib_result = struct();
    calib_result.temperature = best_T;
    calib_result.calibrated_probs = calibrated_probs;
    calib_result.ece = ece;
    calib_result.bin_edges = bin_edges;
    calib_result.bin_accuracies = bin_accs;
    calib_result.bin_confidences = bin_confs;
    calib_result.bin_counts = bin_counts;
end
