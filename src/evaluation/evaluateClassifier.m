function metrics = evaluateClassifier(y_true, y_pred, y_prob)
% EVALUATECLASSIFIER Comprehensive Clinical Diagnostic Performance Metrics
% Computes Sensitivity, Specificity, Precision, F1, Balanced Accuracy,
% Quadratic Weighted Kappa (QWK), ROC-AUC, and Confusion Matrix.

    y_true = y_true(:);
    y_pred = y_pred(:);
    n = numel(y_true);

    % Binary Referable DR Metrics (Grade >= 2 vs Grade < 2)
    is_binary = max(y_true) <= 1;
    if ~is_binary
        y_true_ref = y_true >= 2;
        y_pred_ref = y_pred >= 2;
    else
        y_true_ref = y_true == 1;
        y_pred_ref = y_pred == 1;
    end

    tp = sum(y_pred_ref & y_true_ref);
    fp = sum(y_pred_ref & ~y_true_ref);
    fn = sum(~y_pred_ref & y_true_ref);
    tn = sum(~y_pred_ref & ~y_true_ref);

    sens = tp / max(1, tp + fn);
    spec = tn / max(1, tn + fp);
    prec = tp / max(1, tp + fp);
    f1   = (2 * tp) / max(1, 2 * tp + fp + fn);
    bal_acc = (sens + spec) / 2;

    % ROC-AUC calculation
    if nargin >= 3 && ~isempty(y_prob)
        y_prob = y_prob(:);
        [~, sort_idx] = sort(y_prob, 'descend');
        sorted_true = y_true_ref(sort_idx);
        cum_tp = cumsum(sorted_true) / max(1, sum(sorted_true));
        cum_fp = cumsum(~sorted_true) / max(1, sum(~sorted_true));
        % Trapezoidal integration for AUC
        roc_auc = trapz([0; cum_fp; 1], [0; cum_tp; 1]);
    else
        roc_auc = nan;
    end

    % Multiclass Metrics (if 5-class labels)
    if ~is_binary && numel(unique(y_true)) > 2
        acc_5class = mean(y_true == y_pred);
        conf_mat = confusionmat(y_true, y_pred);
        
        % Quadratic Weighted Kappa (QWK)
        qwk = computeQuadraticWeightedKappa(y_true, y_pred, 5);
    else
        acc_5class = mean(y_true == y_pred);
        conf_mat = confusionmat(y_true_ref, y_pred_ref);
        qwk = nan;
    end

    metrics = struct();
    metrics.sensitivity = sens;
    metrics.specificity = spec;
    metrics.precision = prec;
    metrics.f1_score = f1;
    metrics.balanced_accuracy = bal_acc;
    metrics.roc_auc = roc_auc;
    metrics.accuracy_5class = acc_5class;
    metrics.qwk = qwk;
    metrics.confusion_matrix = conf_mat;
    metrics.tp = tp;
    metrics.fp = fp;
    metrics.fn = fn;
    metrics.tn = tn;
    metrics.num_samples = n;
end

function qwk = computeQuadraticWeightedKappa(y_true, y_pred, num_classes)
    O = confusionmat(y_true, y_pred, 'Order', 0:(num_classes-1));
    n = sum(O(:));
    if n == 0
        qwk = 0;
        return;
    end

    % Weight matrix (quadratic distance)
    [i_grid, j_grid] = meshgrid(0:(num_classes-1), 0:(num_classes-1));
    W = ((i_grid - j_grid).^2) / ((num_classes - 1)^2);

    % Expected matrix under chance agreement
    row_sums = sum(O, 2);
    col_sums = sum(O, 1);
    E = (row_sums * col_sums) / n;

    numerator = sum(sum(W .* O));
    denominator = sum(sum(W .* E));

    if denominator == 0
        qwk = 1.0;
    else
        qwk = 1.0 - (numerator / denominator);
    end
end
