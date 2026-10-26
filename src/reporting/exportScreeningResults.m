function export_table = exportScreeningResults(batch_results, output_csv_path)
% EXPORTSCREENINGRESULTS Export Batch Retinal Screening Results to Structured CSV
% Formats array of diagnostic screening results into a tabular CSV report.

    cfg = shinra_config();
    if nargin < 2 || isempty(output_csv_path)
        output_csv_path = fullfile(cfg.paths.predictions, 'screening_predictions.csv');
    end

    n = numel(batch_results);
    if n == 0
        export_table = table();
        return;
    end

    image_ids = cell(n, 1);
    quality_classes = cell(n, 1);
    quality_scores  = zeros(n, 1);
    predicted_grades = zeros(n, 1);
    predicted_classes = cell(n, 1);
    p_referable_cnn = zeros(n, 1);
    p_referable_fused = zeros(n, 1);
    ma_counts = zeros(n, 1);
    he_areas  = zeros(n, 1);
    ex_areas  = zeros(n, 1);
    dme_risks = zeros(n, 1);
    vessel_densities = zeros(n, 1);
    vessel_tortuosities = zeros(n, 1);
    reliability_scores = zeros(n, 1);
    final_decisions = cell(n, 1);
    urgencies = cell(n, 1);

    for i = 1:n
        res = batch_results(i);
        image_ids{i} = res.image_id;
        quality_classes{i} = res.quality.quality_class;
        quality_scores(i)  = res.quality.score;
        predicted_grades(i) = res.severity.predicted_grade;
        predicted_classes{i} = res.severity.predicted_class_name;
        p_referable_cnn(i) = res.fusion.p_cnn_referable;
        p_referable_fused(i) = res.fusion.p_referable_fused;
        ma_counts(i) = res.lesions.microaneurysms.count;
        he_areas(i)  = res.lesions.hemorrhages.area_fraction;
        ex_areas(i)  = res.lesions.exudates.area_fraction;
        dme_risks(i) = res.lesions.exudates.macular_proximity_risk;
        vessel_densities(i) = res.vessels.vessel_density;
        vessel_tortuosities(i) = res.vessels.mean_tortuosity;
        reliability_scores(i) = res.reliability.reliability_score;
        final_decisions{i} = res.decision.final_decision;
        urgencies{i} = res.decision.urgency;
    end

    export_table = table(image_ids, quality_classes, quality_scores, predicted_grades, predicted_classes, ...
        p_referable_cnn, p_referable_fused, ma_counts, he_areas, ex_areas, dme_risks, ...
        vessel_densities, vessel_tortuosities, reliability_scores, final_decisions, urgencies, ...
        'VariableNames', {'ImageID', 'QualityClass', 'QualityScore', 'PredictedGrade', 'PredictedClass', ...
        'P_Referable_CNN', 'P_Referable_Fused', 'MA_Count', 'HE_Area_Fraction', 'EX_Area_Fraction', 'DME_Risk', ...
        'Vessel_Density', 'Vessel_Tortuosity', 'ReliabilityScore', 'FinalDecision', 'TriageUrgency'});

    writetable(export_table, output_csv_path);
    save(fullfile(cfg.paths.predictions, 'screening_predictions.mat'), 'export_table');
    fprintf('Batch screening results successfully exported to %s\n', output_csv_path);
end
