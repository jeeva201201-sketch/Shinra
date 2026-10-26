function cross_results = runCrossDatasetEvaluation(num_samples_per_set)
% RUNCROSSDATASETEVALUATION Cross-Domain Generalization and Screening Evaluation
% Evaluates distribution of image quality, ungradable rates, anatomical detection,
% and referral rates across IDRiD, DRIVE, and APTOS 2019 cohorts.

    cfg = shinra_config();
    if nargin < 1 || isempty(num_samples_per_set)
        num_samples_per_set = 20;
    end

    fprintf('============================================================\n');
    fprintf('           CROSS-DATASET GENERALIZATION AUDIT               \n');
    fprintf('============================================================\n');

    [idrid_train, ~] = load_idrid();
    [drive_train, ~] = load_drive();
    aptos_table      = load_aptos();

    datasets = {
        'IDRiD (Clinical Train Split)', idrid_train.ImagePath(1:min(num_samples_per_set, height(idrid_train)));
        'DRIVE (Vessel Cohort)',       drive_train.ImagePath(1:min(num_samples_per_set, height(drive_train)));
        'APTOS 2019 (Unlabeled)',      aptos_table.ImagePath(1:min(num_samples_per_set, height(aptos_table)))
    };

    num_ds = size(datasets, 1);
    dataset_names = cell(num_ds, 1);
    mean_quality = zeros(num_ds, 1);
    pct_good     = zeros(num_ds, 1);
    pct_border   = zeros(num_ds, 1);
    pct_ungrad   = zeros(num_ds, 1);
    pct_refer    = zeros(num_ds, 1);

    for d = 1:num_ds
        d_name = datasets{d, 1};
        d_paths = datasets{d, 2};
        n = numel(d_paths);
        dataset_names{d} = d_name;

        q_scores = zeros(n, 1);
        q_classes = cell(n, 1);
        decisions = cell(n, 1);

        for i = 1:n
            img = imread(d_paths{i});
            prep = preprocessRetina(img);
            [q_cls, q_rep] = qualityClassifier(prep.rgb_normalized, prep.fov_mask);
            od_res = detectOpticDisc(prep.rgb_normalized, prep.fov_mask);
            fovea_res = locateFovea(prep.rgb_normalized, od_res, prep.fov_mask);
            vess_res = segmentVessels(prep.rgb_normalized, prep.fov_mask, od_res.mask);
            
            ma_res = detectMicroaneurysms(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
            he_res = detectHemorrhages(prep.rgb_normalized, prep.fov_mask, vess_res.binary_mask, od_res.mask);
            ex_res = detectExudates(prep.rgb_normalized, prep.fov_mask, od_res.mask, fovea_res);
            nv_res = analyzeNeovascularization(vess_res.binary_mask, prep.fov_mask, od_res.mask, vess_res.morphometry);

            lesion_ev = struct(...
                'ma_count', ma_res.count, 'ma_area', ma_res.total_area, ...
                'he_count', he_res.count, 'he_area_fraction', he_res.area_fraction, ...
                'ex_count', ex_res.count, 'ex_area_fraction', ex_res.area_fraction, ...
                'dme_risk', ex_res.macular_proximity_risk, 'nv_score', nv_res.nv_score);

            sev_res = predictDRSeverity(prep, lesion_ev);
            ev_vec  = extractEvidenceVector(sev_res, lesion_ev, vess_res, q_rep);
            fus_res = fuseEvidence(ev_vec, []);
            rel_res = computeReliability(sev_res.confidence, q_rep.score, fus_res.agreement_score);
            dec_res = makeScreeningDecision(q_rep, fus_res, rel_res);

            q_scores(i) = q_rep.score;
            q_classes{i} = q_cls;
            decisions{i} = dec_res.final_decision;
        end

        mean_quality(d) = mean(q_scores);
        pct_good(d)     = (sum(strcmp(q_classes, 'GOOD')) / n) * 100;
        pct_border(d)   = (sum(strcmp(q_classes, 'BORDERLINE')) / n) * 100;
        pct_ungrad(d)   = (sum(strcmp(q_classes, 'UNGRADABLE')) / n) * 100;
        pct_refer(d)    = (sum(strcmp(decisions, 'REFERABLE DR')) / n) * 100;
    end

    cross_results = table(dataset_names, mean_quality, pct_good, pct_border, pct_ungrad, pct_refer, ...
        'VariableNames', {'DatasetCohort', 'Mean_Quality_Score', 'Good_Pct', 'Borderline_Pct', 'Ungradable_Pct', 'Referral_Pct'});

    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('  CROSS-DATASET GENERALIZATION SUMMARY\n');
    fprintf('------------------------------------------------------------------------------------\n');
    disp(cross_results);

    save(fullfile(cfg.paths.metrics, 'cross_dataset_metrics.mat'), 'cross_results');
end
