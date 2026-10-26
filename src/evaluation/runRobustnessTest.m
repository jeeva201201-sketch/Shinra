function robustness_tab = runRobustnessTest(num_test_samples)
% RUNROBUSTNESSTEST Controlled Evaluation of Screening Robustness Under Field Degradation
% Assesses vulnerability of Baseline AI vs SHINRA Quality-Gated Safety Architecture
% across 5 synthetic rural clinical degradation modes:
% 1. Optical / Motion Blur
% 2. Severe Underexposure (Poor flash / dark room)
% 3. Severe Overexposure / Corneal Glare
% 4. Sensor Noise
% 5. Incomplete Retinal Field Occlusion

    cfg = shinra_config();
    fprintf('============================================================\n');
    fprintf('        SHINRA CONTROLLED ROBUSTNESS STRESS-TEST            \n');
    fprintf('============================================================\n');

    [idrid_train_full, ~] = load_idrid();
    [~, val_tab] = split_dataset(idrid_train_full, 0.20, cfg.random_seed);

    if nargin >= 1 && ~isempty(num_test_samples) && num_test_samples < height(val_tab)
        val_tab = val_tab(1:num_test_samples, :);
    end

    n = height(val_tab);
    degradation_types = {'Clean Baseline', 'Optical Blur', 'Severe Underexposure', 'Severe Overexposure', 'Sensor Noise', 'Partial FOV Occlusion'};
    num_deg = numel(degradation_types);

    baseline_confident_errors = zeros(num_deg, 1);
    shinra_recaptured_count   = zeros(num_deg, 1);
    shinra_safe_accuracy      = zeros(num_deg, 1);

    for d = 1:num_deg
        deg_name = degradation_types{d};
        fprintf('Evaluating stress-test condition [%d/%d]: %s...\n', d, num_deg, deg_name);

        b_err = 0;
        s_rec = 0;
        s_correct = 0;
        s_evaluated = 0;

        for i = 1:n
            img_raw = im2double(imread(val_tab.ImagePath{i}));
            y_true_ref = val_tab.Referable_DR(i);

            % Apply Synthetic Field Degradation
            switch d
                case 1 % Clean
                    img_deg = img_raw;
                case 2 % Optical Blur
                    img_deg = imgaussfilt(img_raw, 3.5);
                case 3 % Severe Underexposure
                    img_deg = img_raw * 0.35;
                case 4 % Severe Overexposure
                    img_deg = min(1.0, img_raw * 1.85);
                case 5 % Sensor Noise
                    img_deg = max(0, min(1, img_raw + 0.08 * randn(size(img_raw))));
                case 6 % Partial FOV Occlusion
                    img_deg = img_raw;
                    [h, w, ~] = size(img_deg);
                    img_deg(:, round(w*0.65):end, :) = 0;
            end

            % Baseline Evaluation (No Quality Gate)
            prep_b = struct('rgb_normalized', imresize(img_deg, [512, 512]));
            sev_b = predictDRSeverity(prep_b, []);
            pred_b_ref = sev_b.p_referable >= 0.50;
            if pred_b_ref ~= y_true_ref && sev_b.confidence >= 0.70
                b_err = b_err + 1;
            end

            % SHINRA Evaluation (With Quality Gate & Reliability)
            [fov_m, ~, ~] = detectRetinalFOV(img_deg);
            [q_class, q_rep] = qualityClassifier(img_deg, fov_m);

            if strcmp(q_class, 'UNGRADABLE')
                s_rec = s_rec + 1; % Successfully intercepted by quality gate
            else
                % Process through full pipeline
                prep_s = preprocessRetina(img_deg);
                od_s = detectOpticDisc(prep_s.rgb_normalized, prep_s.fov_mask);
                fovea_s = locateFovea(prep_s.rgb_normalized, od_s, prep_s.fov_mask);
                vess_s = segmentVessels(prep_s.rgb_normalized, prep_s.fov_mask, od_s.mask);
                ma_s = detectMicroaneurysms(prep_s.rgb_normalized, prep_s.fov_mask, vess_s.binary_mask, od_s.mask);
                he_s = detectHemorrhages(prep_s.rgb_normalized, prep_s.fov_mask, vess_s.binary_mask, od_s.mask);
                ex_s = detectExudates(prep_s.rgb_normalized, prep_s.fov_mask, od_s.mask, fovea_s);
                nv_s = analyzeNeovascularization(vess_s.binary_mask, prep_s.fov_mask, od_s.mask, vess_s.morphometry);
                
                lesion_ev = struct(...
                    'ma_count', ma_s.count, 'ma_area', ma_s.total_area, ...
                    'he_count', he_s.count, 'he_area_fraction', he_s.area_fraction, ...
                    'ex_count', ex_s.count, 'ex_area_fraction', ex_s.area_fraction, ...
                    'dme_risk', ex_s.macular_proximity_risk, 'nv_score', nv_s.nv_score);
                
                sev_s = predictDRSeverity(prep_s, lesion_ev);
                ev_vec = extractEvidenceVector(sev_s, lesion_ev, vess_s, q_rep);
                fus_s = fuseEvidence(ev_vec, []);
                rel_s = computeReliability(sev_s.confidence, q_rep.score, fus_s.agreement_score);
                dec_s = makeScreeningDecision(q_rep, fus_s, rel_s);

                pred_s_ref = strcmp(dec_s.final_decision, 'REFERABLE DR');
                s_evaluated = s_evaluated + 1;
                if pred_s_ref == y_true_ref
                    s_correct = s_correct + 1;
                end
            end
        end

        baseline_confident_errors(d) = b_err;
        shinra_recaptured_count(d)   = s_rec;
        if s_evaluated > 0
            shinra_safe_accuracy(d) = (s_correct / s_evaluated) * 100;
        else
            shinra_safe_accuracy(d) = 100; % 100% intercepted
        end
    end

    robustness_tab = table(degradation_types', baseline_confident_errors, shinra_recaptured_count, shinra_safe_accuracy, ...
        'VariableNames', {'DegradationMode', 'Baseline_Confident_Errors', 'SHINRA_Intercepted_Recapture', 'SHINRA_Passed_Safe_Accuracy'});

    fprintf('------------------------------------------------------------------------------------\n');
    fprintf('  SHINRA ROBUSTNESS STRESS-TEST RESULTS\n');
    fprintf('------------------------------------------------------------------------------------\n');
    disp(robustness_tab);

    writetable(robustness_tab, fullfile(cfg.paths.metrics, 'robustness_table.csv'));
    save(fullfile(cfg.paths.metrics, 'robustness_results.mat'), 'robustness_tab');
    fprintf('Robustness results saved to %s\n\n', fullfile(cfg.paths.metrics, 'robustness_table.csv'));
end
