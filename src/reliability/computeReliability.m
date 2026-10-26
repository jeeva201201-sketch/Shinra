function reliability_res = computeReliability(varargin)
% COMPUTERELIABILITY Multi-Factor Engineering Reliability & Uncertainty Scoring
% Computes an objective, transparent five-factor reliability score [0, 100]
% for safety-gated retinal screening triage:
%
% 1. Quality Sub-Score (S_q in [0, 100]): Continuous image quality & defect analysis
% 2. Model Confidence Sub-Score (S_c in [0, 100]): Max posterior probability & Shannon entropy
% 3. Boundary Margin Sub-Score (S_m in [0, 100]): Distance from referable threshold |P_ref - tau*|
% 4. Evidence Completeness Sub-Score (S_comp in [0, 100]): Availability & integrity across sub-pipelines
% 5. Multi-Modal Agreement Sub-Score (S_agree in [0, 100]): Concordance between CNN and lesion evidence
%
% COMPOSITE SCORE:
% R = w_q*S_q + w_c*S_c + w_m*S_m + w_comp*S_comp + w_a*S_agree (Clamped to [0, 100])
% Safety Override: If Quality is UNGRADABLE -> R = min(R, 30.0)
%
% Note: Engineering uncertainty / safety triage framework; not a clinically validated score.

    cfg = shinra_config();

    % -------------------------------------------------------------------------
    % Parse Input Arguments (Supports rich structs or legacy positional scalars)
    % -------------------------------------------------------------------------
    q_rep       = [];
    sev_res     = [];
    fus_res     = [];
    lesion_ev   = [];
    vess_res    = [];
    od_res      = [];
    fovea_res   = [];
    cam_agree   = [];
    custom_cfg  = [];

    if nargin == 1 && isstruct(varargin{1})
        % Single container struct passed
        bundle = varargin{1};
        if isfield(bundle, 'quality'),   q_rep     = bundle.quality;   end
        if isfield(bundle, 'severity'),  sev_res   = bundle.severity;  end
        if isfield(bundle, 'fusion'),    fus_res   = bundle.fusion;    end
        if isfield(bundle, 'lesions'),   lesion_ev = bundle.lesions;   end
        if isfield(bundle, 'vessels'),   vess_res  = bundle.vessels;   end
        if isfield(bundle, 'optic_disc'),od_res    = bundle.optic_disc;end
        if isfield(bundle, 'fovea'),     fovea_res = bundle.fovea;     end
        if isfield(bundle, 'cam_agree'), cam_agree = bundle.cam_agree; end
        if isfield(bundle, 'config'),    custom_cfg= bundle.config;    end
    elseif nargin >= 1 && isstruct(varargin{1})
        % Positional structs
        if nargin >= 1, q_rep     = varargin{1}; end
        if nargin >= 2, sev_res   = varargin{2}; end
        if nargin >= 3, fus_res   = varargin{3}; end
        if nargin >= 4, lesion_ev = varargin{4}; end
        if nargin >= 5, vess_res  = varargin{5}; end
        if nargin >= 6, od_res    = varargin{6}; end
        if nargin >= 7, fovea_res = varargin{7}; end
        if nargin >= 8, cam_agree = varargin{8}; end
        if nargin >= 9, custom_cfg= varargin{9}; end
    elseif nargin >= 1 && (isnumeric(varargin{1}) || isempty(varargin{1}))
        % Legacy positional scalars: computeReliability(confidence, quality_score, agreement_score)
        c_val = 0.80;
        q_val = 75.0;
        a_val = 0.90;
        if nargin >= 1 && ~isempty(varargin{1}), c_val = varargin{1}; end
        if nargin >= 2 && ~isempty(varargin{2}), q_val = varargin{2}; end
        if nargin >= 3 && ~isempty(varargin{3}), a_val = varargin{3}; end

        sev_res = struct('confidence', c_val, 'probabilities', [max(0, 1-c_val), zeros(1, 3), c_val], 'p_referable', c_val, 'expected_grade', c_val * 4);
        q_rep   = struct('score', q_val, 'quality_class', 'GOOD');
        if q_val < cfg.quality.threshold_borderline
            q_rep.quality_class = 'UNGRADABLE';
        elseif q_val < cfg.quality.threshold_good
            q_rep.quality_class = 'BORDERLINE';
        end
        fus_res = struct('p_referable_fused', c_val, 'p_cnn_referable', c_val, 'agreement_score', a_val);
    end

    % -------------------------------------------------------------------------
    % Factor 1: Image Quality Sub-Score (S_q in [0, 100])
    % -------------------------------------------------------------------------
    q_score = 75.0;
    q_class = 'GOOD';
    is_ungradable = false;

    if isstruct(q_rep)
        if isfield(q_rep, 'score') && isnumeric(q_rep.score) && ~isnan(q_rep.score)
            q_score = q_rep.score;
        end
        if isfield(q_rep, 'quality_class') && ischar(q_rep.quality_class)
            q_class = q_rep.quality_class;
        end
    elseif isnumeric(q_rep) && ~isempty(q_rep) && ~isnan(q_rep)
        q_score = q_rep;
        if q_score < cfg.quality.threshold_borderline
            q_class = 'UNGRADABLE';
        elseif q_score < cfg.quality.threshold_good
            q_class = 'BORDERLINE';
        end
    end

    is_ungradable = strcmp(q_class, 'UNGRADABLE');
    S_q = max(0, min(100.0, q_score));
    if is_ungradable
        S_q = min(S_q, 30.0);
    end

    % -------------------------------------------------------------------------
    % Factor 2: Model Confidence & Posterior Entropy Sub-Score (S_c in [0, 100])
    % -------------------------------------------------------------------------
    max_prob = 0.80;
    entropy_val = 0.50;
    entropy_norm = 0.31;
    exp_grade = 2.0;

    if isstruct(sev_res) && isfield(sev_res, 'probabilities') && numel(sev_res.probabilities) == 5 && ~any(isnan(sev_res.probabilities))
        probs = double(sev_res.probabilities(:)');
        probs = max(0, probs);
        if sum(probs) > 0
            probs = probs / sum(probs);
        else
            probs = [0.2, 0.2, 0.2, 0.2, 0.2];
        end

        max_prob = max(probs);
        eps_val = 1e-9;
        entropy_val = -sum(probs .* log(probs + eps_val));
        entropy_norm = max(0, min(1.0, entropy_val / log(5)));
        
        if isfield(sev_res, 'expected_grade') && ~isnan(sev_res.expected_grade)
            exp_grade = sev_res.expected_grade;
        else
            exp_grade = sum((0:4) .* probs);
        end

        % S_c scales positively with max probability and inversely with normalized entropy
        S_c = 100.0 * max(0, min(1.0, 0.60 * max_prob + 0.40 * (1.0 - entropy_norm)));
    elseif isstruct(sev_res) && isfield(sev_res, 'confidence') && ~isnan(sev_res.confidence)
        max_prob = max(0, min(1.0, sev_res.confidence));
        entropy_norm = 1.0 - max_prob;
        entropy_val = entropy_norm * log(5);
        if isfield(sev_res, 'expected_grade') && ~isnan(sev_res.expected_grade)
            exp_grade = sev_res.expected_grade;
        else
            exp_grade = max_prob * 4.0;
        end
        S_c = 100.0 * max_prob;
    else
        S_c = 50.0;
    end

    % -------------------------------------------------------------------------
    % Factor 3: Decision-Boundary Margin Sub-Score (S_m in [0, 100])
    % -------------------------------------------------------------------------
    tau_star = cfg.fusion.referable_threshold;
    if isfield(cfg, 'reliability') && isfield(cfg.reliability, 'boundary_margin')
        margin_delta = cfg.reliability.boundary_margin;
    else
        margin_delta = 0.15;
    end

    p_fused = 0.50;
    if isstruct(fus_res) && isfield(fus_res, 'p_referable_fused') && ~isnan(fus_res.p_referable_fused)
        p_fused = fus_res.p_referable_fused;
    elseif isstruct(sev_res) && isfield(sev_res, 'p_referable') && ~isnan(sev_res.p_referable)
        p_fused = sev_res.p_referable;
    end

    p_fused = max(0, min(1.0, p_fused));
    dist_to_thresh = abs(p_fused - tau_star);
    S_m = 100.0 * min(1.0, dist_to_thresh / max(1e-4, margin_delta));

    % -------------------------------------------------------------------------
    % Factor 4: Evidence Completeness Sub-Score (S_comp in [0, 100])
    % -------------------------------------------------------------------------
    comp = struct();
    comp.quality  = double(isstruct(q_rep) && isfield(q_rep, 'score') && ~isnan(q_rep.score));
    comp.cnn      = double(isstruct(sev_res) && isfield(sev_res, 'probabilities') && ~any(isnan(sev_res.probabilities)));
    comp.anatomy  = double(isstruct(od_res) && isfield(od_res, 'center') && ~isempty(od_res.center));
    comp.vessels  = double(isstruct(vess_res) && isfield(vess_res, 'morphometry') && isfield(vess_res.morphometry, 'vessel_density'));
    comp.lesions  = double(isstruct(lesion_ev) && isfield(lesion_ev, 'ma_count') && ~isnan(lesion_ev.ma_count));
    comp.fusion   = double(isstruct(fus_res) && isfield(fus_res, 'p_referable_fused') && ~isnan(fus_res.p_referable_fused));

    % Evaluate availability
    comp_channels = [comp.quality, comp.cnn, comp.anatomy, comp.vessels, comp.lesions, comp.fusion];
    comp.overall = mean(comp_channels);
    S_comp = 100.0 * comp.overall;

    % -------------------------------------------------------------------------
    % Factor 5: Multi-Modal Agreement Sub-Score (S_agree in [0, 100])
    % -------------------------------------------------------------------------
    raw_agree = 0.90;
    if isstruct(fus_res) && isfield(fus_res, 'agreement_score') && ~isnan(fus_res.agreement_score)
        raw_agree = fus_res.agreement_score;
    end

    % CNN vs Physical Lesion Referral Concordance
    cnn_ref_flag = (p_fused >= tau_star);
    lesion_ref_flag = false;

    if isstruct(lesion_ev)
        ma_cnt  = 0; if isfield(lesion_ev, 'ma_count'), ma_cnt = lesion_ev.ma_count; end
        he_area = 0; if isfield(lesion_ev, 'he_area_fraction'), he_area = lesion_ev.he_area_fraction; end
        ex_area = 0; if isfield(lesion_ev, 'ex_area_fraction'), ex_area = lesion_ev.ex_area_fraction; end
        dme_rk  = 0; if isfield(lesion_ev, 'dme_risk'), dme_rk = lesion_ev.dme_risk; end
        lesion_ref_flag = (ma_cnt >= 5) || (he_area >= 0.001) || (ex_area >= 0.001) || (dme_rk >= 1.0);
    end

    if cnn_ref_flag == lesion_ref_flag
        concordance_score = 1.0;
        agreement_class = 'CONCORDANT';
    else
        concordance_score = max(0.20, 1.0 - abs(p_fused - double(lesion_ref_flag)));
        agreement_class = 'DISCORDANT';
    end

    % Grad-CAM spatial agreement bonus if available
    cam_bonus = 0;
    if isstruct(cam_agree) && isfield(cam_agree, 'dice') && ~isnan(cam_agree.dice)
        cam_bonus = 0.10 * cam_agree.dice;
    end

    S_agree = 100.0 * max(0, min(1.0, 0.50 * raw_agree + 0.50 * concordance_score + cam_bonus));

    % -------------------------------------------------------------------------
    % Composite Score Calculation (R in [0, 100])
    % -------------------------------------------------------------------------
    if isfield(cfg, 'reliability') && isfield(cfg.reliability, 'weights')
        wq   = cfg.reliability.weights.quality;
        wc   = cfg.reliability.weights.confidence;
        wm   = cfg.reliability.weights.margin;
        wcmp = cfg.reliability.weights.completeness;
        wa   = cfg.reliability.weights.agreement;
    else
        wq = 0.20; wc = 0.20; wm = 0.20; wcmp = 0.20; wa = 0.20;
    end

    % Normalize weights
    total_w = wq + wc + wm + wcmp + wa;
    if abs(total_w - 1.0) > 1e-4 && total_w > 0
        wq = wq / total_w; wc = wc / total_w; wm = wm / total_w; wcmp = wcmp / total_w; wa = wa / total_w;
    end

    weights_used = struct('quality', wq, 'confidence', wc, 'margin', wm, 'completeness', wcmp, 'agreement', wa);

    R = wq * S_q + wc * S_c + wm * S_m + wcmp * S_comp + wa * S_agree;

    % Safety Override: Ungradable image restricts maximum reliability score
    if is_ungradable
        R = min(R, 30.0);
    end
    R = max(0, min(100.0, R));

    % -------------------------------------------------------------------------
    % Categorical Uncertainty Level & Computational Confidence Level
    % -------------------------------------------------------------------------
    th_accept   = 60.0;
    th_moderate = 40.0;
    if isfield(cfg, 'reliability')
        if isfield(cfg.reliability, 'threshold_accept'), th_accept = cfg.reliability.threshold_accept; end
        if isfield(cfg.reliability, 'threshold_moderate'), th_moderate = cfg.reliability.threshold_moderate; end
    end

    % Confidence Level
    if R >= th_accept
        conf_level = 'HIGH';
    elseif R >= th_moderate
        conf_level = 'MODERATE';
    else
        conf_level = 'LOW';
    end

    % Uncertainty Level
    if R >= th_accept && dist_to_thresh >= margin_delta
        uncert_level = 'LOW';
    elseif R >= th_moderate || dist_to_thresh >= (margin_delta * 0.5)
        uncert_level = 'MODERATE';
    else
        uncert_level = 'HIGH';
    end

    % -------------------------------------------------------------------------
    % Transparent Reliability Reasons Array
    % -------------------------------------------------------------------------
    reasons = {};

    % Quality reasons
    if strcmp(q_class, 'UNGRADABLE')
        reasons{end+1} = 'Ungradable image quality: optical defects obstruct retinal field';
    elseif strcmp(q_class, 'BORDERLINE')
        reasons{end+1} = 'Borderline image quality: sub-optimal sharpness or exposure';
    else
        reasons{end+1} = 'Good image quality: clear optical focus across retinal field';
    end

    % Margin reasons
    if dist_to_thresh < (margin_delta * 0.5)
        reasons{end+1} = sprintf('Prediction is close to the referral decision boundary (|P_ref - %.2f| = %.3f)', tau_star, dist_to_thresh);
    elseif dist_to_thresh >= margin_delta
        reasons{end+1} = sprintf('Prediction well separated from referral decision boundary (|P_ref - %.2f| = %.3f)', tau_star, dist_to_thresh);
    end

    % Entropy & Confidence reasons
    if entropy_norm > 0.65
        reasons{end+1} = sprintf('High posterior entropy (H_norm = %.2f): ambiguous multi-class distribution', entropy_norm);
    elseif max_prob >= 0.85
        reasons{end+1} = sprintf('High posterior class confidence (max P = %.1f%%)', max_prob * 100);
    elseif max_prob < 0.55
        reasons{end+1} = sprintf('Low maximum class probability (max P = %.1f%%)', max_prob * 100);
    end

    % Completeness reasons
    if comp.overall < 0.80
        reasons{end+1} = 'Evidence channels incomplete: one or more sub-pipelines did not produce valid data';
    else
        reasons{end+1} = 'Evidence channels complete: quality, anatomy, vessels, lesions, and CNN available';
    end

    % Agreement reasons
    if strcmp(agreement_class, 'DISCORDANT')
        reasons{end+1} = 'CNN prediction and physical lesion evidence are discordant';
    else
        reasons{end+1} = 'Multi-modal evidence is concordant between CNN prediction and lesion burden';
    end

    % -------------------------------------------------------------------------
    % Assemble Output Structure
    % -------------------------------------------------------------------------
    reliability_res = struct();
    reliability_res.reliability_score            = R;
    reliability_res.is_reliable                  = (R >= th_accept);
    reliability_res.confidence_level             = conf_level;
    reliability_res.uncertainty_level            = uncert_level;
    reliability_res.quality_status               = q_class;
    reliability_res.evidence_agreement           = agreement_class;
    reliability_res.reliability_reasons          = reasons;
    
    % Sub-Scores
    reliability_res.quality_score_component      = S_q;
    reliability_res.confidence_score_component   = S_c;
    reliability_res.margin_score_component       = S_m;
    reliability_res.completeness_score_component = S_comp;
    reliability_res.agreement_score_component    = S_agree;
    reliability_res.weights_used                 = weights_used;

    % Detailed Diagnostics
    reliability_res.max_probability              = max_prob;
    reliability_res.entropy                      = entropy_val;
    reliability_res.normalized_entropy           = entropy_norm;
    reliability_res.expected_grade               = exp_grade;
    reliability_res.p_ref_fused                  = p_fused;
    reliability_res.tau_star                     = tau_star;
    reliability_res.distance_to_threshold        = dist_to_thresh;
    reliability_res.boundary_margin              = margin_delta;
    reliability_res.completeness                 = comp;
    reliability_res.agreement_score              = raw_agree;

    % Backward-Compatible Legacy Fields
    reliability_res.confidence_factor            = max_prob;
    reliability_res.quality_factor               = S_q / 100.0;
    reliability_res.agreement_factor             = S_agree / 100.0;
end
