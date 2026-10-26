function cfg = shinra_config()
% SHINRA_CONFIG Central Configuration for SHINRA Retinal Screening System
% Returns a configuration structure containing all system parameters,
% directory paths, model hyperparameters, default thresholds, and feature flags.
%
% Designed for MATLAB Desktop and MATLAB Online compatibility (pure relative paths).

    % Root directory resolution
    root_dir = get_project_root();
    
    cfg = struct();
    cfg.version = '1.0.0';
    cfg.system_name = 'SHINRA: Explainable DR Screening System for Rural India';
    cfg.random_seed = 42;
    
    % Directory Structure (relative path resolution via fullfile)
    cfg.paths = struct();
    cfg.paths.root = root_dir;
    cfg.paths.dataset_root = fullfile(root_dir, 'SHINRAdataset');
    cfg.paths.idrid = fullfile(root_dir, 'SHINRAdataset', 'IDRid');
    cfg.paths.drive = fullfile(root_dir, 'SHINRAdataset', 'DRIVE');
    cfg.paths.aptos = fullfile(root_dir, 'SHINRAdataset', 'APTOS 2019');
    
    cfg.paths.models = fullfile(root_dir, 'models');
    cfg.paths.models_quality = fullfile(root_dir, 'models', 'quality');
    cfg.paths.models_vessels = fullfile(root_dir, 'models', 'vessels');
    cfg.paths.models_lesions = fullfile(root_dir, 'models', 'lesions');
    cfg.paths.models_severity = fullfile(root_dir, 'models', 'severity');
    cfg.paths.models_fusion = fullfile(root_dir, 'models', 'fusion');
    
    cfg.paths.results = fullfile(root_dir, 'results');
    cfg.paths.figures = fullfile(root_dir, 'results', 'figures');
    cfg.paths.reports = fullfile(root_dir, 'results', 'reports');
    cfg.paths.metrics = fullfile(root_dir, 'results', 'metrics');
    cfg.paths.predictions = fullfile(root_dir, 'results', 'predictions');
    cfg.paths.logs = fullfile(root_dir, 'results', 'logs');
    
    % Preprocessing Settings
    cfg.preprocessing = struct();
    cfg.preprocessing.target_size = [512, 512]; % Standard fundus analysis resolution
    cfg.preprocessing.crop_margin = 0.02;       % 2% margin around detected FOV
    cfg.preprocessing.clahe_clip_limit = 0.015; % Clip limit for adapthisteq
    cfg.preprocessing.clahe_num_tiles = [8, 8];
    cfg.preprocessing.gauss_sigma_illum = 30;   % Sigma for illumination compensation
    cfg.preprocessing.denoise_median_size = [3, 3];
    
    % Quality Assessment Thresholds (Initial defaults, calibratable from validation)
    cfg.quality = struct();
    cfg.quality.weights = struct(...
        'sharpness', 0.25, ...
        'illumination', 0.20, ...
        'exposure', 0.20, ...
        'contrast', 0.15, ...
        'fov_coverage', 0.20);
    cfg.quality.threshold_good = 65.0;       % Quality score >= 65 -> GOOD
    cfg.quality.threshold_borderline = 40.0; % 40 <= Quality score < 65 -> BORDERLINE
    cfg.quality.laplacian_blur_min = 25.0;   % Minimum Laplacian variance for sharp image
    cfg.quality.fov_coverage_min = 0.50;     % Retinal mask must cover at least 50% of FOV
    cfg.quality.underexposure_max = 0.40;    % Max dark pixel fraction
    cfg.quality.overexposure_max = 0.15;     % Max saturated bright pixel fraction
    
    % Severity Classification (5-Class ICDR DR Scale)
    cfg.severity = struct();
    cfg.severity.class_names = {'No_DR', 'Mild_NPDR', 'Moderate_NPDR', 'Severe_NPDR', 'Proliferative_DR'};
    cfg.severity.num_classes = 5;
    cfg.severity.input_size = [224, 224, 3]; % CNN input dimensions
    cfg.severity.referable_threshold_grade = 2; % Level >= 2 is Referable DR
    cfg.severity.learning_rate = 1e-4;
    cfg.severity.max_epochs = 20;
    cfg.severity.mini_batch_size = 16;
    cfg.severity.l2_regularization = 1e-4;
    
    % Vessel Segmentation (DRIVE)
    cfg.vessels = struct();
    cfg.vessels.target_size = [512, 512];
    cfg.vessels.num_filters = [32, 64, 128, 256];
    cfg.vessels.mini_batch_size = 4;
    cfg.vessels.max_epochs = 30;
    cfg.vessels.learning_rate = 1e-3;
    cfg.vessels.binarization_threshold = 0.45;
    
    % Evidence Fusion & Decision Engine
    cfg.fusion = struct();
    cfg.fusion.referable_threshold = 0.50;           % Historical baseline referral threshold
    cfg.fusion.referable_threshold_candidate = 0.52; % DEVELOPMENT_OPTIMIZED_CANDIDATE threshold
    cfg.fusion.candidate_id = 'DEVELOPMENT_OPTIMIZED_CANDIDATE';
    cfg.fusion.reliability_min_threshold = 0.45;     % Minimum reliability for autonomous screening
    cfg.fusion.hidden_units = [32, 16];
    cfg.fusion.learning_rate = 1e-3;
    cfg.fusion.max_epochs = 100;
    
    % Confidence & Probability Calibration (Fitted strictly on 330 Development Training images)
    cfg.calibration = struct();
    cfg.calibration.method = 'platt';                % 'platt' or 'temperature'
    cfg.calibration.num_bins = 10;
    cfg.calibration.platt = struct(...
        'slope', 1.3106, ...
        'intercept', -1.1412, ...
        'source', '330_dev_train_partition_only');
    cfg.calibration.temperature = struct(...
        'T_opt', 0.818, ...
        'source', '330_dev_train_partition_only');
        
    % Release Configuration: SHINRA_DEVELOPMENT_RELEASE_V1
    cfg.release = struct();
    cfg.release.name = 'SHINRA_DEVELOPMENT_RELEASE_V1';
    cfg.release.status = 'DEVELOPMENT_FROZEN';
    cfg.release.referable_threshold = 0.30;
    cfg.release.reliability_threshold = 60.0;
    cfg.release.calibration_slope = 1.3106;
    cfg.release.calibration_intercept = -1.1412;
    cfg.release.source = 'Fitted on 330 dev train; threshold selected on 83 dev validation';
    
    cfg.fusion.referable_threshold_release_v1 = 0.30;
    
    % Engineering Reliability & Safety-Triage Layer Configuration
    % (Note: Engineering uncertainty & triage thresholds; not clinically validated)
    cfg.reliability = struct();
    cfg.reliability.enabled = true;
    cfg.reliability.threshold_accept = 60.0;      % Reliability score >= 60 -> ACCEPT_SCREENING
    cfg.reliability.threshold_moderate = 40.0;    % 40 <= Score < 60 -> MODERATE / CLINICAL_REVIEW
    cfg.reliability.boundary_margin = 0.15;       % Decision boundary uncertainty margin |P_ref - tau*|
    cfg.reliability.require_quality_gate = true;  % Enforce quality safety gate (UNGRADABLE -> RECAPTURE)
    
    % Five-Factor Multi-Modal Reliability Sub-Score Weights (Sum normalized internally to 1.0)
    cfg.reliability.weights = struct();
    cfg.reliability.weights.quality      = 0.20;  % Image quality sub-score (S_q)
    cfg.reliability.weights.confidence   = 0.20;  % Model confidence / entropy sub-score (S_c)
    cfg.reliability.weights.margin       = 0.20;  % Boundary distance sub-score (S_m)
    cfg.reliability.weights.completeness = 0.20;  % Evidence availability sub-score (S_comp)
    cfg.reliability.weights.agreement    = 0.20;  % Multi-modal concordance sub-score (S_agree)
    
    % Categorical Confidence Level Thresholds [0, 100]
    cfg.reliability.confidence_thresholds = struct();
    cfg.reliability.confidence_thresholds.high     = 70.0; % Score >= 70 -> HIGH confidence
    cfg.reliability.confidence_thresholds.moderate = 45.0; % 45 <= Score < 70 -> MODERATE confidence
    
    % Categorical Uncertainty Level Thresholds [0, 100]
    cfg.reliability.uncertainty_thresholds = struct();
    cfg.reliability.uncertainty_thresholds.low      = 30.0; % Uncertainty <= 30 -> LOW uncertainty
    cfg.reliability.uncertainty_thresholds.moderate = 60.0; % 30 < Uncertainty <= 60 -> MODERATE uncertainty
    
    % Execution Flags (Controls retraining vs pre-trained checkpoint loading)
    cfg.flags = struct();
    cfg.flags.train_quality_model = false;
    cfg.flags.train_vessel_model = false;
    cfg.flags.train_severity_model = false;
    cfg.flags.train_fusion_model = false;
    cfg.flags.run_evaluations = true;
    cfg.flags.save_visualizations = true;
    cfg.flags.verbose = true;
    
    % Simulink Screening System Simulation Parameters (Rural Screening Capacity)
    cfg.simulink = struct();
    cfg.simulink.annual_patient_target = 100000; % 100,000+ patient/year target
    cfg.simulink.operating_days_per_year = 250;
    cfg.simulink.operating_hours_per_day = 8;
    cfg.simulink.num_screening_camps = 25;       % 25 distributed rural primary health centers
    cfg.simulink.mean_arrival_interval_sec = 120; % 1 patient every 2 mins per camp
    cfg.simulink.acquisition_time_sec = 45;
    cfg.simulink.quality_fail_rate = 0.08;       % 8% initial ungradable rate
    cfg.simulink.recapture_success_rate = 0.85;  % 85% recovered after recapture guidance
    cfg.simulink.preprocessing_time_sec = 0.8;
    cfg.simulink.ai_inference_time_sec = 1.2;
    cfg.simulink.evidence_fusion_time_sec = 0.2;
    cfg.simulink.report_generation_time_sec = 0.5;
    cfg.simulink.autonomous_decision_rate = 0.82; % 82% autonomous NOT REFERABLE / REFERABLE
    cfg.simulink.specialist_triage_rate = 0.18;   % 18% complex/uncertain cases referred to human expert
end
