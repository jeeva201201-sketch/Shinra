function main()
% MAIN Master Execution Entry Point for SHINRA Retinal Screening System
% Coordinates environment initialization, dataset validation, optional model training,
% progressive ablation benchmarking, robustness stress-testing, and Simulink throughput modeling.
%
% Fully native for MATLAB Online and desktop MATLAB environments.

    fprintf('================================================================================\n');
    fprintf('           SHINRA: QUALITY-AWARE EXPLAINABLE DR SCREENING SYSTEM               \n');
    fprintf('          Smart India Hackathon - Rural Community Healthcare Deployment         \n');
    fprintf('================================================================================\n\n');

    % 1. Environment & Path Initialization
    status = setup_SHINRA();
    cfg = shinra_config();

    % 2. Dataset Audit & Summary Report
    fprintf('[Phase 1/6] Generating Comprehensive Dataset Audit Report...\n');
    summary_txt = dataset_summary_report();
    fprintf('Audit report saved to results/reports/dataset_inventory_report.txt\n\n');

    % 3. Model Training (Controlled by config flags)
    if cfg.flags.train_vessel_model
        fprintf('[Phase 2/6] Training Retinal Vessel U-Net on DRIVE Dataset...\n');
        trainVesselSegmentation();
    else
        fprintf('[Phase 2/6] Retinal Vessel Model: Using existing checkpoint or morphological engine.\n');
    end

    if cfg.flags.train_severity_model
        fprintf('[Phase 2/6] Training 5-Class Severity CNN on IDRiD Dataset...\n');
        trainSeverityModel();
    else
        fprintf('[Phase 2/6] DR Severity Model: Using existing trained checkpoint.\n');
    end

    if cfg.flags.train_fusion_model
        fprintf('[Phase 2/6] Training Multi-Modal Evidence Fusion Model...\n');
        trainFusionModel();
    else
        fprintf('[Phase 2/6] Evidence Fusion: Using existing checkpoint or calibrated aggregator.\n');
    end

    % 4. Progressive Ablation Study (Benchmark Experiments A through F)
    if cfg.flags.run_evaluations
        fprintf('\n[Phase 3/6] Running Progressive Ablation Benchmark (Experiments A -> F)...\n');
        runAblationStudy(25); % Fast representative evaluation subset

        % 5. Controlled Robustness Stress-Testing
        fprintf('\n[Phase 4/6] Running Synthetic Field Degradation Robustness Test...\n');
        runRobustnessTest(20);

        % 6. Cross-Dataset Generalization Audit
        fprintf('\n[Phase 5/6] Running Cross-Dataset Generalization Audit...\n');
        runCrossDatasetEvaluation(15);
    end

    % 7. Interactive Sample Demonstration
    fprintf('\n[Phase 6/6] Executing Interactive Diagnostic Demo on Sample Retina...\n');
    demo();

    % 8. Simulink Rural Screening Capacity Simulation
    if exist('run_simulink_simulation', 'file') == 2
        fprintf('\nExecuting Simulink Rural Screening Capacity Simulation (100k+ Patients/Year)...\n');
        try
            run_simulink_simulation();
        catch ME
            fprintf('Simulink simulation notice: %s\n', ME.message);
        end
    end

    fprintf('================================================================================\n');
    fprintf('  SHINRA MASTER WORKFLOW EXECUTION COMPLETED SUCCESSFULLY                       \n');
    fprintf('  All reports saved to results/reports/ | Metrics to results/metrics/           \n');
    fprintf('================================================================================\n\n');
end
