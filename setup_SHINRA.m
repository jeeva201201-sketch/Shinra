function status = setup_SHINRA()
% SETUP_SHINRA Initialize the SHINRA Retinal Screening System environment
% Configures MATLAB paths, checks required toolboxes, verifies datasets,
% ensures output directory structures, and displays a comprehensive diagnostic report.
%
% Compatible with MATLAB Desktop (Windows, macOS, Linux) and MATLAB Online.

    fprintf('============================================================\n');
    fprintf('           SHINRA RETINAL SCREENING SYSTEM INITIALIZATION   \n');
    fprintf('   Explainable AI for Diabetic Retinopathy in Rural India   \n');
    fprintf('============================================================\n\n');

    % 1. Determine Project Root Directory
    current_file = mfilename('fullpath');
    if isempty(current_file)
        root_dir = pwd;
    else
        root_dir = fileparts(current_file);
    end
    fprintf('[1/5] Project Root Resolved: %s\n', root_dir);

    % 2. Add Required Folders to MATLAB Path
    folders_to_add = {
        fullfile(root_dir, 'config'), ...
        fullfile(root_dir, 'src', 'data'), ...
        fullfile(root_dir, 'src', 'preprocessing'), ...
        fullfile(root_dir, 'src', 'quality'), ...
        fullfile(root_dir, 'src', 'anatomy'), ...
        fullfile(root_dir, 'src', 'vessels'), ...
        fullfile(root_dir, 'src', 'lesions'), ...
        fullfile(root_dir, 'src', 'severity'), ...
        fullfile(root_dir, 'src', 'fusion'), ...
        fullfile(root_dir, 'src', 'reliability'), ...
        fullfile(root_dir, 'src', 'explainability'), ...
        fullfile(root_dir, 'src', 'evaluation'), ...
        fullfile(root_dir, 'src', 'reporting'), ...
        fullfile(root_dir, 'src', 'utilities'), ...
        fullfile(root_dir, 'models'), ...
        fullfile(root_dir, 'simulink'), ...
        fullfile(root_dir, 'tests')
    };

    for i = 1:numel(folders_to_add)
        if exist(folders_to_add{i}, 'dir')
            addpath(folders_to_add{i});
        end
    end
    fprintf('[2/5] Added %d subdirectories to MATLAB search path.\n', numel(folders_to_add));

    % 3. Verify and Create Output Directories
    required_dirs = {
        fullfile(root_dir, 'models', 'quality'), ...
        fullfile(root_dir, 'models', 'vessels'), ...
        fullfile(root_dir, 'models', 'lesions'), ...
        fullfile(root_dir, 'models', 'severity'), ...
        fullfile(root_dir, 'models', 'fusion'), ...
        fullfile(root_dir, 'results', 'figures', 'preprocessing'), ...
        fullfile(root_dir, 'results', 'figures', 'quality'), ...
        fullfile(root_dir, 'results', 'figures', 'anatomy'), ...
        fullfile(root_dir, 'results', 'figures', 'vessels'), ...
        fullfile(root_dir, 'results', 'figures', 'lesions'), ...
        fullfile(root_dir, 'results', 'figures', 'explainability'), ...
        fullfile(root_dir, 'results', 'figures', 'evaluation'), ...
        fullfile(root_dir, 'results', 'figures', 'robustness'), ...
        fullfile(root_dir, 'results', 'reports'), ...
        fullfile(root_dir, 'results', 'metrics'), ...
        fullfile(root_dir, 'results', 'predictions'), ...
        fullfile(root_dir, 'results', 'logs')
    };

    for i = 1:numel(required_dirs)
        if ~exist(required_dirs{i}, 'dir')
            mkdir(required_dirs{i});
        end
    end
    fprintf('[3/5] Verified runtime and output directories.\n');

    % 4. Detect MATLAB Toolboxes
    fprintf('\n[4/5] MATLAB Toolbox Verification:\n');
    toolboxes = struct();
    toolboxes.ImageProcessing = ~isempty(ver('images'));
    toolboxes.DeepLearning = ~isempty(ver('nnet'));
    toolboxes.ComputerVision = ~isempty(ver('vision'));
    toolboxes.StatisticsML = ~isempty(ver('stats'));
    toolboxes.Simulink = ~isempty(ver('simulink'));

    print_toolbox_status('MATLAB Core Engine', true, 'Core runtime platform');
    print_toolbox_status('Image Processing Toolbox', toolboxes.ImageProcessing, 'FOV detection, CLAHE, morphological filtering');
    print_toolbox_status('Deep Learning Toolbox', toolboxes.DeepLearning, 'U-Net vessel segmentation, CNN severity classifier, Grad-CAM');
    print_toolbox_status('Computer Vision Toolbox', toolboxes.ComputerVision, 'Feature detection, candidate matching');
    print_toolbox_status('Statistics & Machine Learning Toolbox', toolboxes.StatisticsML, 'Evidence fusion, Platt/Temperature calibration, evaluation metrics');
    print_toolbox_status('Simulink', toolboxes.Simulink, 'Rural clinic throughput & screening system capacity modeling');

    % 5. Discover and Validate Local Datasets
    fprintf('\n[5/5] Scanning and Validating Local Datasets:\n');
    dataset_inventory = scan_datasets();

    fprintf('------------------------------------------------------------\n');
    fprintf('  DATASET AUDIT SUMMARY\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('  • IDRiD Dataset:\n');
    fprintf('      - Training Images Found: %d\n', dataset_inventory.idrid.num_train_images);
    fprintf('      - Training Labels Found: %s (%d rows)\n', ...
        yes_no(dataset_inventory.idrid.has_train_labels), dataset_inventory.idrid.num_train_labels);
    fprintf('      - Testing Images Found:  %d\n', dataset_inventory.idrid.num_test_images);
    fprintf('      - Testing Labels Found:  %s (%d rows)\n', ...
        yes_no(dataset_inventory.idrid.has_test_labels), dataset_inventory.idrid.num_test_labels);
    fprintf('      - Lesion Masks Present:  %s\n', yes_no(dataset_inventory.idrid.has_lesion_masks));
    
    fprintf('  • DRIVE Dataset:\n');
    fprintf('      - Training Images Found: %d\n', dataset_inventory.drive.num_train_images);
    fprintf('      - Training Manual Masks: %d\n', dataset_inventory.drive.num_train_masks);
    fprintf('      - Testing Images Found:  %d\n', dataset_inventory.drive.num_test_images);
    fprintf('      - Testing FOV Masks:     %d\n', dataset_inventory.drive.num_test_fov_masks);

    fprintf('  • APTOS 2019 Dataset:\n');
    fprintf('      - Total Images Found:    %d\n', dataset_inventory.aptos.num_images);
    fprintf('      - Training Labels:       %s (Local files used as unlabelled cohort)\n', ...
        yes_no(dataset_inventory.aptos.has_labels));

    % Status structure output
    status = struct();
    status.initialized = true;
    status.root_dir = root_dir;
    status.toolboxes = toolboxes;
    status.datasets = dataset_inventory;
    status.config = shinra_config();

    fprintf('============================================================\n');
    fprintf('  SHINRA INITIALIZATION STATUS: SUCCESSFUL\n');
    fprintf('============================================================\n\n');
end

function print_toolbox_status(name, is_available, purpose)
    if is_available
        fprintf('  [AVAILABLE] %-36s : %s\n', name, purpose);
    else
        fprintf('  [MISSING]   %-36s : %s\n', name, purpose);
    end
end

function str = yes_no(val)
    if val
        str = 'YES';
    else
        str = 'NO';
    end
end
