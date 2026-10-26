function inventory = scan_datasets()
% SCAN_DATASETS Dynamically scan and catalog all available datasets
% Inspects SHINRAdataset/ (IDRiD, DRIVE, APTOS 2019) without hardcoded assumptions.
% Returns a structured catalog of files, paths, and label availability.

    cfg = shinra_config();
    inventory = struct();
    
    % 1. Scan IDRiD Dataset
    inventory.idrid = struct();
    idrid_root = cfg.paths.idrid;
    
    % Search for image subfolders dynamically
    idrid_train_img_dir = fullfile(idrid_root, 'B. Disease Grading', '1. Original Images', 'a. Training Set');
    idrid_test_img_dir  = fullfile(idrid_root, 'B. Disease Grading', '1. Original Images', 'b. Testing Set');
    idrid_labels_dir    = fullfile(idrid_root, 'B. Disease Grading', '2. Groundtruths');
    
    % Fallback directory search if folder names differ
    if ~exist(idrid_train_img_dir, 'dir')
        candidate_dirs = dir(fullfile(idrid_root, '**', '*Training*'));
        for k = 1:numel(candidate_dirs)
            if candidate_dirs(k).isdir && ~strcmp(candidate_dirs(k).name, '.') && ~strcmp(candidate_dirs(k).name, '..')
                idrid_train_img_dir = fullfile(candidate_dirs(k).folder, candidate_dirs(k).name);
                break;
            end
        end
    end
    
    if exist(idrid_train_img_dir, 'dir')
        train_files = dir(fullfile(idrid_train_img_dir, '*.jpg'));
        if isempty(train_files)
            train_files = dir(fullfile(idrid_train_img_dir, '*.png'));
        end
        inventory.idrid.train_image_files = train_files;
        inventory.idrid.num_train_images = numel(train_files);
        inventory.idrid.train_img_dir = idrid_train_img_dir;
    else
        inventory.idrid.train_image_files = [];
        inventory.idrid.num_train_images = 0;
        inventory.idrid.train_img_dir = '';
    end
    
    if exist(idrid_test_img_dir, 'dir')
        test_files = dir(fullfile(idrid_test_img_dir, '*.jpg'));
        if isempty(test_files)
            test_files = dir(fullfile(idrid_test_img_dir, '*.png'));
        end
        inventory.idrid.test_image_files = test_files;
        inventory.idrid.num_test_images = numel(test_files);
        inventory.idrid.test_img_dir = idrid_test_img_dir;
    else
        inventory.idrid.test_image_files = [];
        inventory.idrid.num_test_images = 0;
        inventory.idrid.test_img_dir = '';
    end
    
    % Check for ground truth CSVs
    train_csv = fullfile(idrid_labels_dir, 'a. IDRiD_Disease Grading_Training Labels.csv');
    test_csv  = fullfile(idrid_labels_dir, 'b. IDRiD_Disease Grading_Testing Labels.csv');
    
    if ~exist(train_csv, 'file')
        csv_candidates = dir(fullfile(idrid_root, '**', '*Training*Labels*.csv'));
        if ~isempty(csv_candidates)
            train_csv = fullfile(csv_candidates(1).folder, csv_candidates(1).name);
        end
    end
    
    if ~exist(test_csv, 'file')
        csv_candidates = dir(fullfile(idrid_root, '**', '*Testing*Labels*.csv'));
        if ~isempty(csv_candidates)
            test_csv = fullfile(csv_candidates(1).folder, csv_candidates(1).name);
        end
    end
    
    inventory.idrid.has_train_labels = exist(train_csv, 'file') == 2;
    inventory.idrid.train_labels_file = train_csv;
    if inventory.idrid.has_train_labels
        try
            opts = detectImportOptions(train_csv);
            t_train = readtable(train_csv, opts);
            inventory.idrid.num_train_labels = height(t_train);
        catch
            inventory.idrid.num_train_labels = 0;
        end
    else
        inventory.idrid.num_train_labels = 0;
    end
    
    inventory.idrid.has_test_labels = exist(test_csv, 'file') == 2;
    inventory.idrid.test_labels_file = test_csv;
    if inventory.idrid.has_test_labels
        try
            opts = detectImportOptions(test_csv);
            t_test = readtable(test_csv, opts);
            inventory.idrid.num_test_labels = height(t_test);
        catch
            inventory.idrid.num_test_labels = 0;
        end
    else
        inventory.idrid.num_test_labels = 0;
    end
    
    % Dynamic check for pixel-level lesion segmentation masks (Section A of IDRiD)
    lesion_mask_candidates = dir(fullfile(idrid_root, '**', '*Microaneurysms*'));
    inventory.idrid.has_lesion_masks = ~isempty(lesion_mask_candidates);
    
    % 2. Scan DRIVE Dataset
    inventory.drive = struct();
    drive_root = cfg.paths.drive;
    
    drive_train_img_dir = fullfile(drive_root, 'training', 'images');
    drive_train_man_dir = fullfile(drive_root, 'training', '1st_manual');
    drive_train_msk_dir = fullfile(drive_root, 'training', 'mask');
    drive_test_img_dir  = fullfile(drive_root, 'test', 'images');
    drive_test_msk_dir  = fullfile(drive_root, 'test', 'mask');
    
    inventory.drive.train_img_dir = drive_train_img_dir;
    inventory.drive.train_manual_dir = drive_train_man_dir;
    inventory.drive.train_mask_dir = drive_train_msk_dir;
    inventory.drive.test_img_dir = drive_test_img_dir;
    inventory.drive.test_mask_dir = drive_test_msk_dir;
    
    if exist(drive_train_img_dir, 'dir')
        d_tr_img = dir(fullfile(drive_train_img_dir, '*.tif'));
        inventory.drive.num_train_images = numel(d_tr_img);
        inventory.drive.train_image_files = d_tr_img;
    else
        inventory.drive.num_train_images = 0;
        inventory.drive.train_image_files = [];
    end
    
    if exist(drive_train_man_dir, 'dir')
        d_tr_man = dir(fullfile(drive_train_man_dir, '*.gif'));
        inventory.drive.num_train_masks = numel(d_tr_man);
        inventory.drive.train_manual_files = d_tr_man;
    else
        inventory.drive.num_train_masks = 0;
        inventory.drive.train_manual_files = [];
    end
    
    if exist(drive_test_img_dir, 'dir')
        d_te_img = dir(fullfile(drive_test_img_dir, '*.tif'));
        inventory.drive.num_test_images = numel(d_te_img);
        inventory.drive.test_image_files = d_te_img;
    else
        inventory.drive.num_test_images = 0;
        inventory.drive.test_image_files = [];
    end
    
    if exist(drive_test_msk_dir, 'dir')
        d_te_msk = dir(fullfile(drive_test_msk_dir, '*.gif'));
        inventory.drive.num_test_fov_masks = numel(d_te_msk);
    else
        inventory.drive.num_test_fov_masks = 0;
    end
    
    % 3. Scan APTOS 2019 Dataset
    inventory.aptos = struct();
    aptos_root = cfg.paths.aptos;
    inventory.aptos.root = aptos_root;
    
    if exist(aptos_root, 'dir')
        aptos_files = dir(fullfile(aptos_root, '*.png'));
        if isempty(aptos_files)
            aptos_files = dir(fullfile(aptos_root, '*.jpg'));
        end
        inventory.aptos.num_images = numel(aptos_files);
        inventory.aptos.image_files = aptos_files;
        
        % Check if a CSV exists
        aptos_csv = dir(fullfile(aptos_root, '*.csv'));
        inventory.aptos.has_labels = ~isempty(aptos_csv);
        if inventory.aptos.has_labels
            inventory.aptos.csv_file = fullfile(aptos_csv(1).folder, aptos_csv(1).name);
        else
            inventory.aptos.csv_file = '';
        end
    else
        inventory.aptos.num_images = 0;
        inventory.aptos.image_files = [];
        inventory.aptos.has_labels = false;
        inventory.aptos.csv_file = '';
    end
end
