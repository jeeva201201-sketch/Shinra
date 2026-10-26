function aptos_table = load_aptos()
% LOAD_APTOS Load APTOS 2019 dataset
% Dynamically scans the APTOS directory and constructs a table of images.
% Checks if ground-truth labels exist; if not, marks as unlabelled cohort.

    cfg = shinra_config();
    inventory = scan_datasets();
    
    if inventory.aptos.num_images == 0
        error('APTOS 2019 dataset not found in %s', cfg.paths.aptos);
    end
    
    img_files = inventory.aptos.image_files;
    n = numel(img_files);
    
    image_ids = cell(n, 1);
    image_paths = cell(n, 1);
    
    for i = 1:n
        [~, name, ~] = fileparts(img_files(i).name);
        image_ids{i} = name;
        image_paths{i} = fullfile(img_files(i).folder, img_files(i).name);
    end
    
    if inventory.aptos.has_labels
        % Load CSV labels if present
        opts = detectImportOptions(inventory.aptos.csv_file);
        t_csv = readtable(inventory.aptos.csv_file, opts);
        
        dr_grades = nan(n, 1);
        col_names = t_csv.Properties.VariableNames;
        id_col = col_names{1};
        label_col = col_names{2};
        
        csv_map = containers.Map(string(t_csv.(id_col)), double(t_csv.(label_col)));
        for i = 1:n
            if isKey(csv_map, string(image_ids{i}))
                dr_grades(i) = csv_map(string(image_ids{i}));
            end
        end
        referable_dr = double(dr_grades >= 2);
    else
        % Unlabelled cohort for batch inference & throughput testing
        dr_grades = nan(n, 1);
        referable_dr = nan(n, 1);
    end
    
    aptos_table = table(image_ids, image_paths, dr_grades, referable_dr, ...
        'VariableNames', {'ImageID', 'ImagePath', 'DR_Grade', 'Referable_DR'});
end
