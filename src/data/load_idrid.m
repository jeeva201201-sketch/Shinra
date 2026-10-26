function [train_table, test_table] = load_idrid()
% LOAD_IDRID Load and validate IDRiD dataset tables
% Returns structured tables for IDRiD training and testing splits
% containing ImagePath, ImageID, DR_Grade, DME_Risk, and Referable_DR.

    cfg = shinra_config();
    inventory = scan_datasets();
    
    if inventory.idrid.num_train_images == 0
        error('IDRiD training images not found in %s', cfg.paths.idrid);
    end
    
    % Process Training Set
    train_table = process_split(...
        inventory.idrid.train_img_dir, ...
        inventory.idrid.train_labels_file, ...
        'Training');
        
    % Process Testing Set
    if inventory.idrid.num_test_images > 0
        test_table = process_split(...
            inventory.idrid.test_img_dir, ...
            inventory.idrid.test_labels_file, ...
            'Testing');
    else
        test_table = table();
    end
end

function out_tab = process_split(img_dir, csv_file, split_name)
    if isempty(csv_file) || exist(csv_file, 'file') ~= 2
        warning('Labels CSV missing for IDRiD %s set. Generating unlabelled table.', split_name);
        img_files = dir(fullfile(img_dir, '*.jpg'));
        if isempty(img_files)
            img_files = dir(fullfile(img_dir, '*.png'));
        end
        n = numel(img_files);
        image_ids = cell(n, 1);
        image_paths = cell(n, 1);
        for i = 1:n
            [~, name, ~] = fileparts(img_files(i).name);
            image_ids{i} = name;
            image_paths{i} = fullfile(img_files(i).folder, img_files(i).name);
        end
        out_tab = table(image_ids, image_paths, nan(n,1), nan(n,1), nan(n,1), ...
            'VariableNames', {'ImageID', 'ImagePath', 'DR_Grade', 'DME_Risk', 'Referable_DR'});
        return;
    end
    
    % Read CSV labels
    opts = detectImportOptions(csv_file);
    t_raw = readtable(csv_file, opts);
    
    col_names = t_raw.Properties.VariableNames;
    img_col = col_names{1};
    dr_col  = col_names{2};
    dme_col = col_names{3};
    
    n = height(t_raw);
    image_paths = cell(n, 1);
    valid_mask = true(n, 1);
    
    for i = 1:n
        raw_id = t_raw.(img_col){i};
        if iscell(raw_id)
            raw_id = raw_id{1};
        elseif isnumeric(raw_id)
            raw_id = sprintf('IDRiD_%03d', raw_id);
        end
        raw_id = strtrim(char(raw_id));
        
        % Search for corresponding image file
        candidate_jpg = fullfile(img_dir, [raw_id, '.jpg']);
        candidate_png = fullfile(img_dir, [raw_id, '.png']);
        
        if exist(candidate_jpg, 'file')
            image_paths{i} = candidate_jpg;
        elseif exist(candidate_png, 'file')
            image_paths{i} = candidate_png;
        else
            warning('Image file not found for label ID: %s in %s', raw_id, img_dir);
            valid_mask(i) = false;
        end
    end
    
    image_ids = t_raw.(img_col)(valid_mask);
    image_paths = image_paths(valid_mask);
    dr_grades = double(t_raw.(dr_col)(valid_mask));
    
    if isnumeric(t_raw.(dme_col))
        dme_risks = double(t_raw.(dme_col)(valid_mask));
    else
        dme_risks = nan(numel(image_ids), 1);
    end
    
    % Referable DR: Level >= 2 (Moderate NPDR, Severe NPDR, PDR)
    referable_dr = double(dr_grades >= 2);
    
    out_tab = table(image_ids, image_paths, dr_grades, dme_risks, referable_dr, ...
        'VariableNames', {'ImageID', 'ImagePath', 'DR_Grade', 'DME_Risk', 'Referable_DR'});
end
