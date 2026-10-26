function [drive_train, drive_test] = load_drive()
% LOAD_DRIVE Load and pair DRIVE dataset images and ground-truth masks
% Returns tables for training and test sets containing:
% ImageID, ImagePath, ManualMaskPath, FOVMaskPath.

    cfg = shinra_config();
    inventory = scan_datasets();
    
    if inventory.drive.num_train_images == 0
        error('DRIVE dataset not found in %s', cfg.paths.drive);
    end
    
    % Process Training Set
    drive_train = pair_drive_set(...
        inventory.drive.train_img_dir, ...
        inventory.drive.train_manual_dir, ...
        inventory.drive.train_mask_dir, ...
        true);
        
    % Process Testing Set
    drive_test = pair_drive_set(...
        inventory.drive.test_img_dir, ...
        '', ...
        inventory.drive.test_mask_dir, ...
        false);
end

function tab = pair_drive_set(img_dir, manual_dir, mask_dir, is_train)
    img_files = dir(fullfile(img_dir, '*.tif'));
    n = numel(img_files);
    
    image_ids = cell(n, 1);
    image_paths = cell(n, 1);
    manual_paths = cell(n, 1);
    fov_paths = cell(n, 1);
    
    for i = 1:n
        fname = img_files(i).name;
        [~, base_name, ~] = fileparts(fname);
        image_ids{i} = base_name;
        image_paths{i} = fullfile(img_files(i).folder, fname);
        
        % Extract ID prefix (e.g., '21_training' -> '21', '01_test' -> '01')
        tokens = regexp(base_name, '^(\d+)', 'tokens');
        if ~isempty(tokens)
            id_str = tokens{1}{1};
        else
            id_str = base_name;
        end
        
        % Search corresponding manual mask
        if is_train && ~isempty(manual_dir) && exist(manual_dir, 'dir')
            man_candidate = fullfile(manual_dir, [id_str, '_manual1.gif']);
            if exist(man_candidate, 'file')
                manual_paths{i} = man_candidate;
            else
                manual_paths{i} = '';
            end
        else
            manual_paths{i} = '';
        end
        
        % Search corresponding FOV mask
        if ~isempty(mask_dir) && exist(mask_dir, 'dir')
            if is_train
                fov_candidate = fullfile(mask_dir, [id_str, '_training_mask.gif']);
            else
                fov_candidate = fullfile(mask_dir, [id_str, '_test_mask.gif']);
            end
            
            if exist(fov_candidate, 'file')
                fov_paths{i} = fov_candidate;
            else
                fov_paths{i} = '';
            end
        else
            fov_paths{i} = '';
        end
    end
    
    tab = table(image_ids, image_paths, manual_paths, fov_paths, ...
        'VariableNames', {'ImageID', 'ImagePath', 'ManualMaskPath', 'FOVMaskPath'});
end
