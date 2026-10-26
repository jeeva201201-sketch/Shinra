function [train_sub, val_sub] = split_dataset(data_table, val_ratio, random_seed)
% SPLIT_DATASET Reproducible Stratified Dataset Splitting
% Partitions a dataset table into training and validation subsets
% while maintaining balanced class proportions across DR severity grades.
%
% Prevents data leakage by ensuring strict separation of training & validation.

    if nargin < 2 || isempty(val_ratio)
        val_ratio = 0.20; % 20% validation split default
    end
    if nargin < 3 || isempty(random_seed)
        cfg = shinra_config();
        random_seed = cfg.random_seed;
    end
    
    rng(random_seed, 'twister');
    n = height(data_table);
    
    if ismember('DR_Grade', data_table.Properties.VariableNames) && ~any(isnan(data_table.DR_Grade))
        % Stratified splitting across DR Grades (0 to 4)
        unique_grades = unique(data_table.DR_Grade);
        train_idx = [];
        val_idx = [];
        
        for g = 1:numel(unique_grades)
            grade = unique_grades(g);
            grade_indices = find(data_table.DR_Grade == grade);
            num_samples = numel(grade_indices);
            
            % Random permutation for this class
            shuffled = grade_indices(randperm(num_samples));
            num_val = max(1, round(num_samples * val_ratio));
            
            val_idx = [val_idx; shuffled(1:num_val)];
            train_idx = [train_idx; shuffled(num_val+1:end)];
        end
    else
        % Uniform random split if grades are not available
        shuffled = randperm(n)';
        num_val = max(1, round(n * val_ratio));
        val_idx = shuffled(1:num_val);
        train_idx = shuffled(num_val+1:end);
    end
    
    train_sub = data_table(train_idx, :);
    val_sub   = data_table(val_idx, :);
    
    % Store split metadata
    split_info = struct();
    split_info.total_samples = n;
    split_info.train_samples = height(train_sub);
    split_info.val_samples = height(val_sub);
    split_info.val_ratio = val_ratio;
    split_info.random_seed = random_seed;
    split_info.timestamp = char(datetime('now'));
    
    cfg = shinra_config();
    save(fullfile(cfg.paths.metrics, 'dataset_split_info.mat'), 'split_info');
end
