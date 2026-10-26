function [results_table, batch_results] = run_batch_screening(target_input, max_images)
% RUN_BATCH_SCREENING High-Throughput Batch Retinal Screening Pipeline
% Evaluates an entire directory or datastore of fundus photographs.
% Features resilient error recovery, structured CSV exports, and throughput metrics.
%
% Usage:
%   [results_table, batch_results] = run_batch_screening("SHINRAdataset/APTOS 2019", 50);
%   [results_table, batch_results] = run_batch_screening(); % Uses default APTOS cohort

    cfg = shinra_config();
    if nargin < 1 || isempty(target_input)
        target_input = cfg.paths.aptos;
    end
    if nargin < 2 || isempty(max_images)
        max_images = Inf;
    end

    fprintf('============================================================\n');
    fprintf('           SHINRA HIGH-THROUGHPUT BATCH SCREENING           \n');
    fprintf('============================================================\n');

    % Discover image files
    if ischar(target_input) || isstring(target_input)
        in_path = char(target_input);
        if isfolder(in_path)
            img_files = dir(fullfile(in_path, '*.png'));
            if isempty(img_files)
                img_files = dir(fullfile(in_path, '*.jpg'));
            end
            if isempty(img_files)
                img_files = dir(fullfile(in_path, '*.tif'));
            end
            num_found = numel(img_files);
            file_paths = cell(num_found, 1);
            for k = 1:num_found
                file_paths{k} = fullfile(img_files(k).folder, img_files(k).name);
            end
        elseif isfile(in_path)
            file_paths = {in_path};
        else
            error('Invalid input target: %s', in_path);
        end
    elseif iscell(target_input)
        file_paths = target_input;
    else
        error('Unsupported input type for target_input');
    end

    total_images = min(max_images, numel(file_paths));
    if total_images == 0
        warning('No images found in target: %s', in_path);
        results_table = table();
        batch_results = [];
        return;
    end

    fprintf('Found %d retinal images. Starting batch screening...\n\n', total_images);

    batch_results = [];
    failed_count = 0;
    start_time = tic;

    for i = 1:total_images
        img_p = file_paths{i};
        [~, fname, ~] = fileparts(img_p);
        
        fprintf('[%4d / %4d] Screening: %-25s ... ', i, total_images, fname);
        
        try
            % Execute single image screening (suppress figure saving during high-throughput batch)
            res = run_single_image(img_p, false);
            batch_results = [batch_results; res]; %#ok<AGROW>
            fprintf('Status: %-10s | Decision: %s\n', res.quality.quality_class, res.decision.final_decision);
        catch ME
            failed_count = failed_count + 1;
            fprintf('FAILED: %s\n', ME.message);
        end
    end

    total_time_sec = toc(start_time);
    avg_time_per_image = total_time_sec / max(1, total_images - failed_count);

    fprintf('\n============================================================\n');
    fprintf('  BATCH SCREENING COMPLETE\n');
    fprintf('------------------------------------------------------------\n');
    fprintf('  • Total Screened:       %d images\n', total_images);
    fprintf('  • Successfully Parsed:  %d images\n', total_images - failed_count);
    fprintf('  • Processing Failures:  %d images\n', failed_count);
    fprintf('  • Total Elapsed Time:   %.2f seconds\n', total_time_sec);
    fprintf('  • Average Speed:        %.2f sec / image (%.1f images / min)\n', ...
        avg_time_per_image, 60 / max(0.01, avg_time_per_image));
    fprintf('============================================================\n\n');

    % Export consolidated CSV report
    output_csv = fullfile(cfg.paths.predictions, 'screening_predictions.csv');
    results_table = exportScreeningResults(batch_results, output_csv);
end
