function result = demo(sample_image_path)
% DEMO Quick-Start Interactive Demonstration of the SHINRA Screening System
% Initializes the environment, screens a sample retinal image, prints the
% clinical diagnostic report, and generates the 6-panel explainability figure.
%
% Usage:
%   demo();                          % Uses default IDRiD clinical sample
%   demo("path/to/custom_image.jpg"); % Uses user-specified fundus image

    fprintf('============================================================\n');
    fprintf('           SHINRA RETINAL SCREENING SYSTEM DEMO             \n');
    fprintf('   Smart India Hackathon: Explainable DR in Rural India    \n');
    fprintf('============================================================\n\n');

    % 1. Initialize environment & paths
    setup_SHINRA();
    cfg = shinra_config();

    % 2. Select Sample Image
    if nargin < 1 || isempty(sample_image_path)
        % Default sample from IDRiD training set
        [idrid_train, ~] = load_idrid();
        if height(idrid_train) > 0
            % Select a sample with moderate/severe DR for clear lesion demonstration
            mod_dr_idx = find(idrid_train.DR_Grade == 2, 1);
            if ~isempty(mod_dr_idx)
                sample_image_path = idrid_train.ImagePath{mod_dr_idx};
            else
                sample_image_path = idrid_train.ImagePath{1};
            end
        else
            error('No sample image found in IDRiD dataset.');
        end
    end

    fprintf('Selected Sample Image: %s\n\n', sample_image_path);

    % 3. Run Full Screening Pipeline
    result = run_single_image(sample_image_path, true);

    % 4. Display Report
    fprintf('%s\n', result.report_text);

    if isfield(result, 'panel_figure_path')
        fprintf('Saved 6-Panel Clinical Explainability Figure to:\n  %s\n\n', result.panel_figure_path);
    end
end
