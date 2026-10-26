function app = launch_app()
% LAUNCH_APP Launch the SHINRA Retinal Screening MATLAB App
% Initializes paths and starts the standalone SHINRA App Designer application.

    fprintf('Initializing SHINRA Environment...\n');
    app_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(app_dir);
    
    % Run project setup
    if exist(fullfile(project_root, 'setup_SHINRA.m'), 'file') == 2
        run(fullfile(project_root, 'setup_SHINRA.m'));
    end
    
    % Ensure app folder is on path
    addpath(app_dir);
    
    fprintf('Launching SHINRA Screening Application...\n');
    app = SHINRA_App();
    
    if nargout == 0
        clear app;
    end
end
