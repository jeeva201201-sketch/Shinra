function capture_app_screenshots()
    fprintf('Initializing SHINRA runtime for UI screenshots...\n');
    test_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(test_dir);
    addpath(project_root);
    addpath(fullfile(project_root, 'app'));
    setup_SHINRA();

    img_path = fullfile(project_root, 'SHINRAdataset', 'IDRid', ...
        'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
    fprintf('Running inference on %s...\n', img_path);
    res = runSHINRASingleImage(img_path);

    fprintf('Instantiating SHINRA_App...\n');
    app = SHINRA_App();
    app.UIFigure.Position = [50, 50, 1440, 900];
    app.DiagnosticResult = res;
    app.populateResults(res);
    app.onViewModeChanged();
    drawnow;

    out_dir = fullfile(project_root, 'results', 'reports');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    % 1. Tab 1: Overview
    app.ResultsTabGroup.SelectedTab = app.OverviewTab;
    drawnow;
    pause(0.5);
    tab1_file = fullfile(out_dir, 'screenshot_tab1_overview.png');
    exportapp(app.UIFigure, tab1_file);
    fprintf('Saved Tab 1 screenshot to %s\n', tab1_file);

    % 2. Tab 2: Biomarkers & Reliability
    app.ResultsTabGroup.SelectedTab = app.BiomarkersTab;
    drawnow;
    pause(0.5);
    tab2_file = fullfile(out_dir, 'screenshot_tab2_biomarkers.png');
    exportapp(app.UIFigure, tab2_file);
    fprintf('Saved Tab 2 screenshot to %s\n', tab2_file);

    % 3. Tab 3: Structured Clinical Report
    app.ResultsTabGroup.SelectedTab = app.ReportTab;
    drawnow;
    pause(0.5);
    tab3_file = fullfile(out_dir, 'screenshot_tab3_structured_report.png');
    exportapp(app.UIFigure, tab3_file);
    fprintf('Saved Tab 3 screenshot to %s\n', tab3_file);

    delete(app);
    fprintf('UI screenshot capture complete!\n');
end
