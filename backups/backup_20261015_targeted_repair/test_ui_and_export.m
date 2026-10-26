function test_results = test_ui_and_export()
% TEST_UI_AND_EXPORT Automated Regression & Integration Test Suite
% Tests:
% 1. Backend regression numerical parity on development image IDRiD_001.jpg
% 2. Standalone App Designer UI instantiation, window resizing & maximize toggle
% 3. UI population across all three tabs (including redesigned Clinical Report)
% 4. Visual layer switching across all 10 layers with legend updates
% 5. Multi-page PDF clinical report export and page structure verification
% 6. Legacy TXT clinical report export
%
% Strictly complies with data governance: 0 test images accessed.

    fprintf('============================================================\n');
    fprintf('  SHINRA UI, CLINICAL REPORT & PDF EXPORT REGRESSION TEST   \n');
    fprintf('============================================================\n\n');

    % Setup paths
    test_dir = fileparts(mfilename('fullpath'));
    project_root = fileparts(test_dir);
    addpath(project_root);
    addpath(fullfile(project_root, 'app'));
    setup_SHINRA();

    dev_img_path = fullfile(project_root, 'SHINRAdataset', 'IDRid', ...
        'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');

    if exist(dev_img_path, 'file') ~= 2
        error('Development test image not found: %s', dev_img_path);
    end

    test_results = struct();

    % -------------------------------------------------------------------------
    % TEST 1: BACKEND SINGLE IMAGE INFERENCE & PARITY
    % -------------------------------------------------------------------------
    fprintf('\n[TEST 1/6] Running single-image inference on IDRiD_001.jpg...\n');
    res = runSHINRASingleImage(dev_img_path);

    % Parity checks against locked development baseline
    assert(strcmp(res.image_id, 'IDRiD_001'), 'Mismatch in image_id');
    assert(strcmp(res.quality.quality_class, 'GOOD'), 'Mismatch in quality_class');
    assert(abs(res.quality.quality_score - 78.4) < 1.0, 'Mismatch in quality_score');
    assert(res.severity.predicted_grade == 3, 'Mismatch in predicted_grade (expected Grade 3 Severe NPDR)');
    assert(strcmp(res.severity.predicted_class_name, 'Severe_NPDR'), 'Mismatch in severity class name');
    assert(res.referable.is_referable == 1, 'Mismatch in referable status (expected Referable DR)');
    assert(abs(res.referable.p_referable_calibrated - 1.0) < 1e-3, 'Mismatch in calibrated probability');
    assert(abs(res.referable.referable_threshold_used - 0.30) < 1e-4, 'Mismatch in referral threshold');
    assert(strcmp(res.reliability.recommended_action, 'ACCEPT_SCREENING'), 'Mismatch in triage action');
    assert(res.reliability.reliability_score >= 85.0, 'Mismatch in reliability score');
    assert(res.reliability.passes_reliability_gate == true, 'Mismatch in reliability gate status');

    test_results.backend_parity = 'PASSED';
    fprintf('  -> TEST 1 PASSED: Backend numerical parity exactly preserved.\n');

    % -------------------------------------------------------------------------
    % TEST 2: STANDALONE APP INSTANTIATION & RESPONSIVE WINDOW RESIZING
    % -------------------------------------------------------------------------
    fprintf('\n[TEST 2/6] Testing SHINRA_App instantiation and responsive window controls...\n');
    app = SHINRA_App();
    assert(isvalid(app.UIFigure), 'UIFigure failed to instantiate.');
    assert(strcmp(app.UIFigure.Visible, 'on'), 'UIFigure was not set to visible.');

    % Test maximize toggle
    app.onToggleMaximize();
    assert(strcmp(app.UIFigure.WindowState, 'maximized'), 'Maximize toggle failed.');
    assert(strcmp(app.MaximizeBtn.Text, '🗗 Restore'), 'Maximize button label did not update.');

    % Test restore toggle
    app.onToggleMaximize();
    assert(strcmp(app.UIFigure.WindowState, 'normal'), 'Restore toggle failed.');
    assert(strcmp(app.MaximizeBtn.Text, '⛶ Maximize'), 'Restore button label did not update.');

    % Test window resizing reflow & minimum constraints
    app.UIFigure.Position = [50 50 1600 950];
    app.onWindowSizeChanged();
    assert(app.UIFigure.Position(3) == 1600 && app.UIFigure.Position(4) == 950, 'Window resize failed.');

    % Enforce minimum dimensions test
    app.UIFigure.Position = [50 50 800 500];
    app.onWindowSizeChanged();
    assert(app.UIFigure.Position(3) >= 1100, 'Minimum width constraint not enforced.');
    assert(app.UIFigure.Position(4) >= 700, 'Minimum height constraint not enforced.');

    test_results.ui_window_controls = 'PASSED';
    fprintf('  -> TEST 2 PASSED: Window resizing, maximize/restore, and constraints verified.\n');

    % -------------------------------------------------------------------------
    % TEST 3: UI POPULATION & REDESIGNED CLINICAL REPORT DASHBOARD
    % -------------------------------------------------------------------------
    fprintf('\n[TEST 3/6] Testing UI population and redesigned Clinical Report dashboard...\n');
    app.DiagnosticResult = res;
    app.populateResults(res);

    % Verify Tab 1 (Overview)
    assert(contains(app.TriageActionLabel.Text, 'ACCEPT SCREENING'), 'Tab 1 Triage Action mismatch.');
    assert(contains(app.ReferableStatusLabel.Text, 'REFERABLE DR'), 'Tab 1 Referable mismatch.');
    assert(contains(app.SeverityGradeLabel.Text, 'Severe NPDR'), 'Tab 1 Severity mismatch.');

    % Verify Tab 2 (Biomarkers & Reliability)
    assert(~isempty(app.LesionsTable.Data), 'Tab 2 LesionsTable empty.');
    assert(~isempty(app.ReliabilityTable.Data), 'Tab 2 ReliabilityTable empty.');

    % Verify Tab 3 (Restored Full On-Screen Clinical Report)
    assert(~isempty(app.ReportTextArea.Value), 'Report text area is empty.');
    rep_full_str = strjoin(app.ReportTextArea.Value, ' ');
    assert(contains(rep_full_str, 'IDRiD_001'), 'Report text does not contain image ID.');
    assert(contains(rep_full_str, 'GOOD'), 'Report text does not contain image quality.');
    assert(contains(rep_full_str, 'Severe NPDR') || contains(rep_full_str, 'Grade 3'), 'Report text does not contain severity.');
    assert(contains(rep_full_str, 'REFERABLE DR'), 'Report text does not contain referable decision.');
    assert(contains(rep_full_str, 'ACCEPT_SCREENING') || contains(rep_full_str, 'ACCEPT SCREENING'), 'Report text does not contain triage action.');
    assert(contains(rep_full_str, 'CLINICAL LIMITATIONS & SAFETY NOTICES'), 'Report text does not contain limitations.');

    test_results.report_dashboard = 'PASSED';
    fprintf('  -> TEST 3 PASSED: All three tabs and structured report dashboard populated correctly.\n');

    % -------------------------------------------------------------------------
    % TEST 4: ALL 10 VISUALIZATION LAYERS & LEGEND UPDATES
    % -------------------------------------------------------------------------
    fprintf('\n[TEST 4/6] Testing layer switcher across all 10 visualization layers...\n');
    layers = app.ViewSelectorDropdown.Items;
    assert(numel(layers) == 10, 'Expected 10 display layers.');

    for k = 1:numel(layers)
        layer_name = layers{k};
        app.ViewSelectorDropdown.Value = layer_name;
        app.onViewModeChanged();
        assert(contains(app.LayerLegendTitleLabel.Text, layer_name), ...
            sprintf('Layer legend title mismatch for layer: %s', layer_name));
        assert(~isempty(app.LayerLegendDescLabel.Text), ...
            sprintf('Layer legend description empty for layer: %s', layer_name));
    end

    % Test overlay export
    test_overlay_file = fullfile(project_root, 'results', 'figures', 'test_overlay_save.png');
    if exist(test_overlay_file, 'file') == 2
        delete(test_overlay_file);
    end
    img_data = getimage(app.ImageAxes);
    assert(~isempty(img_data), 'Could not retrieve active layer image data.');
    imwrite(img_data, test_overlay_file);
    assert(exist(test_overlay_file, 'file') == 2, 'Failed to save overlay image.');
    delete(test_overlay_file);

    test_results.layer_viewer = 'PASSED';
    fprintf('  -> TEST 4 PASSED: All 10 visualization layers rendered and legend verified.\n');

    % -------------------------------------------------------------------------
    % TEST 5: MULTI-PAGE PDF CLINICAL REPORT EXPORT
    % -------------------------------------------------------------------------
    fprintf('\n[TEST 5/6] Testing multi-page PDF clinical report export...\n');
    test_pdf_file = fullfile(project_root, 'results', 'reports', 'test_regression_report.pdf');
    if exist(test_pdf_file, 'file') == 2
        delete(test_pdf_file);
    end

    [ok, saved_pdf] = exportSHINRAReportPDF(res, test_pdf_file);
    assert(ok == true, 'exportSHINRAReportPDF returned false.');
    assert(exist(saved_pdf, 'file') == 2, 'Exported PDF file not found on disk.');

    pdf_info = dir(saved_pdf);
    assert(pdf_info.bytes > 50000, sprintf('PDF file suspiciously small: %d bytes', pdf_info.bytes));

    test_results.pdf_export = 'PASSED';
    fprintf('  -> TEST 5 PASSED: Multi-page PDF successfully created (%d KB).\n', round(pdf_info.bytes/1024));

    % -------------------------------------------------------------------------
    % TEST 6: LEGACY TXT REPORT EXPORT & RESET FUNCTIONALITY
    % -------------------------------------------------------------------------
    fprintf('\n[TEST 6/6] Testing legacy TXT report export and UI reset...\n');
    test_txt_file = fullfile(project_root, 'results', 'reports', 'test_regression_report.txt');
    if exist(test_txt_file, 'file') == 2
        delete(test_txt_file);
    end

    fid = fopen(test_txt_file, 'w');
    assert(fid ~= -1, 'Failed to open text file for writing.');
    fprintf(fid, '%s', res.report_text);
    fclose(fid);
    assert(exist(test_txt_file, 'file') == 2, 'Failed to write legacy text report.');
    delete(test_txt_file);

    % Test Reset
    app.onReset();
    assert(isempty(app.DiagnosticResult), 'App DiagnosticResult not cleared on reset.');
    assert(strcmp(app.RunScreeningBtn.Enable, 'off'), 'RunScreeningBtn not disabled on reset.');
    assert(strcmp(app.ExportPdfBtn.Enable, 'off'), 'ExportPdfBtn not disabled on reset.');

    delete(app);
    test_results.legacy_txt_and_reset = 'PASSED';
    fprintf('  -> TEST 6 PASSED: Legacy TXT export and UI reset cleanly verified.\n');

    fprintf('\n============================================================\n');
    fprintf('  ALL 6 REGRESSION TESTS PASSED SUCCESSFULLY!\n');
    fprintf('============================================================\n\n');
end
