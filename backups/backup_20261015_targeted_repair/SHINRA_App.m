classdef SHINRA_App < matlab.apps.AppBase
    % SHINRA_APP Standalone Retinal Screening and Diagnostic Explainability Application
    % Production-grade MATLAB App Designer interface for single-image retinal fundus analysis.
    %
    % Key Capabilities:
    % 1. Image Quality Safety Gate & Tri-State Grading
    % 2. Anatomical Landmark Localization (Optic Disc & Fovea)
    % 3. Deep U-Net Vessel Segmentation & Morphometry
    % 4. Multi-Scale Lesion Biomarker Extraction (MA, HE, EX, DME, NV)
    % 5. 5-Class ICDR Diabetic Retinopathy Severity Grading
    % 6. Calibrated Multimodal Referable DR Classification
    % 7. 5-Factor Reliability Analysis & Clinical Triage
    % 8. Deep Grad-CAM Visual Interpretability Overlays
    % 9. Structured Interactive Clinical Report Dashboard
    % 10. Publication-Grade Multi-Page PDF Clinical Report & Visualization Atlas Export
    % 11. Responsive Window Reflow with Maximize, Minimize, and Restore Support

    properties (Access = public)
        % Main UI Window
        UIFigure                     matlab.ui.Figure
        MainGrid                     matlab.ui.container.GridLayout
        
        % Header Components
        HeaderPanel                  matlab.ui.container.Panel
        HeaderGrid                   matlab.ui.container.GridLayout
        TitleLabel                   matlab.ui.control.Label
        SubtitleLabel                matlab.ui.control.Label
        MaximizeBtn                  matlab.ui.control.Button
        StatusLabel                  matlab.ui.control.Label
        
        % Toolbar Controls
        ControlPanel                 matlab.ui.container.Panel
        ControlGrid                  matlab.ui.container.GridLayout
        SelectImageBtn               matlab.ui.control.Button
        RunScreeningBtn              matlab.ui.control.Button
        ResetBtn                     matlab.ui.control.Button
        ExportPdfBtn                 matlab.ui.control.Button
        SaveTxtBtn                   matlab.ui.control.Button
        SaveImageBtn                 matlab.ui.control.Button
        
        % Body Container
        BodyGrid                     matlab.ui.container.GridLayout
        
        % Left Panel: Retinal Viewer & Overlays
        LeftPanel                    matlab.ui.container.Panel
        LeftGrid                     matlab.ui.container.GridLayout
        ViewSelectorGrid             matlab.ui.container.GridLayout
        ViewSelectorLabel            matlab.ui.control.Label
        ViewSelectorDropdown         matlab.ui.control.DropDown
        FitZoomBtn                   matlab.ui.control.Button
        ImageAxes                    matlab.ui.control.UIAxes
        ImageInfoLabel               matlab.ui.control.Label
        LayerLegendPanel             matlab.ui.container.Panel
        LayerLegendGrid              matlab.ui.container.GridLayout
        LayerLegendTitleLabel        matlab.ui.control.Label
        LayerLegendDescLabel         matlab.ui.control.Label
        
        % Right Panel: Diagnostic Results Tabs
        RightPanel                   matlab.ui.container.Panel
        RightGrid                    matlab.ui.container.GridLayout
        ResultsTabGroup              matlab.ui.container.TabGroup
        OverviewTab                  matlab.ui.container.Tab
        BiomarkersTab                matlab.ui.container.Tab
        ReportTab                    matlab.ui.container.Tab
        
        % Tab 1: Overview Tab Components
        OverviewGrid                 matlab.ui.container.GridLayout
        TriagePanel                  matlab.ui.container.Panel
        TriageActionLabel            matlab.ui.control.Label
        TriageDetailLabel            matlab.ui.control.Label
        
        ReferablePanel               matlab.ui.container.Panel
        ReferableTitleLabel          matlab.ui.control.Label
        ReferableStatusLabel         matlab.ui.control.Label
        ReferableProbLabel           matlab.ui.control.Label
        ReferableGateLabel           matlab.ui.control.Label
        
        SeverityPanel                matlab.ui.container.Panel
        SeverityTitleLabel           matlab.ui.control.Label
        SeverityGradeLabel           matlab.ui.control.Label
        SeverityScoreLabel           matlab.ui.control.Label
        SeverityAxes                 matlab.ui.control.UIAxes
        
        GatesPanel                   matlab.ui.container.Panel
        QualityGateLabel             matlab.ui.control.Label
        ReliabilityGateLabel         matlab.ui.control.Label
        
        % Tab 2: Biomarkers & Reliability Breakdown Tab Components
        BiomarkersGrid               matlab.ui.container.GridLayout
        LesionsTable                 matlab.ui.control.Table
        ReliabilityTable             matlab.ui.control.Table
        BiomarkerSummaryLabel        matlab.ui.control.Label
        ReliabilitySummaryLabel      matlab.ui.control.Label
        
        % Tab 3: Restored Clinical Report Tab Components
        ReportGrid                   matlab.ui.container.GridLayout
        ReportHeaderLabel            matlab.ui.control.Label
        ReportTextArea               matlab.ui.control.TextArea
        
        % State Data
        CurrentImagePath             char = ''
        RawImage                     = []
        DiagnosticResult             = []
    end

    methods (Access = public)

        function createComponents(app)
            % CREATECOMPONENTS Initialize and build UI layout with responsive reflow

            % 1. Main Figure Window
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [50 50 1420 880];
            app.UIFigure.Name = 'SHINRA — Explainable AI Retinal Screening System for Rural India';
            app.UIFigure.Color = [0.94 0.95 0.97];
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.SizeChangedFcn = @(src, event) onWindowSizeChanged(app);
            app.UIFigure.CloseRequestFcn = @(src, event) delete(app);

            % 2. Master Grid Layout
            app.MainGrid = uigridlayout(app.UIFigure, [3, 1]);
            app.MainGrid.RowHeight = {72, 48, '1x'};
            app.MainGrid.Padding = [12 12 12 12];
            app.MainGrid.RowSpacing = 8;

            % 3. Header Panel
            app.HeaderPanel = uipanel(app.MainGrid);
            app.HeaderPanel.BackgroundColor = [0.10 0.18 0.28];
            app.HeaderPanel.BorderType = 'none';

            app.HeaderGrid = uigridlayout(app.HeaderPanel, [2, 3]);
            app.HeaderGrid.ColumnWidth = {'1x', 110, 260};
            app.HeaderGrid.RowHeight = {36, 22};
            app.HeaderGrid.Padding = [16 6 16 6];
            app.HeaderGrid.RowSpacing = 2;

            app.TitleLabel = uilabel(app.HeaderGrid);
            app.TitleLabel.Text = 'SHINRA | Retinal Fundus Screening & Diagnostic Interpretability Engine';
            app.TitleLabel.FontName = 'Segoe UI';
            app.TitleLabel.FontSize = 18;
            app.TitleLabel.FontWeight = 'bold';
            app.TitleLabel.FontColor = [1.0 1.0 1.0];

            app.SubtitleLabel = uilabel(app.HeaderGrid);
            app.SubtitleLabel.Layout.Row = 2;
            app.SubtitleLabel.Layout.Column = 1;
            app.SubtitleLabel.Text = 'Autonomous Quality Safety Gate • Calibrated DR Severity • 5-Factor Reliability • Clinical Triage';
            app.SubtitleLabel.FontName = 'Segoe UI';
            app.SubtitleLabel.FontSize = 11;
            app.SubtitleLabel.FontColor = [0.75 0.85 0.95];

            % Maximize / Restore Toggle Button
            app.MaximizeBtn = uibutton(app.HeaderGrid, 'push');
            app.MaximizeBtn.Layout.Row = 1;
            app.MaximizeBtn.Layout.Column = 2;
            app.MaximizeBtn.Text = '⛶ Maximize';
            app.MaximizeBtn.FontName = 'Segoe UI';
            app.MaximizeBtn.FontSize = 11;
            app.MaximizeBtn.FontWeight = 'bold';
            app.MaximizeBtn.BackgroundColor = [0.18 0.28 0.42];
            app.MaximizeBtn.FontColor = [1 1 1];
            app.MaximizeBtn.Tooltip = 'Toggle window maximize / restore state';
            app.MaximizeBtn.ButtonPushedFcn = @(src, event) onToggleMaximize(app);

            % Live Status Indicator
            app.StatusLabel = uilabel(app.HeaderGrid);
            app.StatusLabel.Layout.Row = 1;
            app.StatusLabel.Layout.Column = 3;
            app.StatusLabel.HorizontalAlignment = 'right';
            app.StatusLabel.Text = '● System Status: Ready';
            app.StatusLabel.FontName = 'Segoe UI';
            app.StatusLabel.FontSize = 12;
            app.StatusLabel.FontWeight = 'bold';
            app.StatusLabel.FontColor = [0.30 0.90 0.50];

            % 4. Control Toolbar Panel
            app.ControlPanel = uipanel(app.MainGrid);
            app.ControlPanel.BackgroundColor = [1.0 1.0 1.0];
            app.ControlPanel.BorderType = 'line';

            app.ControlGrid = uigridlayout(app.ControlPanel, [1, 7]);
            app.ControlGrid.ColumnWidth = {140, 195, 95, 185, 145, 175, '1x'};
            app.ControlGrid.Padding = [10 6 10 6];
            app.ControlGrid.ColumnSpacing = 8;

            app.SelectImageBtn = uibutton(app.ControlGrid, 'push');
            app.SelectImageBtn.Text = '📁 Select Image...';
            app.SelectImageBtn.FontName = 'Segoe UI';
            app.SelectImageBtn.FontSize = 12;
            app.SelectImageBtn.FontWeight = 'bold';
            app.SelectImageBtn.BackgroundColor = [0.22 0.44 0.72];
            app.SelectImageBtn.FontColor = [1 1 1];
            app.SelectImageBtn.ButtonPushedFcn = @(src, event) onSelectImage(app);

            app.RunScreeningBtn = uibutton(app.ControlGrid, 'push');
            app.RunScreeningBtn.Text = '▶ Run SHINRA Screening';
            app.RunScreeningBtn.FontName = 'Segoe UI';
            app.RunScreeningBtn.FontSize = 12;
            app.RunScreeningBtn.FontWeight = 'bold';
            app.RunScreeningBtn.BackgroundColor = [0.18 0.65 0.35];
            app.RunScreeningBtn.FontColor = [1 1 1];
            app.RunScreeningBtn.Enable = 'off';
            app.RunScreeningBtn.ButtonPushedFcn = @(src, event) onRunScreening(app);

            app.ResetBtn = uibutton(app.ControlGrid, 'push');
            app.ResetBtn.Text = '↺ Reset';
            app.ResetBtn.FontName = 'Segoe UI';
            app.ResetBtn.FontSize = 12;
            app.ResetBtn.BackgroundColor = [0.90 0.92 0.95];
            app.ResetBtn.ButtonPushedFcn = @(src, event) onReset(app);

            % Primary Multi-Page PDF Export Button
            app.ExportPdfBtn = uibutton(app.ControlGrid, 'push');
            app.ExportPdfBtn.Text = '📄 Export PDF Report';
            app.ExportPdfBtn.FontName = 'Segoe UI';
            app.ExportPdfBtn.FontSize = 12;
            app.ExportPdfBtn.FontWeight = 'bold';
            app.ExportPdfBtn.BackgroundColor = [0.65 0.20 0.20];
            app.ExportPdfBtn.FontColor = [1 1 1];
            app.ExportPdfBtn.Enable = 'off';
            app.ExportPdfBtn.Tooltip = 'Export comprehensive multi-page clinical screening PDF with visualization atlas';
            app.ExportPdfBtn.ButtonPushedFcn = @(src, event) onExportPdfReport(app);

            % Preserved Legacy TXT Report Export Button
            app.SaveTxtBtn = uibutton(app.ControlGrid, 'push');
            app.SaveTxtBtn.Text = '💾 Save TXT Report';
            app.SaveTxtBtn.FontName = 'Segoe UI';
            app.SaveTxtBtn.FontSize = 12;
            app.SaveTxtBtn.Enable = 'off';
            app.SaveTxtBtn.Tooltip = 'Save plain-text clinical screening summary';
            app.SaveTxtBtn.ButtonPushedFcn = @(src, event) onSaveTxtReport(app);

            % Save Currently Active Overlay Image
            app.SaveImageBtn = uibutton(app.ControlGrid, 'push');
            app.SaveImageBtn.Text = '📷 Save Active Overlay';
            app.SaveImageBtn.FontName = 'Segoe UI';
            app.SaveImageBtn.FontSize = 12;
            app.SaveImageBtn.Enable = 'off';
            app.SaveImageBtn.Tooltip = 'Save high-resolution image of currently viewed layer';
            app.SaveImageBtn.ButtonPushedFcn = @(src, event) onSaveImage(app);

            % 5. Main Body Grid (Left: Image / Right: Diagnostic Results)
            app.BodyGrid = uigridlayout(app.MainGrid, [1, 2]);
            app.BodyGrid.ColumnWidth = {'1.2x', '1x'};
            app.BodyGrid.Padding = [0 0 0 0];
            app.BodyGrid.ColumnSpacing = 10;

            % ----------------- LEFT PANEL: IMAGING & OVERLAYS -----------------
            app.LeftPanel = uipanel(app.BodyGrid);
            app.LeftPanel.Title = 'Retinal Fundus Visualizer & Clinical Overlays';
            app.LeftPanel.FontName = 'Segoe UI';
            app.LeftPanel.FontSize = 12;
            app.LeftPanel.FontWeight = 'bold';
            app.LeftPanel.BackgroundColor = [1.0 1.0 1.0];

            app.LeftGrid = uigridlayout(app.LeftPanel, [4, 1]);
            app.LeftGrid.RowHeight = {36, '1x', 24, 75};
            app.LeftGrid.Padding = [10 10 10 10];
            app.LeftGrid.RowSpacing = 6;

            % View selector controls with Fit-to-view button
            app.ViewSelectorGrid = uigridlayout(app.LeftGrid, [1, 3]);
            app.ViewSelectorGrid.ColumnWidth = {95, '1x', 105};
            app.ViewSelectorGrid.Padding = [0 0 0 0];
            app.ViewSelectorGrid.ColumnSpacing = 6;

            app.ViewSelectorLabel = uilabel(app.ViewSelectorGrid);
            app.ViewSelectorLabel.Text = 'Display Layer:';
            app.ViewSelectorLabel.FontName = 'Segoe UI';
            app.ViewSelectorLabel.FontSize = 12;
            app.ViewSelectorLabel.FontWeight = 'bold';

            app.ViewSelectorDropdown = uidropdown(app.ViewSelectorGrid);
            app.ViewSelectorDropdown.Items = {
                'Original Fundus Photograph', ...
                'Preprocessed & CLAHE Enhanced', ...
                'Retinal Anatomy (Optic Disc & Fovea)', ...
                'Retinal Vessel Tree (Deep U-Net)', ...
                'Microaneurysms Detection (Red)', ...
                'Hemorrhages Detection (Magenta)', ...
                'Exudates & DME Risk Map (Yellow)', ...
                'Multi-Lesion Composite Map', ...
                'Grad-CAM Deep Activation Heatmap', ...
                'Grad-CAM Clinical Explainability Overlay'
            };
            app.ViewSelectorDropdown.Value = 'Original Fundus Photograph';
            app.ViewSelectorDropdown.FontName = 'Segoe UI';
            app.ViewSelectorDropdown.FontSize = 12;
            app.ViewSelectorDropdown.ValueChangedFcn = @(src, event) onViewModeChanged(app);

            app.FitZoomBtn = uibutton(app.ViewSelectorGrid, 'push');
            app.FitZoomBtn.Text = '🔍 Fit View';
            app.FitZoomBtn.FontName = 'Segoe UI';
            app.FitZoomBtn.FontSize = 11;
            app.FitZoomBtn.Tooltip = 'Reset image view to full frame aspect ratio';
            app.FitZoomBtn.ButtonPushedFcn = @(src, event) onFitView(app);

            % Main Display Axes
            app.ImageAxes = uiaxes(app.LeftGrid);
            app.ImageAxes.Toolbar.Visible = 'on';
            app.ImageAxes.XTick = [];
            app.ImageAxes.YTick = [];
            app.ImageAxes.Box = 'on';
            app.ImageAxes.Color = [0.05 0.05 0.07];
            title(app.ImageAxes, 'No Retinal Image Loaded', 'Color', [0.7 0.7 0.7], 'FontName', 'Segoe UI');

            % Image Metadata Info Label
            app.ImageInfoLabel = uilabel(app.LeftGrid);
            app.ImageInfoLabel.Text = 'Image: None loaded | Resolution: -- | Quality: -- | Grad-CAM Agreement: --';
            app.ImageInfoLabel.FontName = 'Segoe UI';
            app.ImageInfoLabel.FontSize = 11;
            app.ImageInfoLabel.FontColor = [0.4 0.4 0.4];

            % Dynamic Layer Legend & Clinical Interpretation Card
            app.LayerLegendPanel = uipanel(app.LeftGrid);
            app.LayerLegendPanel.BackgroundColor = [0.96 0.97 0.99];
            app.LayerLegendPanel.BorderType = 'line';

            app.LayerLegendGrid = uigridlayout(app.LayerLegendPanel, [2, 1]);
            app.LayerLegendGrid.RowHeight = {22, 38};
            app.LayerLegendGrid.Padding = [10 6 10 6];
            app.LayerLegendGrid.RowSpacing = 2;

            app.LayerLegendTitleLabel = uilabel(app.LayerLegendGrid);
            app.LayerLegendTitleLabel.Text = 'Active Layer: Original Fundus Photograph';
            app.LayerLegendTitleLabel.FontName = 'Segoe UI';
            app.LayerLegendTitleLabel.FontSize = 11;
            app.LayerLegendTitleLabel.FontWeight = 'bold';
            app.LayerLegendTitleLabel.FontColor = [0.10 0.18 0.28];

            app.LayerLegendDescLabel = uilabel(app.LayerLegendGrid);
            app.LayerLegendDescLabel.Text = 'True-color 45-degree field-of-view retinal fundus photograph as acquired from the camera.';
            app.LayerLegendDescLabel.FontName = 'Segoe UI';
            app.LayerLegendDescLabel.FontSize = 10;
            app.LayerLegendDescLabel.FontColor = [0.35 0.35 0.35];

            % ----------------- RIGHT PANEL: RESULTS & TRIAGE -----------------
            app.RightPanel = uipanel(app.BodyGrid);
            app.RightPanel.Title = 'Diagnostic Results, Reliability & Clinical Triage';
            app.RightPanel.FontName = 'Segoe UI';
            app.RightPanel.FontSize = 12;
            app.RightPanel.FontWeight = 'bold';
            app.RightPanel.BackgroundColor = [1.0 1.0 1.0];

            app.RightGrid = uigridlayout(app.RightPanel, [1, 1]);
            app.RightGrid.Padding = [6 6 6 6];

            app.ResultsTabGroup = uitabgroup(app.RightGrid);

            % ----------------- TAB 1: OVERVIEW & DECISION -----------------
            app.OverviewTab = uitab(app.ResultsTabGroup, 'Title', 'Screening & Triage');
            app.OverviewGrid = uigridlayout(app.OverviewTab, [4, 1]);
            app.OverviewGrid.RowHeight = {70, 85, 175, 65};
            app.OverviewGrid.Padding = [8 8 8 8];
            app.OverviewGrid.RowSpacing = 8;

            % 1. Triage Action Panel (Color Coded Banner)
            app.TriagePanel = uipanel(app.OverviewGrid);
            app.TriagePanel.BackgroundColor = [0.93 0.94 0.96];
            app.TriagePanel.BorderType = 'line';
            triageGrid = uigridlayout(app.TriagePanel, [2, 1]);
            triageGrid.RowHeight = {26, 26};
            triageGrid.Padding = [12 6 12 6];
            triageGrid.RowSpacing = 2;

            app.TriageActionLabel = uilabel(triageGrid);
            app.TriageActionLabel.Text = 'TRIAGE STATUS: AWAITING INFERENCE';
            app.TriageActionLabel.FontName = 'Segoe UI';
            app.TriageActionLabel.FontSize = 14;
            app.TriageActionLabel.FontWeight = 'bold';
            app.TriageActionLabel.FontColor = [0.3 0.3 0.3];

            app.TriageDetailLabel = uilabel(triageGrid);
            app.TriageDetailLabel.Text = 'Select an image and click "Run SHINRA Screening" to initiate analysis.';
            app.TriageDetailLabel.FontName = 'Segoe UI';
            app.TriageDetailLabel.FontSize = 11;
            app.TriageDetailLabel.FontColor = [0.4 0.4 0.4];

            % 2. Referable DR Status Panel
            app.ReferablePanel = uipanel(app.OverviewGrid);
            app.ReferablePanel.BackgroundColor = [0.97 0.98 1.0];
            refGrid = uigridlayout(app.ReferablePanel, [2, 2]);
            refGrid.ColumnWidth = {'1x', '1x'};
            refGrid.RowHeight = {24, 38};
            refGrid.Padding = [12 8 12 8];
            refGrid.RowSpacing = 2;

            app.ReferableTitleLabel = uilabel(refGrid);
            app.ReferableTitleLabel.Text = 'Referable DR Classification:';
            app.ReferableTitleLabel.FontName = 'Segoe UI';
            app.ReferableTitleLabel.FontSize = 12;
            app.ReferableTitleLabel.FontWeight = 'bold';

            app.ReferableGateLabel = uilabel(refGrid);
            app.ReferableGateLabel.HorizontalAlignment = 'right';
            app.ReferableGateLabel.Text = 'Referral Threshold: τ_ref = 0.30';
            app.ReferableGateLabel.FontName = 'Segoe UI';
            app.ReferableGateLabel.FontSize = 11;
            app.ReferableGateLabel.FontColor = [0.4 0.4 0.4];

            app.ReferableStatusLabel = uilabel(refGrid);
            app.ReferableStatusLabel.Layout.Row = 2;
            app.ReferableStatusLabel.Layout.Column = 1;
            app.ReferableStatusLabel.Text = '--';
            app.ReferableStatusLabel.FontName = 'Segoe UI';
            app.ReferableStatusLabel.FontSize = 16;
            app.ReferableStatusLabel.FontWeight = 'bold';

            app.ReferableProbLabel = uilabel(refGrid);
            app.ReferableProbLabel.Layout.Row = 2;
            app.ReferableProbLabel.Layout.Column = 2;
            app.ReferableProbLabel.HorizontalAlignment = 'right';
            app.ReferableProbLabel.Text = 'Calibrated P(ref): --';
            app.ReferableProbLabel.FontName = 'Segoe UI';
            app.ReferableProbLabel.FontSize = 13;

            % 3. Severity Grading Panel & Probability Chart
            app.SeverityPanel = uipanel(app.OverviewGrid);
            app.SeverityPanel.BackgroundColor = [1.0 1.0 1.0];
            sevGrid = uigridlayout(app.SeverityPanel, [2, 2]);
            sevGrid.ColumnWidth = {'1x', '1.3x'};
            sevGrid.RowHeight = {24, '1x'};
            sevGrid.Padding = [10 6 10 6];

            app.SeverityTitleLabel = uilabel(sevGrid);
            app.SeverityTitleLabel.Text = '5-Class ICDR Severity:';
            app.SeverityTitleLabel.FontName = 'Segoe UI';
            app.SeverityTitleLabel.FontSize = 12;
            app.SeverityTitleLabel.FontWeight = 'bold';

            sevDetailsSubgrid = uigridlayout(sevGrid, [3, 1]);
            sevDetailsSubgrid.Layout.Row = 2;
            sevDetailsSubgrid.Layout.Column = 1;
            sevDetailsSubgrid.Padding = [0 0 0 0];
            sevDetailsSubgrid.RowSpacing = 2;

            app.SeverityGradeLabel = uilabel(sevDetailsSubgrid);
            app.SeverityGradeLabel.Text = 'Grade: --';
            app.SeverityGradeLabel.FontName = 'Segoe UI';
            app.SeverityGradeLabel.FontSize = 14;
            app.SeverityGradeLabel.FontWeight = 'bold';

            app.SeverityScoreLabel = uilabel(sevDetailsSubgrid);
            app.SeverityScoreLabel.Text = 'Ordinal Score E[G]: --\nConfidence: --';
            app.SeverityScoreLabel.FontName = 'Segoe UI';
            app.SeverityScoreLabel.FontSize = 11;
            app.SeverityScoreLabel.FontColor = [0.3 0.3 0.3];

            app.SeverityAxes = uiaxes(sevGrid);
            app.SeverityAxes.Layout.Row = 2;
            app.SeverityAxes.Layout.Column = 2;
            app.SeverityAxes.Toolbar.Visible = 'off';
            app.SeverityAxes.FontName = 'Segoe UI';
            app.SeverityAxes.FontSize = 9;
            app.SeverityAxes.Box = 'on';
            title(app.SeverityAxes, 'Grade Posterior Probabilities', 'FontSize', 9, 'FontWeight', 'normal');

            % 4. Safety Gates Summary Panel
            app.GatesPanel = uipanel(app.OverviewGrid);
            app.GatesPanel.BackgroundColor = [0.96 0.97 0.98];
            gatesGrid = uigridlayout(app.GatesPanel, [1, 2]);
            gatesGrid.ColumnWidth = {'1x', '1x'};
            gatesGrid.Padding = [10 6 10 6];

            app.QualityGateLabel = uilabel(gatesGrid);
            app.QualityGateLabel.Text = 'Quality Safety Gate: --\nScore: -- | Class: --';
            app.QualityGateLabel.FontName = 'Segoe UI';
            app.QualityGateLabel.FontSize = 11;

            app.ReliabilityGateLabel = uilabel(gatesGrid);
            app.ReliabilityGateLabel.Text = 'Reliability Safety Gate (τ ≥ 60.0):\nScore: -- | Gate: --';
            app.ReliabilityGateLabel.FontName = 'Segoe UI';
            app.ReliabilityGateLabel.FontSize = 11;

            % ----------------- TAB 2: BIOMARKERS & RELIABILITY -----------------
            app.BiomarkersTab = uitab(app.ResultsTabGroup, 'Title', 'Biomarkers & Reliability');
            app.BiomarkersGrid = uigridlayout(app.BiomarkersTab, [4, 1]);
            app.BiomarkersGrid.RowHeight = {22, 110, 22, 130};
            app.BiomarkersGrid.Padding = [8 8 8 8];
            app.BiomarkersGrid.RowSpacing = 4;

            app.BiomarkerSummaryLabel = uilabel(app.BiomarkersGrid);
            app.BiomarkerSummaryLabel.Text = 'Extracted Retinal Biomarkers:';
            app.BiomarkerSummaryLabel.FontName = 'Segoe UI';
            app.BiomarkerSummaryLabel.FontSize = 12;
            app.BiomarkerSummaryLabel.FontWeight = 'bold';

            app.LesionsTable = uitable(app.BiomarkersGrid);
            app.LesionsTable.ColumnName = {'Biomarker', 'Measurement', 'Clinical Significance'};
            app.LesionsTable.ColumnWidth = {130, 110, 'auto'};
            app.LesionsTable.RowName = {};
            app.LesionsTable.FontName = 'Segoe UI';
            app.LesionsTable.FontSize = 11;

            app.ReliabilitySummaryLabel = uilabel(app.BiomarkersGrid);
            app.ReliabilitySummaryLabel.Text = '5-Factor Reliability Analysis Breakdown:';
            app.ReliabilitySummaryLabel.FontName = 'Segoe UI';
            app.ReliabilitySummaryLabel.FontSize = 12;
            app.ReliabilitySummaryLabel.FontWeight = 'bold';

            app.ReliabilityTable = uitable(app.BiomarkersGrid);
            app.ReliabilityTable.ColumnName = {'Factor', 'Weight', 'Score [0-100]', 'Assessment'};
            app.ReliabilityTable.ColumnWidth = {140, 65, 95, 'auto'};
            app.ReliabilityTable.RowName = {};
            app.ReliabilityTable.FontName = 'Segoe UI';
            app.ReliabilityTable.FontSize = 11;

            % ----------------- TAB 3: RESTORED CLINICAL REPORT -----------------
            app.ReportTab = uitab(app.ResultsTabGroup, 'Title', 'Clinical Report');
            app.ReportGrid = uigridlayout(app.ReportTab, [2, 1]);
            app.ReportGrid.RowHeight = {30, '1x'};
            app.ReportGrid.ColumnWidth = {'1x'};
            app.ReportGrid.Padding = [8 8 8 8];
            app.ReportGrid.RowSpacing = 6;

            app.ReportHeaderLabel = uilabel(app.ReportGrid);
            app.ReportHeaderLabel.Text = 'Standardized Clinical Screening Report (Preview & Audit Trail)';
            app.ReportHeaderLabel.FontName = 'Segoe UI';
            app.ReportHeaderLabel.FontSize = 12;
            app.ReportHeaderLabel.FontWeight = 'bold';
            app.ReportHeaderLabel.FontColor = [0.10 0.18 0.28];

            app.ReportTextArea = uitextarea(app.ReportGrid);
            app.ReportTextArea.Editable = 'off';
            app.ReportTextArea.FontName = 'Consolas';
            app.ReportTextArea.FontSize = 11;
            app.ReportTextArea.FontColor = [0.10 0.12 0.18];
            app.ReportTextArea.BackgroundColor = [1.0 1.0 1.0];
            app.ReportTextArea.Value = {'No screening report generated. Select an image and run screening.'};

            % Initial state reset
            app.onReset();

            % Show UIFigure
            app.UIFigure.Visible = 'on';
        end

        % =================================================================
        % WINDOW MANAGEMENT CALLBACKS
        % =================================================================

        function onToggleMaximize(app)
            % ONTOGGLEMAXIMIZE Toggle window maximize and normal restored state
            if strcmp(app.UIFigure.WindowState, 'maximized')
                app.UIFigure.WindowState = 'normal';
                app.MaximizeBtn.Text = '⛶ Maximize';
                app.MaximizeBtn.Tooltip = 'Maximize application window to full screen';
            else
                app.UIFigure.WindowState = 'maximized';
                app.MaximizeBtn.Text = '🗗 Restore';
                app.MaximizeBtn.Tooltip = 'Restore application window to standard dimensions';
            end
        end

        function onWindowSizeChanged(app)
            % ONWINDOWSIZECHANGED Enforce sensible minimum dimensions and reflow
            pos = app.UIFigure.Position;
            min_w = 1100;
            min_h = 700;
            need_adjust = false;
            
            if pos(3) < min_w
                pos(3) = min_w;
                need_adjust = true;
            end
            if pos(4) < min_h
                pos(4) = min_h;
                need_adjust = true;
            end
            
            if need_adjust && strcmp(app.UIFigure.WindowState, 'normal')
                app.UIFigure.Position = pos;
            end

            % Update maximize button state text if window was resized or maximized externally
            if strcmp(app.UIFigure.WindowState, 'maximized')
                app.MaximizeBtn.Text = '🗗 Restore';
            else
                app.MaximizeBtn.Text = '⛶ Maximize';
            end
        end

        function onFitView(app)
            % ONFITVIEW Reset axes limits and aspect ratio to full frame
            axis(app.ImageAxes, 'image');
            axis(app.ImageAxes, 'off');
        end

        % =================================================================
        % SCREENING WORKFLOW CALLBACKS
        % =================================================================

        function onSelectImage(app)
            % ONSELECTIMAGE Open file browser and load preview
            filters = {'*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.mat', 'All Retinal Image Formats (*.jpg, *.png, *.tif, *.mat)'; ...
                       '*.*', 'All Files (*.*)'};
            
            default_dir = pwd;
            idrid_train_dir = fullfile(pwd, 'SHINRAdataset', 'IDRid', 'B. Disease Grading', '1. Original Images', 'a. Training Set');
            if exist(idrid_train_dir, 'dir') == 7
                default_dir = idrid_train_dir;
            end

            [fname, fpath] = uigetfile(filters, 'Select Retinal Fundus Image', default_dir);
            if isequal(fname, 0) || isequal(fpath, 0)
                return;
            end

            selected_path = fullfile(fpath, fname);
            try
                raw = imread(selected_path);
                
                % Clear previous results immediately upon new image selection
                app.onReset();

                app.CurrentImagePath = selected_path;
                app.RawImage = raw;
                app.DiagnosticResult = [];

                % Display preview
                cla(app.ImageAxes);
                imagesc(app.ImageAxes, raw);
                axis(app.ImageAxes, 'image');
                axis(app.ImageAxes, 'off');
                title(app.ImageAxes, sprintf('Loaded: %s', fname), 'Color', [0.1 0.1 0.1], 'FontName', 'Segoe UI', 'Interpreter', 'none');

                [h, w, c] = size(raw);
                app.ImageInfoLabel.Text = sprintf('Image: %s | Dimensions: %dx%d (%d channels) | Ready to screen', fname, w, h, c);
                app.RunScreeningBtn.Enable = 'on';
                app.ExportPdfBtn.Enable = 'off';
                app.SaveTxtBtn.Enable = 'off';
                app.SaveImageBtn.Enable = 'off';
                                                app.ViewSelectorDropdown.Value = 'Original Fundus Photograph';
                app.updateLayerLegend('Original Fundus Photograph');

                app.updateStatus('Image Loaded Successfully. Click "Run SHINRA Screening".', [0.1 0.5 0.8]);
            catch ME
                uialert(app.UIFigure, sprintf('Failed to load image:\n%s', ME.message), 'Image Load Error');
            end
        end

        function onRunScreening(app)
            % ONRUNSCREENING Execute master SHINRA inference pipeline
            if isempty(app.CurrentImagePath) && isempty(app.RawImage)
                return;
            end

            app.updateStatus('Screening in progress... Executing Quality Gate & Deep Inference...', [0.85 0.55 0.10]);
            app.RunScreeningBtn.Enable = 'off';
            app.SelectImageBtn.Enable = 'off';
            drawnow;

            try
                % Call master reusable inference function
                if ~isempty(app.CurrentImagePath) && exist(app.CurrentImagePath, 'file') == 2
                    res = runSHINRASingleImage(app.CurrentImagePath);
                else
                    res = runSHINRASingleImage(app.RawImage);
                end

                app.DiagnosticResult = res;

                % Populate all UI tabs and cards
                app.populateResults(res);

                % Update active visual layer view
                app.onViewModeChanged();

                % Enable export buttons
                app.ExportPdfBtn.Enable = 'on';
                app.SaveTxtBtn.Enable = 'on';
                app.SaveImageBtn.Enable = 'on';
                                                app.RunScreeningBtn.Enable = 'on';
                app.SelectImageBtn.Enable = 'on';

                app.updateStatus('● SHINRA Screening Completed Successfully', [0.15 0.75 0.35]);

            catch ME
                app.RunScreeningBtn.Enable = 'on';
                app.SelectImageBtn.Enable = 'on';
                app.updateStatus('● Error occurred during screening', [0.85 0.20 0.20]);
                uialert(app.UIFigure, sprintf('Inference Error:\n%s\n\nStack:\n%s', ME.message, getReport(ME)), 'SHINRA Pipeline Error');
            end
        end

        function populateResults(app, res)
            % POPULATERESULTS Update all diagnostic cards, tables, charts, and report dashboard

            % -------------------------------------------------------------
            % TAB 1: OVERVIEW & DECISION
            % -------------------------------------------------------------
            % 1. Triage Banner
            triage_action = res.reliability.recommended_action;
            app.TriageActionLabel.Text = sprintf('ACTION: %s', upper(strrep(triage_action, '_', ' ')));
            app.TriageDetailLabel.Text = res.reliability.action_reason;

            switch triage_action
                case 'ACCEPT_SCREENING'
                    app.TriagePanel.BackgroundColor = [0.88 0.96 0.88];
                    app.TriageActionLabel.FontColor = [0.08 0.55 0.18];
                case 'RECAPTURE'
                    app.TriagePanel.BackgroundColor = [0.98 0.88 0.88];
                    app.TriageActionLabel.FontColor = [0.75 0.12 0.12];
                case 'CLINICAL_REVIEW'
                    app.TriagePanel.BackgroundColor = [0.99 0.94 0.82];
                    app.TriageActionLabel.FontColor = [0.70 0.45 0.05];
                otherwise
                    app.TriagePanel.BackgroundColor = [0.92 0.93 0.95];
                    app.TriageActionLabel.FontColor = [0.35 0.35 0.40];
            end

            % 2. Referable DR Status Card
            if res.referable.is_referable
                app.ReferableStatusLabel.Text = 'REFERABLE DR (Positive)';
                app.ReferableStatusLabel.FontColor = [0.80 0.15 0.15];
            else
                app.ReferableStatusLabel.Text = 'NON-REFERABLE DR (Negative)';
                app.ReferableStatusLabel.FontColor = [0.10 0.60 0.25];
            end

            app.ReferableProbLabel.Text = sprintf('P(Calibrated) = %.3f (Raw = %.3f)', ...
                res.referable.p_referable_calibrated, res.referable.p_referable_raw);

            % 3. 5-Class Severity Card & Probability Chart
            grade_names = {'No DR (Grade 0)', 'Mild NPDR (Grade 1)', 'Moderate NPDR (Grade 2)', 'Severe NPDR (Grade 3)', 'PDR (Grade 4)'};
            pred_g = res.severity.predicted_grade;
            app.SeverityGradeLabel.Text = sprintf('%s', grade_names{pred_g + 1});
            if pred_g >= 2
                app.SeverityGradeLabel.FontColor = [0.80 0.15 0.15];
            elseif pred_g == 1
                app.SeverityGradeLabel.FontColor = [0.75 0.50 0.05];
            else
                app.SeverityGradeLabel.FontColor = [0.10 0.60 0.25];
            end

            app.SeverityScoreLabel.Text = sprintf('Expected Ordinal Score E[G] = %.2f\nClassification Confidence = %.1f%%', ...
                res.severity.expected_grade, res.severity.confidence * 100);

            % Overview Bar Chart
            cla(app.SeverityAxes);
            probs = res.severity.probabilities(:)';
            y_pos = 0:4;
            barh(app.SeverityAxes, y_pos, probs * 100, 'FaceColor', [0.22 0.45 0.72], 'EdgeColor', 'none');
            app.SeverityAxes.YDir = 'reverse';
            app.SeverityAxes.YTick = 0:4;
            app.SeverityAxes.YTickLabel = {'G0', 'G1', 'G2', 'G3', 'G4'};
            app.SeverityAxes.XLim = [0, 100];
            app.SeverityAxes.XTick = [0, 50, 100];
            xlabel(app.SeverityAxes, 'Prob (%)', 'FontSize', 8);
            grid(app.SeverityAxes, 'on');

            % 4. Safety Gates Summary
            app.QualityGateLabel.Text = sprintf('Quality Gate: %s (Score: %.1f/100)\nArtifacts: %s | Retake Advice: %s', ...
                res.quality.quality_class, res.quality.quality_score, ...
                bool2str(res.quality.has_severe_artifacts, 'Severe Detected', 'Clear'), ...
                res.quality.recapture_guidance);

            app.ReliabilityGateLabel.Text = sprintf('Reliability Gate (τ ≥ 60.0): %s\nScore: %.1f/100 | Tier: %s', ...
                bool2str(res.reliability.passes_reliability_gate, 'PASSED', 'TRIGGERED REVIEW'), ...
                res.reliability.reliability_score, res.reliability.confidence_tier);

            % -------------------------------------------------------------
            % TAB 2: BIOMARKERS & RELIABILITY BREAKDOWN
            % -------------------------------------------------------------
            lesion_data = {
                'Microaneurysms', sprintf('%d lesions (Area: %d px)', res.lesions.microaneurysms.count, res.lesions.microaneurysms.total_area), 'Early NPDR hallmark'; ...
                'Hemorrhages', sprintf('%d lesions (Area: %.2f%%)', res.lesions.hemorrhages.count, res.lesions.hemorrhages.area_fraction*100), 'Vascular rupture marker'; ...
                'Hard Exudates', sprintf('%d lesions (Area: %.2f%%)', res.lesions.exudates.count, res.lesions.exudates.area_fraction*100), 'Lipid leakage / DME indicator'; ...
                'DME Macular Risk', sprintf('Risk Score: %.2f', res.lesions.exudates.macular_proximity_risk), 'Macular threat assessment'; ...
                'Neovascularization', sprintf('Biomarker Score: %.2f', res.lesions.neovascularization.nv_score), 'Exploratory PDR vessel tortuosity'
            };
            app.LesionsTable.Data = lesion_data;

            rel_breakdown = res.reliability.subscores;
            rel_data = {
                'Image Quality (Sq)', '25%', sprintf('%.1f', rel_breakdown.quality), res.quality.quality_class; ...
                'Classifier Confidence (Sc)', '25%', sprintf('%.1f', rel_breakdown.confidence), sprintf('Margin: %.2f', res.severity.confidence); ...
                'Boundary Margin (Sm)', '20%', sprintf('%.1f', rel_breakdown.margin), sprintf('Separation: %.3f', abs(res.referable.p_referable_calibrated - 0.30)); ...
                'Biomarker Completeness (Scomp)', '15%', sprintf('%.1f', rel_breakdown.completeness), 'Anatomy & Lesion consistency'; ...
                'Grad-CAM Alignment (Sagree)', '15%', sprintf('%.1f', rel_breakdown.agreement), sprintf('Dice: %.3f', res.explainability.agreement.dice)
            };
            app.ReliabilityTable.Data = rel_data;

            % -------------------------------------------------------------
            % TAB 3: RESTORED CLINICAL REPORT DISPLAY
            % -------------------------------------------------------------
            if isfield(res, 'report_text') && ~isempty(res.report_text)
                report_lines = splitlines(res.report_text);
                app.ReportTextArea.Value = report_lines;
            else
                app.ReportTextArea.Value = {'Report text unavailable.'};
            end

            % Update image info label strip
            app.ImageInfoLabel.Text = sprintf('Image ID: %s | Quality: %s (%.1f) | Severity: %s | CAM-Lesion Dice: %.3f', ...
                res.image_id, res.quality.quality_class, res.quality.quality_score, ...
                grade_names{pred_g + 1}, res.explainability.agreement.dice);
        end

        function onViewModeChanged(app)
            % ONVIEWMODECHANGED Render selected layer on image axes and update legend
            if isempty(app.DiagnosticResult)
                if ~isempty(app.RawImage)
                    cla(app.ImageAxes);
                    imagesc(app.ImageAxes, app.RawImage);
                    axis(app.ImageAxes, 'image');
                    axis(app.ImageAxes, 'off');
                end
                return;
            end

            mode = app.ViewSelectorDropdown.Value;
            res = app.DiagnosticResult;
            img_to_show = [];
            title_text = mode;

            switch mode
                case 'Original Fundus Photograph'
                    img_to_show = res.overlays.original_rgb;
                case 'Preprocessed & CLAHE Enhanced'
                    img_to_show = res.overlays.preprocessed_rgb;
                case 'Retinal Anatomy (Optic Disc & Fovea)'
                    img_to_show = res.overlays.anatomy;
                case 'Retinal Vessel Tree (Deep U-Net)'
                    img_to_show = res.overlays.vessels;
                case 'Microaneurysms Detection (Red)'
                    img_to_show = res.overlays.microaneurysms;
                case 'Hemorrhages Detection (Magenta)'
                    img_to_show = res.overlays.hemorrhages;
                case 'Exudates & DME Risk Map (Yellow)'
                    img_to_show = res.overlays.exudates;
                case 'Multi-Lesion Composite Map'
                    img_to_show = res.overlays.multi_lesions;
                case 'Grad-CAM Deep Activation Heatmap'
                    img_to_show = res.overlays.gradcam_heatmap;
                case 'Grad-CAM Clinical Explainability Overlay'
                    img_to_show = res.overlays.gradcam_overlay;
                otherwise
                    img_to_show = res.overlays.original_rgb;
            end

            if ~isempty(img_to_show)
                cla(app.ImageAxes);
                imagesc(app.ImageAxes, img_to_show);
                axis(app.ImageAxes, 'image');
                axis(app.ImageAxes, 'off');
                title(app.ImageAxes, title_text, 'Color', [0.15 0.15 0.15], 'FontName', 'Segoe UI', 'FontSize', 11);
            end

            app.updateLayerLegend(mode);
        end

        function updateLayerLegend(app, mode)
            % UPDATELAYERLEGEND Refresh dynamic clinical legend below image axes
            app.LayerLegendTitleLabel.Text = sprintf('Active Layer: %s', mode);
            switch mode
                case 'Original Fundus Photograph'
                    app.LayerLegendDescLabel.Text = 'True-color 45-degree field-of-view retinal photograph as acquired from the fundus camera.';
                case 'Preprocessed & CLAHE Enhanced'
                    app.LayerLegendDescLabel.Text = 'Illumination shading corrected via Gaussian filter; green-channel CLAHE enhanced for microvascular clarity.';
                case 'Retinal Anatomy (Optic Disc & Fovea)'
                    app.LayerLegendDescLabel.Text = 'Cyan ring: Optic Disc boundary. Amber ring: Macular zone (2.5x Optic Disc radius) threatening central vision.';
                case 'Retinal Vessel Tree (Deep U-Net)'
                    app.LayerLegendDescLabel.Text = 'Green overlay: Segmented retinal blood vessel network produced by trained Deep U-Net architecture.';
                case 'Microaneurysms Detection (Red)'
                    app.LayerLegendDescLabel.Text = 'Red spots: Detected microaneurysm candidates from morphological top-hat filtering (Earliest NPDR hallmark).';
                case 'Hemorrhages Detection (Magenta)'
                    app.LayerLegendDescLabel.Text = 'Magenta regions: Detected intraretinal blot and flame hemorrhages indicating capillary rupture.';
                case 'Exudates & DME Risk Map (Yellow)'
                    app.LayerLegendDescLabel.Text = 'Yellow regions: Detected hard lipid exudates. Circles indicate concentric proximity risk zones to fovea.';
                case 'Multi-Lesion Composite Map'
                    app.LayerLegendDescLabel.Text = 'Integrated clinical biomarker map: Red = Microaneurysms | Magenta = Hemorrhages | Yellow = Hard Exudates.';
                case 'Grad-CAM Deep Activation Heatmap'
                    app.LayerLegendDescLabel.Text = 'Jet colormap: SqueezeNet V3 convolutional class activation map (Red = high model attention, Blue = low).';
                case 'Grad-CAM Clinical Explainability Overlay'
                    app.LayerLegendDescLabel.Text = 'Alpha-blended neural attention overlay (50% CAM heatmap + 50% normalized fundus) for spatial audit.';
                otherwise
                    app.LayerLegendDescLabel.Text = 'Displaying selected retinal fundus inspection layer.';
            end
        end

        function onReset(app)
            % ONRESET Clear state and reset all UI controls to default
            app.CurrentImagePath = '';
            app.RawImage = [];
            app.DiagnosticResult = [];

            cla(app.ImageAxes);
            app.ImageAxes.Color = [0.05 0.05 0.07];
            title(app.ImageAxes, 'No Retinal Image Loaded', 'Color', [0.7 0.7 0.7], 'FontName', 'Segoe UI');

            cla(app.SeverityAxes);
            title(app.SeverityAxes, 'Grade Posterior Probabilities', 'FontSize', 9, 'FontWeight', 'normal');

            app.ImageInfoLabel.Text = 'Image: None loaded | Resolution: -- | Quality: -- | Grad-CAM Agreement: --';
            app.updateLayerLegend('Original Fundus Photograph');

            % Reset Tab 1
            app.TriagePanel.BackgroundColor = [0.93 0.94 0.96];
            app.TriageActionLabel.Text = 'TRIAGE STATUS: AWAITING INFERENCE';
            app.TriageActionLabel.FontColor = [0.3 0.3 0.3];
            app.TriageDetailLabel.Text = 'Select an image and click "Run SHINRA Screening" to initiate analysis.';

            app.ReferableStatusLabel.Text = '--';
            app.ReferableStatusLabel.FontColor = [0.2 0.2 0.2];
            app.ReferableProbLabel.Text = 'Calibrated P(ref): --';

            app.SeverityGradeLabel.Text = 'Grade: --';
            app.SeverityGradeLabel.FontColor = [0.2 0.2 0.2];
            app.SeverityScoreLabel.Text = 'Ordinal Score E[G]: --\nConfidence: --';

            app.QualityGateLabel.Text = 'Quality Safety Gate: --\nScore: -- | Class: --';
            app.ReliabilityGateLabel.Text = 'Reliability Safety Gate (τ ≥ 60.0):\nScore: -- | Gate: --';

            % Reset Tab 2
            app.LesionsTable.Data = {};
            app.ReliabilityTable.Data = {};

            % Reset Tab 3
            app.ReportTextArea.Value = {'No screening report generated. Select an image and run screening.'};

            % Buttons
            app.RunScreeningBtn.Enable = 'off';
            app.ExportPdfBtn.Enable = 'off';
            app.SaveTxtBtn.Enable = 'off';
            app.SaveImageBtn.Enable = 'off';
                                    app.ViewSelectorDropdown.Value = 'Original Fundus Photograph';

            app.updateStatus('System Ready. Please select a retinal fundus image.', [0.30 0.85 0.45]);
        end

        % =================================================================
        % EXPORT CALLBACKS
        % =================================================================

        function onExportPdfReport(app)
            % ONEXPORTPDFREPORT Export multi-page clinical screening report as PDF
            if isempty(app.DiagnosticResult)
                return;
            end

            app.updateStatus('Exporting Multi-Page PDF Clinical Report...', [0.85 0.55 0.10]);
            try
                default_name = sprintf('SHINRA_Report_%s.pdf', app.DiagnosticResult.image_id);
                reports_dir = fullfile(pwd, 'results', 'reports');
                if ~exist(reports_dir, 'dir')
                    mkdir(reports_dir);
                end
                default_target = fullfile(reports_dir, default_name);

                [fname, fpath] = uiputfile('*.pdf', 'Export Clinical Screening Report (PDF)', default_target);
                if isequal(fname, 0) || isequal(fpath, 0)
                    app.updateStatus('PDF Export cancelled.', [0.4 0.4 0.4]);
                    return;
                end

                target_file = fullfile(fpath, fname);
                [ok, saved_path] = exportSHINRAReportPDF(app.DiagnosticResult, target_file);
                if ok
                    app.updateStatus(sprintf('Multi-Page PDF Report Saved: %s', fname), [0.15 0.65 0.35]);
                    uialert(app.UIFigure, sprintf('Clinical Screening Report successfully exported to:\n%s', saved_path), ...
                        'PDF Export Complete', 'Icon', 'success');
                else
                    error('PDF generation returned false.');
                end
            catch ME
                app.updateStatus('PDF Export failed.', [0.85 0.20 0.20]);
                uialert(app.UIFigure, sprintf('Failed to export PDF report:\n%s', ME.message), 'PDF Export Error', 'Icon', 'error');
            end
        end

        function onSaveTxtReport(app)
            % ONSAVETXTREPORT Preserved legacy plain-text clinical report export
            if isempty(app.DiagnosticResult)
                return;
            end

            default_name = sprintf('SHINRA_Report_%s.txt', app.DiagnosticResult.image_id);
            reports_dir = fullfile(pwd, 'results', 'reports');
            if ~exist(reports_dir, 'dir')
                mkdir(reports_dir);
            end
            default_target = fullfile(reports_dir, default_name);

            [fname, fpath] = uiputfile('*.txt', 'Save Legacy Plain-Text Clinical Report', default_target);
            if isequal(fname, 0) || isequal(fpath, 0)
                return;
            end

            target_file = fullfile(fpath, fname);
            try
                fid = fopen(target_file, 'w');
                if fid == -1
                    error('Could not open file for writing: %s', target_file);
                end
                fprintf(fid, '%s', app.DiagnosticResult.report_text);
                fclose(fid);
                app.updateStatus(sprintf('Plain-Text Report Saved to %s', fname), [0.15 0.65 0.35]);
            catch ME
                uialert(app.UIFigure, sprintf('Failed to save plain-text report:\n%s', ME.message), 'Save Error', 'Icon', 'error');
            end
        end

        function onSaveImage(app)
            % ONSAVEIMAGE Export current active overlay to high-resolution PNG
            if isempty(app.DiagnosticResult)
                return;
            end

            mode_str = strrep(lower(app.ViewSelectorDropdown.Value), ' ', '_');
            mode_str = regexprep(mode_str, '[^a-z0-9_]', '');
            default_name = sprintf('SHINRA_%s_%s.png', app.DiagnosticResult.image_id, mode_str);

            figures_dir = fullfile(pwd, 'results', 'figures');
            if ~exist(figures_dir, 'dir')
                mkdir(figures_dir);
            end
            default_target = fullfile(figures_dir, default_name);

            [fname, fpath] = uiputfile('*.png', 'Save Active Retinal Overlay', default_target);
            if isequal(fname, 0) || isequal(fpath, 0)
                return;
            end

            target_file = fullfile(fpath, fname);
            try
                img_data = getimage(app.ImageAxes);
                if isempty(img_data)
                    f = getframe(app.ImageAxes);
                    img_data = f.cdata;
                end
                imwrite(img_data, target_file);
                app.updateStatus(sprintf('Annotated Overlay Saved: %s', fname), [0.15 0.65 0.35]);
            catch ME
                uialert(app.UIFigure, sprintf('Failed to save image overlay:\n%s', ME.message), 'Save Error', 'Icon', 'error');
            end
        end

        function updateStatus(app, status_text, text_color)
            % UPDATESTATUS Update the header status label
            if nargin < 3
                text_color = [1.0 1.0 1.0];
            end
            app.StatusLabel.Text = status_text;
            app.StatusLabel.FontColor = text_color;
            drawnow;
        end
    end

    % App initialization and construction
    methods (Access = public)

        function app = SHINRA_App()
            % Constructor
            createComponents(app);
            registerApp(app, app.UIFigure);
        end

        function delete(app)
            % Destructor
            delete(app.UIFigure);
        end
    end
end

function str = bool2str(val, true_str, false_str)
    if val
        str = true_str;
    else
        str = false_str;
    end
end

function out = iif(cond, val_true, val_false)
    if cond
        out = val_true;
    else
        out = val_false;
    end
end
