function diag_report = diagnoseTrainedVesselUNet(net_in)
% DIAGNOSETRAINEDVESSELUNET Comprehensive Runtime Diagnostic for Trained U-Net Object
% Inspects the actual MATLAB R2026a network object representation, layer graph,
% terminal layers, and evaluates a single-image semanticseg / predict test on 25_training.
%
% Usage:
%   diagnoseTrainedVesselUNet(net)
%   or
%   diag_report = diagnoseTrainedVesselUNet();

    cfg = shinra_config();
    fprintf('========================================================================\n');
    fprintf('       SHINRA: TRAINED U-NET RUNTIME OBJECT & API DIAGNOSTIC            \n');
    fprintf('========================================================================\n\n');

    model_file = fullfile(cfg.paths.models_vessels, 'vessel_unet.mat');
    checkpoint_exists = (exist(model_file, 'file') == 2);

    % 1. Resolve Network Object
    if nargin >= 1 && ~isempty(net_in)
        net = net_in;
        source_str = 'Passed directly from workspace argument';
    elseif checkpoint_exists
        loaded = load(model_file);
        if isfield(loaded, 'net')
            net = loaded.net;
            source_str = sprintf('Loaded from checkpoint: %s', model_file);
        else
            error('SHINRA:NoNetVariable', 'Checkpoint %s exists but does not contain a "net" variable.', model_file);
        end
    else
        error('SHINRA:NoNetworkFound', ...
            'No network object passed and checkpoint not found at: %s\nPlease pass the network object: diagnoseTrainedVesselUNet(net)', ...
            model_file);
    end

    diag_report = struct();
    diag_report.source = source_str;
    diag_report.checkpoint_exists = checkpoint_exists;
    diag_report.checkpoint_path = model_file;

    % -------------------------------------------------------------------------
    % 1 & 2. Class and Properties of net
    % -------------------------------------------------------------------------
    net_class = class(net);
    diag_report.net_class = net_class;

    fprintf('1. NETWORK OBJECT IDENTITY & PROPERTIES:\n');
    fprintf('  • Source:                 %s\n', source_str);
    fprintf('  • Class of net:           %s\n', net_class);

    prop_names = properties(net);
    diag_report.properties = prop_names;
    fprintf('  • Public Properties:      [%s]\n', strjoin(prop_names, ', '));

    is_dag = isa(net, 'DAGNetwork') || isa(net, 'nnet.cnn.DAGNetwork');
    is_dl = isa(net, 'dlnetwork');
    is_series = isa(net, 'SeriesNetwork') || isa(net, 'nnet.cnn.SeriesNetwork');

    diag_report.is_dag = is_dag;
    diag_report.is_dl = is_dl;
    diag_report.is_series = is_series;

    fprintf('  • Is DAGNetwork:          %s\n', boolToStr(is_dag));
    fprintf('  • Is dlnetwork:           %s\n', boolToStr(is_dl));
    fprintf('  • Is SeriesNetwork:       %s\n\n', boolToStr(is_series));

    % -------------------------------------------------------------------------
    % 3 & 4. Layer-by-Layer Inspection
    % -------------------------------------------------------------------------
    has_layers_prop = isprop(net, 'Layers') || isfield(net, 'Layers');
    diag_report.has_layers = has_layers_prop;

    has_pixel_class_layer = false;
    has_class_layer = false;
    has_softmax_layer = false;
    pixel_layer_info = struct();
    key_layers = {};

    if has_layers_prop
        num_layers = numel(net.Layers);
        diag_report.num_layers = num_layers;
        fprintf('2. FULL LAYER-BY-LAYER TABLE (%d Layers Total):\n', num_layers);
        fprintf('-----------------------------------------------------------------------------------------\n');
        fprintf('  Idx   Layer Name                     Layer Class                                      \n');
        fprintf('-----------------------------------------------------------------------------------------\n');

        layer_table_cells = cell(num_layers, 3);
        for i = 1:num_layers
            l = net.Layers(i);
            l_name = l.Name;
            l_class = class(l);

            layer_table_cells{i, 1} = i;
            layer_table_cells{i, 2} = l_name;
            layer_table_cells{i, 3} = l_class;

            fprintf('  %3d   %-30s %-45s\n', i, l_name, l_class);

            % Detect Key Layer Types
            if contains(l_class, 'PixelClassificationLayer', 'IgnoreCase', true)
                has_pixel_class_layer = true;
                pixel_layer_info.index = i;
                pixel_layer_info.name = l_name;
                pixel_layer_info.class = l_class;
                if isprop(l, 'Classes')
                    pixel_layer_info.classes = cellstr(l.Classes);
                else
                    pixel_layer_info.classes = {};
                end
                key_layers{end+1} = sprintf('[PIXEL CLASSIFICATION] Layer %d: %s', i, l_name); %#ok<AGROW>
            end
            if contains(l_class, 'ClassificationLayer', 'IgnoreCase', true) && ~contains(l_class, 'Pixel')
                has_class_layer = true;
                key_layers{end+1} = sprintf('[IMAGE CLASSIFICATION] Layer %d: %s', i, l_name); %#ok<AGROW>
            end
            if contains(l_class, 'Softmax', 'IgnoreCase', true)
                has_softmax_layer = true;
                key_layers{end+1} = sprintf('[SOFTMAX] Layer %d: %s', i, l_name); %#ok<AGROW>
            end
            if contains(l_class, 'DepthConcatenation', 'IgnoreCase', true)
                key_layers{end+1} = sprintf('[SKIP CONCAT] Layer %d: %s', i, l_name); %#ok<AGROW>
            end
            if contains(l_class, 'TransposedConvolution', 'IgnoreCase', true)
                key_layers{end+1} = sprintf('[TRANSPOSED CONV] Layer %d: %s', i, l_name); %#ok<AGROW>
            end
            if contains(l_class, 'ImageInputLayer', 'IgnoreCase', true)
                key_layers{end+1} = sprintf('[INPUT LAYER] Layer %d: %s (InputSize: [%s])', ...
                    i, l_name, num2str(l.InputSize)); %#ok<AGROW>
            end
        end
        fprintf('-----------------------------------------------------------------------------------------\n\n');
        diag_report.layer_table = layer_table_cells;
    else
        diag_report.num_layers = 0;
        fprintf('2. NETWORK HAS NO DIRECT .Layers PROPERTY.\n\n');
    end

    diag_report.has_pixel_classification_layer = has_pixel_class_layer;
    diag_report.has_classification_layer = has_class_layer;
    diag_report.has_softmax_layer = has_softmax_layer;
    diag_report.pixel_layer_info = pixel_layer_info;

    fprintf('3. KEY ARCHITECTURAL MILESTONES FOUND:\n');
    if ~isempty(key_layers)
        for k = 1:numel(key_layers)
            fprintf('  • %s\n', key_layers{k});
        end
    else
        fprintf('  • No standard milestone layers detected.\n');
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 5 & 6. Network Graph, Inputs, and Outputs Inspection
    % -------------------------------------------------------------------------
    fprintf('4. NETWORK GRAPH, INPUTS & OUTPUTS:\n');
    if isprop(net, 'InputNames')
        fprintf('  • Input Names:            [%s]\n', strjoin(net.InputNames, ', '));
        diag_report.input_names = net.InputNames;
    end
    if isprop(net, 'OutputNames')
        fprintf('  • Output Names:           [%s]\n', strjoin(net.OutputNames, ', '));
        diag_report.output_names = net.OutputNames;
    end
    if isprop(net, 'Connections')
        num_conns = height(net.Connections);
        fprintf('  • Total Connections:      %d graph edges\n', num_conns);
        diag_report.num_connections = num_conns;
    end
    if is_dl && isprop(net, 'Learnables')
        fprintf('  • dlnetwork Learnables:   %d parameters\n', height(net.Learnables));
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 7. Inspect Checkpoint Variables (if MAT file exists)
    % -------------------------------------------------------------------------
    fprintf('5. CHECKPOINT MAT FILE INSPECTION:\n');
    if checkpoint_exists
        loaded_vars = whos('-file', model_file);
        var_names = {loaded_vars.name};
        fprintf('  • Checkpoint File:        %s\n', model_file);
        fprintf('  • Stored Variables:       [%s]\n', strjoin(var_names, ', '));
        for vi = 1:numel(loaded_vars)
            fprintf('    - %-20s: %-15s [%s] (%.2f KB)\n', ...
                loaded_vars(vi).name, loaded_vars(vi).class, ...
                num2str(loaded_vars(vi).size), loaded_vars(vi).bytes / 1024);
        end
        diag_report.checkpoint_variables = var_names;
    else
        fprintf('  • Checkpoint File:        NOT FOUND on disk (%s)\n', model_file);
        diag_report.checkpoint_variables = {};
    end
    fprintf('\n');

    % -------------------------------------------------------------------------
    % 8 & 9. Single-Image Dry-Run Test (Image 25_training)
    % -------------------------------------------------------------------------
    fprintf('6. SINGLE-IMAGE INFERENCE DRY-RUN TEST (25_training):\n');
    test_id = '25_training';
    drive_root = cfg.paths.drive;
    img_dir = fullfile(drive_root, 'training', 'images');
    fov_dir = fullfile(drive_root, 'training', 'mask');

    img_path = fullfile(img_dir, [test_id, '.tif']);
    if ~exist(img_path, 'file'); img_path = fullfile(img_dir, [test_id, '.png']); end
    fov_path = fullfile(fov_dir, '25_training_mask.gif');

    target_size = [512, 512];
    semanticseg_pass = false;
    semanticseg_error_msg = 'Not attempted';
    predict_pass = false;
    predict_output_shape = [];

    if exist(img_path, 'file') == 2
        img_rgb = imread(img_path);
        if exist(fov_path, 'file') == 2
            fov_mask = imread(fov_path) > 0;
        else
            fov_mask = true(size(img_rgb, 1), size(img_rgb, 2));
        end

        g_chan = im2double(img_rgb(:, :, 2));
        g_clahe = adapthisteq(g_chan, 'ClipLimit', 0.015, 'NumTiles', [8, 8]);
        g_clahe(~fov_mask) = 0;
        test_in = single(imresize(g_clahe, target_size, 'bicubic'));

        % Test A: semanticseg(test_in, net)
        fprintf('  [A] Testing: semanticseg(test_in, net)...\n');
        try
            pred_cat = semanticseg(test_in, net);
            semanticseg_pass = true;
            semanticseg_error_msg = 'NONE (Success)';
            cats_found = categories(pred_cat);
            num_vessels = sum(pred_cat(:) == 'Vessel');
            fprintf('      -> RESULT: SUCCESS (Output Size: [%s], Class: %s, Categories: [%s], Vessel Pixels: %d)\n', ...
                num2str(size(pred_cat)), class(pred_cat), strjoin(cats_found, ', '), num_vessels);
        catch ME_sem
            semanticseg_pass = false;
            semanticseg_error_msg = sprintf('%s (%s)', ME_sem.message, ME_sem.identifier);
            fprintf('      -> RESULT: FAILED\n');
            fprintf('         Identifier: %s\n', ME_sem.identifier);
            fprintf('         Message:    %s\n', ME_sem.message);
        end

        % Test B: predict(net, test_in)
        fprintf('  [B] Testing: predict(net, test_in)...\n');
        try
            raw_pred = predict(net, test_in);
            predict_pass = true;
            predict_output_shape = size(raw_pred);
            fprintf('      -> RESULT: SUCCESS (Output Size: [%s], Class: %s, Min: %.4f, Max: %.4f)\n', ...
                num2str(size(raw_pred)), class(raw_pred), min(raw_pred(:)), max(raw_pred(:)));
        catch ME_pred
            predict_pass = false;
            fprintf('      -> RESULT: FAILED\n');
            fprintf('         Identifier: %s\n', ME_pred.identifier);
            fprintf('         Message:    %s\n', ME_pred.message);
        end
    else
        fprintf('  [ERROR] Test image %s not found in %s\n', test_id, img_dir);
        semanticseg_error_msg = 'Test image 25_training not found';
    end
    fprintf('\n');

    diag_report.semanticseg_pass = semanticseg_pass;
    diag_report.semanticseg_error = semanticseg_error_msg;
    diag_report.predict_pass = predict_pass;
    diag_report.predict_output_shape = predict_output_shape;

    % -------------------------------------------------------------------------
    % 10. Summary Report as Requested
    % -------------------------------------------------------------------------
    fprintf('========================================================================\n');
    fprintf('                    DIAGNOSTIC EXECUTIVE SUMMARY                        \n');
    fprintf('========================================================================\n');
    fprintf('ACTUAL NETWORK CLASS:             %s\n', net_class);
    if has_layers_prop
        fprintf('NUMBER OF LAYERS:                 %d\n', numel(net.Layers));
    else
        fprintf('NUMBER OF LAYERS:                 N/A\n');
    end
    if isprop(net, 'OutputNames')
        fprintf('OUTPUT/TERMINAL STRUCTURE:        [%s]\n', strjoin(net.OutputNames, ', '));
    else
        fprintf('OUTPUT/TERMINAL STRUCTURE:        N/A\n');
    end
    fprintf('PIXEL CLASSIFICATION LAYER FOUND: %s\n', boolToStr(has_pixel_class_layer));
    if semanticseg_pass
        fprintf('SEMANTICSEG DRY-RUN:              PASS\n');
    else
        fprintf('SEMANTICSEG DRY-RUN:              FAIL\n');
    end
    fprintf('EXACT ERROR IF FAIL:              %s\n', semanticseg_error_msg);
    fprintf('CHECKPOINT EXISTS:                %s\n', boolToStr(checkpoint_exists));
    fprintf('========================================================================\n\n');
end

function s = boolToStr(b)
    if b
        s = 'YES';
    else
        s = 'NO';
    end
end
