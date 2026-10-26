test_dir = fileparts(mfilename('fullpath'));
project_root = fileparts(test_dir);
addpath(project_root);
cd(project_root);
setup_SHINRA();

dev_img_path = fullfile(project_root, 'SHINRAdataset', 'IDRid', 'B. Disease Grading', '1. Original Images', 'a. Training Set', 'IDRiD_001.jpg');
fprintf('Running single-image inference on %s...\n', dev_img_path);
res = runSHINRASingleImage(dev_img_path);

layers_info = {
    1, 'Original Fundus Photograph', res.overlays.original_rgb;
    2, 'Preprocessed / CLAHE Enhanced', res.overlays.preprocessed_rgb;
    3, 'Retinal Anatomy (OD & Fovea)', res.overlays.anatomy;
    4, 'Retinal Vessel Segmentation', res.overlays.vessels;
    5, 'Microaneurysm Detection', res.overlays.microaneurysms;
    6, 'Hemorrhage Detection', res.overlays.hemorrhages;
    7, 'Exudates / DME Risk', res.overlays.exudates;
    8, 'Multi-Lesion Composite', res.overlays.multi_lesions;
    9, 'Grad-CAM Attention Heatmap', res.overlays.gradcam_heatmap;
    10, 'Grad-CAM / Fundus Blend', res.overlays.gradcam_overlay
};

out_dir = fullfile('results', 'figures', 'debug_layers');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

fprintf('\n%-5s | %-35s | %-15s | %-8s | %-6s | %-6s | %s\n', ...
    'Slot', 'Layer Name', 'Size', 'Class', 'Min', 'Max', 'Status');
fprintf('%s\n', repmat('-', [1, 95]));

for k = 1:size(layers_info, 1)
    slot = layers_info{k, 1};
    name = layers_info{k, 2};
    img = layers_info{k, 3};
    
    sz = size(img);
    cls = class(img);
    mn = double(min(img(:)));
    mx = double(max(img(:)));
    
    is_valid = ~isempty(img) && ndims(img) == 3 && sz(3) == 3 && ~any(isnan(img(:))) && mx > mn;
    status_str = 'VALID';
    if ~is_valid
        status_str = 'INVALID / BLANK';
    end
    
    fprintf('%-5d | %-35s | %-15s | %-8s | %-6.2f | %-6.2f | %s\n', ...
        slot, name, mat2str(sz), cls, mn, mx, status_str);
    
    % Save image to test rendering
    if isa(img, 'double')
        img_to_save = uint8(round(min(max(img, 0), 1) * 255));
    else
        img_to_save = img;
    end
    imwrite(img_to_save, fullfile(out_dir, sprintf('layer_%02d.png', slot)));
end

fprintf('\nAll 10 layer test images written to: %s\n', out_dir);
