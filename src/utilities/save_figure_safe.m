function save_figure_safe(fig_handle, output_path)
% SAVE_FIGURE_SAFE Headless Figure Exporter Compatible with MATLAB Online
% Saves figures cleanly to disk in PNG format without requiring GUI display.

    [parent_dir, ~, ~] = fileparts(output_path);
    if ~exist(parent_dir, 'dir')
        mkdir(parent_dir);
    end

    try
        exportgraphics(fig_handle, output_path, 'Resolution', 150);
    catch
        try
            saveas(fig_handle, output_path);
        catch ME
            warning('Failed to save figure to %s: %s', output_path, ME.message);
        end
    end
end
