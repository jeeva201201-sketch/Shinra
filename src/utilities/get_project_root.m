function root_dir = get_project_root()
% GET_PROJECT_ROOT Dynamically resolves the project root directory
% Returns the absolute path to the SHINRA root directory without hardcoding.
% Works seamlessly across Windows, Linux, macOS, and MATLAB Online.

    % Resolve relative to this function location (src/utilities/get_project_root.m)
    current_file = mfilename('fullpath');
    if isempty(current_file)
        % Fallback if executed in workspace root
        root_dir = pwd;
    else
        utils_dir = fileparts(current_file);
        src_dir = fileparts(utils_dir);
        root_dir = fileparts(src_dir);
    end
end
