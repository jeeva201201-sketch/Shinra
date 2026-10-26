function nv_result = analyzeNeovascularization(vessel_mask, fov_mask, od_mask, morpho)
% ANALYZENEOVASCULARIZATION Exploratory Analysis of Proliferative Neovascularization
% Extracts vascular architectural biomarkers of new vessel proliferation
% (NVD at Optic Disc and NVE Elsewhere) via fine-vessel clustering and high tortuosity.
%
% Note: Documented as an exploratory feature-based screening biomarker.

    if nargin < 2 || isempty(fov_mask)
        fov_mask = true(size(vessel_mask));
    end
    if nargin < 3 || isempty(od_mask)
        od_mask = false(size(vessel_mask));
    end
    if nargin < 4 || isempty(morpho)
        morpho = vesselMorphometry(vessel_mask, fov_mask, od_mask);
    end

    [rows, cols] = size(vessel_mask);
    nv_result = struct();

    % 1. Thin / Fine vessel extraction (exclude primary main vessel trunks)
    trunk_vessels = imopen(vessel_mask, strel('disk', 2));
    fine_vessels  = vessel_mask & ~trunk_vessels;

    % 2. Optic Disc Neovascularization (NVD) Analysis
    % Neovascularization within 1 disc diameter of OD boundary
    od_peridisc_zone = imdilate(od_mask, strel('disk', 15));
    nvd_fine_vessels = fine_vessels & od_peridisc_zone;
    nvd_density = sum(nvd_fine_vessels(:)) / max(1, sum(od_peridisc_zone(:)));

    % 3. Peripheral Neovascularization (NVE) Cluster Detection
    % Dense cluster of fine, tortuous, disorganized vessels
    fine_density_map = imfilter(double(fine_vessels), fspecial('gaussian', 25, 6), 'replicate');
    fine_density_map(~fov_mask) = 0;
    
    nve_clusters = fine_density_map > 0.15;
    nve_clusters = bwareaopen(nve_clusters, 40);

    % 4. Composite Neovascularization Risk Score [0, 1]
    % Weighted combination of NVD density, NVE clustering, and vessel tortuosity
    tortuosity_factor = max(0, min(1.0, (morpho.mean_tortuosity - 1.1) / 0.4));
    branching_factor  = min(1.0, morpho.branching_density / 8.0);
    nvd_factor        = min(1.0, nvd_density / 0.12);
    nve_factor        = min(1.0, sum(nve_clusters(:)) / 500);

    nv_score = 0.35 * nvd_factor + 0.30 * nve_factor + 0.20 * tortuosity_factor + 0.15 * branching_factor;
    nv_score = max(0, min(1.0, nv_score));

    nv_result.nv_score = nv_score;
    nv_result.is_nv_suspected = nv_score >= 0.50;
    nv_result.nvd_density = nvd_density;
    nv_result.candidate_mask = (nvd_fine_vessels | nve_clusters) & fov_mask;
    nv_result.confidence = min(0.90, 0.4 + 0.5 * nv_score);
    nv_result.status = 'EXPLORATORY_BIOMARKER';
end
