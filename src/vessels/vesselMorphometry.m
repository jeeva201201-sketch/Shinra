function morpho = vesselMorphometry(vessel_mask, fov_mask, od_mask)
% VESSELMORPHOMETRY Quantitative Extraction of Retinal Vascular Architecture Biomarkers
% Computes vessel density, total skeleton length, branching points,
% caliber variation, and tortuosity indices across the retinal vascular tree.

    if nargin < 2 || isempty(fov_mask)
        fov_mask = true(size(vessel_mask));
    end
    if nargin < 3 || isempty(od_mask)
        od_mask = false(size(vessel_mask));
    end

    morpho = struct();
    retinal_area = max(1, sum(fov_mask(:)));

    % 1. Retinal Vessel Area Density
    vessel_pixels = sum(vessel_mask(:) & fov_mask(:));
    morpho.vessel_density = vessel_pixels / retinal_area;

    % 2. Vessel Skeletonization
    skeleton = bwmorph(vessel_mask & fov_mask, 'skel', Inf);
    skeleton = bwmorph(skeleton, 'spur', 3); % Prune minor spurs
    morpho.total_vessel_length = sum(skeleton(:));

    % 3. Branching Points & End Points
    branch_pts = bwmorph(skeleton, 'branchpoints');
    end_pts    = bwmorph(skeleton, 'endpoints');
    morpho.num_branch_points = sum(branch_pts(:));
    morpho.num_end_points    = sum(end_pts(:));
    morpho.branching_density = (morpho.num_branch_points / retinal_area) * 1e4;

    % 4. Mean Vessel Caliber (Effective width in pixels)
    if morpho.total_vessel_length > 0
        morpho.mean_vessel_caliber = vessel_pixels / morpho.total_vessel_length;
    else
        morpho.mean_vessel_caliber = 0;
    end

    % 5. Vascular Tortuosity Index
    % Ratio of actual vessel length between branches to straight-line Euclidean distance
    cc = bwconncomp(skeleton - branch_pts);
    if cc.NumObjects > 5
        tortuosity_list = zeros(cc.NumObjects, 1);
        valid_cnt = 0;
        for k = 1:cc.NumObjects
            idx = cc.PixelIdxList{k};
            if numel(idx) > 10 % Sufficient length for tortuosity calculation
                [sy, sx] = ind2sub(size(skeleton), idx);
                % Arc length is approximately number of pixels
                arc_len = numel(idx);
                % Chord length between endpoints
                chord_len = sqrt((sx(1) - sx(end))^2 + (sy(1) - sy(end))^2);
                if chord_len > 2
                    valid_cnt = valid_cnt + 1;
                    tortuosity_list(valid_cnt) = arc_len / chord_len;
                end
            end
        end
        if valid_cnt > 0
            morpho.mean_tortuosity = mean(tortuosity_list(1:valid_cnt));
        else
            morpho.mean_tortuosity = 1.0;
        end
    else
        morpho.mean_tortuosity = 1.0;
    end
end
