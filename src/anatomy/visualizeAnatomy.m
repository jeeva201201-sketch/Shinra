function overlay_img = visualizeAnatomy(img, od_result, fovea_result)
% VISUALIZEANATOMY Generate Color Overlays for Optic Disc and Foveal Anatomy
% Draws the detected Optic Disc boundary (cyan), center marker,
% Foveal Avascular Zone (yellow crosshair), and Macular Danger Zone (amber circle).

    if ~isa(img, 'double')
        img = im2double(img);
    end
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end

    overlay_img = img;
    [rows, cols, ~] = size(img);

    % Draw Optic Disc Contour (Cyan: [0, 1, 1])
    od_boundary = bwperim(od_result.mask);
    od_boundary_dilated = imdilate(od_boundary, strel('disk', 2));
    
    for c = 1:3
        ch = overlay_img(:, :, c);
        if c == 1
            ch(od_boundary_dilated) = 0.0;
        else
            ch(od_boundary_dilated) = 1.0;
        end
        overlay_img(:, :, c) = ch;
    end

    % Draw Macula Boundary (Amber: [1, 0.7, 0])
    macula_boundary = bwperim(fovea_result.macula_mask);
    macula_dilated = imdilate(macula_boundary, strel('disk', 1));
    
    for c = 1:3
        ch = overlay_img(:, :, c);
        if c == 1
            ch(macula_dilated) = 1.0;
        elseif c == 2
            ch(macula_dilated) = 0.7;
        else
            ch(macula_dilated) = 0.0;
        end
        overlay_img(:, :, c) = ch;
    end

    % Draw Fovea Crosshair (Yellow: [1, 1, 0])
    fx = fovea_result.center(1);
    fy = fovea_result.center(2);
    cross_len = 10;
    
    for dx = -cross_len:cross_len
        cx = fx + dx;
        if cx >= 1 && cx <= cols && fy >= 1 && fy <= rows
            overlay_img(fy, cx, :) = [1, 1, 0];
        end
    end
    for dy = -cross_len:cross_len
        cy = fy + dy;
        if fx >= 1 && fx <= cols && cy >= 1 && cy <= rows
            overlay_img(cy, fx, :) = [1, 1, 0];
        end
    end
end
