clear all; close all; clear classes; clc;

%% Set flags.
rng('shuffle');
inspect_only = false;
random_length = true;
limit_length = true;
limit_min_length = false;

%% Set Strings
TRAIN_NAME = '200x100_rl_limited_ubuntu_wonderit_fixed'
TRAIN_ID = '0001'

IMAGE_PATH = '200x100_rl_limited_ubuntu_wonderit_fixed/test/11111100001111110011_100.tiff'
IMAGE_ID = '11111100001111110011_100'

%% Solve the system.
gray = [0.5 0.5 0.5];  % [r g b]
flux_y = 2000;
flux_x1 = -1000; flux_x2 = 1000;

if ~inspect_only
	%% Visualize the solution.
	figure
	clear opts
	opts.withobjsrc = true;
	opts.withabs = false;  % true: abs(solution), false: real(solution)
	opts.withpml = false;  % true: show PML, false: do not show PML
	opts.withgrid = false;
	z_loc = 5;
    
    result_array = [];
    
    imageArray = imread(IMAGE_PATH);
    imageWidth = size(imageArray, 2);
    imageHeight = size(imageArray, 1);
    rectWidth = 10;
    
    randomBoxArray = []
    for i = 1 : 10 : imageWidth
        rectHeight = 0;

        for j = 1 : 1 : imageHeight
            if imageArray(j, i)
                rectHeight = rectHeight + 1;
            end
        end
        
        x_start = 100 * i/10 - 1000;
        x_end = 100 * i/10 - 900;

        y_start = 0;
        y_end = rectHeight * 10;
        if rectHeight > 0
            fprintf('x_start : %d, x_end: %d, height : %d', x_start, x_end, rectHeight);
            randomBoxArray = [randomBoxArray , Box([x_start, x_end; y_start, y_end; 0, 10])]
        end
    end
    

    %% Calculate the power flux through the slit.
    P_arr = []

    wavelength = [400:50:1550]
    tic; % TIC, pair 1
    for ii = 1:1:length(wavelength)
         [E, H, obj_array, src_array, J] = maxwell_run(...
        'OSC', 1e-9, wavelength(ii), ...
        'DOM', {'vacuum', 'none', 1.0}, [-1000, 1000; -1000, 2600; 0, 10], 10, BC.p, [100 100 0],...
        'OBJ', ...
            {'vacuum', 'b', 1.0}, Rectangle(Axis.y, flux_y, [0 10; flux_x1 flux_x2]), ...
            {'CRC/Ag', gray}, randomBoxArray, ...
        'SRCM', PlaneSrc(Axis.y, -500, Axis.z), ...
        inspect_only);

        if ii == 1
            vis2d(E{Axis.x}, Axis.z, z_loc, obj_array, src_array, opts)
        end

        power = powerflux_patch(E, H, Axis.y, flux_y, [0 10; flux_x1 flux_x2]);

        P_arr(ii,:) = power;
    end

    result_array = [result_array; transpose(P_arr)];

    averageTime = toc/length(wavelength); % TOC, pair1
    fprintf('TotalTimeElapsed: %.2f, AverageTimeElapsed: %.2f\n', averageTime * length(wavelength), averageTime );

    csvFileName = sprintf('%s/%s.csv', TRAIN_NAME, IMAGE_ID);
    csvwrite(csvFileName,result_array)
    
end
