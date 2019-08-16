clear all; close all; clear classes; clc;

%% Set flags.
inspect_only = false;
random_length = true;
limit_length = true;

%% Set Strings
TRAIN_NAME = '200x100_rl_fixed'
TRAIN_ID = '0001'

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
% 	opts.withinterp = false;
% 	opts.cscale = 1e-1;
	z_loc = 5;
    
    loop_number = 1000;
    result_array = [];
    random_max = 2^20-1;
    random_row = 20;
    max_length = 100;
    
    for n = 1 : 1 : loop_number
        
        randomBoxArray = []
        objectNumber = randi([1 random_max]); % 2^20-1

        binNumber = dec2bin(objectNumber, random_row);
        fprintf('Get Random Number = %d\n', objectNumber);
        fprintf('Decimal to Binary = %s\n', binNumber);
        
        % Initialize image array
        imageArray = zeros(100,200);
        
        % Initialize Random Limit Length
        random_max_length = randi([max_length/2 max_length]);
        max_length_for_this_loop = max_length;
        
        if limit_length
            max_length_for_this_loop = random_max_length;
        end
        
        fprintf('Max Length : %d\n', max_length_for_this_loop);
        
        % Initialize shift length for vertical align = center
        shift_length = (max_length - max_length_for_this_loop) / 2
        
        for i = 0 : 1 : strlength(binNumber)-1
%             fprintf('loop result = %c\n', binNumber(i+1));

            if binNumber(i+1) == '1'
                x_start = 100 * i - 1000;
                x_end = 100 * i - 900;
                
                y_start = 0;
                y_end = max_length_for_this_loop * 10;
                
                % Add Random Length for Rectangles
                if random_length
                    rect_len = randi([round(max_length_for_this_loop/2) max_length_for_this_loop]);
                    rect_loc = randi([0 max_length_for_this_loop-rect_len]);
                    
                    y_start = (rect_loc + shift_length) * 10;
                    y_end = (rect_loc + rect_len + shift_length) * 10;
                end
                fprintf('rect_length : %d, rect_loc : %d \n', rect_len, rect_loc);
                
                x_image_start = 10 * i+1;
                x_image_end = 10 * i + 10;
                y_image_start = y_start/10+1;
                y_image_end = y_end/10;
%                 imageArray(1:10, 1:10) = 1;
                imageArray(y_image_start:y_image_end,x_image_start:x_image_end) = 1;
                
%                 fprintf('result made on : %c\n', binNumber(i+1));
                randomBoxArray = [randomBoxArray , Box([x_start, x_end; y_start, y_end; 0, 10])]
            end
        end
        
        %% Image Array to TIFF File
        foldername = sprintf('%s/%s', TRAIN_NAME, TRAIN_ID);
        mkdir(foldername);
        filename = sprintf('%s/%s/%d.tiff', TRAIN_NAME, TRAIN_ID, n);
        imwrite(logical(flipud(imageArray)), filename, 'tif');

        %% Calculate the power flux through the slit.
        P_arr = []

        wavelength = [400:50:1550]
        tic; % TIC, pair 1
        for ii = 1:1:length(wavelength)
%             tStart = tic; % TIC, pair 2
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
            
            % Calculate as percentage
%             power = power / 2500;
%             fprintf('power = %e\n', power);

            P_arr(ii,:) = power;
    %          clear E, H, obj_array, src_array, J;
%             tElapsed = toc(tStart); % TOC, pair2

%             fprintf('n : %d, P : %e, TimeElapsed: %.2f\n', wavelength(ii), power, tElapsed );
        end
        
        image_id = objectNumber;
        if random_length
            image_id = n;
        end
    
        P_arr = [image_id; P_arr];
        
        
        result_array = [result_array; transpose(P_arr)];

        averageTime = toc/length(wavelength); % TOC, pair1
        fprintf('ITERATION NUM : %d , TotalTimeElapsed: %.2f, AverageTimeElapsed: %.2f\n',n, averageTime * length(wavelength), averageTime );
    %     plot(wavelength,P_arr)
        
    end
    csvFileName = sprintf('%s/%s.csv', TRAIN_NAME, TRAIN_ID);
    csvwrite(csvFileName,result_array)
    
    % test
    
    
end
