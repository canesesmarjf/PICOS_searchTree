% Main_1 post process:
clear all
close all
clc

% Load raw data:
data_folder = "../input_files/";
x_p = readmatrix(data_folder + "Step_1_x_p" + ".csv");
v_p = readmatrix(data_folder + "Step_1_v_p" + ".csv");

% Normalize data:
x_norm = 1;
v_norm = max(max(v_p));

x_p = x_p/x_norm;
v_p = v_p/v_norm;

% Get data produced by C++ code:
xx = 9;
data_folder = "../output_files/main_1/";
particle_count = readmatrix(data_folder + "leaf_v_" + "p_count" ...
    + "_xx_" + string(xx) + ".csv");
node_center = readmatrix(data_folder + "leaf_v_" + "node_center" ...
    + "_xx_" + string(xx) + ".csv");
node_dim = readmatrix(data_folder + "leaf_v_" + "node_dim" ...
    + "_xx_" + string(xx) + ".csv");
x_q = readmatrix(data_folder + "x_q" + ".csv");
xx = xx + 1;

% Derived quantities:
dx = mean(diff(x_q));

% Set the minimum particle count to use for reampling a node:
min_count = 7;

% Lets focus on an individual leaf_v:
z1 = x_q(xx) - dx/2;
z2 = x_q(xx) + dx/2;
rng = find(x_p > z1 & x_p < z2);

figure('color','w')
hold on
box on
axis image
xlim([-1,1])
ylim([0,1])
for vv = 1:numel(particle_count)
    hsq(vv) = plotSquare(node_center(vv,:), node_dim(vv,:), particle_count(vv),min_count);
    set(hsq(vv),'lineWidth',1);
end
hold on
plot(v_p(rng,1),v_p(rng,2),'k.','MarkerSize',5);
title("xx = " + string(xx))


% Calculate how many new memory locations are to be released:
rng_count = find(particle_count > min_count);
free_mem = sum(particle_count(rng_count) - 6);
total_mem = sum(particle_count);

disp("Critical particle count per node is " + string(min_count));
disp("Total number of particles is " + string(total_mem));
disp("Free memory locations is " + string(free_mem));
disp("Particles left if all free mems taken is " + string(total_mem - free_mem));


data_folder = "../output_files/main_1/";
p_count = readmatrix(data_folder + "leaf_x_p_count" + ".csv");
mean_p_count = mean(p_count);

figure;
box on
hold on
bar(x_q,p_count)
plot(x_q(xx),p_count(xx),'r.')
line([min(x_q),max(x_q)],[1,1]*mean_p_count)

%% Functions:
% =========================================================================
function [] = plot_binary_tree_grid(ax,kx,ky,kz)
    % Produce grid:
    grids.x = linspace(-1,+1,2^kx + 1);
    grids.y = linspace(-1,+1,2^ky +1);
    grids.z = linspace(0,+1,2^kz + 1);
    
    disp("Total nodes: " + num2str(2^(kx+ky+kz)))
    
    % Draw grid:
    ax.XTick = grids.x(1:1:end); ax.XTickLabel = grids.x(1:1:end);
    ax.YTick = grids.y(1:1:end); ax.YTickLabel = grids.y(1:1:end);
    ax.ZTick = grids.z(1:1:end); ax.ZTickLabel = grids.z(1:1:end);
    grid on;
end

function h = plotSquare(center, dimensions, particle_count, min_count)
    % center: a vector [x, y] representing the coordinates of the center
    % dimensions: a vector [dx, dy] representing the width and height of the square

    % Extract center coordinates
    x = center(1);
    y = center(2);

    % Extract dimensions
    dx = dimensions(1);
    dy = dimensions(2);

    % Calculate the coordinates of the square's vertices
    verticesX = [x-dx/2, x+dx/2, x+dx/2, x-dx/2, x-dx/2];
    verticesY = [y-dy/2, y-dy/2, y+dy/2, y+dy/2, y-dy/2];

    % Plot the square
    hold on;  % Optional: maintain existing plot
    if (particle_count > min_count)
       h = plot(verticesX, verticesY, 'r-', 'LineWidth', 4);
       fill(verticesX, verticesY, 'r', 'LineWidth',1);
    else
       h = plot(verticesX, verticesY, 'b-', 'LineWidth', 1);
    end

    % Annotate particle_count:
    text(x,y,num2str(particle_count),'HorizontalAlignment',...
        'center','VerticalAlignment','middle','FontSize',6)

    hold off; % Optional: release hold on existing plot

    % Optional: Adjust figure axes if necessary
    axis equal; % Set equal scaling for x and y axes
    % Adjust xlim and ylim as needed to fit the square within the figure
    % xlim([xmin, xmax]);
    % ylim([ymin, ymax]);

    % Optional: Add labels or title to the plot
    % xlabel('X');
    % ylabel('Y');
    % title('Square Plot');
end