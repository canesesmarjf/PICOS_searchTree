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

data_folder = "../output_files/main_1/";
p_count = readmatrix(data_folder + "p_count" + ".csv");

z1 = -0.4375;
z2 = -0.375;
rng = find(x_p > z1 & x_p < z2);

figure('color','w')
plot3(x_p(rng),v_p(rng,1),v_p(rng,2),'k.','MarkerSize',5)
axis image
xlim([-1,1])
ylim([-1,1])
zlim([0,1])
plot_binary_tree_grid(gca,5-2,6-2,5-2)
view([90,0])

figure;
bar(p_count)

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