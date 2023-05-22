% Main_1 post process:
clear all
close all
clc

% Load raw data:
data_folder = "../input_files/";
x_p = readmatrix(data_folder + "Step_1_x_p" + ".csv");
v_p = readmatrix(data_folder + "Step_1_v_p" + ".csv");
a_p = readmatrix(data_folder + "Step_1_a_p" + ".csv");

% Normalize data:
x_norm = 1;
v_norm = max(max(v_p));

x_p = x_p/x_norm;
v_p = v_p/v_norm;

% Get data produced by C++ code:
xx = 19;
data_folder = "../output_files/main_1/";
particle_count = readmatrix(data_folder + "leaf_v_" + "p_count" ...
    + "_xx_" + string(xx) + ".csv");
node_center = readmatrix(data_folder + "leaf_v_" + "node_center" ...
    + "_xx_" + string(xx) + ".csv");
node_dim = readmatrix(data_folder + "leaf_v_" + "node_dim" ...
    + "_xx_" + string(xx) + ".csv");
x_q = readmatrix(data_folder + "x_q" + ".csv");
xx = xx + 1;

x_pn = readmatrix(data_folder + "x_p_new" + ".csv");
v_pn = readmatrix(data_folder + "v_p_new" + ".csv");
a_pn = readmatrix(data_folder + "a_p_new" + ".csv");

x_pn = x_pn/x_norm;
v_pn = v_pn/v_norm;

% Derived quantities:
dx = mean(diff(x_q));

% Set the minimum particle count to use for reampling a node:
min_count = 7;

% Lets focus on an individual leaf_v:
z1 = x_q(xx) - dx/2;
z2 = x_q(xx) + dx/2;
rng = find(x_p > z1 & x_p < z2);
rng_new = find(x_pn > z1 & x_pn < z2);

% Grid depths:
kx = 6;
ky = 6;
kz = 6;
n_skip = 3;

figure('color','w')
hold on
box on
% axis image
% xlim([-1,1])
% ylim([0,1])
for vv = 1:numel(particle_count)
    hsq(vv) = plotSquare(node_center(vv,:), node_dim(vv,:), particle_count(vv),min_count);
    set(hsq(vv),'lineWidth',1);
end
hold on
plot(v_p(rng,1),v_p(rng,2),'k.','MarkerSize',15);
plot(v_pn(rng_new,1),v_pn(rng_new,2),'g.','MarkerSize',8);
title("xx = " + string(xx))
plot_binary_tree_grid(gca,kx,ky,kz,n_skip);
axis image
xlim([-1,1])
ylim([0,1])

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
p_count_new = readmatrix(data_folder + "leaf_x_p_count_new" + ".csv");
mean_p_count = mean(p_count);

figure;
box on
hold on
bar(x_q,p_count)
plot(x_q,p_count_new,'ro-')
plot(x_q(xx),p_count(xx),'r.')
line([min(x_q),max(x_q)],[1,1]*mean_p_count)

%% Comparing phase space data:

% Plot results:
% =========================================================================
rng = find(x_pn ~= -1);

IONS.x_p = x_p;
IONS.v_p = v_p;
IONS_new.x_p = x_pn;
IONS_new.v_p = v_pn;

var{1} = IONS;
var{2} = IONS_new;

kx = 5;
ky = 6;
kz = 6;
n_skip = 3;

for rr = 1:2
    figure('color','w');
    plot_increase_size(1.5,1.5);

    % Set up the first subplot (3D view)
    subplot(2,2,1);
    box on;
    plot3(var{rr}.x_p,var{rr}.v_p(:,1),var{rr}.v_p(:,2),'k.','MarkerSize',1);
    xlabel('$x$','Interpreter','latex','FontSize',18);
    ylabel('$v_{\parallel}$','Interpreter','latex','FontSize',18);
    zlabel('$v_{\perp}$','Interpreter','latex','FontSize',18);
    view([-30,30]);
    xlim([-1,1])
    ylim([-1,1])
    zlim([0,1])
    title('3D view','Interpreter','latex','FontSize',14);
    plot_binary_tree_grid(gca,kx,ky,kz,n_skip);

    % Set up the second subplot (xy view)
    subplot(2,2,2);
    box on;
    plot(var{rr}.x_p,var{rr}.v_p(:,1),'k.','MarkerSize',1);
    xlabel('$x$','Interpreter','latex','FontSize',18);
    ylabel('$v_{\parallel}$','Interpreter','latex','FontSize',18);
    title('xy view','Interpreter','latex','FontSize',14);
    plot_binary_tree_grid(gca,kx,ky,kz,n_skip);
    
    % Set up the third subplot (xz view)
    subplot(2,2,3);
    box on;
    plot(var{rr}.x_p,var{rr}.v_p(:,2),'k.','MarkerSize',1);
    xlabel('$x$','Interpreter','latex','FontSize',18);
    ylabel('$v_{\perp}$','Interpreter','latex','FontSize',18);
    title('xz view','Interpreter','latex','FontSize',14);
    plot_binary_tree_grid(gca,kx,ky,kz,n_skip);
    
    % Set up the fourth subplot (yz view)
    subplot(2,2,4);
    box on;
    plot(var{rr}.v_p(:,1),var{rr}.v_p(:,2),'k.','MarkerSize',1);
    xlabel('$v_{\parallel}$','Interpreter','latex','FontSize',18);
    ylabel('$v_{\perp}$','Interpreter','latex','FontSize',18);
    title('yz view','Interpreter','latex','FontSize',14);
    plot_binary_tree_grid(gca,kx,ky,kz,n_skip);
end

%% Functions:
% =========================================================================
function [] = plot_binary_tree_grid(ax,kx,ky,kz,n_skip)
    % Produce grid:
    grids.x = linspace(-1,+1,2^kx + 1);
    grids.y = linspace(-1,+1,2^ky +1);
    grids.z = linspace(0,+1,2^kz + 1);
    
    disp("Total nodes: " + num2str(2^(kx+ky+kz)))
    
    % Draw grid:
    ax.XTick = grids.x(1:1:end);
    ax.YTick = grids.y(1:1:end); ax.YTickLabel = grids.y(1:1:end);
    ax.ZTick = grids.z(1:1:end); ax.ZTickLabel = grids.z(1:1:end);
    grid on;

    ticks = get(gca,'Xtick');
    ticklabels = cellstr(num2str(ticks'));
    rng = mod(1:numel(ticklabels),n_skip) ~= 0;
    ticklabels(rng) = {''};
    set(gca,'XTickLabel',ticklabels);

    ticks = get(gca,'Ytick');
    ticklabels = cellstr(num2str(ticks'));
    rng = mod(1:numel(ticklabels),n_skip) ~= 0;
    ticklabels(rng) = {''};
    set(gca,'YTickLabel',ticklabels);
   
end

function [] = plot_increase_size(sfx,sfy)
    set(gcf,'Position',get(gcf,'Position').*[1 1 sfx sfy]);
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

    % Min dim:
    dx_min = min(dx);
    dy_min = min(dy);

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