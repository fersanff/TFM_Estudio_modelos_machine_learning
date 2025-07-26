clear all; close all; clc;

set(groot, ...
    'defaultAxesFontSize',      12, ...          % = 12 pt de tu doc
    'defaultTextFontSize',      12, ...
    'defaultLegendFontSize',    12, ...
    'defaultAxesFontName',      'Times New Roman', ...
    'defaultTextFontName',      'Times New Roman', ...
    'defaultLegendFontName',    'Times New Roman', ...
    'defaultAxesTickLabelInterpreter', 'latex', ...  % usa $...$
    'defaultTextInterpreter',          'latex', ...
    'defaultLegendInterpreter',        'latex', ...
    'defaultLineMarkerSize',     8);   % círculos/estrellas bien visibles
% Cargo los scripts de cada modelo
addpath(genpath('../dataset_bin'))
addpath(genpath('../Modelos_Pinball/Modelos_PSVM'));
addpath(genpath('../Modelos_Pinball/Modelos_Preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_PSVM'));

rng(1); % Para reproducibilidad

%% 1. PARÁMETROS

% Generación de datos
n_puntos_clase = 50;
media_clase1 = [-2, -2];
media_clase2 = [2, 2];
cov_matrix = [1, 0.5; 0.5, 1];

% Parametros de los modelos
C = 1;
epsilon = 0.1;
tau = 0.5;
kerneltype = 'linear';
sigma = 1.0;
d = 2;

% Parámetros de ruido y visualización
% Nivel 0 para el caso sin ruido, seguido de 4 realizaciones con la misma varianza.
noise_levels = [0, 0.05, 0.05, 0.05, 0.05];
% Todos los hiperplanos serán negros, pero con diferente estilo de línea.
styles = {'-', '--', ':', '-.', '--'}; % Estilos de línea para distinguir
legend_entries = cell(1, length(noise_levels));
% Nombres en las leyendas
legend_entries{1} = 'Sin Ruido';
for i=2:length(noise_levels)
    legend_entries{i} = 'Ruido';
end

%% 2. GENERACIÓN DE DATOS SINTÉTICOS ORIGINALES
X_orig = [mvnrnd(media_clase1, cov_matrix, n_puntos_clase);
          mvnrnd(media_clase2, cov_matrix, n_puntos_clase)];
y = [-ones(n_puntos_clase, 1); ones(n_puntos_clase, 1)];

%% 3. PREPARACIÓN DE LA MALLA PARA VISUALIZACIÓN
padding = 1.5;
grid_points = 100;
x1_range = linspace(min(X_orig(:,1))-padding, max(X_orig(:,1))+padding, grid_points);
x2_range = linspace(min(X_orig(:,2))-padding, max(X_orig(:,2))+padding, grid_points);
[X1_grid, X2_grid] = meshgrid(x1_range, x2_range);
X_test = [X1_grid(:), X2_grid(:)];

%% 4. CREACIÓN DE LA FIGURA Y BUCLE DE ANÁLISIS

fig = figure('Name', 'Estabilidad de Modelos Frente a Ruido', 'Position', [100, 100, 1200, 1000]);

% --- Preparación de los subplots ---
model_names = {sprintf('SVM Hinge (C=%.1f)',C), ...
               sprintf('SVM Pinball (tau=%.1f, C=%.1f)', tau,C), ...
               sprintf('PSVM Hinge (eps=%.1f,C=%.1f)', epsilon,C), ...
               sprintf('PSVM Pinball (eps=%.1f, tau=%.1f,C=%.1f)', epsilon, tau,C)};
axes_handles = gobjects(4, 1);
line_handles = gobjects(length(noise_levels), 4); % Filas para ruido, Columnas para modelo

for k = 1:4 % Bucle para crear los 4 subplots
    axes_handles(k) = subplot(2, 2, k);
    gscatter(axes_handles(k), X_orig(:,1), X_orig(:,2), y, [0.6 0.6 1; 1 0.6 0.6], 'o+'); % Colores más suaves
    hold(axes_handles(k), 'on');
    title(axes_handles(k), model_names{k});
    xlabel(axes_handles(k), 'Característica 1');
    ylabel(axes_handles(k), 'Característica 2');
end

% --- Bucle principal para añadir las fronteras ---
for i = 1:length(noise_levels)
    current_noise = noise_levels(i);
    
    % Generar datos con el ruido actual. Crucial: randn() se llama cada
    % vez, generando una NUEVA realización de ruido.
    if current_noise > 0
        noise_matrix = sqrt(current_noise) * randn(size(X_orig));
        X = X_orig + noise_matrix;
    else
        X = X_orig; % Caso base sin ruido
    end
    
    fprintf('--- Entrenando para la realización %d (Ruido: %.2f) ---\n', i, current_noise);
    
    % --- Modelo 1: SVM Hinge ---
    [~, ~, Fvals, ~] = svm_dual_quadprog_hinge(X, y, kerneltype, C, sigma, d, X_test);
    [~, line_handles(i, 1)] = contour(axes_handles(1), X1_grid, X2_grid, reshape(Fvals, size(X1_grid)), [0 0], 'k', 'LineStyle', styles{i}, 'LineWidth', 1.5);
    
    % --- Modelo 2: SVM Pinball ---
    [~, ~, Fvals, ~] = svm_soft_margin_quadprog(X, y, kerneltype, tau, C, sigma, d, X_test);
    [~, line_handles(i, 2)] = contour(axes_handles(2), X1_grid, X2_grid, reshape(Fvals, size(X1_grid)), [0 0], 'k', 'LineStyle', styles{i}, 'LineWidth', 1.5);
    
    % --- Modelo 3: PSVM Hinge ---
    [~, ~, Probs, ~] = psvm_dual_quadprog_hinge(X, y, kerneltype, C, epsilon, d, sigma, X_test);
    [~, line_handles(i, 3)] = contour(axes_handles(3), X1_grid, X2_grid, reshape(Probs, size(X1_grid)), [0 0], 'k', 'LineStyle', styles{i}, 'LineWidth', 1.5);

    % --- Modelo 4: PSVM Pinball ---
    [~, ~, Probs, ~] = psvm_dual_quadprog_pinball(X, y, kerneltype, C, epsilon, d, sigma, tau, X_test);
    [~, line_handles(i, 4)] = contour(axes_handles(4), X1_grid, X2_grid, reshape(Probs, size(X1_grid)), [0 0], 'k', 'LineStyle', styles{i}, 'LineWidth', 1.5);
end

% --- Finalizar plots y añadir leyendas ---
for k = 1:4
    hold(axes_handles(k), 'off');
    axis(axes_handles(k), 'tight');
    grid(axes_handles(k), 'on');
end

% Añadir una leyenda común a toda la figura. Usamos los handles del primer plot.
lgd = legend(line_handles(:,1), legend_entries);
lgd = legend(line_handles(:,2), legend_entries);
lgd = legend(line_handles(:,3), legend_entries);
lgd = legend(line_handles(:,4), legend_entries);

disp('--- Análisis completado ---');