% Visualización de PSVM primal (lineal) y dual con varios kernels
clear; clc; close all;

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
% Defino las rutas a los proyectos
addpath(genpath('../dataset_bin'))
addpath(genpath('../Modelos_Hinge/Modelos_preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_PSVM'));

%% Datos no separables 
rng(42)
n = 30;
X1 = 2 * rand(n, 2);         
X2 = 2 * rand(n, 2) + 3;
X_sep = [X1; X2];
y_sep = [ones(n,1); -ones(n,1)];
n = 5;
X2_mc = 2 * rand(n, 2);         
X1_mc = 2 * rand(n, 2) + 3;
X_mc = [X1_mc; X2_mc];
X = [X_sep ; X_mc];
y = [y_sep;ones(n,1); -ones(n,1)];
figure(1)
gscatter(X(:,1), X(:,2), y, 'rb', 'xo');
axis equal;
grid on;
legend("C1","C2","Datos mal clasificados","Frontera de decision","Margenes")
%% Parámetros comunes
C       = 1;
epsilon = 0.1;
sigma   = 1;    % para RBF y Gaussian
d       = 2;    % grado para polinómico
%% Malla para visualizar fronteras
margin   = 1;
x1range  = linspace(min(X(:,1))-margin, max(X(:,1))+margin, 200);
x2range  = linspace(min(X(:,2))-margin, max(X(:,2))+margin, 200);
[X1, X2] = meshgrid(x1range, x2range);
gridPts  = [X1(:), X2(:)];

%% 1) PSVM Primal (solo lineal)
[w_lp, b_lp, ~, ~,optval_p] = psvm_primal(X, y, C, epsilon);
% f(x) = w'*x + b - 0.5
Fp = reshape(gridPts*w_lp + b_lp - 0.5, size(X1));

figure('Name','PSVM Primal (Lineal)');
gscatter(X(:,1), X(:,2), y, 'rb', 'xo'); hold on;
contour(X1, X2, Fp, [0 0], 'k-', 'LineWidth',2);        % frontera
contour(X1, X2, Fp, [-0.5 -0.5], 'k--','LineWidth',1);   % margen inferior
contour(X1, X2, Fp, [0.5 0.5], 'k--','LineWidth',1);     % margen superior
title('PSVM Primal — Kernel Lineal');
legend('+1','-1','Frontera','Márgenes');
axis equal; grid on;


%% 2) PSVM (varios kernels)
kernelTypes = {"linear","rbf","gaussian","poly"};
titles      = {'Dual Lineal','Dual RBF','Dual Gauss','Dual Poli (d=2)'};
figure('Name','PSVM Dual — Varios Kernels');
for k = 1:4
    kt = kernelTypes{k};
    % resolvemos PSVM dual (siempre devuelve w y b)
    [w_pts, b_pts,f_vals,pred_vals] = psvm_dual_quadprog_hinge(X, y, kt, C, epsilon,d, sigma,gridPts);
    [w_score, b_score,f_score,pred_score] = psvm_dual_quadprog_hinge(X, y, kt,C,epsilon, d, sigma,X);
    
    % Frontera de decision
    F = reshape(f_vals, size(X1));
    % Calculo del score
    y_pred = sign(f_score);
    acc = mean(y_pred == y);

    % pintamos
    subplot(2,2,k);
    gscatter(X(:,1), X(:,2), y, 'rb','xo'); hold on;
    contour(X1, X2, F,    [0 0],   'k-',  'LineWidth',2);
    title(titles{k});
    axis equal; grid on;
    legend('+1','-1','Frontera')
    fprintf("Acurracy del modelo %f\n",acc);
end
