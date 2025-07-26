clc
clear
%% Ajustes globales de fuente y marcadores  -----------------------------
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
%%Defino los datos para el plot
%Datos separables
rng(42)
n = 30;
X1 = 2 * rand(n, 2);
X2 = 2 * rand(n, 2) + 3;
X_sep = [X1; X2];
y_sep = [ones(n,1); -ones(n,1)];
figure(1)
gscatter(X_sep(:,1), X_sep(:,2), y_sep, 'rb', 'xo');
axis equal;
grid on;

%Datos no separables
n = 5;
X2_mc = 2 * rand(n, 2);
X1_mc = 2 * rand(n, 2) + 3;
X_mc = [X1_mc; X2_mc];
X = [X_sep ; X_mc];
y = [y_sep;ones(n,1); -ones(n,1)];
figure(2)
gscatter(X(:,1), X(:,2), y, 'rb', 'xo');
hold on;
plot(X_mc(:,1), X_mc(:,2), 'mo', 'MarkerSize', 12, 'LineWidth', 1.5)
axis equal;
grid on;
legend("C1","C2","Datos mal clasificados","Frontera de decision","Margenes")

%% Modelos SVM
% Parámetros
C = 1.5;
sigma = 1;
d = 2;
%Defino los datos para hacer el mallado
% Crear una malla de puntos para evaluar la función de decisión
x1Range = linspace(min(X_sep(:,1))-1, max(X_sep(:,1))+1, 100);
x2Range = linspace(min(X_sep(:,2))-1, max(X_sep(:,2))+1, 100);
[X1, X2] = meshgrid(x1Range, x2Range);
gridPts_sep = [X1(:), X2(:)];

[~, w,f_vals,~] = svm_dual_quadprog_hinge(X_sep, y_sep, "linear", C, sigma, d,gridPts_sep);
F = reshape(f_vals, size(X1));
margin  = norm(w);
%Separables
figure(3)
gscatter(X_sep(:,1), X_sep(:,2), y_sep, 'rb','xo'); hold on
contour(X1, X2, F,  [0 0],     'k-',  'LineWidth',2)   % Frontera
contour(X1, X2, F,  [-0.23 -0.23],'k--', 'LineWidth',1)  % Márgen –
contour(X1, X2, F,  [0.29 0.29], 'k--', 'LineWidth',1)  % Márgen +
title('Modelo SVM Hard Margin')
axis equal, grid on
legend({'C1','C2','Frontera','Márgen −','Márgen +'}, 'Location','best')

x1Range = linspace(min(X(:,1))-1, max(X(:,1))+1, 100);
x2Range = linspace(min(X(:,2))-1, max(X(:,2))+1, 100);
[X1, X2] = meshgrid(x1Range, x2Range);
gridPts = [X1(:), X2(:)];
[w_pts, b_pts,f_vals,~] = svm_dual_quadprog_hinge(X, y, "linear", C, sigma, d,gridPts);
F = reshape(f_vals, size(X1));
%Separables
figure(4)
gscatter(X(:,1), X(:,2), y, 'rb','xo'); hold on
contour(X1, X2, F,  [0 0],     'k-',  'LineWidth',2)
contour(X1, X2, F,  [-0.5 -0.5],'k--', 'LineWidth',1)
contour(X1, X2, F,  [0.5  0.5], 'k--', 'LineWidth',1)
title('Modelo SVM Soft Margin')
axis equal, grid on
legend({'C1','C2','Frontera','Márgen −','Márgen +'}, 'Location','best')
%% 2) SVM (varios kernels)
kernelTypes = {"linear","rbf","gaussian","poly"};
titles      = {'Dual Lineal','Dual RBF','Dual Gauss','Dual Poli (d=2)'};
figure('Name','PSVM Dual — Varios Kernels');
for k = 1:4
    kt = kernelTypes{k};
    % resolvemos PSVM dual (siempre devuelve w y b)
    [w_pts, b_pts,f_vals,~] = svm_dual_quadprog_hinge(X, y, kt, C, sigma, d,gridPts);
    [w_score, b_score,~,pred] = svm_dual_quadprog_hinge(X, y, kt, C, sigma, d,X);

    % Frontera de decision
    F = reshape(f_vals, size(X1));
    % Calculo del score
    acc = mean(pred == y);

    % pintamos
    subplot(2,2,k);
    gscatter(X(:,1), X(:,2), y, 'rb','xo'); hold on;
    contour(X1, X2, F,    [0 0],   'k-',  'LineWidth',2);
    contour(X1, X2, F,   [-0.5 -0.5],'k--','LineWidth',1);
    contour(X1, X2, F,   [ 0.5  0.5],'k--','LineWidth',1);
    title(titles{k});
    axis equal; grid on;
    legend('+1','-1','Frontera','Márgenes');
    fprintf("Acurracy del modelo %f\n",acc);

end
