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
%% Defino las rutas a los proyectos
addpath(genpath('../dataset_bin'))
addpath(genpath('../Modelos_Pinball/Modelos_PSVM'));
addpath(genpath('../Modelos_Pinball/Modelos_Preliminares'));
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
tau = 0.5; %La mediana
kernel="linear";
%% Malla para visualizar fronteras
margin   = 1;
x1range  = linspace(min(X(:,1))-margin, max(X(:,1))+margin, 200);
x2range  = linspace(min(X(:,2))-margin, max(X(:,2))+margin, 200);
[X1, X2] = meshgrid(x1range, x2range);
gridPts  = [X1(:), X2(:)];

%% 1) SVM Pinball (varios kernels)
kernelTypes = {"linear","rbf","gaussian","poly"};
titles      = {'Dual Lineal','Dual RBF','Dual Gauss','Dual Poli (d=2)'};
figure('Name','PSVM Dual — Varios Kernels');
for k = 1:4
    kt = kernelTypes{k};

    % resolvemos PSVM dual (siempre devuelve w y b)
    [~,~,Fvals,~] = svm_soft_margin_quadprog(X, y, kt, tau, C, sigma, d,gridPts);
    [~,~,Fscore,pred] = svm_soft_margin_quadprog(X, y, kt, tau, C, sigma, d,X);
    
    % Frontera de decision
    F = reshape(Fvals, size(X1));
    % Calculo del score
    acc = mean(pred == y);
    % pintamos
    subplot(2,2,k);
    gscatter(X(:,1), X(:,2), y, 'rb','xo'); hold on;
    contour(X1, X2, F,    [0 0],   'k-',  'LineWidth',2);
    title(titles{k});
    axis equal; grid on;
    legend('+1','-1','Frontera','Márgenes');
    fprintf("Acurracy del modelo %f\n",acc);
end
