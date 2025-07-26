clc
clear
addpath(genpath('../dataset_bin'))
addpath(genpath('../Modelos_Pinball/Modelos_PSVM'));
addpath(genpath('../Modelos_Pinball/Modelos_Preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_PSVM'));

%% 1) Carga datos
datasets     = {'sonar.mat','sonar'; ...
    'breastcancer.mat','breastcancer'; ...
    'ionosphereN.mat','ionosphere'};
dataset_file = datasets{1,1};   % selecciona breastcancer
dataset_name = datasets{1,2};
load(dataset_file);
[m, n] = size(X);
fInd    = 1:n;
CV      = 10;
fold_size = floor(m/CV);

%Hiperparametros
kerneltype  = 'linear';
C           = 2^4;
epsilon     = 2^-1;
d           = 2;
sigma       = 0.75;

k=1;
tst = perm(k:fold_size:m);
trn = setdiff(1:m, tst);
Xtr = X(trn, fInd);   Ytr = Y(trn);
Xt  = X(tst',fInd);   Yt  = Y(tst');
%% 1) Modelo base: entrena en (Xtr,Ytr) y obtiene scores sobre Xt
[param_u, ~, ~, ~] = psvm_dual_quadprog_hinge(...
    Xtr, Ytr, kerneltype, C, epsilon, d, sigma, Xt);

T     = 100;
noiseLevel  = 0.05;
sigma_noise = sqrt(noiseLevel);
norm_u = norm(param_u,2);
rob=0;
rng(2025, 'twister');
for i = 1:T
    % 2) Perturbo solo los datos de entrenamiento
    noise = sigma_noise .* randn(size(X));
    X_noise = X + noise;
    k = 1;
    tst = perm(k:fold_size:m);
    trn = setdiff(1:m, tst);
    Xtr = X_noise(trn, fInd);   Ytr = Y(trn);
    Xt  = X_noise(tst',fInd);   Yt  = Y(tst');

    % 3) Entreno sobre Xtr_n e infiero sobre el mismo Xt
[param_i, ~, ~, ~] = psvm_dual_quadprog_hinge(...
        Xtr, Ytr, kerneltype, C, epsilon, d, sigma, Xt);
    norm_u_i=norm( param_i - param_u, 2 );
    rob = norm_u_i/norm_u+rob;
end

%% 4) Métrica de robustez sobre los scores de la frontera
Nrob = rob/T;
fprintf('Robustez de frontera (test fold %d): %.4f\n', k, Nrob);
