clc
clear all
addpath(genpath('../dataset_bin'))
addpath(genpath('../Modelos_Pinball/Modelos_PSVM'));
addpath(genpath('../Modelos_Pinball/Modelos_Preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_preliminares'));
addpath(genpath('../Modelos_Hinge/Modelos_PSVM'));

datasets = {
    'sonar.mat', 'sonar';
    'breastcancer.mat', 'breastcancer';
    'ionosphereN.mat', 'ionosphere'
    };

dataset_file = datasets{1, 1};
dataset_name = datasets{1, 2};
disp(dataset_file)
load(dataset_file);

%Se añade ruido
noiseLevel = 0.05;
sigma_noise = sqrt(noiseLevel);
noise  = sigma_noise .* randn(size(X));
X = X + noise; %Ruido Gaussiano añadido a el dataset
[m, n] = size(X);
fInd = 1:n;
CV = 10;
fold_size = floor(m / CV);

kerneltypes = ["linear","rbf","gaussian","poly"];
sigma = 0.75;
d = 2;
C = 2^(1);
kerneltype = 'poly';
tau = 0.2;
for k = 1:CV
    start_idx = (k-1)*fold_size + 1;
    if k == CV
        end_idx = m;
    else
        end_idx = k*fold_size;
    end
    tst = perm(k:fold_size:m);
    trn = setdiff(1:m, tst);
    Ytr = Y(trn,:);
    Xtr = X(trn, fInd);
    Yt = Y(tst',:);
    Xt = X(tst', fInd);

    [~, ~, ~, Prediction] = svm_soft_margin2(Xtr, Ytr, kerneltype, tau ,C, sigma, d, Xt);
    [AUC(k), Accu(k)] = medi_auc_accu(Prediction, Yt);
end
fprintf("AUC %d Accuracy %d \n",mean(AUC),mean(Accu));



