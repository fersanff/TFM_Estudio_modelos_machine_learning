% Cross Validation PSVM (linear kernel)
% Validación cruzada para el metodo PSVM
%
%% Carga BBDD
clc
clear all
addpath(genpath('../dataset_bin'))
addpath(genpath('../Modelos_Pinball/Modelos_PSVM'));
addpath(genpath('../Modelos_Pinball/Modelos_Preliminares'));
%load 'exa_bmpm.mat'
%load ejm2_prob200.mat
%load ejm_wang1.mat
%load Ejm_one_dim.mat
%load Peces_ju_anch.mat
% load Peces_ju_sard.mat
% load Peces_anch_sard.mat

%load 'sonar.mat';
%load 'breastcancer.mat';
load('ionosphereN.mat');

%load('heart_statlogN.mat');
%load('bupa_liverN.mat');
%load('ionosphereN.mat');
%load('breastcancer')
%load('australian.mat');
%load('diabetes.mat');
%load('german_credit.mat');
%load('splice.mat');
%load x18data.mat; % Flare-M
%load x23data.mat; % Yeast4
%load yeast3.mat
%load('titanic.mat');
%load segment0_n.mat
%load('image_n.mat');
%load('waveformBin.mat');
%load 'phoneme.mat'
%load('ring_n.mat')
%%
%X=norm(X);
[m n]=size(X);
fInd = [ 1:n ];%<--- return all features

tau = 0.75;
C = 2^1;
Epsi = 0.01;

kerneltypes = [ "linear","rbf","gaussian","poly" ];
kerneltype = kerneltypes(2);
sigma=0.75;
d=2;

%Añado ruido
noiseLevel  = 0.05;
sigma_noise = sqrt(noiseLevel);
noise = sigma_noise .* randn(size(X));
X_noise = X + noise;

rng(0);  % para reproducibilidad
permIdx = randperm(m);
splitPoint = floor(0.8 * m);

trainIdx = permIdx(1:splitPoint);
testIdx  = permIdx(splitPoint+1:end);

Xtr = X(trainIdx, fInd);
Ytr = Y(trainIdx);

Xt  = X(testIdx, fInd);
Yt  = Y(testIdx);
C = 2^-7;
[~,~,~,Prediction] = psvm_dual_quadprog_hinge(Xtr, Ytr, kerneltype,C,Epsi, d,sigma,Xt);
[AUC,Accu]=medi_auc_accu(Prediction,Yt);
fprintf("AUC %f, Accuracy %f\n",AUC,Accu);
