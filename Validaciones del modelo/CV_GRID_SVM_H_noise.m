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

for dset = 1:size(datasets,1)
    dataset_file = datasets{dset, 1};
    dataset_name = datasets{dset, 2};
    load(dataset_file);
    noiseLevel = 0.05;
    sigma_noise = sqrt(noiseLevel);
    noise  = sigma_noise .* randn(size(X));
    X = X + noise; %Ruido Gaussiano añadido a el dataset
    [m, n] = size(X);
    fInd = 1:n;
    CV = 10;
    fold_size = floor(m / CV);

    Cl = -7; Ch = 7;
    kerneltypes = ["linear","rbf","gaussian","poly"];
    sigma = 0.75;
    d = 2;

    AUCMATRIX = zeros(Ch-Cl+1, 1);
    ACCUMATRIX = zeros(Ch-Cl+1, 1);

    for kerneltype = kerneltypes
        for j = Cl:Ch
            C = 2^j;
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

                [~, ~, ~, Prediction] = svm_dual_quadprog_hinge(Xtr, Ytr, kerneltype, C, sigma, d, Xt);
                [AUC(k), Accu(k)] = medi_auc_accu(Prediction, Yt);
            end
            AUCMATRIX(j-Cl+1) = mean(AUC);
            ACCUMATRIX(j-Cl+1) = mean(Accu);
        end

        file1 = sprintf("Data_SVM_H_noise\\ACCUMATRIX_%s_%s.mat", kerneltype, dataset_name);
        file2 = sprintf("Data_SVM_H_noise\\AUCMATRIX_%s_%s.mat", kerneltype, dataset_name);
        save(file1, 'ACCUMATRIX');
        save(file2, 'AUCMATRIX');
    end
end
