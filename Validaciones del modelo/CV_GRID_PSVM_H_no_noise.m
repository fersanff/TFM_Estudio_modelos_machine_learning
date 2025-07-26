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

    [m n]=size(X);
    fInd = 1:n;
    CV=10;
    fold_size = floor(m / CV);

    Cl=-7; Ch=7;
    Ceps_l=-7; Ceps_h=0;

    kerneltypes = ["linear","rbf","gaussian","poly"];
    sigma = 0.75;
    d=2;

    AUCMATRIX = zeros(Ceps_h-Ceps_l+1, Ch-Cl+1);
    ACCUMATRIX = zeros(Ceps_h-Ceps_l+1, Ch-Cl+1);

    for kerneltype = kerneltypes
        for i = Ceps_l:Ceps_h
            Epsi = 2^i;
            for j = Cl:Ch
                C = 2^j;
                for k=1:CV
                    start_idx = (k-1)*fold_size + 1;
                    if k == CV
                        end_idx = m;
                    else
                        end_idx = k*fold_size;
                    end
                    tst = perm(k:fold_size:m);
                    trn = setdiff(1:m, tst);
                    Ytr = Y(trn,:);
                    Xtr = X(trn,fInd);
                    Yt = Y(tst',:);
                    Xt = X(tst',fInd);
                    [~, ~, ~, Prediction] = psvm_dual_quadprog_hinge(Xtr, Ytr, kerneltype, C, Epsi, sigma, d, Xt);
                    [AUC(k), Accu(k)] = medi_auc_accu(Prediction, Yt);
                end
                AUCMATRIX(i-Ceps_l+1, j-Cl+1) = mean(AUC);
                ACCUMATRIX(i-Ceps_l+1, j-Cl+1) = mean(Accu);
            end
        end

        file1 = sprintf("Data_PSVM_H\\ACCUMATRIX_%s_%s.mat",kerneltype, dataset_name);
        file2 = sprintf("Data_PSVM_H\\AUCMATRIX_%s_%s.mat",kerneltype, dataset_name);
        save(file1, 'ACCUMATRIX');
        save(file2, 'AUCMATRIX');
    end
end
