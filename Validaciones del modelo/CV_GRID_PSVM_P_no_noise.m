% Cross Validation PSVM (linear kernel)
% Validación cruzada para el metodo PSVM
%
%% Carga BBDD
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
    fInd = [ 1:n ];%<--- return all features
    CV=10; % validacion cruzada cantidad de folds
    fold_size = floor(m / CV); %Tamaño a recorrer en cada fold
    taus = [1,0.5,0.2,0.1];
    Clt=1; %Cota inferior para tau, Paper Shao et al usa -8
    Cht=length(taus);  % Cota superior para tau, Paper Shao et al usa 8

    Cl=-7; %Cota inferior para C, Paper Shao et al usa -8
    Ch=7;  % Cota superior para C, Paper Shao et al usa 8

    Ceps_l=-7; %Cota inferior para eps, Paper Shao et al usa -8
    Ceps_h=0; %Cota superior para eps, Paper Shao et al usa 0

    AUCMATRIX=zeros(Ceps_h-Ceps_l+1,Ch-Cl+1,Cht-Clt+1);
    ACCUMATRIX=zeros(Ceps_h-Ceps_l+1,Ch-Cl+1,Cht-Clt+1);
    kerneltypes = ["linear","rbf","gaussian","poly"];
    sigma = 0.75;
    d=2;
    for kerneltype = kerneltypes
        for i=Ceps_l:Ceps_h
            Epsi = 2^i;
            for j=Cl:Ch
                C = 2^j;
                for p = Clt:Cht
                    tau = taus(p);
                    for k=1:CV
                        start_idx = (k-1)*fold_size + 1;
                        if k == CV
                            end_idx = m;
                        else
                            end_idx = k*fold_size;
                        end
                        tst=perm(k:fold_size:m);
                        trn=setdiff(1:m,tst);
                        Ytr=Y(trn,:);
                        Xtr=X(trn,fInd);
                        Yt=Y(tst',:);
                        Xt=X(tst',fInd);
                        [~,~,~,Prediction] = psvm_dual_quadprog_pinball(Xtr, Ytr, kerneltype,C,Epsi, d,sigma,tau,Xt);
                        [AUC(k),Accu(k)]=medi_auc_accu(Prediction,Yt);
                    end
                    AUCMATRIX(i-Ceps_l+1,j-Cl+1,p-Clt+1)=mean(AUC);
                    ACCUMATRIX(i-Ceps_l+1,j-Cl+1,p-Clt+1)=mean(Accu);
                end
            end
        end
        file1 = sprintf("Data_PSVM_P\\ACCUMATRIX_%s_%s.mat",kerneltype, dataset_name);
        file2 = sprintf("Data_PSVM_P\\AUCMATRIX_%s_%s.mat",kerneltype, dataset_name);
        save(file1, 'ACCUMATRIX');
        save(file2, 'AUCMATRIX');
    end
end
