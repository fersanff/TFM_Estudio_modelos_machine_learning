function [param, b, prob, pred] = psvm_dual_quadprog_pinball(X, y, kerneltype, C, epsilon,d,sigma,tau,X_test)
% PSVM dual vía CVX
% INPUTS:
%   X          : m×n matriz de entrenamiento
%   y          : m×1 vector de etiquetas (+1 / −1)
%   Xtest      : t×n matriz de datos para test
%   kerneltype : 'linear','poly','rbf','gaussian'
%   C          : parámetro de penalización
%   epsilon    : ε > 0
% OUTPUTS:
%   w    : n×1 vector de pesos (vacío si kernel no es lineal)
%   b    : sesgo
%   prob : t×1 scores crudos   (w⊤x + b ó suma kernel + b)
%   pred : t×1 predicciones ±1

[m, ~] = size(X);
e=ones(m,1);
kernel_f = kernel_fun(kerneltype,sigma,d);
K = zeros(m);
for i = 1:m
    for j = i:m
        K(i,j) = kernel_f(X(i,:), X(j,:));
        K(j,i) = K(i,j);
    end
end
K1 = K .* (y * y');    % elemento a elemento, K1(i,j)=y_i y_j K(i,j)
K2 = K .* y;           % cada columna j multiplicada por y_j
K3 = K * diag(y);      % cada fila i multiplicada por y_i

Q = [ K1,   K2,  -K2;
      K3,    K,   -K;
     -K3,  -K,    K ];
Q = (Q + Q')/2;         % asegurar simetría numérica
%Se cambia el signo porque se quiere maximizar la función y quadprog solo
%minimiza
f = [ -0.5*(y + epsilon);    % coeficiente para α
       zeros(m,1);           % coeficiente para β
       +e ];         % coeficiente para γ  (−(−γ)=+γ)
%   Aeq * x = beq  impone  sum_i (y_i α_i + β_i − γ_i) = 0
Aeq = [ y',  e',  -e' ];  

% cotas:  -τC/ε ≤ α ≤ C/ε,   β ≥ 0,  γ ≥ 0 
lb = [ (-tau*C/epsilon)*e; %Esta restriccion corresponde a alfa
        zeros(m,1); %Esta restriccion corresponde a beta
        zeros(m,1) ]; %Esta restriccion corresponde a gamma
ub = [  +    (C/epsilon)*e;
        Inf*e;
        Inf*e ];
[x,~,~,~,mult] = quadprog(Q, f, [], [], Aeq, 0, lb, ub, []);
alfa = x(1:m);
beta_var  = x(m+1:2*m);
gamma_var = x(2*m+1:3*m);
b     = mult.eqlin;     % el multiplicador de Aeq·x=0
% 4) Recuperar b y coeficientes combinados
coef = alfa.*y + beta_var - gamma_var;
% 5) Calcular w solo si el kernel es lineal
if strcmpi(kerneltype,'linear')
    w = X' * coef;
else
    w = [];
end
param = [w;b];
param = param(:);
% 6) Calcular scores sobre X
if ~isempty(w)
    % lineal: f = X*w + b
    prob = X_test * w + b - 0.5;
else
    prob = zeros(size(X_test,1), 1);
    for i = 1:size(X_test,1)
        % Evaluar el kernel entre el punto de la malla y cada dato de entrenamiento
        K_vals = zeros(size(X,1),1);
        for j = 1:size(X,1)
            K_vals(j) = kernel_f(X(j,:), X_test(i,:));
        end
        % Función de decisión.
        prob(i) = sum(coef.*K_vals) + b - 0.5;
    end
end

% 7) Umbral y predicción final
pred = sign(prob);
end
