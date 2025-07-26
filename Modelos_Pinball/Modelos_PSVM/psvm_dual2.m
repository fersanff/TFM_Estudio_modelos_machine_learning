function [w, b, prob, pred] = psvm_dual2(X, y, kerneltype, C, epsilon,d,sigma,tau,X_test)
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
kernel_f = kernel_fun(kerneltype,sigma,d);
K = zeros(m);
for i = 1:m
    for j = i:m
        K(i,j) = kernel_f(X(i,:), X(j,:));
        K(j,i) = K(i,j);
    end
end

% 3) Resolver el dual con CVX
% m = length(y);
% K, y, tau, zeta, epsilon ya definidos

cvx_begin quiet
variables lambda_var(m) beta_var(m) mu_var(m) gamma_var(m)
dual variable b
% Objetivo
maximize( ...
    sum( 0.5*(lambda_var-mu_var).*(y + epsilon) - gamma_var ) ...
    - 0.5*quad_form( (lambda_var-mu_var).*y + beta_var - gamma_var, K ) );
% Restricciones
subject to
% Balance dual
b: sum( (lambda_var-mu_var).*y + beta_var - gamma_var ) == 0;
% Caja coupling = C
lambda_var == (C/epsilon)-(mu_var/tau);
% No negatividad
beta_var   >= 0;
gamma_var  >= 0;
mu_var >= 0;
lambda_var >= 0;
cvx_end
% 4) Recuperar b y coeficientes combinados
alfa = lambda_var-mu_var;
coef = alfa.*y + beta_var - gamma_var;
b = -b;
% 5) Calcular w solo si el kernel es lineal
if strcmpi(kerneltype,'linear')
    w = X' * coef;
else
    w = [];
end
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
