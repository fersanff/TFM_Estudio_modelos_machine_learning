function [param, b, prob, pred] = psvm_dual_quadprog_hinge(X, y, kerneltype, C, epsilon,d,sigma,X_test)
% PSVM dual hinge vía CVX
% INPUTS:
%   X          : m×n matriz de entrenamiento
%   y          : m×1 vector de etiquetas (+1 / −1)
%   kerneltype : 'linear','poly','rbf','gaussian'
%   C          : parámetro de penalización
%   epsilon    : ε > 0
%   Xtest      : t×n matriz de datos para test
% OUTPUTS:
%   w    : n×1 vector de pesos (vacío si kernel no es lineal)
%   b    : sesgo
%   prob : t×1 scores crudos   (w⊤x + b ó suma kernel + b)
%   pred : t×1 predicciones ±1

[m, ~] = size(X);
e=ones(m,1);

kernel_f = kernel_fun(kerneltype,d,sigma);
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
Q = Q + 1e-8*eye(3*m);  % pequeño término para estabilidad
f=[-0.5*y-0.5*epsilon*e; zeros(m,1); e];
Ae=[y', e', -e'];
Cu=[(C/epsilon)*e;Inf*e;Inf*e];

[sol,~,~,~,b_var]= quadprog(Q,f,[],[],Ae,0,zeros(3*m,1),Cu);

lambda = sol(1:m);
mu_var  = sol(m+1:2*m);
alfa = sol(2*m+1:3*m);
b     = b_var.eqlin;
coef = y .* lambda + mu_var - alfa;
% 5) Calcular w solo si el kernel es lineal
if strcmpi(kerneltype,'linear')
    w = X' * coef;
else
    w = [];
end
param = [w;b];
param = param(:);
% 6) Calcular scores sobre X_test
if ~isempty(w)
    % lineal: f = prob*w + b
    prob=X_test*w + b - 0.5;
else
    prob = zeros(size(X_test,1), 1);
    for i = 1:size(X_test,1)
        % Evaluar el kernel entre el punto de la malla y cada dato de entrenamiento
        K_vals = zeros(size(X,1),1);
        for j = 1:size(X,1)
            K_vals(j) = kernel_f(X(j,:), X_test(i,:));
        end
        % Función de decisión: suma de (lambda .* y .* kernel) + b
        prob(i) = sum(coef.*K_vals) + b-0.5;
    end
end

% 7) Umbral y predicción final
pred = sign(prob);

end
