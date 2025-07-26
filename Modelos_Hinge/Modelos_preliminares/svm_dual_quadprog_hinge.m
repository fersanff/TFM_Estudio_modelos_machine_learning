function [param,w,Fvals,pred] = svm_dual_quadprog_hinge(X,y,kernel,C,sigma,d,Xtest)
% Modelo SVM Hinge (dual) con kernels
% Salidas:
%   w   : vector de peso primal (2×1 si lineal, [] si no-lineal)
%   b   : sesgo escalar
%   F   : decision values f(x_i) para cada muestra de X (m×1)
%   acc : accuracy = mean(sign(F)==y)
[m,~] = size(X);
e=ones(m,1);

% 1) Matriz de kernel
K = zeros(m);
kf = kernel_fun(kernel,sigma,d);
for i = 1:m
    for j = i:m
        K(i,j) = kf(X(i,:), X(j,:));
        K(j,i) = K(i,j);
    end
end

% 2) Q (m×m) y f (m×1)
Q = (y*y').*K;
Q = (Q + Q')/2 + 1e-8*eye(m);   % simetría y estabilidad
f = -ones(m,1);

% 3) Igualdad y cotas
Aeq = y'; beq = 0;
lb  = 0;
ub  = +C    * e;

[lambda,~,~,~,~] = quadprog(Q, f, [], [], Aeq, beq, lb, ub, []);

% 5) Extraer b
tol = 1e-6;
sv = find(lambda > tol);          % soportes donde lambda_i>0
b_vals = - ( (lambda .* y)' * K(:, sv) );  % 1×|sv|
b = mean(b_vals);

% 3) w primal (sólo si lineal)
if strcmpi(kernel,'linear')
    w = X' * (lambda .* y);   % 2×1
else
    w = [];                  % no lo usamos en el espacio original
end
param = [w;b];
param = param(:);
%Calculo la frontera
if ~isempty(w)
    Fvals = Xtest * w + b;
else
    Fvals = zeros(size(Xtest,1), 1);
    for i = 1:size(Xtest,1)
        % Evaluar el kernel entre el punto de la malla y cada dato de entrenamiento
        K_vals = zeros(size(X,1),1);
        for j = 1:size(X,1)
            K_vals(j) = kf(X(j,:), Xtest(i,:));
        end
        % Función de decisión: suma de (lambda .* y .* kernel) + b
        Fvals(i) = sum(lambda .* y .* K_vals) + b;
    end
end
% Calculo del score
pred = sign(Fvals);
end
