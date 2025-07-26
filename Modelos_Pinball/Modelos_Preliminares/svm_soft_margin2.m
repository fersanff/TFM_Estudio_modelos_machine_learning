function [w,b,Fvals,pred] = svm_soft_margin2(X,y,kernel,tau,C,sigma,d,Xtest)
% Modelo SVM Pinball (dual) con kernels
% Salidas:
%   w   : vector de peso primal (2×1 si lineal, [] si no-lineal)
%   b   : sesgo escalar
%   F   : decision values f(x_i) para cada muestra de X (m×1)
%   acc : accuracy = mean(sign(F)==y)

[m,~] = size(X);
% 1) Matriz de kernel
K = zeros(m);
kf = kernel_fun(kernel,sigma,d);
for i = 1:m
    for j = i:m
        K(i,j) = kf(X(i,:), X(j,:));
        K(j,i) = K(i,j);
    end
end

% 2) Resolvemos el dual con CVX
Q = (y*y').*K;
cvx_begin quiet
variables lambda(m) mu_var(m)
maximize( sum(lambda) - 0.5*lambda'*Q*lambda )
subject to
y' * lambda == 0;
-tau*C <= lambda;
C >= lambda;
cvx_end

% 3) w primal (sólo si lineal)
if strcmpi(kernel,'linear')
    w = X' * (lambda .* y);   % 2×1
else
    w = [];                  % no lo usamos en el espacio original
end

% 5) Cálculo de b vía KKT y promediado sobre soportes
tol = 1e-6;
alfa = lambda-mu_var;
sv  = find( (alfa>tol) | (mu_var>tol) );  % vector |sv|×1
b_vals = - (lambda .* y)' * K(:, sv);    % 1×|S| con b_j^*
b = mean(b_vals(:));
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
pred = sign(Fvals);
end
