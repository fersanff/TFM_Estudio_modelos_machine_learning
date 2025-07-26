function [w,b,lambda,Fvals] = svm_softmargin_dual(x,y,C,Xtest)
%Funcion del SVM con margen duro en su forma primal
%Entrada:
% x dato en dimension R^N
% y etiqueta o clase a la que pertenece cada dato
% Salida:
% w la pendiente o vector de pesos del hiperplano
% b el sesgo o la distancia al origen de la recta
% lambda multiplicador de Lagrange
[n,~] = size(x);
tol = 1e-4;
Q = (x * x') .* (y * y');
cvx_begin quiet
variables lambda(n);
variables expr;
maximize(sum(lambda)- lambda' * Q * lambda / 2);
subject to
lambda >= 0;
C >= lambda;
lambda' * y == 0;
cvx_end
sv_ind = lambda > tol;
w = x' * (lambda .* y);
b = mean(y(sv_ind)-x(sv_ind,:) * w);
%Calculo la frontera

% w es 2×1, aplicamos f(x)=w^T x + b
Fvals = Xtest * w + b;       % (nG×2)*(2×1) + scalar

end