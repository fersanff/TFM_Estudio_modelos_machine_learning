function [w,b] = svm_hardmargin_dual(x,y)
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
cvx_begin
    variables lambda(n);
    variables expr;
    maximize(sum(lambda)-  0.5 * quad_form(lambda, Q));
    subject to
        lambda >= 0;
        lambda' * y == 0;
cvx_end
sv_ind = lambda > tol;
w = x' * (lambda .* y);
b = mean(y(sv_ind)-x(sv_ind,:) * w);
end