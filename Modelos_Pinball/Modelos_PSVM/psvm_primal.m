function [w,b,prob,pred] = psvm_primal(X, y_label, C, epsilon, tau)
% PSVM primal con kernel lineal (estilo CVX similar a svm_softmargin_dual)
% INPUT:
%   x       : n×d matriz de datos
%   y       : n×1 vector de etiquetas (+1 / –1)
%   C       : parámetro de penalización
%   epsilon : parámetro ε
% OUTPUT:
%   w  : d×1 vector de pesos
%   b  : sesgo
%   xi : variables de holgura (n×1)
[n, d] = size(X);

cvx_begin quiet
    variables w(d) b xi(n)
    minimize( 0.5 * sum_square(w) + C * sum(xi) / epsilon )
    subject to
        y_label .* (X*w + b - 0.5) >= 0.5*epsilon - xi;
        y_label .* (X*w + b - 0.5) <= 0.5*epsilon + (1/tau).*xi;
        X*w + b >= 0;
        X*w + b <= 1;
        xi >= 0;
cvx_end
prob = X*w + b- 0.5;
pred = sign(prob);
end
