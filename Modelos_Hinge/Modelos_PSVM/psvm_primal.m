function [w,b,prob,pred,primVal] = psvm_primal(X, y, C, epsilon)
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

cvx_begin
    variables w(d) b xi(n)
    minimize( 0.5 * sum_square(w) + C * sum(xi) / epsilon )
    subject to
        % condición de separación suave PSVM
        y .* (X*w + b - 0.5) >= 0.5*epsilon - xi;
        % restricciones de banda [0, 1]
        X*w + b >= 0;
        X*w + b <= 1;
        % holguras no negativas
        xi >= 0;
cvx_end
primVal = cvx_optval;
prob=X*w + b;
pred=sign(prob-0.5);
end
