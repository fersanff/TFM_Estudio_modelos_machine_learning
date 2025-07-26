function [w,b] = svm_soft_margin(x,y,C)
%Funcion del SVM con margen suave en su forma primal
%Entrada:
% x dato en dimension R^N
% y etiqueta o clase a la que pertenece cada dato
% C Parametro de relajacion
% Salida:
% w la pendiente o vector de pesos del hiperplano
% b el sesgo o la distancia al origen de la recta
[n,m] = size(x);
cvx_begin
    variables w(m) xi(n) b;
    minimize( 0.5 * norm(w) + C*sum(xi.^2)); %Equivalente a ||w||^2 
    subject to
        y.*(x*w+b)>=1-xi;
        xi >= 0;
cvx_end
end