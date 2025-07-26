function [w,b] = svm_hard_margin(x,y)
%Funcion del SVM con margen duro en su forma primal
%Entrada:
% x dato en dimension R^N
% y etiqueta o clase a la que pertenece cada dato
% Salida:
% w la pendiente o vector de pesos del hiperplano
% b el sesgo o la distancia al origen de la recta
[n,m] = size(x);
cvx_begin
    variables w(m) b;
    minimize( 0.5 * norm(w) ); %Equivalente a ||w||^2 
    subject to
        y.*(x*w+b)>=1;
cvx_end
end