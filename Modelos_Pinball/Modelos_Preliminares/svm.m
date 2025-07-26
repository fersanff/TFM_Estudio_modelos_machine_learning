function [w,b,delta] = svm(X,y,kernel,tau,C,sigma,d,X_test,y_test)
% Modelo implementado para el svm, en este modelo se aplicara por defecto un modelo svm
% de margen suave, se usaran tambien las formulaciones duales del problema.
% Entradas:
% X: Datos de entrada.
% y: Las etiquetas de los datos de entrada.
% kernel: Tipo de kernel que se utilizara para el problema.
% C: Parametro de relajacion.
% sigma: Parametro opcional en el caso de que se use un kernel especifico
% d: Parametro opcional en el caso de que se use un kernel especifico
% Salidas:
% w: pendiente del hiperplano de separacion
% b: sesgo del hiperplano de separacion
w = 0;
switch lower(kernel)
    case "linear"
        [w,b,~] = svm_soft_margin(X,y,kernel,tau,C,[],[],X_test,y_test);
    case "rbf"
        if isempty(sigma)
            error("Para kernel 'rbf', el parámetro 'sigma' es obligatorio.");
        else
            [w,b,delta]=svm_soft_margin(X,y,kernel,tau,C,sigma,[],X_test,y_test);
        end
    case "gaussian"
        if isempty(sigma)
            error("Para kernel 'gaussian', el parámetro 'sigma' es obligatorio.");
        else
            [w,b,delta]=svm_soft_margin(X,y,kernel,tau,C,sigma,[],X_test,y_test);
        end
    case "poly"
        if isempty(d)
            error("Para kernel 'poly', el parámetro 'd' es obligatorio.");
        else
            [w,b,delta]=svm_soft_margin(X,y,kernel,tau,C,[],d,X_test,y_test);
        end
    otherwise
        error("Tipo de kernel no reconocido. Usa 'linear', 'rbf', 'gaussian' o 'poly'.");
end

end