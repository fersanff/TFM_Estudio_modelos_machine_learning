function [w,b,lambda,f_vals] = svm(X,y,kernel,C,sigma,d,Xtest)
% Modelo implementado para el svm, en este modelo se aplicara por defecto un modelo svm
% de margen suave, se usaran tambien las formulaciones duales del problema.
% Entradas:
% X: Datos de entrada.
% y: Las etiquetas de los datos de entrada.
% kernel: Tipo de kernel que se utilizara para el problema.
% C: Parametro de relajacion.
% sigma: Parametro opcional en el caso de que se use un kernel especifico
% d: Parametro opcional en el caso de que se use un kernel especifico
% Xtest: Conjunto de puntos donde evaluar la funcion
% Salidas:
% w: pendiente del hiperplano de separacion
% b: sesgo del hiperplano de separacion
w = 0;
switch lower(kernel)
    case "linear"
        [w,b,lambda,f_vals] = svm_softmargin_dual(X,y,C,Xtest);
    case "rbf"
        if isempty(sigma)
            error("Para kernel 'rbf', el parámetro 'sigma' es obligatorio.");
        else
            [b,lambda,f_vals]=svm_kernel_sm(X,y,kernel,C,sigma,[],Xtest);
        end
    case "gaussian"
        if isempty(sigma)
            error("Para kernel 'gaussian', el parámetro 'sigma' es obligatorio.");
        else
            [b,lambda,f_vals]=svm_kernel_sm(X,y,kernel,C,sigma,[],Xtest);
        end
    case "poly"
        if isempty(d)
            error("Para kernel 'poly', el parámetro 'd' es obligatorio.");
        else
            [b,lambda,f_vals]=svm_kernel_sm(X,y,kernel,C,[],d,Xtest);
        end
    otherwise
        error("Tipo de kernel no reconocido. Usa 'linear', 'rbf', 'gaussian' o 'poly'.");
end

end