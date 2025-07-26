function [b,lambda,Fvals] = svm_kernel_sm(X,y,kernel,C,sigma,d,Xtest)
% Modelo svm con kernels, por defecto se utiliza el margen suave
% Entrada:
% X: Datos de entrada
% y: etiquetas de los datos de entrada
% c: parametro de relajacion
% Salida:
% w la pendiente o vector de pesos del hiperplano
% b el sesgo o la distancia al origen de la recta
[num,~] = size(X);
K = zeros(num);
kernel_f = kernel_fun(kernel,sigma,d);
for i=1:num
    for j=i:num
        K(i,j)=kernel_f(X(i,:),X(j,:));
        K(j,i)=K(i,j);
    end
end
Q = (y*y').*K;
cvx_begin quiet
variable lambda(num);
maximize(sum(lambda)-lambda' * Q * lambda/ 2);
subject to
lambda >= 0;
lambda <= C;
y' * lambda == 0;
cvx_end

ind = lambda > 1e-4 & lambda < C-1e-4;
w = K' * (lambda .* y);
b = mean(y(ind)-K(ind,:) * (lambda.* y));

%Calculo frontera
Fvals = zeros(size(Xtest,1), 1);
for i = 1:size(Xtest,1)
    % Evaluar el kernel entre el punto de la malla y cada dato de entrenamiento
    K_vals = zeros(size(X,1),1);
    for j = 1:size(X,1)
        K_vals(j) = kernel_f(X(j,:), Xtest(i,:));
    end
    % Función de decisión: suma de (lambda .* y .* kernel) + b
    Fvals(i) = sum(lambda .* y .* K_vals) + b;
end

end