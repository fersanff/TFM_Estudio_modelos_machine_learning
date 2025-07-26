function kernel = kernel_fun(kerneltype, sigma,d)
% Funcion para seleccionar el tipo de kernel con parámetros opcionales
% INPUT:
%   kerneltype: Tipo de kernel ('linear','poly','rbf','gaussian')
%   Parámetros opcionales (Name,Value):
%     'd'     : grado del polinómico (default = 2)
%     'sigma' : ancho de banda para rbf/gaussian (default = 1)
% OUTPUT:
%   kernel: Función anónima K(x1,x2) correspondiente al kernel
% --- 2) Selección del kernel ---
switch kerneltype
    case "linear"
        % K(x,y) = x·y'
        kernel = @(x1, x2) x1 * x2';

    case "poly"
        % K(x,y) = (x·y')^d
        kernel = @(x1, x2) (x1 * x2').^d;

    case "rbf"
        % K(x,y) = exp(-||x - y||^2 / sigma^2)
        kernel = @(x1, x2) exp(-norm(x1 - x2)^2 / (sigma^2));

    case "gaussian"
        % K(x,y) = (1/(sqrt(2π)*σ)) * exp(-||x-y||^2/(2σ^2))
        kernel = @(x1, x2) (1./(sqrt(2*pi)*sigma)) * ...
            exp(-norm(x1 - x2)^2./(2*sigma^2));

    otherwise
        error("Tipo de kernel no reconocido. Usa 'linear','poly','rbf' o 'gaussian'.");
end

end
