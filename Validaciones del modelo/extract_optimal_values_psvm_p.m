clc
clear
folder = 'Data_PSVM_P'; % Cambia esta ruta por la carpeta que quieras procesar
files = dir(fullfile(folder, '*.mat')); % Lista todos los .mat

for i = 1:length(files)
    filepath = fullfile(folder, files(i).name);
    data = load(filepath); % Carga la estructura con variables

    % Suponemos que hay ACCUMATRIX o AUCMATRIX dentro
    fields = fieldnames(data);
    matrix = data.(fields{1}); % Carga la primera variable que encuentra

    % Aplanamos la matriz a un vector para buscar el máximo
    values = matrix(:);

    % Obtener valor máximo y su índice lineal
    [max_val, lin_idx] = max(values);

    % Detectar dimensiones de la matriz y convertir índice lineal
    d = ndims(matrix);
    sz = size(matrix);
    switch d
        case 2
            [row, col] = ind2sub(sz, lin_idx);
            pos_str = sprintf('fila %d, columna %d', row, col);
        case 3
            [row, col, page] = ind2sub(sz, lin_idx);
            pos_str = sprintf('fila %d, columna %d, capa %d', row, col, page);
        otherwise
            % Para matrices de más de 3 dimensiones:
            subs = cell(1, d);
            [subs{:}] = ind2sub(sz, lin_idx);
            coords = sprintf('%d,', cell2mat(subs));
            coords(end) = [];  % quitar la última coma
            pos_str = sprintf('índices [%s]', coords);
    end

    % Calcular métricas adicionales
    mean_val = mean(values);
    std_val  = std(values);
    n        = numel(values);

    % Intervalo de confianza del 95%
    z       = 1.96;               % para 95%
    sem     = std_val / sqrt(n);  % error estándar de la media
    ci_low  = mean_val - z * sem;
    ci_high = mean_val + z * sem;

    % Mostrar resultados
    fprintf('Archivo: %s\n', files(i).name);
    fprintf('  Máximo: %.4f (%s)\n', max_val, pos_str);
    fprintf('  Promedio: %.4f\n', mean_val);
    fprintf('  IC 95%%: [%.4f, %.4f]\n\n', ci_low, ci_high);
end
