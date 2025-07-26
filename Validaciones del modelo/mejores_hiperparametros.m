clc
clear

folder = 'Data_PSVM_H';

% Listar archivos de ACCU y AUC
filesACC = dir(fullfile(folder, 'ACCUMATRIX_*'));
filesAUC = dir(fullfile(folder, 'AUCMATRIX_*'));

% Rango de exponentes para C: 2^(Ceps_l) ... 2^(Ceps_h)
Ceps_l = -7;
Ceps_h =  0;

% Estructura para almacenar los mejores resultados por base
bestMetrics = struct();

%% Procesar ACCURACY
for i = 1:numel(filesACC)
    fname = filesACC(i).name;
    filepath = fullfile(folder, fname);
    data = load(filepath);
    
    % Obtener la matriz 3D: (epsilon × C × tau)
    flds = fieldnames(data);
    M3 = data.(flds{1});
    
    % Colapsar dimensiones de epsilon y tau: quedar con máximo sobre ambas
    M1 = max(M3, [], 1);        % reduce epsilon → 1 × C × tau
    M2 = squeeze(max(M1, [], 3));  % reduce tau → 1 × C, luego squeeze → 1×C vector
    
    % Encontrar máximo de Accuracy sobre C
    [accMax, jCol] = max(M2);
    
    % Reconstruir hiperparámetro C
    Cexp = Ceps_l + (jCol - 1);
    C    = 2^Cexp;
    
    % Extraer base de datos y kernel desde el nombre de archivo:
    %   "ACCUMATRIX_<kernel>_<base>.mat"
    parts  = split(fname, '_');
    base   = erase(parts{3}, '.mat');
    kernel = parts{2};
    
    % Inicializar si es la primera vez que aparece esta base
    if ~isfield(bestMetrics, base)
        bestMetrics.(base).bestAUC.value = -inf;
        bestMetrics.(base).bestACC.value = -inf;
    end
    
    % Actualizar si este Accuracy es mejor
    if accMax > bestMetrics.(base).bestACC.value
        bestMetrics.(base).bestACC.value  = accMax;
        bestMetrics.(base).bestACC.kernel = kernel;
        bestMetrics.(base).bestACC.C      = C;
    end
end

%% Procesar AUC
for i = 1:numel(filesAUC)
    fname = filesAUC(i).name;
    filepath = fullfile(folder, fname);
    data = load(filepath);
    
    % Obtener la matriz 3D: (epsilon × C × tau)
    flds = fieldnames(data);
    M3 = data.(flds{1});
    
    % Colapsar dimensiones de epsilon y tau: quedar con máximo sobre ambas
    M1 = max(M3, [], 1);           % reduce epsilon → 1 × C × tau
    M2 = squeeze(max(M1, [], 3));  % reduce tau → 1 × C, luego squeeze → 1×C vector
    
    % Encontrar máximo de AUC sobre C
    [aucMax, jCol] = max(M2);
    
    % Reconstruir hiperparámetro C
    Cexp = Ceps_l + (jCol - 1);
    C    = 2^Cexp;
    
    % Extraer base de datos y kernel desde el nombre de archivo:
    %   "AUCMATRIX_<kernel>_<base>.mat"
    parts  = split(fname, '_');
    base   = erase(parts{3}, '.mat');
    kernel = parts{2};
    
    % Inicializar si es la primera vez que aparece esta base
    if ~isfield(bestMetrics, base)
        bestMetrics.(base).bestAUC.value = -inf;
        bestMetrics.(base).bestACC.value = -inf;
    end
    
    % Actualizar si este AUC es mejor
    if aucMax > bestMetrics.(base).bestAUC.value
        bestMetrics.(base).bestAUC.value  = aucMax;
        bestMetrics.(base).bestAUC.kernel = kernel;
        bestMetrics.(base).bestAUC.C      = C;
    end
end

%% Mostrar resumen final
bases = fieldnames(bestMetrics);
for k = 1:numel(bases)
    b = bases{k};
    info = bestMetrics.(b);
    fprintf('Base: %s\n', b);
    fprintf('  Mejor Accuracy: %.6f\n', info.bestACC.value);
    fprintf('    Kernel: %s\n', info.bestACC.kernel);
    fprintf('    C     : %g\n', info.bestACC.C);
    fprintf('  Mejor AUC     : %.6f\n', info.bestAUC.value);
    fprintf('    Kernel: %s\n', info.bestAUC.kernel);
    fprintf('    C     : %g\n\n', info.bestAUC.C);
end
