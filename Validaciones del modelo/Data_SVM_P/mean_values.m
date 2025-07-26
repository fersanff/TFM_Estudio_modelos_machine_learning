% Nivel de confianza deseado (por ejemplo 0.95 para el 95%):
confLevel = 0.95;

data = AUCMATRIX(:);
%% === Cálculo de la media y el intervalo de confianza ===
n = numel(data);                % Tamaño de la muestra
mu_hat = mean(data);            % Media muestral
s = std(data);                  % Desviación estándar muestral (por defecto, normalización n-1)
se = s / sqrt(n);               % Error estándar de la media

alpha = 1 - confLevel;          
df = n - 1;                     % Grados de libertad para la t de Student
tcrit = tinv(1 - alpha/2, df);  % Valor crítico de t para el intervalo bilateral

% Intervalo de confianza al nivel especificado:
ci_lower = mu_hat - tcrit * se;
ci_upper = mu_hat + tcrit * se;

%% === Salida de resultados ===
fprintf('Número de observaciones: %d\n', n);
fprintf('Media muestral: %.4f\n', mu_hat);
fprintf('Desviación estándar muestral: %.4f\n', s);
fprintf('Error estándar de la media: %.4f\n', se);
fprintf('Nivel de confianza: %.1f%%\n', confLevel*100);
fprintf("Intervalo de confianza %.4f\n",tcrit * se)
fprintf('Intervalo de confianza: [%.4f, %.4f]\n', ci_lower, ci_upper);