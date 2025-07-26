function Fvals = evalDecision(X,y,kt,kernel,w, b,lambda)
    m    = size(X,1);
    Fvals = zeros(m,1);
    if strcmpi(kt,'linear')
        % lineal: simple producto matricial
        Fvals = X * w + b;
    else
        % no lineal: sumamos lambda .* y .* K_vals
        for ii = 1:m
            % punto a evaluar
            xj     = X(ii,:);
            K_vals = zeros(m,1);
            for jj = 1:m
                K_vals(jj) = kernel(X(jj,:), xj);
            end
            Fvals(ii) = sum(lambda .* y .* K_vals) + b;
        end
    end
end