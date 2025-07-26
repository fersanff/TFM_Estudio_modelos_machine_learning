function [w, b] = svm_hard(data, labels)
    % INPUT
    % data: num-by-dim matrix. num is the number of data points,
    % dim is the dimension of a point
    % labels: num-by-1 vector, specifying the class that each point
    % belongs to.
    % either be +1 or be -1
    % OUTPUT
    % w: dim-by-1 vector, the normal direction of hyperplane
    % b: a scalar, the bias
    
    [num, dim] = size(data);
    cvx_begin
        variables w(dim) b;
        minimize(norm(w));
        subject to
        labels .* (data * w + b) >= 1;
    cvx_end
end