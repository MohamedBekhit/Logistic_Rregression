function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

h_theta = sigmoid(X * theta);

    for ind=1:m
        J = J + ((-1)*(y(ind))*(log(h_theta(ind))) - (1 - y(ind)) * log(1 - h_theta(ind)));
    end
J = (1/m)*J;

% log_h_theta = log(h_theta);
% log_1_h_theta = log(1 - h_theta);
% temp = (-1) .* y .* log_h_theta - (1 - y) .* log_1_h_theta;
% J = (1/m) * sum(temp);

n = length(theta); 
    for j = 1:n
        for i = 1:m
            grad(j, 1) = grad(j, 1) + (h_theta(i) - y(i)) * X(i, j);
        end
    end
grad = (1/m) * grad;

end
