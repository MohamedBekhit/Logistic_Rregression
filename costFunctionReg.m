function [J_reg, grad_reg] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta, 1);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
grad_reg = zeros(size(theta));

h_theta = sigmoid(X * theta);

%COST FUNCTION
for ind=1:m
    J = J + ((-1)*(y(ind))*(log(h_theta(ind))) - (1 - y(ind)) * log(1 - h_theta(ind)));
end
J = (1/m)*J;
reg = 0;
for ind = 2:n
    reg = reg + (theta(ind))^2;
end
J_reg = J + reg * lambda / (2*m);


%GRADIENT

for j = 1:n
    for i = 1:m
        grad(j, 1) = grad(j, 1) + (h_theta(i) - y(i)) * X(i, j);
    end
end
grad = (1/m) * grad;
grad_reg(1, 1) = grad(1, 1);
for j = 2:n
    grad_reg(j, 1) = grad(j, 1) + lambda * theta(j) / m;
end

end
