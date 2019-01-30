function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

temp = exp((-1).*z);
g = 1./(1 + temp);





% =============================================================

end
