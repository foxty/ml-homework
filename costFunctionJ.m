function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing our training exapmles
% y is the class labels

m = size(X, 1);         % number of training exapmles
predictions = X*theta;   % predictions of hypothesis on all m
sqrErrors = (predictions-y).^2;

J = 1/(2*m) * sum(sqrErrors);