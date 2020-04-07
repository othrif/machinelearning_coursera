clear ; close all; clc

load ('ex5data1.mat');
m = size(X, 1);
p = 8;

X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);
X_poly = [ones(m, 1), X_poly];

X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

lambda = 3
[theta] = trainLinearReg(X_poly, y, lambda);
[J_train, gradtmp] = linearRegCostFunction(X_poly, y, theta, 0);
[J_val, gradtmp] = linearRegCostFunction(X_poly_val, yval, theta, 0);
[J_test, gradtmp] = linearRegCostFunction(X_poly_test, ytest, theta, 0);

J_train
J_val
J_test
