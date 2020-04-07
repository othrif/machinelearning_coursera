clear ; close all; clc

load ('ex5data1.mat');
m = size(X, 1);
lambda = 0.01

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];

m = size(X_poly, 1);
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
itr = 50
for i = 1:m
    error_train_tmp=0;
    error_val_tmp=0;
    for k = 1:itr
        sel = randperm(i);
        Xtrain = X_poly(sel, :);
        ytrain = y(sel);
        n = size(Xtrain, 1);
        nval = size(X_poly_val, 1);
        [theta] = trainLinearReg(Xtrain, ytrain, lambda);
        [error_train_tmp, gradtmp] = linearRegCostFunction(Xtrain, ytrain, theta, 0);
        [error_val_tmp, gradtmp] = linearRegCostFunction(X_poly_val, yval, theta, 0);
        error_train(i) = error_train(i) + 1/itr * error_train_tmp;
        error_val(i) = error_val(i) + 1/itr * error_val_tmp;
    end

end

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])


pause;