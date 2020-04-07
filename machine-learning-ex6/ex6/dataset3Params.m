function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;
search = false;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
if (search)
N = [0.01 0.03 0.1 0.3 1 3 10 30];
min_err = 999.;
for i_C =  1 : length(N)
    for i_sigma =  1 : length(N)
        tmp_C = N(i_C);
        tmp_sigma = N(i_sigma);
        model= svmTrain(X, y, tmp_C, @(x1, x2) gaussianKernel(x1, x2, tmp_sigma));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        %printf("C=%.2f, sigma=%.2f => err=%.4f\n", tmp_C, tmp_sigma, err);
        if (err < min_err)
            min_err = err;
            C = tmp_C ;
            sigma = tmp_sigma;
            printf(">>  C=%.2f, sigma=%.2f with error %.4f\n", C, sigma,min_err)
        endif
end
end
printf("Final C=%.2f, sigma=%.2f with error %.4f\n", C, sigma,min_err)
endif

% =========================================================================

