clear ; close all; clc


load('ex3data1.mat');
m = size(X, 1);
rand_indices = randperm(m);
%sel = X(rand_indices(1:100),:);
%sel = X(1:100,:);
%displayData(sel);
%pause;

%theta_t = [-2; -1; 1; 2];
%X_t = [ones(5,1) reshape(1:15,5,3)/10]
%y_t = ([1;0;1;0;1] >= 0.5);
%lambda_t = 3;
%[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);
%J
%grad

num_labels = 10;
lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
pred = predictOneVsAll(all_theta, X);