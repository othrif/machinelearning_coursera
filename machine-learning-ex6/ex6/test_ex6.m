clear ; close all; clc
load('ex6data3.mat');
plotData(X, y);
load('ex6data3.mat');
[C, sigma] = dataset3Params(X, y, Xval, yval);
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
visualizeBoundary(X, y, model);
pause;