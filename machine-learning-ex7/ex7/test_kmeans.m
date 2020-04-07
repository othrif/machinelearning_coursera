clear ; close all; clc

load('ex7data2.mat')
K = 3; % 3 Centroids
initial_centroids = [3 3; 6 2; 8 5];
centroids = initial_centroids
max_iters = 10;
idx = zeros(size(X,1));
for i=1:max_iters
    idx = findClosestCentroids(X, centroids);
    centroids = computeCentroids(X, idx, K);
end
