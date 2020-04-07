clear ; close all; clc
A = double(imread('bird_small.png')); A = A / 255;
K = 16;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
size(X);
initial_centroids = kMeansInitCentroids(X, K);
max_iters = 3;
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);
centroids
