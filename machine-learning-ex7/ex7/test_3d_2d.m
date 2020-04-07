clear ; close all; clc
A = double(imread('bird_small.png'));
A = A / 255;
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 16;
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, 1);
sel = floor(rand(1000, 1) * size(X, 1)) + 1;
size(X, 1)
size(sel)
palette = hsv(K);
colors = palette(idx(sel), :);

figure;
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');

[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
Z = projectData(X_norm, U, 2);

figure;
plotDataPoints(Z(sel, :), idx(sel), K);
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');
pause;