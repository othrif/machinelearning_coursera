clear ; close all; clc
load ('ex7faces.mat');
figure;
displayData(X(1:100, :));
[X_norm, mu, sigma] = featureNormalize(X);
[U, S] = pca(X_norm);
figure;
displayData(U(:, 1:36)');
K = 100;
Z = projectData(X_norm, U, K);
X_rec  = recoverData(Z, U, K);

subplot(1, 2, 1);
displayData(X_norm(1:10,:));
title('Original faces');
axis square;
subplot(1, 2, 2);
displayData(X_rec(1:10,:));
title('Recovered faces');
axis square;
pause;