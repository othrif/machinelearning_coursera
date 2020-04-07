printf("quick tests...\n")

printf("load data\n")
data = load('ex1data2.txt');
X=data(:,1:2);
y=data(:,3);
m = length(y);

[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];

alpha = 1*3;
num_iters = 400;

theta = zeros(size(X,2), 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);


figure;
%theta = zeros(size(X,2), 1);
%[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
%plot(1:numel(J_history), J_history, '-g', 'LineWidth', 2);

labels = {};
colororder = get (gca, "colororder");
numAlphas = 7;
for i=1:numAlphas
    alpha = alpha/3;
    fprintf("Plot alpha=%.3f\n", alpha)
    theta = zeros(size(X,2), 1);
    [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
    h = plot(1:numel(J_history), J_history, plotColors( plotColorsI,:), 'LineWidth', 2, sprintf(";alpha=%d;",alpha)); hold on;
    set (h, "color", colororder(i,:));
    labels = {labels{:}, ["alpha ", num2str(i)]};
end
hold off;
title('Convergence of the cost function J(theta) for different learning rates alpha');
xlabel('Number of iterations');
ylabel('Cost J');
