printf("quick tests...\n")

printf("load data\n")
data = load('ex1data1.txt');
x=data(:,1);
y=data(:,2);
m=length(y);
printf("training sample is: %d\n", m)
X = [ones(m, 1), data(:,1)];
theta = zeros(2, 1);
iterations = 1500;
alpha = 0.01;


fprintf('\nTesting the cost function ...\n')
% compute and display initial cost
J = computeCost(X, y, theta);
fprintf('With theta = [0 ; 0]\nCost computed = %f\n', J);
fprintf('Expected cost value (approx) 32.07\n');


plotData(X(:,2), y);
hold on;
theta = gradientDescent(X, y, theta, alpha, iterations);
plot(X(:,2), X*theta, 'g-')


