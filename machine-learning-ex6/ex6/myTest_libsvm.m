% Use of libsvm library

%% Initialization
clear ; close all; clc
addpath('libsvm-3.24/matlab/')


%% =============== Part 1: Simple test  ================

% Loading data
fprintf('Loading and Visualizing Data ...\n')
[trainlabels, trainfeatures] = libsvmread('exDataforlibsvm/twofeature.txt');

% Plot training data
plotData(trainfeatures, trainlabels);

% Training Linear SVM
fprintf('\nTraining Linear SVM ...\n')

% Try changing -c value below and see how the decision boundary changes
model = svmtrain(trainlabels, trainfeatures, '-s 0 -t 0 -c 100');
visualizeBoundaryLinear_libsvm(trainfeatures, trainlabels, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =============== Part 2: Train spam classifier ================

acc_training = []

% Train with 50 examples
fprintf('Loading Data with 50 training emails...\n')
[trainlabels, trainfeatures] = libsvmread('exDataforlibsvm/email_train-50.txt');
model = svmtrain(trainlabels, trainfeatures, '-s 0 -t 0 -c 1');
[testlabels, testfeatures] = libsvmread('exDataforlibsvm/email_test.txt');
[predicted_label, accuracy, decision_values]= svmpredict(testlabels, testfeatures, model);
fprintf('Accuracy with 50 training emails = %.2f%%\n', accuracy(1));
acc_training = [acc_training; accuracy(1)];
% Train with 100 examples
fprintf('Loading Data with 100 training emails...\n')
[trainlabels, trainfeatures] = libsvmread('exDataforlibsvm/email_train-100.txt');
model = svmtrain(trainlabels, trainfeatures, '-s 0 -t 0 -c 1');
[testlabels, testfeatures] = libsvmread('exDataforlibsvm/email_test.txt');
[predicted_label, accuracy, decision_values]= svmpredict(testlabels, testfeatures, model);
fprintf('Accuracy with 100 training emails = %.2f%%\n', accuracy(1));
acc_training = [acc_training; accuracy(1)];
% Train with 400 examples
fprintf('Loading Data with 400 training emails...\n')
[trainlabels, trainfeatures] = libsvmread('exDataforlibsvm/email_train-400.txt');
model = svmtrain(trainlabels, trainfeatures, '-s 0 -t 0 -c 1');
[testlabels, testfeatures] = libsvmread('exDataforlibsvm/email_test.txt');
[predicted_label, accuracy, decision_values]= svmpredict(testlabels, testfeatures, model);
fprintf('Accuracy with 400 training emails = %.2f%%\n', accuracy(1));
acc_training = [acc_training; accuracy(1)];
% Train with all examples
fprintf('Loading Data with all training emails...\n')
[trainlabels, trainfeatures] = libsvmread('exDataforlibsvm/email_train-all.txt');
model = svmtrain(trainlabels, trainfeatures, '-s 0 -t 0 -c 1');
[testlabels, testfeatures] = libsvmread('exDataforlibsvm/email_test.txt');
[predicted_label, accuracy, decision_values]= svmpredict(testlabels, testfeatures, model);
fprintf('Accuracy with all training emails = %.2f%%\n', accuracy(1));
acc_training = [acc_training; accuracy(1)];

plot(acc_training);
hold on;
fprintf('Program paused. Press enter to continue.\n');
pause;
