%  availabe in environment: mXtrain  mytrain
load('othSpamTrain.mat');
X = mXtrain;
y = mytrain;
C = 0.1;
model = svmTrain(X, y, C, @linearKernel);
p = svmPredict(model, X);
fprintf('Training Accuracy: %f\n', mean(double(p == y)) * 100);
% availabe in environment: mXtest  mytest
load('othSpamTest.mat');
fprintf('\nEvaluating the trained Linear SVM on a test set ...\n')
p = svmPredict(model, mXtest);
fprintf('Test Accuracy: %f\n', mean(double(p == mytest)) * 100);




filename = 'spamSample.txt';
% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);
fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');



