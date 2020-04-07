%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Othmane Rifki
%% Dec 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% My email classifier written in Matlab / Octave
%% Goal of this code is to return X, y, Xtest, ytest, Xcv, ycv that represent the training, testing and cross validation sets
%% These sets will be constructed by the following:
%% Read files containing a spam/non-spam email
%% Process each file to reduce the file content to uniform key terms (i.e. http://mywebsite.com > httpaddr)
%% Match each key term from the email to an index from the dictionary (i.e.  hello my name is  > 89 882 9 992)
%% Form feature vector with the number of elements equal to the dictionary, 1 when a term appears in the email and 0 otherwise
%%      (i.e.) X(i)  = [1 0 0 0 1 1 0 1 0 .... 0 0 0 1 0 1]
%% Repeat the above strategy to form a vector X with dimensions m x n where m is the number of examples and n the dimension of the feature vector
%% Split the data randomly to obtain training , cross validation (for optimizing parameters like C and sigma), and testing sets
%% Analyze the processed emails and create a dictionary with the most repeated emails

clear ; close all; clc

Xoth=[];
yoth=[];
%listOfFiles_nonspam = readFiles('spamcorpus/all_nonspam/');
listOfFiles_nonspam = readFiles('test-nonspam/');
fprintf('Number of non-spam emails processed: %d\n', length(listOfFiles_nonspam));
%listOfFiles_spam = readFiles('spamcorpus/all_spam/');
listOfFiles_spam = readFiles('test-spam/');
fprintf('Number of spam emails processed: %d\n', length(listOfFiles_spam));

for i=1:length(listOfFiles_nonspam)
    % Non-spam
    fprintf("\n\nNon-spam %d/%d: %s\n", i, length(listOfFiles_nonspam), listOfFiles_nonspam{i});
    file_contents_nonspam = readFile(listOfFiles_nonspam{i});
    word_indices_nonspam  = processEmail(file_contents_nonspam);
    features_nonspam      = emailFeatures(word_indices_nonspam);
    Xoth = [Xoth; features_nonspam'];
    yoth = [yoth; 0];
    % Print Stats
    fprintf('Word Indices: \n');
    fprintf(' %d', word_indices_nonspam);
    fprintf('\n\n');
    fprintf('Length of feature vector: %d\n', length(features_nonspam));
    fprintf('Number of non-zero entries: %d\n', sum(features_nonspam > 0));
end
for i=1:length(listOfFiles_spam)
    % Non-spam
    fprintf("\n\nSpam %d/%d: %s\n", i, length(listOfFiles_spam), listOfFiles_spam{i});
    file_contents_spam = readFile(listOfFiles_spam{i});
    word_indices_spam  = processEmail(file_contents_spam);
    features_spam      = emailFeatures(word_indices_spam);
    Xoth = [Xoth; features_spam'];
    yoth = [yoth; 1];
    % Print Stats
    fprintf('Word Indices: \n');
    fprintf(' %d', word_indices_spam);
    fprintf('\n\n');
    fprintf('Length of feature vector: %d\n', length(features_spam));
    fprintf('Number of non-zero entries: %d\n', sum(features_spam > 0));
end

%ratio_spam_nonspam = (listOfFiles_spam)/(listOfFiles_nonspam)
%ratio_train_other = 0.2
%num_train_nonspam = 1000
%num_test_nonspam = ratio_train_other * num_train_nonspam
%num_xval_nonspam = ratio_train_other * num_train_nonspam
%num_train_spam = ratio_spam_nonspam * num_train_nonspam
%num_test_spam =  ratio_spam_nonspam * num_test_nonspam
%num_xval_spam =  ratio_spam_nonspam * num_xval_nonspam


m = size(Xoth, 1);
% use fraction of the examples for training
perm = randperm(m);
train = perm(1:ceil(m*0.6));
cv = perm(ceil(m*0.6+1):ceil(m*0.8));
test = perm(ceil(m*0.8):end);
% use fraction for training and fraction for testing
mXtrain=Xoth(train, :);
mytrain=yoth(train,:);
mXcv=Xoth(test,:);
mycv=yoth(test,:);
mXtest=Xoth(test,:);
mytest=yoth(test,:);

save('othSpamTrain.mat', 'mXtrain', 'mytrain')
save('othSpamCV.mat', 'mXcv', 'mycv')
save('othSpamTest.mat', 'mXtest', 'mytest')

%load('spamTrain.mat');
%C = 0.1;
%model = svmTrain(Xoth, yoth, C, @linearKernel);
%p = svmPredict(model, Xoth)
%yoth
%fprintf('Test Accuracy: %f\n', mean(double(p == yoth)) * 100);


