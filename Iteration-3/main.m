clc;
clear;
close all;

data = readtable('G3.csv');

[X, Y] = preprocessing(data);

idx = clustering(X);

results = classification(X, Y);

evaluation(results);

disp('FINAL ACCURACIES:');
disp(['Decision Tree: ', num2str(results.DecisionTree.accuracy)]);
disp(['SVM: ', num2str(results.SVM.accuracy)]);
disp(['KNN: ', num2str(results.KNN.accuracy)]);
nn_results = neural_network(X, Y);

disp(['Neural Network Accuracy: ', num2str(nn_results.accuracy)]);
