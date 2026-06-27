clc; clear; close all;

% ==============================
% Load dataset
% ==============================
data = readtable('G3.csv');

% ==============================
% Preprocessing
% ==============================
[X, Y] = preprocessing(data);

% ==============================
% Machine Learning Models
% ==============================
svm_results = svm_model(X, Y);
knn_results = knn_model(X, Y);
dt_results  = decision_tree_model(X, Y);
nn_results  = neural_network_model(X, Y);

% ==============================
% Clustering (Unsupervised)
% ==============================
clustering_kmeans(X);

% ==============================
% Display final accuracies
% ==============================
fprintf('\nFINAL ACCURACIES:\n');
fprintf('SVM: %.4f\n', svm_results.accuracy);
fprintf('KNN: %.4f\n', knn_results.accuracy);
fprintf('Decision Tree: %.4f\n', dt_results.accuracy);
fprintf('Neural Network: %.4f\n', nn_results.accuracy);
