clc; clear; close all;

% =========================
% Load data
% =========================
data = readtable('G3.csv');   % adjust if needed

% =========================
% Preprocessing
% =========================
[X, Y] = preprocessing(data);

% =========================
% Feature Selection (Aggressive)
% =========================
[X_fs, fs_idx] = feature_selection(X, Y, 14);  % keep most features


% =========================
% CLUSTERING
% =========================
clustering_kmeans(X_fs, Y);

% =========================
% SVM Classification (Aggressive)
% =========================
svm_results = classification_aggressive(X_fs, Y);
disp('SVM Accuracy (Aggressive):');
disp(svm_results.accuracy);

figure;
confusionchart(svm_results.Y_true, svm_results.Y_pred);
title('SVM Confusion Matrix (Aggressive)');

% =========================
% NEURAL NETWORK (Aggressive)
% =========================
nn_results = neural_network_aggressive(X_fs, Y);
disp('Neural Network Accuracy (Aggressive):');
disp(nn_results.accuracy);

figure;
confusionchart(nn_results.Y_true, nn_results.Y_pred);
title('Neural Network Confusion Matrix (Aggressive)');
