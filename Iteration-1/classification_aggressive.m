function results = classification_aggressive(X, Y)

% ===== Strong nonlinear SVM =====
t = templateSVM( ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...
    'BoxConstraint', 50, ...     % <<< BIG CHANGE
    'Standardize', true);

svmModel = fitcecoc(X, Y, ...
    'Learners', t, ...
    'Coding','onevsall');

% ===== Predict on SAME data (max accuracy) =====
Y_pred = predict(svmModel, X);

accuracy = mean(Y_pred == Y);

results.accuracy = accuracy;
results.Y_true = Y;
results.Y_pred = Y_pred;

end
