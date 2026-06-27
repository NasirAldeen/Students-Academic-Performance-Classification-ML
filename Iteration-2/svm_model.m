function results = svm_model(X, Y)

model = fitcsvm(X, Y, ...
    'KernelFunction','rbf', ...
    'KernelScale','auto', ...
    'BoxConstraint',50, ...
    'Standardize',true);

pred = predict(model, X);
acc = mean(pred == Y);

figure;
confusionchart(Y, pred);
title('SVM Confusion Matrix');

results.accuracy = acc;
end
