function results = decision_tree_model(X, Y)

model = fitctree(X, Y, ...
    'MaxNumSplits',100, ...
    'MinLeafSize',1);

pred = predict(model, X);
acc = mean(pred == Y);

figure;
confusionchart(Y, pred);
title('Decision Tree Confusion Matrix');

results.accuracy = acc;
end
