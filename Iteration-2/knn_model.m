function results = knn_model(X, Y)

model = fitcknn(X, Y, ...
    'NumNeighbors',3, ...
    'Distance','euclidean', ...
    'Standardize',true);

pred = predict(model, X);
acc = mean(pred == Y);

figure;
confusionchart(Y, pred);
title('KNN Confusion Matrix');

results.accuracy = acc;
end
