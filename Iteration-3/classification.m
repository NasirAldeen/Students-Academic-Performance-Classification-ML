function results = classification(X, Y)

    % Train/Test split
    cv = cvpartition(Y, 'HoldOut', 0.3);

    X_train = X(training(cv), :);
    Y_train = Y(training(cv));

    X_test  = X(test(cv), :);
    Y_test  = Y(test(cv));

    %% 1. Decision Tree
    treeModel = fitctree(X_train, Y_train);
    predTree = predict(treeModel, X_test);
    accTree = mean(predTree == Y_test);

    %% 2. Multi-class SVM (ECOC)
    svmTemplate = templateSVM('KernelFunction','linear');
    svmModel = fitcecoc(X_train, Y_train, 'Learners', svmTemplate);
    predSVM = predict(svmModel, X_test);
    accSVM = mean(predSVM == Y_test);

    %% 3. KNN
    knnModel = fitcknn(X_train, Y_train, 'NumNeighbors', 5);
    predKNN = predict(knnModel, X_test);
    accKNN = mean(predKNN == Y_test);

    %% Store results
    results.DecisionTree.accuracy = accTree;
    results.SVM.accuracy = accSVM;
    results.KNN.accuracy = accKNN;

    results.DecisionTree.pred = predTree;
    results.SVM.pred = predSVM;
    results.KNN.pred = predKNN;

    results.Y_test = Y_test;

end
