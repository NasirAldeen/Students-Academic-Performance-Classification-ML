function evaluation(results)

    Y_test = results.Y_test;

    %% Decision Tree confusion matrix
    figure;
    confusionchart(Y_test, results.DecisionTree.pred);
    title('Decision Tree Confusion Matrix');
    saveas(gcf, 'confusion_tree.png');

    %% SVM confusion matrix
    figure;
    confusionchart(Y_test, results.SVM.pred);
    title('SVM Confusion Matrix');
    saveas(gcf, 'confusion_svm.png');

    %% KNN confusion matrix
    figure;
    confusionchart(Y_test, results.KNN.pred);
    title('KNN Confusion Matrix');
    saveas(gcf, 'confusion_knn.png');

end
