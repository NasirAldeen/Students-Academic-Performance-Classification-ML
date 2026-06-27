function nn_results = neural_network(X, Y)

    % Train/Test split
    cv = cvpartition(Y, 'HoldOut', 0.3);

    X_train = full(X(training(cv), :)');
X_test  = full(X(test(cv), :)');

Y_train = Y(training(cv))';
Y_test  = Y(test(cv))';

T_train = full(ind2vec(Y_train));


    % Create neural network
    net = patternnet(10);  % 10 hidden neurons
    net.trainParam.showWindow = false;

    % Train network
    net = train(net, X_train, T_train);

    % Predict
    Y_pred = net(X_test);
    Y_pred_class = vec2ind(Y_pred);

    % Accuracy
    accNN = mean(Y_pred_class == Y_test);

    % Store results
    nn_results.accuracy = accNN;
    nn_results.pred = Y_pred_class;
    nn_results.true = Y_test;

    % Confusion matrix
    figure;
    confusionchart(Y_test, Y_pred_class);
    title('Neural Network Confusion Matrix');
    saveas(gcf, 'confusion_nn.png');

end
