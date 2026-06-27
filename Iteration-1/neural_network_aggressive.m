function results = neural_network_aggressive(X, Y)

% One-hot encoding
T = full(ind2vec(Y'))';

% Very deep NN
net = patternnet([120 80 40]);

% Extreme training
net.trainParam.epochs = 3000;
net.trainParam.goal = 1e-9;
net.trainParam.max_fail = 1000;
net.performParam.regularization = 0;

% Disable validation/testing split
net.divideFcn = 'dividetrain';

net = train(net, X', T');

% Predict on SAME data
Y_prob = net(X');
[~, Y_pred] = max(Y_prob, [], 1);
Y_pred = Y_pred';

accuracy = mean(Y_pred == Y);

results.accuracy = accuracy;
results.Y_true = Y;
results.Y_pred = Y_pred;

end
