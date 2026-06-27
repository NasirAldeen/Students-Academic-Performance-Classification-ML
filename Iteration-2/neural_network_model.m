function results = neural_network_model(X, Y)

T = full(ind2vec(Y')) ;

net = patternnet([64 32 16]);
net.trainFcn = 'trainscg';
net.performFcn = 'crossentropy';
net.divideFcn = 'dividetrain';   % training only

net = train(net, X', T);

outputs = net(X');
[~, pred] = max(outputs);

pred = pred';
acc = mean(pred == Y);

figure;
confusionchart(Y, pred);
title('Neural Network Confusion Matrix');

results.accuracy = acc;
end
