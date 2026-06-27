function clustering_kmeans(X, Y)
% K-Means + Elbow + PCA visualization

% -------------------------
% Elbow Method
% -------------------------
maxK = 10;
inertia = zeros(maxK,1);

for k = 1:maxK
    [~, ~, sumd] = kmeans(X, k, ...
        'Replicates', 10, ...
        'MaxIter', 500, ...
        'Display','off');
    inertia(k) = sum(sumd);
end

figure;
plot(1:maxK, inertia, '-o','LineWidth',1.5);
xlabel('Number of Clusters (k)');
ylabel('Inertia');
title('Elbow Method for Optimal k');
grid on;

% -------------------------
% Choose k = number of classes
% -------------------------
k = numel(unique(Y));

% -------------------------
% K-Means Clustering
% -------------------------
[idx, C] = kmeans(X, k, ...
    'Replicates', 20, ...
    'MaxIter', 1000);

% -------------------------
% PCA for Visualization
% -------------------------
[coeff, score] = pca(X);

figure;
gscatter(score(:,1), score(:,2), idx);
hold on;
plot(C(:,1), C(:,2), 'kx','MarkerSize',15,'LineWidth',3);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('K-Means Clustering of Students Performance (PCA)');
grid on;
hold off;

end
