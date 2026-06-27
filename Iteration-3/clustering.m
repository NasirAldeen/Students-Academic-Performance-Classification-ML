function idx = clustering(X)

    % Number of clusters (Low, Middle, High)
    k = 3;

    % Apply K-means clustering
    [idx, C] = kmeans(X, k, 'Replicates', 10);

    % Reduce dimensions for visualization using PCA
    [coeff, score] = pca(X);

    % Plot clusters
    figure;
    gscatter(score(:,1), score(:,2), idx);
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    title('K-Means Clustering of Students Performance');
    grid on;

    % Save figure for later use in paper
    saveas(gcf, 'clusters.png');

end
