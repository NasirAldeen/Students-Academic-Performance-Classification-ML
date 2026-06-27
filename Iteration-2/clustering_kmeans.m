function clustering_kmeans(X)

[coeff, score] = pca(X);

k = 2;
idx = kmeans(score(:,1:2), k);

figure;
gscatter(score(:,1), score(:,2), idx);
xlabel('Principal Component 1');
ylabel('Principal Component 2');
title('K-Means Clustering (PCA)');

end
