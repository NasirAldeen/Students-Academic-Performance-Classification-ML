function [X_fs, idx] = feature_selection(X, Y, k)
% Chi-square feature selection (top-k)

[idx, scores] = fscchi2(X, Y); %#ok<ASGLU>
idx = idx(1:k);
X_fs = X(:, idx);

end
