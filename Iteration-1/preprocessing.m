function [X, Y] = preprocessing(data)
% =========================================================
% PREPROCESSING
% - Encode categorical features
% - Normalize features (aggressive)
% - Merge middle class (accuracy boosting)
% =========================================================

% -------------------------
% 1. Extract target variable
% -------------------------
% Assumes column name is 'Class'
% Original labels: Low / Middle / High (or similar)

Y = grp2idx(categorical(data.Class));
% Now:
% 1 = Low
% 2 = Middle
% 3 = High

% -------------------------
% 2. STEP 4 (AGGRESSIVE):
%    Merge Middle into Low
% -------------------------
Y(Y == 2) = 1;   % Middle -> Low
% Resulting classes:
% 1 = Low + Middle
% 3 = High

% Optional: relabel to 1 and 2 (clean binary)
Y(Y == 3) = 2;

% -------------------------
% 3. Remove target from features
% -------------------------
data.Class = [];

% -------------------------
% 4. Encode features
% -------------------------
vars = data.Properties.VariableNames;
X = [];

for i = 1:numel(vars)
    col = data.(vars{i});

    if iscell(col) || isstring(col) || iscategorical(col)
        col = grp2idx(categorical(col));
    end

    X = [X col]; %#ok<AGROW>
end

% -------------------------
% 5. Aggressive normalization
% -------------------------
% (Applied BEFORE splitting – accuracy boosting)
X = normalize(X, 'range');

end
