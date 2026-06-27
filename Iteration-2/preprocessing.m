function [X, Y] = preprocessing(data)

% Target variable
Y = grp2idx(categorical(data.Class));

% Merge middle class (class consolidation)
Y(Y == 2) = 1;     % Middle → Low
Y(Y == 3) = 2;     % High → 2-class system

% Remove target
data.Class = [];

% Encode features
vars = data.Properties.VariableNames;
X = [];

for i = 1:numel(vars)
    col = data.(vars{i});
    if iscell(col) || isstring(col) || iscategorical(col)
        col = grp2idx(categorical(col));
    end
    X = [X col]; %#ok<AGROW>
end

% Normalize features
X = normalize(X, 'range');

end
