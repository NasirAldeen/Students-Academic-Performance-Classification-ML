function [X, Y] = preprocessing(data)

    % Remove missing values
    data = rmmissing(data);

    % Convert categorical variables to numeric
    vars = data.Properties.VariableNames;
    for i = 1:length(vars)
        if iscell(data.(vars{i})) || iscategorical(data.(vars{i}))
            data.(vars{i}) = grp2idx(categorical(data.(vars{i})));
        end
    end

    % Target variable: Class (Low / Middle / High)
    Y = data.Class;

    % Remove target from features
    data.Class = [];

    % Convert features to matrix
    X = table2array(data);

    % Normalize features
    X = normalize(X);

end
