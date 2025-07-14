function [tensor, sample_weights] = generate_tensor(data, profiles_list, profile_sizes,input_cols, target_cols)
% This function transforms tabular time-series data into a 3D dlarray format
% suitable for training a recurrent neural network. For each profile (sequence),
% it aligns input features at time step t with target temperatures at time step t+1,
% enabling the model to learn next-step forecasting.

%   Copyright 2025 The MathWorks, Inc.
numInputs = numel(input_cols);
numTargets = numel(target_cols);
max_profile_length = max(profile_sizes.GroupCount(ismember(profile_sizes.profile_id, profiles_list)));
% Initialize tensor with NaNs
tensor = dlarray(nan(numInputs+numTargets,length(profiles_list),max_profile_length-1),'CBT');

% Fill the tensor with data
for i = 1:length(profiles_list)
    pid = profiles_list(i);
    df = data(data.profile_id == pid, :);
    % Convert table to array and drop 'profile_id' column
    % apply timestep shift for output columns:
    % inputs at time step k are used to predict the outputs at timestep k+1
    tensor(1:numInputs, i, 1:height(df)-1) = table2array(df(1:end-1, input_cols))';
    tensor(numInputs+1:end, i, 1:height(df)-1) = table2array(df(2:end, target_cols))';
    % if not applying timeshift, this would just be
    % tensor(:, i, 1:height(df)) = table2array(df(:, setdiff(df.Properties.VariableNames, {'profile_id'},'stable')))';
end

% Create sample weights
sample_weights = ~isnan(tensor(1, :, :));

% Replace NaNs with zeros
tensor(isnan(tensor)) = 0;

% Convert to single precision
tensor = single(tensor);
end