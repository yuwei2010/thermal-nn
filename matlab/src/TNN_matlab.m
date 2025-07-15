%% Data setup
rng(0) % for reproducibility
proj = currentProject;
% assuming that the measures_v2.csv from Kaggle is in a folder /data/input
path_to_csv = fullfile(proj.RootFolder,'data', 'input', 'measures_v2.csv');
data = readtable(path_to_csv); % read in the data as a table

% define the four target temperatures
target_cols = {'pm', 'stator_yoke', 'stator_tooth', 'stator_winding'};
input_cols = setdiff(data.Properties.VariableNames, ['profile_id',target_cols],'stable');
temperature_cols = [target_cols, {'ambient', 'coolant'}];
test_profiles = [60, 62, 74]; %profile ids reserved for the test data
all_profiles = unique(data.profile_id,'stable');
train_profiles = setdiff(all_profiles, test_profiles);


% Normalize the temperatures: divide by 200
scale = 200;
non_temperature_cols = setdiff(data.Properties.VariableNames, [temperature_cols, {'profile_id'}]);
for i = 1:length(temperature_cols)
    data.(temperature_cols{i}) = data.(temperature_cols{i}) / scale;
end

% standardize non  temperature columns (divide by maximum values)
for i = 1:length(non_temperature_cols)
    col_max = max(abs(data.(non_temperature_cols{i})));
    data.(non_temperature_cols{i}) = data.(non_temperature_cols{i}) / col_max;
end

% Extra features (Feature Engineering)
if all(ismember({'i_d', 'i_q', 'u_d', 'u_q'}, data.Properties.VariableNames))
    data.i_s = sqrt(data.i_d.^2 + data.i_q.^2);
    data.u_s = sqrt(data.u_d.^2 + data.u_q.^2);
    input_cols = [input_cols,'i_s','u_s'];
end

%% Store data in 3D array (channel - batch - time format)

% how many time steps are in each profile?
profile_sizes = varfun(@length, data, 'GroupingVariables', 'profile_id');
profile_sizes = profile_sizes(:, {'profile_id', 'GroupCount'});

% Generate tensors in TBC (= time/batch/channel) format
[train_tensor, train_sample_weights] = generate_tensor(data, train_profiles, profile_sizes,input_cols,target_cols);
[test_tensor, test_sample_weights] = generate_tensor(data, test_profiles, profile_sizes,input_cols,target_cols);

%% Setup of the network architecture
% Define the inputs
numInputs = length(input_cols);
tbptt_size = 512;

% construct the TNN model
% We can optionally use an optimized mode that explicitly manually defines the
% backward operation. This will significantly improve performance, but
% requires additional work:

optimizedMode = true;
first_model_input = networkDataLayout([numel(input_cols),size(train_tensor,2),tbptt_size],'CBT');
second_model_input = networkDataLayout([numel(target_cols),size(train_tensor,2)],'CB');

if optimizedMode
    optimizedtnn = optimizedTNNLayer(input_cols,target_cols,temperature_cols);
    model = dlnetwork(optimizedtnn, first_model_input, second_model_input);
else

    % standard mode: Here, the first iteration will take several minutes
    % but subsequent iterations will be faster

    if (isMATLABReleaseOlderThan('R2025a'))
        error('MATLAB version R2025a or newer is required to run non-optimized mode.')
    end
    warning('Using unoptimized version. Completion of the first epoch make take several minutes!')


    first_cell_input = networkDataLayout([numel(input_cols),size(train_tensor,2)],'CB');
    second_cell_input = networkDataLayout([numel(target_cols),size(train_tensor,2)],'CB');
    Cell = TNNCell(input_cols,temperature_cols,target_cols);
    Cell = generateNetworks(Cell);
    networkCell = dlnetwork(Cell,first_cell_input,second_cell_input);

    DEL = DiffEqLayer(networkCell);
    model = dlnetwork(DEL,first_model_input,second_model_input);
end

% Initialize parameters for Adam optimizer
learningRate = 1e-3;
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;
averageGrad = [];
averageSqGrad = [];
iteration = 0;

%% Training loop
n_epochs = 100;
n_batches = ceil(size(train_tensor, 3) / tbptt_size);

% dlaccelerate is key for training performance
myModelLoss = dlaccelerate(@lossFun);
maxIterations = n_epochs * n_batches;

for epoch = 1:n_epochs
    % Initialize the hidden state with the targets of the first time step
    tic
    hidden = squeeze(train_tensor(end-length(target_cols)+1:end, :,1 ));
    epochLosses = zeros(n_batches,1); % array for storing losses per iteration
    % inner loop: iterate over minibatches
    for i = 1:n_batches
        iteration = iteration + 1;
        batch_input = train_tensor(1:numInputs,:,(i-1)*tbptt_size+1:min(i*tbptt_size,end));
        batch_target = train_tensor(numInputs+1:end,:,(i-1)*tbptt_size+1:min(i*tbptt_size,end));
        sample_weights = train_sample_weights(1,:,(i-1)*tbptt_size+1:min(i*tbptt_size,end));

        % loss and gradient calculation
        [weighted_loss,gradients,hidden] = dlfeval(myModelLoss, ...
            model,batch_input,batch_target,hidden,sample_weights);
        % solver step
        [model,averageGrad,averageSqGrad] = adamupdate(model,gradients,averageGrad,averageSqGrad,iteration,...
            learningRate,beta1,beta2,epsilon);
        % store the losses
        epochLosses(i) = extractdata(weighted_loss);

    end
    % Reduce learning rate
    if epoch == 75
        learningRate = learningRate * 0.5;
    end

    % Display progress: Time and average loss per iteration
    meanLoss = mean(epochLosses);
    fprintf('Epoch %d/%d, speed: %1.2fs/epoch, average epoch loss: %.2e\n', epoch, n_epochs,toc,meanLoss );
end

%% run prediction on test data set
inferenceFun = @(inputs,hiddenState)model.predict(inputs,hiddenState);
initialCondition = squeeze(test_tensor(numInputs+1:end,:,1));
[pred, hidden] = inferenceFun(test_tensor(1:numInputs,:,:), initialCondition);
prediction = extractdata(pred);
%% Visualize performance
profileLengths = profile_sizes.GroupCount(ismember(profile_sizes.profile_id,test_profiles))-1; % 60,62,74
generatePlots(profileLengths,test_tensor,prediction,scale,target_cols)

%% Simulink export
% export the model to Simulink layer blocks
if (isMATLABReleaseOlderThan('R2024b'))
    error('MATLAB version R2024b or newer is required to export model to Simulink.')
end
modelParams= exportToSimulink(model);

% select one of the three test cases 60,62,74
testID = 1;

% prepare input data for usage with From-Workspace blocks
simulinkInput = generateSimulinkInputs(data,test_profiles(testID),input_cols);
stopTime =  simulinkInput.time(end);
open_system('TNN_model');
simOut = sim('TNN_model');
simData =squeeze(simOut.yout{1}.Values.Data);
% compute the maximum deviation between MATLAB and Simulink:
% should be in the order of machine precision
allDeviations = abs(simData-squeeze(pred(:,testID,1:profileLengths(testID))));
fprintf('Max deviation Simulink vs Matlab (Test profile %d): %1.4e\n',testID,max(allDeviations(:)));
%   Copyright 2025 The MathWorks, Inc.