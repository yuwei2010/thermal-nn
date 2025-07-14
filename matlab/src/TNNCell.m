classdef TNNCell <  nnet.layer.Layer
    % TNNCell, forward pass for a single time step
    %   Copyright 2025 The MathWorks, Inc.
    properties
        sample_time (1,1) double = 0.5; % in seconds
        output_size (1,1) double
        adj_mat (:,:) double {mustBePositive,mustBeInteger}
        temp_idcs (:,1) double {mustBePositive,mustBeInteger}
        nontemp_idcs (:,1) double {mustBePositive,mustBeInteger}
        input_cols (:,1) string
        target_cols (:,1) string
        temperature_cols (:,1) string
    end

    properties (Learnable)
        conductance_net
        ploss
        caps
    end

    methods

        function obj = TNNCell(input_cols,temperature_cols,target_cols)

            obj.NumInputs = 2;
            obj.output_size = length(target_cols);
            n_temps = length(temperature_cols);

            % Populate adjacency matrix
            obj.adj_mat = ones(n_temps, n_temps);
            tril_idx = find(tril(ones(n_temps),-1));
            adj_idx_arr = 1:(0.5*n_temps*(n_temps-1));
            obj.adj_mat(tril_idx) = adj_idx_arr;
            obj.adj_mat = obj.adj_mat + obj.adj_mat'-1;
            obj.adj_mat = obj.adj_mat(1:obj.output_size, :);

            obj.input_cols = strtrim(string(input_cols))';
            obj.target_cols = strtrim(string(target_cols))';
            obj.temperature_cols = strtrim(string(temperature_cols))';

            % Indices for temperature and non-temperature columns
            obj.temp_idcs = find(ismember(obj.input_cols, obj.temperature_cols));
            obj.nontemp_idcs = find(~ismember(obj.input_cols, [obj.temperature_cols; "profile_id"]));

        end
        function obj = generateNetworks(obj)

            n_temps = length(obj.temperature_cols);
            n_conds = 0.5 * n_temps * (n_temps - 1);
            numNeurons = 16;

            % per default, just use one dense layer + sigmoid activations
            obj.conductance_net = dlnetwork([featureInputLayer(length(obj.input_cols) + obj.output_size),...
                fullyConnectedLayer(n_conds,Name = "conduc_fc1"),sigmoidLayer]);

            % per default, use two dense layers + tanh activations
            obj.ploss = dlnetwork([featureInputLayer(length(obj.input_cols) + obj.output_size),...
                fullyConnectedLayer(numNeurons,Name = "ploss_fc1"),...
                tanhLayer,...
                fullyConnectedLayer(obj.output_size,Name="ploss_fc2")]);

            obj.caps = dlarray(randn(obj.output_size, 1,'single') * 0.5 - 9.2); % Initialize caps

        end


        function out = predict(obj, inp, prev_out)
            temps = [prev_out; inp(obj.temp_idcs,:)];
            sub_nn_inp = [inp; prev_out];

            % Conductance network forward pass
            conducts = abs(predict(obj.conductance_net, sub_nn_inp'));

            % Power loss network forward pass
            power_loss = abs(predict(obj.ploss, sub_nn_inp'));

            % Calculate temperature differences
            szTemps = size(temps);
            szPrevout = size(prev_out);

            % The following two commented lines are equivalent to the lines
            % below:
            % conductArray = reshape(conducts(:,obj.adj_mat),size(conducts,1),size(obj.adj_mat,1),size(obj.adj_mat,2));
            % temp_diffs = sum((reshape(temps',szTemps(2),1,szTemps(1)) - prev_out') .* conductArray, 3);

            tmp = (repelem(temps',1,szPrevout(1))-repmat(prev_out',1,szTemps(1))).*conducts(:,obj.adj_mat);
            temp_diffs= sum(reshape(tmp,szTemps(2),szPrevout(1),szTemps(1)),3);

            % Calculate output
            out = prev_out + obj.sample_time.* exp(obj.caps) .* (temp_diffs + power_loss)';
            out = max(min(out, 5), -1); % Clip output
        end
    end
end
