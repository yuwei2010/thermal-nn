classdef optimizedTNNLayer <  nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable
    % optimizedTNNLayer   Thermal Neural Network layer
    %   Copyright 2025 The MathWorks, Inc.
    properties
        output_size (1,1) double
        nontemp_idcs (:,1) double {mustBePositive,mustBeInteger}
        input_cols (:,1) string
        target_cols (:,1) string
        temperature_cols (:,1) string
        diffFcn (1,1) 
    end

    properties (Learnable)
        W1_cn
        b1_cn
        W1_pl
        b1_pl
        W2_pl
        b2_pl
        caps
    end

    methods
        function this = optimizedTNNLayer(input_cols,target_cols,temperature_cols)
            this.NumInputs = 2;
            this.NumOutputs = 2;
            this.output_size = length(target_cols);
            n_temps = length(temperature_cols);

            % Populate adjacency matrix
            adj_mat = ones(n_temps, n_temps);
            tril_idx = find(tril(ones(n_temps),-1));
            adj_idx_arr = 1:(0.5*n_temps*(n_temps-1));
            adj_mat(tril_idx) = adj_idx_arr;
            adj_mat = adj_mat + adj_mat'-1;
            adj_mat = adj_mat(1:this.output_size, :);

            this.input_cols = strtrim(string(input_cols))';
            this.target_cols = strtrim(string(target_cols))';
            this.temperature_cols = strtrim(string(temperature_cols))';

            % Indices for temperature and non-temperature columns
            temp_idcs = find(ismember(this.input_cols, this.temperature_cols));
            this.nontemp_idcs = find(~ismember(this.input_cols, [this.temperature_cols; "profile_id"]));
            this = initAllLearnables(this);
            this.diffFcn = tnnWithBackward(adj_mat, temp_idcs);
        end

        function this = initAllLearnables(this)

            n_temps = length(this.temperature_cols);
            n_conds = 0.5 * n_temps * (n_temps - 1);
            numNeurons = 16;
            % per default, just use one dense layer + sigmoid activations
            this.W1_cn = initGlorot(n_conds,length(this.input_cols) + this.output_size);
            this.b1_cn = dlarray(zeros(n_conds,1,'single'));

            % per default, use two dense layers + tanh activations
            this.W1_pl = initGlorot(numNeurons,length(this.input_cols) + this.output_size);
            this.b1_pl = dlarray(zeros(numNeurons,1,'single'));
            this.W2_pl = initGlorot(this.output_size,numNeurons);
            this.b2_pl = dlarray(zeros(this.output_size,1,'single'));

            this.caps = dlarray(randn(this.output_size, 1,'single') * 0.5 - 9.2); % Initialize caps
        end

        function [X, h] = predict(this, X, h)
            [X, h] = this.diffFcn(X, h, this.W1_cn, this.b1_cn, this.W1_pl, this.b1_pl, this.W2_pl, this.b2_pl, this.caps);
            X = dlarray(X,'CBT');
            h = dlarray(h, 'CBT');
        end

        function [condNet,plossNet] = exportDlnetworks(this)

            condNet = dlnetwork([featureInputLayer(length(this.input_cols) + this.output_size),...
                fullyConnectedLayer(size(this.W1_cn,1),Weights=this.W1_cn,Bias = this.b1_cn),...
                sigmoidLayer]);
            plossNet = dlnetwork([featureInputLayer(length(this.input_cols) + this.output_size),...
                   fullyConnectedLayer(16,Weights = this.W1_pl,Bias = this.b1_pl),...
                tanhLayer,...
                fullyConnectedLayer(this.output_size,Weights = this.W2_pl,Bias = this.b2_pl)]);

        end

    end
end

function W = initGlorot(numNeurons,numInputs)
W = dlarray(sqrt(6./(numInputs+numNeurons))*(2*rand(numNeurons,numInputs,'single')-1));
end
