classdef DiffEqLayer <  nnet.layer.Layer & nnet.layer.Formattable
    % wraps TNNCell
    % performs prediction over the time steps
    %   Copyright 2025 The MathWorks, Inc.
    properties (Learnable)
        Cell
    end

    methods
        function obj = DiffEqLayer(cell)
            % Constructor to store the cell
            obj.Cell = cell;
            obj.NumInputs = 2;
            obj.NumOutputs = 2;
        end

        function [outputs, state] = predict(obj, input, state)
            % Initialize cell array for outputs
            numSteps = size(input, 3); % Assuming input is [time, batch, features]
            % preallocate the outputs:
            outputs = dlarray(zeros(size(state,1),size(input,2),size(input,3),'single'),'CBT');
            Cell = obj.Cell;

            % Iterate over each time step
            for idx = 1:numSteps
                outputs(:,:,idx) = Cell.predict(squeeze(input(:, :, idx)),state);
                state = squeeze(outputs(:,:,idx));
            end

        end
    end
end


