function [weighted_loss,gradients,hidden] = lossFun(model,batch_input,batch_target, ...
    hidden,sample_weights)
% loss function: weighted MSE loss
%   Copyright 2025 The MathWorks, Inc.

% forward pass
[output, hidden] = predict(model, batch_input, hidden);
% Compute loss
loss = l2loss(output, batch_target, Reduction  = "none");
% Sample weighting
weighted_loss = sum((loss .* sample_weights) ./ sum(sample_weights, 'all'),'all');
% gradient calculation
gradients= dlgradient(weighted_loss,model.Learnables,EnableHigherDerivatives=false);
end