function simulinkInput = generateSimulinkInputs(data,testID,input_cols)

% structure with time format for the From Workspace block:
% values field is an array of format numChannels by 1 by numTimeSteps
numInputs = numel(input_cols);
values = single(data{data.profile_id==testID,input_cols});
profLength = size(values,1)-1;
simulinkInput.time=0:profLength-1;
simulinkInput.signals.values = reshape(values(1:end-1,:)',numInputs,1,profLength);
simulinkInput.signals.dimensions=[numInputs 1];
end