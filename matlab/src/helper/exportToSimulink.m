function modelParams= exportToSimulink(model)
% export a trained model to Simulink layer blocks
proj = currentProject;
prjRoot = proj.RootFolder;
if isa(model.Layers(1),'optimizedTNNLayer')
    [condNet,plossNet] = exportDlnetworks(model.Layers(1));
    modelParams.sampleTime = model.Layers(1).diffFcn.sample_time;
    modelParams.adj_mat =  model.Layers(1).diffFcn.adj_mat;
    modelParams.temp_idcs = model.Layers(1).diffFcn.temp_idcs;
    modelParams.caps = model.Layers(1).caps;

else
    condNet = model.Layers(1).Cell.Layers(1).conductance_net;
    plossNet = model.Layers(1).Cell.Layers(1).ploss;

    modelParams.sampleTime = model.Layers(1).Cell.Layers(1).sample_time;
    modelParams.adj_mat =  model.Layers(1).Cell.Layers(1).adj_mat;
    modelParams.temp_idcs = model.Layers(1).Cell.Layers(1).temp_idcs;
    modelParams.caps = model.Layers(1).Cell.Layers(1).caps;
end

saveDir = fullfile(prjRoot,"src");
exportNetwork(condNet,"cond_net",saveDir)
exportNetwork(plossNet,"ploss_net",saveDir)
end

function exportNetwork(network,modelName,saveDir)
if exist(modelName,'file')
    answer = questdlg(modelName+" already exists. Overwrite?",'Overwrite file?','Yes','Keep both', 'Do nothing','Yes');
    if strcmp(answer,'Yes')
        bdclose(modelName)
        delete(which(modelName+".slx"))
    elseif strcmp(answer,'Do nothing')
        return           
    end
end
exportNetworkToSimulink(network,ModelName =modelName ,OpenSystem = false, ...
    InputDataType='single',ModelPath=saveDir,SaveNetworkInModelWorkSpace = true);
end