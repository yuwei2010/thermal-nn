prj = currentProject;
answer = 'No';
path_to_csv = fullfile(prj.RootFolder,'data', 'input', 'measures_v2.csv');

if ~exist(path_to_csv,"file")
        answer = questdlg("No dataset detected. Download the" + newline + "'Electric Motor Temperature Data Set' from MathWorks Supportfiles?"+newline+...
            "https://ssd.mathworks.com/supportfiles?", ...
            'Missing data set','Yes','No','Yes');
    switch answer
        case 'Yes'
            loadEMTDS(prj);
        case 'No'
            disp("No data will be downloaded, will not be able to run training or inference.")
    end

end
edit('TNN_matlab.m')

function loadEMTDS(prj)
URL = "https://ssd.mathworks.com/supportfiles/EMTDS/data/raw/ElectricMotorTemperatureDataSet.zip";
disp('Downloading the Electric Temperature Motor Data Set. This may take a few moments.')
disp('Acknowledgement: Wilhelm Kirchgässner and Oliver Wallscheid, and Joachim Böcker, Electric Motor Temperature Data Set, Paderborn, 2021. doi:10.34740/KAGGLE/DSV/2161054.')
unzip(URL,fullfile(prj.RootFolder,"data","input"))
disp('Download complete.')
end