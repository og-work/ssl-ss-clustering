%This function calculates the kernel from data. 
% *Inputs:
% BASE_PATH
% Base path for directory based on system/platform
% 
% data:
% matrix NxD of features  where N: number of points and D: feature dimension
% i.e. Points are arranged along the rows
% 
% *Outputs:
% kernel:
% Kernel computed from the data

function outKernel = functionGetKernel(inBASE_PATH, inData, inKernelType, inDatasetPath)

% Precompute Kernel Matrix
% Suggest precompute chi2 kernel and save it for re-training model
addpath(genpath(sprintf('%s/codes/matlab-stuff/tree-based-zsl', inBASE_PATH)));

if 0%exist(fullfile(inDatasetPath, 'linear_kernel_awa_train_dataset_vgg.mat'),'file')
    disp('Loading the precomputed kernel...')
    temp = load(strcat(inDatasetPath, 'linear_kernel_awa_train_dataset_vgg.mat'));
    outKernel = temp.outKernel;
else
    %%% Doesn't exist precomputed chi2 kernel
    features = [];
    if 1 %Para.norm_flag
        %%% Normalize Feature Vector and Label Vector
        temp.FeatureMatrix = func_NormalizeFeatureMatrix(inData);
    else
        temp.FeatureMatrix = inData;
    end
    
    features = [features temp.FeatureMatrix];
    features(isnan(features)) = 0;
    
    %% Compute Chi2 Kernel
    %   it may take long time to compute Chi2 kernel
    disp('Compuing the kernel...')
    outKernel = func_PrecomputeKernel(features, features, inKernelType);
    %save(sprintf('%s/linear_kernel_awa_train_dataset_vgg.mat', inDatasetPath), 'outKernel', '-v7.3');

end