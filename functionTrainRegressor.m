% This function obtains the multidimensional regression function.
% Input data of D dimension is mapped to Q dimension using regressor function
% which is learned using SVM. Learned regressor is again applied to input data
% *Inputs:
% inData:
% matrix NxD of features  where N: number of points and D: feature dimension
% i.e. Points are arranged along the rows
%
% inDatasetLabels:
% vector of size N containing labels of N data points
%
% inAttributes:
% matrix of size NxQ where there are Q dim attributes arranged along rows for
% N data points
%
% inBASE_PATH:
% Base path for directory
%
% inUseKernelisedData:
% Flag for using/not using kernelised data for SVM
% *Outputs:
% outMappedVectors:
% Mapped input data using learned SVM regressor.

function outMappedVectors = functionTrainRegressor(inData, inDatasetLabels, inAttributes, inBASE_PATH, inUseKernelisedData)
addpath(genpath(sprintf('%s/codes/matlab-stuff/tree-based-zsl', inBASE_PATH)));

kernelData = double(inData);
Data.D_tr = kernelData;%(Data.tr_sample_ind, Data.tr_sample_ind);
Data.D_ts = kernelData;%(Data.ts_sample_ind, Data.tr_sample_ind);

if inUseKernelisedData
%     Generate RBF Chi2 Kernel Matrix
%     normalizer for RBF kernel
%     temp.A = mean(mean(kernelData));
%     Data.D_tr = exp(- Data.D_tr/temp.A);
%     Data.D_ts = exp(- Data.D_ts/temp.A);
end

temp.SS = sum(inAttributes.^2,2);
temp.label_k = sqrt(size(inAttributes,2)./temp.SS);
attributes = double(repmat(temp.label_k, 1, size(inAttributes,2)) .* inAttributes);

% Training Support Vector Regression model for each dimension
% Parameters:
% -s
% 0 -- linear: u'*v
% 1 -- polynomial: (gamma*u'*v + coef0)^degree
% 2 -- radial basis function: exp(-gamma*|u-v|^2)
% 3 -- sigmoid: tanh(gamma*u'*v + coef0)
% 4 -- precomputed kernel (kernel values in training_set_file)
if inUseKernelisedData
    kernel = 4;
else
    kernel = 0;
end
%TODO: Need to tune Para.C

%[C gamma] = functionGetParaUsingCrossvalidation(Data.D_tr, attributes, inUseKernelisedData);

Para.C = 0.001;
outMappedVectors = zeros(size(inData, 1), size(inAttributes, 2));

for d = 1:size(attributes, 2)
    tic;
    if inUseKernelisedData
        %If precomputed kernel is used
        model{d} = libsvmtrain(attributes(:,d),[(1:size(Data.D_tr,1))' Data.D_tr],sprintf('-s 3 -t %d -c %f -h 0', kernel, Para.C)); % -s 3
    else
        %If data is used directly
        model{d} = libsvmtrain(attributes(:,d), Data.D_tr, sprintf('-s 3 -t %d -c %f -h 0', kernel, Para.C)); % -s 3
        %ts_LabelVec_hat(:,d) = libsvmpredict(zeros(size(Data.D_ts,1),1),[(1:size(Data.D_ts,1))' Data.D_ts],model{d});
    end
    outMappedVectors(:,d) = libsvmpredict(zeros(size(Data.D_tr,1),1), Data.D_tr, model{d});
    toc;
    fprintf('Finish %d th dimension\n',d)
end

