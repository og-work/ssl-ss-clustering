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
%
%inIdicesOfTrainingSamples

%inIndicesOfTestingSamples
% *Outputs:
% outMappedVectors:
% Mapped input data using learned SVM regressor.
% outRegressorFunction
%
% outSemanticEmbeddingsTest

function [outSemanticEmbeddingsTrain, outRegressorFunction, outSemanticEmbeddingsTest] = functionTrainRegressor(varargin)

%Parse inputs
if nargin == 6
    inData = varargin{1};
    inAttributes = varargin{2};
    inBASE_PATH = varargin{3};
    inUseKernelisedData = varargin{4};
    inIdicesOfTrainingSamples = varargin{5};
    inIndicesOfTestingSamples = varargin{6};
else
    fprintf('\n Not enough number of input arguments to functionTrainRegressor()')
end

%addpath(genpath(sprintf('%s/codes/matlab-stuff/tree-based-zsl', inBASE_PATH)));
kernelData = double(inData);

if inUseKernelisedData
    Data.D_tr = kernelData(inIdicesOfTrainingSamples, inIdicesOfTrainingSamples);
    Data.D_ts = kernelData(inIndicesOfTestingSamples, inIdicesOfTrainingSamples);
    kernel = 4;
    %     Generate RBF Chi2 Kernel Matrix
    %     normalizer for RBF kernel
    %     temp.A = mean(mean(kernelData));
    %     Data.D_tr = exp(- Data.D_tr/temp.A);
    %     Data.D_ts = exp(- Data.D_ts/temp.A);
else
    Data.D_tr = kernelData(:, inIdicesOfTrainingSamples);
    Data.D_ts = kernelData(:, inIndicesOfTestingSamples);
    kernel = 0;
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
%TODO: Need to tune Para.C
%[C gamma] = functionGetParaUsingCrossvalidation(Data.D_tr, attributes, inUseKernelisedData);
Para.C = 1.5;
% outSemanticEmbeddingsTest = zeros(size(Data.D_ts, 2), size(attributes, 2));
% outSemanticEmbeddingsTrain = zeros(size(Data.D_tr, 2), size(attributes, 2));
model = cell(size(attributes, 2), 1);

for d = 1:size(attributes, 2)
    tic;
    if inUseKernelisedData
        %If precomputed kernel is used
        model{d} = libsvmtrain(attributes(:,d),[(1:size(Data.D_tr,1))' Data.D_tr],sprintf('-s 3 -t %d -c %f -h 0', kernel, Para.C)); % -s 3
        outSemanticEmbeddingsTest(:,d) = libsvmpredict(zeros(size(Data.D_ts,1),1),[(1:size(Data.D_ts,1))' Data.D_ts], model{d});
        outSemanticEmbeddingsTrain(:,d) = libsvmpredict(zeros(size(Data.D_tr,1),1),[(1:size(Data.D_tr,1))' Data.D_tr], model{d});
    else
        %If data is used directly
        model{d} = libsvmtrain(attributes(:,d), Data.D_tr', sprintf('-s 3 -t %d -c %f -h 0', kernel, Para.C)); % -s 3
        outSemanticEmbeddingsTrain(:,d) = libsvmpredict(zeros(size(Data.D_tr,2),1), Data.D_tr', model{d});
        outSemanticEmbeddingsTest(:,d) = libsvmpredict(zeros(size(Data.D_ts,2),1), Data.D_ts', model{d});
    end
    toc;
    fprintf('Finish %d th dimension\n',d)
end
outRegressorFunction = model;

