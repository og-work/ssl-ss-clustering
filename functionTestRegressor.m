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

function outMappedVector = functionTestRegressor(inData, inRegressorFunction)

for d = 1:length(inRegressorFunction)
    outMappedVector(d, 1) = svmpredict(zeros(1, 1), double(inData), inRegressorFunction{d});
end

