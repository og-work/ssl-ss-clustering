function clusteringModel = functionClusterData(inVggFeatures, inDatasetLabels, ...
    inNUMBER_OF_CLUSTERS, inNUMBER_OF_CLASSES, inClassLabels)

USE_SUBSET_OF_FEATURES = 0;
disp('K-means clustering of features of dataset ...');
subSetOfAllFeatures = zeros(0,0);
offset = 0;
stride = 15;
subsetOfFeatures = [];
perClassFeatures = 90; % This number will change as per dataset. This number should be less than the minumum number of features
%across all classes

for i = 1:inNUMBER_OF_CLASSES
    classFeatures = inVggFeatures(:, find(inDatasetLabels == inClassLabels(i)));
    if USE_SUBSET_OF_FEATURES
        subsetOfFeatures = [subsetOfFeatures classFeatures(:, 1:perClassFeatures)];
    else
        subsetOfFeatures = [subsetOfFeatures classFeatures(:, :)];
    end
    i
end
%clear subSetOfAllFeatures
% subSetOfAllFeatures = allFeatures(:, startFeatureIndexTrain(4):5:endFeatureIndexTrain(4));
[clusterCenters, clusterAssignmentsOfData] = vl_kmeans(subsetOfFeatures, inNUMBER_OF_CLUSTERS, 'Initialization', 'plusplus');
% Using Matlab kmeans
% [clusterAssignmentsOfData, clusterCenters] = kmeans(subSetOfAllFeatures', NUMBER_OF_CLUSTERS);
% clusterCenters = clusterCenters';

if USE_SUBSET_OF_FEATURES
    %     a = reshape(clusterAssignmentsOfData, [], perClassFeatures);
    %     clusteringModel = mode(a, 2);
else
    for i = 1:inNUMBER_OF_CLASSES
        clusteringModel.classClusterAssignment(inClassLabels(i), 1) = mode(clusterAssignmentsOfData(inDatasetLabels == inClassLabels(i)));
        % Number of samples per class
        clusteringModel.classClusterAssignment(inClassLabels(i), 2) = length(clusterAssignmentsOfData(inDatasetLabels == inClassLabels(i)));
    end
end

clusteringModel.clusterCenters = clusterCenters;
