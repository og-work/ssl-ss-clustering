function targetDomainEmbeddingsTest = functionGetTargetDomainEmbedding(inTest_id, inSemanticEmbeddingsTest, inNumberOfValidClusters, inSsClusteringModel, ...
    inMappingG, inRemappedSeenPrototypes, inIndexOfRemappedSeenPrototypes)

targetDomainEmbeddingsTest = zeros(length(inIndexOfRemappedSeenPrototypes), length(inTest_id), inNumberOfValidClusters);
numberOfSeenClasses = length(inIndexOfRemappedSeenPrototypes);
[sorted sortingOrder] = sort(inIndexOfRemappedSeenPrototypes);

for i = 1:length(inTest_id)
    %Find the cluster to which test sample belongs
    %tmp = ssClusteringModel.clusterCenters;
    %clusterCenters = tmp(:, validClusterIndices);
    %distMat =  clusterCenters - repmat(vggFeatures(:,test_id(i)), 1, numberOfValidClusters);
    %[distance clusterAssignment] = min(sqrt(sum(distMat.^2, 1)));
    %distance = sqrt(sum(distMat.^2, 1));
    %distance = distance./sum(distance);
    %weighted_d = 0;
    %semanticEmbedding = functionTestRegressor(vggFeatures(:,test_id(i))', regressorFunction);
    diff1 = repmat(inSemanticEmbeddingsTest(i, :)', 1, inNumberOfValidClusters) - inSsClusteringModel.clusterCenters;
    weights1 = sum(diff1.^2, 1)/sum(sum(diff1.^2, 1));
    scoresAcrossClusters = [];
    
    for clusterIndex = 1:inNumberOfValidClusters
        outReMappedVector = functionTestRegressor(inSemanticEmbeddingsTest(i, :), inMappingG(:, clusterIndex));
        diff2 = repmat(outReMappedVector, 1, numberOfSeenClasses) - inRemappedSeenPrototypes;
        weights2 = sum(diff2.^2, 1)/sum(sum(diff2.^2, 1));
        targetDomainEmbeddingsTest(:, i, clusterIndex) = weights1(clusterIndex) * weights2(sortingOrder)';
    end
    
%     margins = [margins; max(scoresAcrossClusters, [], 1)];%weighted_d];
end