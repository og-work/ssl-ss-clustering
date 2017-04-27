function targetDomainEmbeddingsTest = functionGetTargetDomainEmbedding(inTest_id, inSemanticEmbeddingsTest, inNumberOfValidClusters, inSsClusteringModel, ...
    inMappingG, inRemappedSeenPrototypes, inIndexOfRemappedSeenPrototypes)

targetDomainEmbeddingsTest = zeros(length(inIndexOfRemappedSeenPrototypes), length(inTest_id), 1);%inNumberOfValidClusters);
numberOfSeenClasses = length(inIndexOfRemappedSeenPrototypes);
[sorted sortingOrder] = sort(inIndexOfRemappedSeenPrototypes);

for i = 1:length(inTest_id)
    diff1 = repmat(inSemanticEmbeddingsTest(i, :)', 1, inNumberOfValidClusters) - inSsClusteringModel.clusterCenters;
    weights1 = sum(diff1.^2, 1)/sum(sum(diff1.^2, 1));
    scoresAcrossClusters = [];
    weightedEmbedding = zeros(numberOfSeenClasses, 1);
    
    for clusterIndex = 1:inNumberOfValidClusters
        outReMappedVector = functionTestRegressor(inSemanticEmbeddingsTest(i, :), inMappingG(:, clusterIndex));
        diff2 = repmat(outReMappedVector, 1, numberOfSeenClasses) - inRemappedSeenPrototypes;
        seenClassProportions = sum(diff2.^2, 1)/sum(sum(diff2.^2, 1));
        weightedEmbedding = weightedEmbedding + weights1(clusterIndex) * seenClassProportions(sortingOrder)';
    end
    targetDomainEmbeddingsTest(:, i) = weightedEmbedding;
%     margins = [margins; max(scoresAcrossClusters, [], 1)];%weighted_d];
end