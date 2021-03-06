
clc;
clear
close all;

% 1: Linux Laptop
% 2: Windows laptop
% 3: Linux Desktop
% 4: Windows Desktop
SYSTEM_PLATFORM = 4;
BASE_PATH = '';
listDatasets = {'AwA', 'Pascal-Yahoo'};
DATASET_ID = 2;
DATASET = listDatasets{DATASET_ID};
%Select kernels from the following
listOfKernelTypes = {'chisq', 'cosine', 'linear', 'rbf', 'rbfchisq'};
kernelType = listOfKernelTypes{4};
useKernelisedData = 3;
numberOfClusters = 2;
listFileNamesMappedAttributes = {'awa_mappedAllAttributes', 'apy_mappedAllAttributes'};
listFileNamesMappedAttributesLabels = {'awa_mappedAllAttributeLabels', 'apy_mappedAllAttributeLabels'};
fileNameMappedAttributes = listFileNamesMappedAttributes{DATASET_ID};
fileNameMappedAttributesLabels = listFileNamesMappedAttributesLabels{DATASET_ID};
%Enable/add required tool boxes
addPath = 0;
BASE_PATH = functionSemantic_similaity_env_setup(SYSTEM_PLATFORM, addPath);
VIEW_TSNE = 0;

%% START >> Load data
if(strcmp(DATASET, 'AwA'))
    dataset_path = sprintf('%s/data/code-data/semantic-similarity/precomputed-features-AwA/', BASE_PATH);
    %'data/code-data/semantic-similarity/cnn-features
    %temp = load(sprintf('%s/data/code-data/semantic-similarity/cnn-features/AwA/feat-imagenet-vgg-verydeep-19.mat', BASE_PATH));
    temp = load(sprintf('%s/AwA_All_vgg19Features.mat', dataset_path));
    vggFeatures = temp.vggFeatures;
    attributes = load(sprintf('%s/AwA_All_ClassLabelPhraseDict.mat', dataset_path));
    temp = load(sprintf('%s/AwA_All_DatasetLabels.mat', dataset_path));
    datasetLabels = temp.datasetLabels';
    vggFeatures = vggFeatures';
    attributes = attributes.phrasevec_mat';
    NUMBER_OF_CLASSES = 50;
    %Default setting of AwA
    %defaultTrainClassLabels = [1:7, 9:16, 18:21, 23, 25, 27:32, 35:36, 39, 41:50];
    % From dataset
    defaultTestClassLabels = [8 17 22 24 26 33 34 37 38 40];
    % from semsnticdemo
    %defaultTestClassLabels = [25 39 15 6 42 14 18 48 34 24];
    numberOfSamplesPerTrainClass = 20;%92;%150 apy, 92 AwA
elseif (strcmp(DATASET, 'Pascal-Yahoo'))
    dataset_path = sprintf('%s/data/code-data/semantic-similarity/cnn-features/aPY/', BASE_PATH);
    load(sprintf('%s/class_attributes.mat', dataset_path));
    load(sprintf('%s/cnn_feat_imagenet-vgg-verydeep-19.mat', dataset_path));
    datasetLabels = labels;
    clear labels;
    vggFeatures = cnn_feat;
    attributes = class_attributes';
    NUMBER_OF_CLASSES = 32;
    % From dataset
    %defaultTestClassLabels = [1 2 5 6 21 8 10 12 32];%[21:32];
    defaultTestClassLabels = [21:32];
    numberOfSamplesPerTrainClass = 150; %150 apy, 92 AwA
    classNames = {'1 aeroplane', '2 bicycle', '3 bird', '4 boat', '5 bottle', '6 bus', '7 car', '8 cat', ...
        '9 chair','10 cow','11 diningtable','12 dog','13 horse','14 motorbike','15 person','16 pottedplant','17 sheep',...
        '18 sofa','19 train','20 tvmonitor','21 donkey','22 monkey','23 goat','24 wolf','25 jetski','26 zebra','27 centaur','28 mug',...
        '29 statue','30 building','31 bag','32 carriage'}';
else
    sprintf('No Dataset selected ...')
end

%% END >> Load data

%% Start >> Clustering of data
labels = zeros(1, NUMBER_OF_CLASSES);
labels(defaultTestClassLabels) = 1;
labels = 1. - labels;
defaultTrainClassLabels = find(labels);

leaveKOut = 1;
mappedAllAttributes = [];
mappedAllAttributeLabels = [];
mappedAllAttributeLabelsVisualisation = [];

%Get training class features
vggFeaturesTraining = [];
labelsTrainingData = [];
labelsTestingData = [];
indicesOfTrainingSamples = [];
indicesOfTestingSamples = [];

for classInd = 1:length(defaultTrainClassLabels)
    tmp = (datasetLabels == defaultTrainClassLabels(classInd));
    indicesOfTrainingSamples = [indicesOfTrainingSamples; find(tmp)];
    vggFeaturesTraining = [vggFeaturesTraining vggFeatures(:, find(tmp))];
    labelsTrainingData = [labelsTrainingData; defaultTrainClassLabels(classInd) * ones(sum(tmp), 1)];
end

for classInd = 1:length(defaultTestClassLabels)
    tmp = (datasetLabels == defaultTestClassLabels(classInd));
    indicesOfTestingSamples = [indicesOfTestingSamples; find(tmp)];
    labelsTestingData = [labelsTestingData; defaultTestClassLabels(classInd) * ones(sum(tmp), 1)];
end

if useKernelisedData
    %kernelData = functionGetKernel(BASE_PATH, vggFeaturesTraining', kernelType, dataset_path);
    kernelFullData = functionGetKernel(BASE_PATH, vggFeatures', kernelType, dataset_path);
end

kernelTrainData = kernelFullData(indicesOfTrainingSamples, indicesOfTrainingSamples);
kernelTestData = kernelFullData(indicesOfTestingSamples, indicesOfTrainingSamples);

if 1%~exist(fullfile(sprintf('%s/%s.mat', dataset_path, fileNameMappedAttributes)),'file')
    for ind = 1:1%length(datasetLabels) - leaveKOut;
        leaveOutDatasetLabels = labelsTrainingData;
        %Assign 0 label for left out samples
        leaveOutDatasetLabels((ind - 1)*leaveKOut + 1 : (ind - 1)*leaveKOut + leaveKOut) = 0;
        %Find array of 0-1 in which 0 corresponds to left out sample
        tempB = (leaveOutDatasetLabels~=0);
        %create new labels array contianing only non-left-out samples
        leaveOutDatasetLabels = leaveOutDatasetLabels(tempB == 1);
        attributesMat = [];
        mappedAttributeLabels = [];
        mappedAttributeLabelsVisualisation = [];
        tempC = [];
        
        for c_tr = 1:length(defaultTrainClassLabels)
            tmp1 = (leaveOutDatasetLabels == defaultTrainClassLabels(c_tr));
            col1 = find(tmp1);
            col1 = col1(1:numberOfSamplesPerTrainClass);
            tempC = [tempC; col1];
            %Prepare attribute matrix which contains attribute vec for each data
            %point in leaveOutData
            % Extract Features for each train class
            numberOfSamplesOfClass(c_tr) = numberOfSamplesPerTrainClass;%sum(leaveOutDatasetLabels==defaultTrainClassLabels(c_tr));
            attributesMat = [attributesMat; repmat(attributes(:, defaultTrainClassLabels(c_tr))', numberOfSamplesOfClass(c_tr), 1)];
            %tr_sample_ind = tr_sample_ind + tr_sample_class_ind;
            mappedAttributeLabels = [mappedAttributeLabels; defaultTrainClassLabels(c_tr) * ones(numberOfSamplesOfClass(c_tr), 1)];
            mappedAttributeLabelsVisualisation = [mappedAttributeLabelsVisualisation; c_tr * ones(numberOfSamplesOfClass(c_tr), 1)];
            col1=[];tmp1=[];
        end
        
        indicesOfTrainingSamplesLeaveOut = [1:length(tempC)];
        indicesOfTestingSamplesTmp = [length(tempC) + 1: length(tempC) + length(indicesOfTestingSamples)];
        tempC = [tempC; indicesOfTestingSamples];
        
        %leaveOutData contains non-left-out training samples and all testing samples
        if useKernelisedData
            leaveOutData = kernelFullData(tempC, tempC);
        else
            %***TODO Correction here: should be vggFeatures
            leaveOutData = vggFeaturesTraining(:, tempC);
        end
        
        %Train regressor
        [mappedAttributes mappingF semanticEmbeddingsTest]= functionTrainRegressor(leaveOutData', ...
            attributesMat, BASE_PATH, useKernelisedData, indicesOfTrainingSamplesLeaveOut, indicesOfTestingSamplesTmp);
        mappedAllAttributes = [mappedAllAttributes; mappedAttributes];
        mappedAllAttributeLabels = [mappedAllAttributeLabels; mappedAttributeLabels];
        mappedAllAttributeLabelsVisualisation = [mappedAllAttributeLabelsVisualisation; mappedAttributeLabelsVisualisation];
        leaveOutDatasetLabels = [];
        tempB = [];
        leaveOutData = [];
        %attributesMat = [];
        %mappedAttributeLabels = [];
        save(sprintf('%s/%s.mat',dataset_path, fileNameMappedAttributes), 'mappedAllAttributes');
        save(sprintf('%s/%s.mat',dataset_path, fileNameMappedAttributesLabels), 'mappedAllAttributeLabels');
    end
else
    temp1 = load(sprintf('%s/%s.mat',dataset_path, fileNameMappedAttributes));
    mappedAllAttributes = temp1.mappedAllAttributes;
    temp2 = load(sprintf('%s/%s.mat',dataset_path, fileNameMappedAttributesLabels));
    mappedAllAttributeLabels = temp2.mappedAllAttributeLabels;
end

ssClusteringModel = functionClusterData(mappedAllAttributes', mappedAllAttributeLabels, ...
    numberOfClusters, length(defaultTrainClassLabels), defaultTrainClassLabels);
%% End >> Clustering of data

if VIEW_TSNE
    % Plot clustered points
    semanticEmbeddingFullData = [mappedAllAttributes; semanticEmbeddingsTest];
    semanticLabelsFullData = [mappedAttributeLabels; labelsTestingData];
    %funtionTSNEVisualisation(semanticEmbeddingsTest', labelsTestingData', length(defaultTestClassLabels));
    
    labelsAsClassNames = functionGetLabelsAsClassNames(classNames, semanticLabelsFullData);
    mappedData = [];
    mappedData = funtionTSNEVisualisation(semanticEmbeddingFullData');
    figureTitle = sprintf('Seen and Unseen class samples MAPPED');
    functionMyScatterPlot(mappedData, labelsAsClassNames', NUMBER_OF_CLASSES, figureTitle);
    labelsAsClassNames = [];
    
    labelsAsClassNames = functionGetLabelsAsClassNames(classNames, mappedAllAttributeLabelsVisualisation);
    mappedData = [];
    mappedData = funtionTSNEVisualisation(mappedAllAttributes');
    figureTitle = sprintf('Seen class samples MAPPED using f');
    functionMyScatterPlot(mappedData, labelsAsClassNames', length(defaultTrainClassLabels), figureTitle);
    labelsAsClassNames = [];
end

%% START >> Semantic to semantic mapping
useKernelisedData = 0;
mappingG = [];
b = 1;
indexOfRemappedSeenPrototypes = [];

for clusterIndex = 1:numberOfClusters
    allClassesInCluster = find(ssClusteringModel.classClusterAssignment(:, 1) == clusterIndex);
    trainClassIndex = find(ismember(defaultTrainClassLabels, allClassesInCluster));
    %testClassIndex = defaultTestClassLabels; %find(ismember(defaultTestClassLabels, allClassesInCluster));
    trainClassLabels = defaultTrainClassLabels(trainClassIndex);
    indicesOfTrainClassAttributes = find(ismember(mappedAllAttributeLabels, trainClassLabels));
    remappSource = mappedAllAttributes(indicesOfTrainClassAttributes, :);
    remappTarget = attributesMat(indicesOfTrainClassAttributes, :);
    [reMappedAttributes regressor reMappedSemanticEmbeddingsTest]= functionTrainRegressor(remappSource', ...
        remappTarget, BASE_PATH, useKernelisedData, [1:size(remappSource, 1)], []);
    mappingG = [mappingG regressor];
    reMappedAllAttributesLabels = [];
    indexOfRemappedSeenPrototypes = [indexOfRemappedSeenPrototypes trainClassLabels];
    
    for m = 1:length(trainClassLabels)
        % tmpClassLabel = m + length(defaultTrainClassLabels);
        reMappedAttributesLabels = trainClassLabels(m)*ones(sum(mappedAllAttributeLabels == trainClassLabels(m)), 1);
        reMappedAllAttributesLabels = [reMappedAllAttributesLabels; reMappedAttributesLabels];
        startI = (m - 1) * numberOfSamplesPerTrainClass + 1;
        endI = (m - 1) * numberOfSamplesPerTrainClass + numberOfSamplesPerTrainClass;
        remappedSeenPrototypes(:, b) = mean(reMappedAttributes(startI:endI, :))';
        b = b + 1;
    end
    
    if VIEW_TSNE 
%         funtionTSNEVisualisation([mappedAllAttributes; reMappedAttributes]', ...
%             [mappedAllAttributeLabelsVisualisation; reMappedAllAttributesLabels]', tmpClassLabel);
        labelsAsClassNames = functionGetLabelsAsClassNames(classNames, reMappedAllAttributesLabels);
        mappedData = [];
        mappedData = funtionTSNEVisualisation(reMappedAttributes');
        figureTitle = sprintf('Cluster %d : Seen class samples RE-MAPPED using g%d', clusterIndex, clusterIndex);
        functionMyScatterPlot(mappedData, labelsAsClassNames', length(trainClassLabels), figureTitle);
        labelsAsClassNames = [];
    end
end
%%END >> Semantic to semantic mapping


%NN
margins = [];
test_id = find(ismember(datasetLabels, defaultTestClassLabels));
for i = 1:length(test_id)
    diff = repmat(semanticEmbeddingsTest(i, :)', 1, length(defaultTestClassLabels)) - attributes(:, defaultTestClassLabels);
    nnScores = sum(diff.^2, 1)/sum(sum(diff.^2, 1));
    %     scoresAcrossClusters =  reshape(targetDomainEmbeddingsTest(:, i, :), length(defaultTrainClassLabels), numberOfClusters)'...
    %         * histogramsAllClasses(:,defaultTestClassLabels);
    margins = [margins; max(nnScores, [], 1)];
end
%%% classify
[margin id] = max(margins, [], 2);
a = (defaultTestClassLabels(id));
b = datasetLabels(test_id);
if ~sum(size(a) == size(b))
    a = a';
end
acc = 100*sum(a == b)/length(test_id)
margins = [];
meanAcc = mean(acc)
%NN

if 0
%Training
validClusterIndex = 1;
validClusterIndices = [];
histogramsUnseenClasses = functionGetSourceDomainEmbedding(defaultTestClassLabels, defaultTrainClassLabels, attributes);

%Testing
margins = [];
test_id = find(ismember(datasetLabels, defaultTestClassLabels));
targetDomainEmbeddingsTest = functionGetTargetDomainEmbedding(test_id, semanticEmbeddingsTest,...
    numberOfClusters, ssClusteringModel, mappingG, remappedSeenPrototypes, indexOfRemappedSeenPrototypes);

for i = 1:length(test_id)
    scoresAcrossClusters =  targetDomainEmbeddingsTest(:, i)'...
        * histogramsUnseenClasses(:,defaultTestClassLabels);
    %     scoresAcrossClusters =  reshape(targetDomainEmbeddingsTest(:, i, :), length(defaultTrainClassLabels), numberOfClusters)'...
    %         * histogramsAllClasses(:,defaultTestClassLabels);
    margins = [margins; max(scoresAcrossClusters, [], 1)];
end

%%% classify
[margin id] = max(margins, [], 2);
a = (defaultTestClassLabels(id));
b = datasetLabels(test_id);
if ~sum(size(a) == size(b))
    a = a';
end
acc = 100*sum(a == b)/length(test_id)
margins = [];
meanAcc = mean(acc)
%% END >> Testing
end
