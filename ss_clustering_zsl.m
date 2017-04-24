
clc;
clear
%close all;

% 1: Linux Laptop
% 2: Windows laptop
% 3: Linux Desktop
% 4: Windows Desktop
SYSTEM_PLATFORM = 1;
BASE_PATH = '';
listDatasets = {'AwA', 'Pascal-Yahoo'};
DATASET_ID = 2;
DATASET = listDatasets{DATASET_ID};
%Select kernels from the following
listOfKernelTypes = {'chisq', 'cosine', 'linear', 'rbf', 'rbfchisq'};
kernelType = listOfKernelTypes{4};
useKernelisedData = 1;

%Enable/add required tool boxes
addPath = 1;
BASE_PATH = functionSemantic_similaity_env_setup(SYSTEM_PLATFORM, addPath);

%% START >> Load data
if 1
    if(strcmp(DATASET, 'AwA'))
        dataset_path = sprintf('%s/data/code-data/semantic-similarity/precomputed-features-AwA/', BASE_PATH);
        %'data/code-data/semantic-similarity/cnn-features
        %temp = load(sprintf('%s/data/code-data/semantic-similarity/cnn-features/AwA/feat-imagenet-vgg-verydeep-19.mat', BASE_PATH));
        temp = load(sprintf('%s/AwA_All_vgg19Features.mat', dataset_path));
        vggFeatures = temp.vggFeatures;
        attributes = load(sprintf('%s/AwA_All_ClassLabelPhraseDict.mat', dataset_path));
        temp = load(sprintf('%s/AwA_All_DatasetLabels.mat', dataset_path));
        datasetLabels = temp.datasetLabels;
        vggFeatures = vggFeatures';
        attributes = attributes.phrasevec_mat';
        NUMBER_OF_CLASSES = 50;
        %Default setting of AwA
        %defaultTrainClassLabels = [1:7, 9:16, 18:21, 23, 25, 27:32, 35:36, 39, 41:50];
        % From dataset
        defaultTestClassLabels = [8 17 22 24 26 33 34 37 38 40];
        % from semsnticdemo
        %defaultTestClassLabels = [25 39 15 6 42 14 18 48 34 24];
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
        defaultTestClassLabels = [21:32];
        
    else
        sprintf('No Dataset selected ...')
    end
end
%% END >> Load data


%% Start >> Clustering of data
numberOfClusters = 3;
labels = zeros(1, NUMBER_OF_CLASSES);
labels(defaultTestClassLabels) = 1;
labels = 1. - labels;
defaultTrainClassLabels = find(labels);

leaveKOut = 1;
mappedAllAttributes = [];
mappedAllAttributeLabels = [];

% *********************This is temporary. Remove afterwards
%**********************************************************
%**********************************************************
%reducing dimension of attribute vectors for faster processing
%attributes = attributes(1:30, :);
%**********************************************************
%Get training class features
vggFeaturesTraining = [];
labelsTrainingData = [];

for classInd = 1:length(defaultTrainClassLabels)
    tmp = (datasetLabels == defaultTrainClassLabels(classInd));
    vggFeaturesTraining = [vggFeaturesTraining vggFeatures(:, find(tmp))];
    labelsTrainingData = [labelsTrainingData; defaultTrainClassLabels(classInd) * ones(sum(tmp), 1)];
end

if useKernelisedData
    kernelData = functionGetKernel(BASE_PATH, vggFeaturesTraining', kernelType, dataset_path);
    %kernelData = ones(size(vggFeatures, 2), size(vggFeatures, 2));
end

listFileNamesMappedAttributes = {'awa_mappedAllAttributes', 'apy_mappedAllAttributes'};
listFileNamesMappedAttributesLabels = {'awa_mappedAllAttributeLabels', 'apy_mappedAllAttributeLabels'};
fileNameMappedAttributes = listFileNamesMappedAttributes{DATASET_ID};
fileNameMappedAttributesLabels = listFileNamesMappedAttributesLabels{DATASET_ID};
numberOfSamplesPerTrainClass = 92;

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
            col1=[];tmp1=[];
        end
        
        %create new data contianing only non-left-out samples
        %leaveOutData = vggFeatures(:,find(tempB == 1));
        if useKernelisedData
            leaveOutData = kernelData(tempC, tempC);
        else
            leaveOutData = vggFeaturesTraining(:, tempC);
        end
        
        %Train regressor
        mappedAttributes = functionTrainRegressor(leaveOutData', leaveOutDatasetLabels, ...
            attributesMat, BASE_PATH, useKernelisedData);
        mappedAllAttributes = [mappedAllAttributes; mappedAttributes];
        mappedAllAttributeLabels = [mappedAllAttributeLabels; mappedAttributeLabels];
        leaveOutDatasetLabels = [];
        tempB = [];
        leaveOutData = [];
        attributesMat = [];
        mappedAttributeLabels = [];
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

% Plot clustered points
funtionTSNEVisualisation(mappedAllAttributes', mappedAllAttributeLabels', length(defaultTrainClassLabels));


%% START >> Training
validClusterIndex = 1;
validClusterIndices = [];

for clusterIndex = 1:numberOfClusters
    allClassesInCluster = find(ssClusteringModel.classClusterAssignment(:, 1) == clusterIndex);
    trainClassIndex = find(ismember(defaultTrainClassLabels, allClassesInCluster));
    %testClassIndex = defaultTestClassLabels; %find(ismember(defaultTestClassLabels, allClassesInCluster));
    trainClassLabels = defaultTrainClassLabels(trainClassIndex);
    testClassLabels = defaultTestClassLabels;% defaultTestClassLabels(testClassIndex);
    clusterInfo(clusterIndex).trainClasses = [trainClassLabels zeros(1, NUMBER_OF_CLASSES - length(trainClassLabels))];
    clusterInfo(clusterIndex).testClasses = [testClassLabels zeros(1, NUMBER_OF_CLASSES - length(testClassLabels))];
    
    if ~isempty(trainClassLabels)
        %%  train
        % Templates are empirical mean embedding per class
        Templates = zeros(size(vggFeatures,1), length(trainClassLabels), 'single');
        for i = 1:length(trainClassLabels)
            Templates(:,i) = mean(vggFeatures(:, datasetLabels==trainClassLabels(i)), 2);
        end
        
        %% source domain
        A = attributes;
        
        %%% linear kernel   (H + 1e-3)
        %A = A ./ repmat(sqrt(sum(A.^2, 2))+eps, [1 size(A,2)]);
        A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);%original
        A = A' * A;
        
        %%% projection
        H = A(trainClassLabels, trainClassLabels);
        F = A(trainClassLabels, :);
        alpha = zeros(length(trainClassLabels), size(F,2));
        
        %%% qpas is a quadratic programming solver.
        %You can change it to any QP solver you have (e.g. the default matlab QP solver)
        % using only equality constrint and lower bound
        
        for i = 1:size(F,2)
            f = -F(:,i);
            %alpha(:,i) = qpas(double(H + 1e1*eye(size(H,2))), double(f),[],[], ...
            %    ones(1,length(f)),1,zeros(length(f),1));
            alpha(:, i) = quadprog(double(H + 1e1*eye(size(H,2))), double(f), [], [], ones(1,length(f)), 1, zeros(length(f),1));
        end
        
        %%  target domain
        train_id = find(ismember(datasetLabels, trainClassLabels));
        x = zeros(4096, length(trainClassLabels), length(trainClassLabels), 'single');
        for i = 1:length(train_id)
            %     d = min(repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)]), Templates);    % intersection
            d = max(0, repmat(vggFeatures(:,train_id(i)), [1 size(Templates,2)])-Templates);    % ReLU
            x(:,:,datasetLabels(train_id(i))==trainClassLabels) = ...
                x(:,:,datasetLabels(train_id(i))==trainClassLabels) + single(d*alpha(:,trainClassLabels));
        end
        
        y = [];
        
        for i = 1:length(trainClassLabels)
            d = -ones(size(alpha,2), 1);
            d(trainClassLabels(i)) = 1;
            y = [y; d(trainClassLabels)];
        end
        x = reshape(x, size(x,1), []);
        
        %%% train svms
        %%% svmocas is an svm solver. You can change it to any svm solver you have (e.g. liblinear)
        
        maxval = max(abs(x(:)));
        %rand_id = randsample(find(y==-1), 50);
        
        %[w b stat] = svmocas(x./maxval, 1, double(y), 1e1);
        features = x./maxval;
        
        svmModel = train(double(y), sparse(double(features')), '-c 10');
        w = (svmModel.w)';
        
        %%% update models
        % Alternately learn on Templates and w
        for iter = 1:1
            iter
            %%% gradient
            grad = zeros(length(w), size(alpha,1));
            for i = 1:length(train_id)
                %         d = min(repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)]), Templates);
                d = max(0, repmat(vggFeatures(:,train_id(i)), [1 size(Templates,2)])-Templates);
                val = (w' * d) * alpha;
                y = -ones(1, size(alpha,2));
                y(datasetLabels(train_id(i))) = 1;
                dec = single(val.*y<0);
                if any(dec==1)
                    grad = grad + (w*((dec.*y)*alpha'))...
                        .* single(repmat(vggFeatures(:,train_id(i)), [1 size(alpha,1)])<Templates);     % gradient needs to be adjusted for intersection
                end
            end
            %     Templates = max(0, Templates + 1e-2*grad./length(train_id));     % no l1: 1e0, with l1: 1e-3
            Templates = max(0, Templates - 1e-3*grad./length(train_id));
            
            
            %%% sample data
            x = zeros(4096, length(trainClassLabels), length(trainClassLabels), 'single');
            for i = 1:length(train_id)
                %         d = min(repmat(cnn_feat(:,train_id(i)), [1 size(Templates,2)]), Templates);
                d = max(0, repmat(vggFeatures(:,train_id(i)), [1 size(Templates,2)])-Templates);
                x(:,:,datasetLabels(train_id(i))==trainClassLabels) = x(:,:,datasetLabels(train_id(i))==trainClassLabels) + single(d*alpha(:,trainClassLabels));
            end
            y = [];
            for i = 1:length(trainClassLabels)
                d = -ones(size(alpha,2), 1);
                d(trainClassLabels(i)) = 1;
                y = [y; d(trainClassLabels)];
            end
            x = reshape(x, size(x,1), []);
            
            %%% train svms
            maxval = max(abs(x(:)));
            %[w b stat] = svmocas(x./maxval, 1, double(y(:)), 1e1);
            
            features = x./maxval;
            svmModel = train(double(y), sparse(double(features')), '-c 10');
            w = (svmModel.w)';
        end
        
        clusterInfo(validClusterIndex).w = w;
        clusterInfo(validClusterIndex).alpha = alpha;
        clusterInfo(validClusterIndex).Templates = Templates;
        validClusterIndex = validClusterIndex + 1;
        validClusterIndices = [validClusterIndices clusterIndex];
    else
        sprintf('Skipping cluster %d', clusterIndex)
    end
end
numberOfValidClusters = validClusterIndex - 1;
%% END >> Training


%% START >> Testing
margins = [];
test_id = find(ismember(datasetLabels, testClassLabels));

for i = 1:length(test_id)
    
    %Find the cluster to which test sample belongs
    %tmp = ssClusteringModel.clusterCenters;
    %clusterCenters = tmp(:, validClusterIndices);
    %distMat =  clusterCenters - repmat(vggFeatures(:,test_id(i)), 1, numberOfValidClusters);
    %[distance clusterAssignment] = min(sqrt(sum(distMat.^2, 1)));
    %distance = sqrt(sum(distMat.^2, 1));
    %distance = distance./sum(distance);
    %weighted_d = 0;
    scoresAcrossClusters = [];
    for clusterIndex = 1:numberOfValidClusters
        Templates = clusterInfo(clusterIndex).Templates;
        w = clusterInfo(clusterIndex).w;
        alpha = clusterInfo(clusterIndex).alpha;
        d = max(0, repmat(vggFeatures(:,test_id(i)), [1 size(Templates,2)])- Templates);
        d = w' * d * alpha(:,testClassLabels);
        %weighted_d = weighted_d + d * distance(clusterIndex);
        scoresAcrossClusters = [scoresAcrossClusters; d];
    end
    
    margins = [margins; max(scoresAcrossClusters, [], 1)];%weighted_d];
end
%%% classify
[margin id] = max(margins, [], 2);
a = (testClassLabels(id));
b = datasetLabels(test_id);
if ~sum(size(a) == size(b))
    a = a';
end
acc = sum(a == b)/length(test_id)
margins = [];
meanAcc = mean(acc)
%% END >> Testing


%% START >> Save results
results.clusterInfo = clusterInfo;
results.accuracy = meanAcc;
results.numberOfClusters = numberOfClusters;
results.numberOfValidClusters = numberOfValidClusters;
%save(sprintf('%s/data/code-data/semantic-similarity/results/results_itr_%d_AwA_clstrs_%d.mat', ...
%    BASE_PATH, expIter, numberOfClusters), 'results');
%% END >> Save results

