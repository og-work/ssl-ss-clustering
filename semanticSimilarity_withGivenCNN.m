
clc;
clear
close all;
USE_LAPTOP = 1;
BASE_PATH = '';

if USE_LAPTOP
    %For laptop
    BASE_PATH = '/home/omy/Documents/omkar-server-backup-20feb2017/Documents/Study/PhD-Research/';
else
    %For desktop
    BASE_PATH = '/nfs4/omkar/Documents/study/phd-research/';
end
addpath(genpath(sprintf('%s/codes/third-party-softwares/ocas/libocas_v097/', BASE_PATH)));
addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/svm/liblinear', BASE_PATH)));
run(sprintf('%s/codes/third-party-softwares-codes/VL-feat/VLFEATROOT/vlfeat-0.9.20/toolbox/vl_setup', BASE_PATH));
disp('Enabling VL feat toolbox');
addpath(genpath(sprintf('%s/codes/third-party-softwares-codes/t-sne/', BASE_PATH)));

defaultTrainClassLabels = [];
defaultTestClassLabels = [];
if 1
    dataset_path = sprintf('%s/codes/matlab-stuff/tree-based-zsl/Demo/Data/AwA/', BASE_PATH);
    %'data/code-data/semantic-similarity/cnn-features
    temp = load(sprintf('%s/data/code-data/semantic-similarity/cnn-features/AwA/feat-imagenet-vgg-verydeep-19.mat', BASE_PATH));
    %temp = load(sprintf('%s/AwA_All_vgg19Features.mat', dataset_path));
    vggFeatures = [temp.train_feat temp.test_feat];
    datasetLabels = [temp.train_labels; temp.test_labels];
    attributes = load(sprintf('%s/AwA_All_ClassLabelPhraseDict.mat', dataset_path));
    %temp = load(sprintf('%s/AwA_All_DatasetLabels.mat', dataset_path));
    %datasetLabels = temp.datasetLabels;
end

%vggFeatures = vggFeatures';

%% Start >> Clustering of data
numberOfClusters = 1;
NUMBER_OF_CLASSES = 50;
clusteringModel = functionClusterData(vggFeatures, datasetLabels, numberOfClusters, NUMBER_OF_CLASSES);
%% End >> Clustering of data


k=1;
labels = zeros(1, 50);
%Default setting of AwA
%defaultTrainClassLabels = [1:7, 9:16, 18:21, 23, 25, 27:32, 35:36, 39, 41:50];
%defaultTestClassLabels = [8 17 22 24 26 33 34 37 38 40]; % From dataset
defaultTestClassLabels = [25 39 15 6 42 14 18 48 34 24];
labels(defaultTestClassLabels) = 1;
labels = 1. - labels;
defaultTrainClassLabels = find(labels);
clusterInfo = zeros(NUMBER_OF_CLASSES, 2*numberOfClusters);

for i = 1:numberOfClusters
    
    allClassesInCluster = find(clusteringModel(:, 1) == i);
    trainClassIndex = find(ismember(defaultTrainClassLabels, allClassesInCluster));
    testClassIndex = find(ismember(defaultTestClassLabels, allClassesInCluster));
    trainClassLabels = defaultTrainClassLabels(trainClassIndex);
    testClassLabels = defaultTestClassLabels(testClassIndex);
    clusterInfo(:, 2*(k - 1) + 1) = [trainClassLabels zeros(1, NUMBER_OF_CLASSES - length(trainClassLabels))];
    clusterInfo(:, 2*(k - 1) + 2) = [testClassLabels zeros(1, NUMBER_OF_CLASSES - length(testClassLabels))];
  
    if ~isempty(testClassLabels)
        %%  train
        % Templates are empirical mean embedding per class
        Templates = zeros(size(vggFeatures,1), length(trainClassLabels), 'single');
        for i = 1:length(trainClassLabels)
            Templates(:,i) = mean(vggFeatures(:, datasetLabels==trainClassLabels(i)), 2);
        end
        
        %% source domain
        A = attributes.phrasevec_mat';
        
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
        
        %%  test
        margins = [];
        test_id = find(ismember(datasetLabels, testClassLabels));
        for i = 1:length(test_id)
            %     d = min(repmat(cnn_feat(:,test_id(i)), [1 size(Templates,2)]), Templates);
            d = max(0, repmat(vggFeatures(:,test_id(i)), [1 size(Templates,2)])-Templates);
            d = w' * d * alpha(:,testClassLabels);
            margins = [margins; d];
        end
        %%% classify
        [margin id] = max(margins, [], 2);
        a = (testClassLabels(id));
        b = datasetLabels(test_id);
        if ~sum(size(a) == size(b))
            a = a';
        end
        acc(k) = sum(a == b)/length(test_id)
        k=k+1;
        margins = [];
        
    else
       sprintf('Skipping cluster %d', k) 
    end
end

meanAcc = mean(acc)