
clear 
clc
close all
% 1: Linux Laptop, 2: Windows laptop, 3: Linux Desktop 4: Windows Desktop
    SYSTEM_PLATFORM = 1;
    BASE_PATH = '';
    listDatasets = {'AwA', 'Pascal-Yahoo'};
    DATASET = listDatasets{1};
    addPath = 1;
    BASE_PATH = functionSemantic_similaity_env_setup(SYSTEM_PLATFORM, addPath);
dataset_path = sprintf('%s/data/code-data/semantic-similarity/cnn-features/AwA/', BASE_PATH);;

load(sprintf('%s/classes.mat', dataset_path));
load(sprintf('%s/predicateMatrixContinuous.mat', dataset_path));     % better
% load(sprintf('%s/predicateMatrixBinary.mat', dataset_path));
load(sprintf('%s/trainTestSplit.mat', dataset_path));

%%
if exist(sprintf('%s/feat-imagenet-vgg-verydeep-19.mat', dataset_path))
    load(sprintf('%s/feat-imagenet-vgg-verydeep-19.mat', dataset_path), 'train_feat', 'train_labels', 'test_feat', 'test_labels');
    train_feat = train_feat ./ repmat(sum(train_feat, 1), [size(train_feat,1) 1]);
    test_feat = test_feat ./ repmat(sum(test_feat, 1), [size(test_feat,1) 1]);
else
    load(sprintf('%s/matFiles/feat-imagenet-vgg-verydeep-19.mat', dataset_path));

    %  load features
    feat_names = {'imagenet-vgg-verydeep-19'};    %   best

    %%% training
    train_feat = [];
    train_labels = [];
    parfor i = 1:length(trainClassLabels)
        X = [];
        for j = 1:length(feat_names)
            x = [];
            files = dir(sprintf('%s/Features/%s/%s/*.mat', dataset_path, feat_names{j}, trainClasses{1}{i}));
            files(1) = [];
            for k = 1:length(files)
                d = myLoadCNN(sprintf('%s/Features/%s/%s/%s', dataset_path, feat_names{j}, ...
                    trainClasses{1}{i}, files(k).name));
                x = [x d];            
            end
            X = [X; x];
        end
        train_feat = [train_feat X];
        train_labels = [train_labels; trainClassLabels(i)*ones(length(files), 1)];
    end

    %%% test
    test_feat = [];
    test_labels = [];
    parfor i = 1:length(testClassLabels)
        X = [];
        for j = 1:length(feat_names)
            x = [];
            files = dir(sprintf('%s/Features/%s/%s/*.mat', dataset_path, feat_names{j}, testClasses{1}{i}));
            files(1) = [];
            for k = 1:length(files)
                d = myLoadCNN(sprintf('%s/Features/%s/%s/%s', dataset_path, feat_names{j}, ...
                    testClasses{1}{i}, files(k).name));
                x = [x d];                        
            end
            X = [X; x];
        end
        test_feat = [test_feat X];
        test_labels = [test_labels; testClassLabels(i)*ones(length(files), 1)];
    end

    feat_names = {'decaf'};

    %%% training
    train_feat = [];
    train_labels = [];
    parfor i = 1:length(trainClassLabels)
        X = [];
        for j = 1:length(feat_names)
            x = [];
            files = dir(sprintf('%s/Features/%s/%s/*.txt', dataset_path, feat_names{j}, trainClasses{1}{i}));
            for k = 1:length(files)
                fid = fopen(sprintf('%s/Features/%s/%s/%s', dataset_path, feat_names{j}, ...
                    trainClasses{1}{i}, files(k).name));
                txt = textscan(fid, '%f');
                d = single(txt{1});
                d = d ./ sum(d,1);
                x = [x d];
                fclose(fid);
            end
            X = [X; x];
        end
        train_feat = [train_feat X];
        train_labels = [train_labels; trainClassLabels(i)*ones(length(files), 1)];
    end

    %%% test
    test_feat = [];
    test_labels = [];
    parfor i = 1:length(testClassLabels)
        X = [];
        for j = 1:length(feat_names)
            x = [];
            files = dir(sprintf('%s/Features/%s/%s/*.txt', dataset_path, feat_names{j}, testClasses{1}{i}));
            for k = 1:length(files)
                fid = fopen(sprintf('%s/Features/%s/%s/%s', dataset_path, feat_names{j}, ...
                    testClasses{1}{i}, files(k).name));
                txt = textscan(fid, '%f');
                d = single(txt{1});
                d = d ./ sum(d,1);
                x = [x d];            
                fclose(fid);
            end
            X = [X; x];
        end
        test_feat = [test_feat X];
        test_labels = [test_labels; testClassLabels(i)*ones(length(files), 1)];
    end

    save(sprintf('%s/matFiles/feat.mat', dataset_path), 'train_feat', 'train_labels', 'test_feat', 'test_labels', '-v7.3');
end

%%  train
Templates = zeros(size(train_feat,1), length(trainClassLabels), 'single');
for i = 1:length(trainClassLabels)
    Templates(:,i) = mean(train_feat(:,train_labels==trainClassLabels(i)), 2);
end

%% source domain
% A = single(predicateMatrixBinary');
A = single(predicateMatrixContinuous');
B = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);
B = B' * B;

%%% liniear kernel (H + 1e1)
A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);
A = A' * A;

%%% projection
H = A(trainClassLabels, trainClassLabels);
F = A(trainClassLabels, :);
alpha = zeros(length(trainClassLabels), size(F,2));
for i = 1:size(F,2)
    f = -F(:,i);    
   % alpha(:,i) = qpas(double(H + 1e1*eye(size(H,2))),double(f),[],[], ...
   %     ones(1,length(f)),1,zeros(length(f),1));
    alpha(:, i) = quadprog(double(H + 1e1*eye(size(H,2))), double(f), [], [], ones(1,length(f)), 1, zeros(length(f),1));

end

%%  target domain
Ns = 1;
%Ns = 2;
mkdir(sprintf('%s/tmp/svm_train_data/', dataset_path));
allVectorsd = [];
x = [];

parfor i = 1:length(train_labels)
    d = min(repmat(train_feat(:,i), [1 size(Templates,2)]), Templates);
%     d = max(0, repmat(train_feat(:,i), [1 size(Templates,2)])-Templates);
    
    L = 1:50;
    L(L==train_labels(i)) = [];
    cid = randsample(L, Ns);    
    
    %d = d*(repmat(alpha(:,train_labels(i)), [1 Ns])-alpha(:,cid)) ...
    %    ./repmat(1-B(train_labels(i), cid), [size(d,1) 1]);
    d = d*(repmat(alpha(:,train_labels(i)), [1 Ns])-alpha(:,cid)) ...
        ./repmat(1-B(train_labels(i), cid), [size(d,1) 1]);
    mySaveSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i), d);
    %x = [x, d];
    i;
end

parfor i = 1:length(train_labels)
    d = myLoadSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i));
    x = [x, d.d];
    i
end

maxval = max(abs(x(:)));
x = x./maxval;
y = ones(size(x,2), 1);
sid = randsample(length(y), round(length(y)/2));
x(:,sid) = -x(:,sid);
y(sid) = -y(sid);

%%% train svms
%[w b stat] = svmocas(x, 1, double(y), 1e0);
features = x;
svmModel = train(double(y), sparse(double(features')), '-c 1');
w = (svmModel.w)';

% % %%%  test
% % margins = [];
% % for i = 1:length(test_labels)    
% %     d = min(repmat(test_feat(:,i), [1 size(Templates,2)]), Templates);
% % %     d = max(0, repmat(test_feat(:,i), [1 size(Templates,2)])-Templates); 
% %     d = w' * d * alpha(:,testClassLabels);
% %     margins = [margins; d];
% % end
% % %%% classify
% % [margin id] = max(margins, [], 2);
% % ACC = sum(testClassLabels(id)==test_labels)/length(test_labels);

for iter = 1:5
    
    %%% gradient
    grad = zeros(length(w), size(alpha,1));
    for i = 1:length(train_labels)   
        d = min(repmat(train_feat(:,i), [1 size(Templates,2)]), Templates); 
%         d = max(0, repmat(train_feat(:,i), [1 size(Templates,2)])-Templates); 
        val = (w' * d) * alpha;
        val = val(train_labels(i)) - val;
        dec = single(val<0);
        if any(dec==1)
            grad = grad + (w*(alpha(:,train_labels(i))-alpha*dec'/sum(dec))')...
                .* single(repmat(train_feat(:,i), [1 size(alpha,1)])<Templates);            
        end        
    end
    Templates = max(0, Templates + 1e-2*grad./length(train_labels));
%     Templates = max(0, Templates - 1e-2*grad./length(train_labels));
        

    %%% sample data
    Ns = 1;
    mkdir(sprintf('%s/tmp/svm_train_data/', dataset_path));
    parfor i = 1:length(train_labels)
        d = min(repmat(train_feat(:,i), [1 size(Templates,2)]), Templates);
%         d = max(0, repmat(train_feat(:,i), [1 size(Templates,2)])-Templates); 
        
        L = 1:50;
        L(L==train_labels(i)) = [];
        cid = randsample(L, Ns);    

        d = d*(repmat(alpha(:,train_labels(i)), [1 Ns])-alpha(:,cid)) ...
            ./repmat(1-B(train_labels(i), cid), [size(d,1) 1]);
        mySaveSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i), d);

    end

    x = [];
    parfor i = 1:length(train_labels)
        d = myLoadSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i));
        x = [x, d.d];
    end

    maxval = max(abs(x(:)));
    x = x./maxval;
    y = ones(size(x,2), 1);
    sid = randsample(length(y), round(length(y)/2));
    x(:,sid) = -x(:,sid);
    y(sid) = -y(sid);

    %%% train svms
    %[w b stat] = svmocas(x, 1, double(y), 1e0);
    %features = x./maxval;
    svmModel = train(double(y), sparse(double(features')), '-c 1');
    w = (svmModel.w)';


%     %%  test
%     margins = [];
%     for i = 1:length(test_labels)    
%         d = min(repmat(test_feat(:,i), [1 size(Templates,2)]), Templates);
% %         d = max(0, repmat(test_feat(:,i), [1 size(Templates,2)])-Templates); 
%         d = w' * d * alpha(:,testClassLabels);
%         margins = [margins; d];
%     end
%     %%% classify
%     [margin id] = max(margins, [], 2);
%     acc = sum(testClassLabels(id)==test_labels)/length(test_labels);
%     
%     ACC = [ACC, acc]
end

    %%  test
    margins = [];
    for i = 1:length(test_labels)    
        d = min(repmat(test_feat(:,i), [1 size(Templates,2)]), Templates);
%         d = max(0, repmat(test_feat(:,i), [1 size(Templates,2)])-Templates); 
        d = w' * d * alpha(:,testClassLabels);
        margins = [margins; d];
    end
    %%% classify
    [margin id] = max(margins, [], 2);
    acc = sum(testClassLabels(id)==test_labels)/length(test_labels);
    
   

% %%  train
% Templates = zeros(size(train_feat,1), length(trainClassLabels), 'single');
% for i = 1:length(trainClassLabels)
%     Templates(:,i) = mean(train_feat(:,train_labels==trainClassLabels(i)), 2);
% end
% 
% %%% encode classes in target domain
% A = predicateMatrixContinuous';
% % A = single(predicateMatrixBinary');
% A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);
% A = A' * A;
% H = A(trainClassLabels, trainClassLabels);
% F = A(trainClassLabels, :);
% alpha = zeros(length(trainClassLabels), size(F,2));
% for i = 1:size(F,2)
%     f = -F(:,i);    
%     alpha(:,i) = qpas(double(H + 1e-2*eye(size(H,2))),double(f),[],[], ...
%         ones(1,length(f)),1,zeros(length(f),1));    
% end
% alpha(abs(alpha)<1e-3) = 0;
% alpha = alpha ./ repmat(sum(alpha,1), [size(alpha,1) 1]);
% 
% %%% sample data
% % Ns = 1;
% Ns = 2;
% mkdir(sprintf('%s/tmp/svm_train_data/', dataset_path));
% parfor i = 1:length(train_labels)
%     d = min(repmat(train_feat(:,i), [1 size(Templates,2)]), Templates);
%     L = 1:50;
%     L(L==train_labels(i)) = [];
%     cid = randsample(L, Ns);    
%     
%     d = d*(repmat(alpha(:,train_labels(i)), [1 Ns])-alpha(:,cid)) ...
%         ./repmat(1-A(train_labels(i), cid), [size(d,1) 1]);
%     mySaveSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i), d);
% 
% %     x = [];
% %     for j = 1:Ns
% %         d1 = d .* repmat(alpha(:,train_labels(i))'-alpha(:,cid(j))', [size(d,1) 1]);
% %         x = [x single(d1(:)./(1-A(train_labels(i), cid(j))))];
% %     end
% %     mySaveSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i), sparse(double(x)));
% end
% 
% x = [];
% parfor i = 1:length(train_labels)
%     d = myLoadSVMData(sprintf('%s/tmp/svm_train_data/%d.mat', dataset_path, i));
%     x = [x, d];
% end
% 
% % x = single(full(x));
% maxval = max(abs(x(:)));
% x = x./maxval;
% y = ones(size(x,2), 1);
% sid = randsample(length(y), round(length(y)/2));
% x(:,sid) = -x(:,sid);
% y(sid) = -y(sid);
% % K = [(1:size(x,2))' x'*x];
% % cmd = '-t 4 -c 5 -h 0';
% % model = svmtrain(y, double(K), cmd);
% % w = x(:,model.sv_indices) * model.sv_coef;
% 
% % y = ones(size(x,2), 1);
% % sid = randsample(length(y), round(length(y)/2));
% % x(:,sid) = -x(:,sid);
% % y(sid) = -y(sid);
% % K = [(1:size(x,2))' x'*x];
% % cmd = '-t 4 -c 5';
% % model = svmtrain(y, double(K), cmd);
% % w = x(:,model.sv_indices) * model.sv_coef;
% 
% % y = ones(size(x,2), 1);
% % sid = randsample(length(y), round(length(y)/2));
% % x(:,sid) = -x(:,sid);
% % y(sid) = -y(sid);
% % 
% % %%% train svms
% % maxval = max(abs(x(:)));
% % x = x./maxval;
% C = 1e3*ones(length(y), 1);
% C(y==1) = C(y==1) * sum(y==-1)/sum(y==1);    
% [w b stat] = svmocas(x, 1, double(y), C);
% % w = w ./ maxval;
% % save(sprintf('%s/results/w.mat', dataset_path), 'w', 'b', '-v7.3');
% % clear x;
% 
% %%  test
% margins = [];
% parfor i = 1:length(test_labels)    
%     d = w' * min(repmat(test_feat(:,i), [1 size(Templates,2)]), Templates);
% %     d = sum(w .* min(repmat(test_feat(:,i), [1 size(Templates,2)]), Templates), 1);
%     d = d * alpha(:, testClassLabels);    
%     margins = [margins; d];
% end
% %%% classify
% [margin id] = max(margins, [], 2);
% acc = sum(testClassLabels(id)==test_labels)/length(test_labels)
%     
% % save(sprintf('%s/results/result.mat', dataset_path), 'acc', 'margins', '-v7.3');
% 
% 
