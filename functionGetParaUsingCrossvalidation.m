function [outC, outGamma] = functionGetParaUsingCrossvalidation(inData, inLabels, inUseKernelisedData)
disp('        Cross Validation ...')
%# grid of parameters
folds = 3;
[C,gamma] = meshgrid(-5:2:15, -15:2:3);
cv_acc = zeros(numel(C) * size(inLabels, 2), 1);
k = 1;

for d = 1:5%size(inLabels, 2)
    for i=1:numel(C)
        tic;
        fprintf('d: %d i: %d', d, i)
        if inUseKernelisedData
            %If precomputed kernel is used                              
            cv_acc(k) = libsvmtrain(inLabels(:,d), [(1:size(inData,1))' inData], sprintf('-s 3 -c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
        else
            %If data is used directly
            cv_acc(k) = libsvmtrain(inLabels(:,d), inData, sprintf('-s 3 -c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
        end
        k = k + 1;
    end
    
    toc;
    fprintf('Finish %d th dimension and %d th itertion out of %d Accuracy: %f\n', d, i, numel(C), cv_acc(k))
end

%pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%contour plot of paramter selection
figure;
contour(C, gamma, reshape(cv_acc,size(C))), colorbar
hold on
plot(C(idx), gamma(idx), 'rx')
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

%# now you can train you model using best_C and best_gamma
outC = 2^C(idx);
outGamma = 2^gamma(idx);