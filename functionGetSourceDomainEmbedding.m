function outHistogramAllClasses = functionGetSourceDomainEmbedding(inVggFeatures, inTrainClassLabels, ...
    inDatasetLabels, inAttributes)

% Templates are phi(s) i.e. target domain embedding  function/mapping
Templates = zeros(size(inVggFeatures,1), length(inTrainClassLabels), 'single');
for i = 1:length(inTrainClassLabels)
    Templates(:,i) = mean(inVggFeatures(:, inDatasetLabels==inTrainClassLabels(i)), 2);
end

%% source domain
A = inAttributes;

%%% linear kernel   (H + 1e-3)
%A = A ./ repmat(sqrt(sum(A.^2, 2))+eps, [1 size(A,2)]);
A = A ./ repmat(sqrt(sum(A.^2, 1))+eps, [size(A,1) 1]);%original
A = A' * A;

%%% projection
H = A(inTrainClassLabels, inTrainClassLabels);
F = A(inTrainClassLabels, :);
alpha = zeros(length(inTrainClassLabels), size(F,2));

%%% qpas is a quadratic programming solver.
%You can change it to any QP solver you have (e.g. the default matlab QP solver)
% using only equality constrint and lower bound

for i = 1:size(F,2)
    f = -F(:,i);
    %alpha(:,i) = qpas(double(H + 1e1*eye(size(H,2))), double(f),[],[], ...
    %    ones(1,length(f)),1,zeros(length(f),1));
    alpha(:, i) = quadprog(double(H + 1e1*eye(size(H,2))), double(f), [], [], ones(1,length(f)), 1, zeros(length(f),1));
end

outHistogramAllClasses = alpha;
