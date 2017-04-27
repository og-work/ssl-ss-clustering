function outHistogramUnseenClasses = functionGetSourceDomainEmbedding(inTestClassLabels, inTrainClassLabels, inAttributes)

outHistogramUnseenClasses = zeros(size(inAttributes, 2), length(inTrainClassLabels));

for p = 1:length(inTestClassLabels)
    diff = repmat(inAttributes(:, inTestClassLabels(p)), 1, length(inTrainClassLabels)) - inAttributes(:, inTrainClassLabels);
    outHistogramUnseenClasses(inTestClassLabels(p), :) = sum(diff.^2, 1)/sum(sum(diff.^2, 1));
end
outHistogramUnseenClasses = outHistogramUnseenClasses';

if 0
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

end
