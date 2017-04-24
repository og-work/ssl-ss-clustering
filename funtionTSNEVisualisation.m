% Author: Omkar Anil Gune, IIT Bombay, India
% This is a wrapper function.
% This function is used to visualise higher dimension data using T-SNE tool.
% Inputs::
% inFeatures:
% D x N matrix with N data points each of D dimension
% inLabels:
% 1 vector containing labels of each of N data points.
% inNumberOfClasses:
% Number of classes in data

function funtionTSNEVisualisation(inFeatures, inLabels, inNumberOfClasses)
%Set parameters
no_dims = 2;
initial_dims = 50;
perplexity = 30;
% Run t-SNE
mappedX = tsne(inFeatures', [], no_dims, initial_dims, perplexity);
% Plot results
markerString = '';

for p = 1:inNumberOfClasses
    if rem(p, 5)== 0
        markerString = strcat(markerString, 'o');
    elseif rem(p, 5) == 1
        markerString = strcat(markerString, 'x');
    elseif rem(p, 5) == 2
        markerString = strcat(markerString, 'd');
    elseif rem(p, 5) == 3
        markerString = strcat(markerString, 's');
    elseif rem(p, 5) == 4
        markerString = strcat(markerString, 'h');
    end
end

colorMap = hsv(inNumberOfClasses);

figure;
gscatter(mappedX(:,1), mappedX(:,2), inLabels, colorMap, markerString);
