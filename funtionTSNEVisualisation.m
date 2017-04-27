% Author: Omkar Anil Gune, IIT Bombay, India
% This is a wrapper function.
% This function is used to visualise higher dimension data using T-SNE tool.
% Inputs::
% inFeatures:
% D x N matrix with N data points each of D dimension
% inLabels:
% 1 X N vector containing labels of each of N data points.
% inNumberOfClasses:
% Number of classes in data

function [mappedX] = funtionTSNEVisualisation(inFeatures)
%Set parameters
no_dims = 2;
initial_dims = 50;
perplexity = 30;
% Run t-SNE
mappedX = tsne(inFeatures', [], no_dims, initial_dims, perplexity);
