function functionMyScatterPlot(inMappedData, inLabels, inNumberOfClasses, inFigureTitle)

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
gscatter(inMappedData(:,1), inMappedData(:,2), inLabels, colorMap, markerString);
title(inFigureTitle)

