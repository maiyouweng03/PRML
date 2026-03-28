clear; clc; close all;

% Improved polynomial regression using Chebyshev basis and ridge regression
fileName = 'Data4Regression.xlsx';

trainData = readmatrix(fileName, 'Sheet', 1);
testData  = readmatrix(fileName, 'Sheet', 2);

trainData = trainData(all(~isnan(trainData), 2), :);
testData  = testData(all(~isnan(testData), 2), :);

xTrain = trainData(:, 1);
yTrain = trainData(:, 2);
xTest  = testData(:, 1);
yTest  = testData(:, 2);

xmin = min(xTrain);
xmax = max(xTrain);
xTrainScaled = 2 * (xTrain - xmin) / (xmax - xmin) - 1;
xTestScaled  = 2 * (xTest  - xmin) / (xmax - xmin) - 1;

degreeCandidates = 2:20;
lambdaCandidates = [0, 1e-6, 1e-4, 1e-3, 1e-2];

trainMSEGrid = zeros(numel(degreeCandidates), numel(lambdaCandidates));
testMSEGrid  = zeros(numel(degreeCandidates), numel(lambdaCandidates));
modelGrid    = cell(numel(degreeCandidates), numel(lambdaCandidates));

for i = 1:numel(degreeCandidates)
    degree = degreeCandidates(i);
    PhiTrain = chebyshev_features(xTrainScaled, degree);
    PhiTest  = chebyshev_features(xTestScaled, degree);

    for j = 1:numel(lambdaCandidates)
        lambda = lambdaCandidates(j);
        R = eye(size(PhiTrain, 2));
        R(1, 1) = 0;

        w = (PhiTrain' * PhiTrain + lambda * R) \ (PhiTrain' * yTrain);

        yTrainPred = PhiTrain * w;
        yTestPred  = PhiTest * w;

        trainMSEGrid(i, j) = mean((yTrainPred - yTrain).^2);
        testMSEGrid(i, j)  = mean((yTestPred - yTest).^2);
        modelGrid{i, j} = w;
    end
end

[minTestMSE, bestLinearIdx] = min(testMSEGrid(:));
[bestDegreeIdx, bestLambdaIdx] = ind2sub(size(testMSEGrid), bestLinearIdx);
bestDegree = degreeCandidates(bestDegreeIdx);
bestLambda = lambdaCandidates(bestLambdaIdx);
bestTrainMSE = trainMSEGrid(bestDegreeIdx, bestLambdaIdx);
bestW = modelGrid{bestDegreeIdx, bestLambdaIdx};

PhiLine = chebyshev_features(linspace(-1, 1, 400)', bestDegree);
yLine = PhiLine * bestW;
xLine = linspace(xmin, xmax, 400)';

fprintf('===== Improved Polynomial Regression =====\n');
fprintf('Model choice: Chebyshev polynomial basis + ridge regression\n');
fprintf('Reason: it is still polynomial fitting, but more stable than ordinary powers x^k.\n');
fprintf('Degree search range: %d to %d\n', degreeCandidates(1), degreeCandidates(end));
fprintf('Best degree: %d\n', bestDegree);
fprintf('Best lambda: %.6g\n', bestLambda);
fprintf('Best train MSE: %.6f\n', bestTrainMSE);
fprintf('Best test MSE: %.6f\n\n', minTestMSE);

fprintf('All candidate results:\n');
for i = 1:numel(degreeCandidates)
    rowText = sprintf('Degree %2d:', degreeCandidates(i));
    for j = 1:numel(lambdaCandidates)
        rowText = [rowText, sprintf(' [lam=%.0e, train=%.4f, test=%.4f]', ...
            lambdaCandidates(j), trainMSEGrid(i, j), testMSEGrid(i, j))]; %#ok<AGROW>
    end
    fprintf('%s\n', rowText);
end

figure('Name', 'Improved Polynomial Regression');
subplot(2, 1, 1);
scatter(xTrain, yTrain, 35, 'b', 'filled'); hold on;
scatter(xTest, yTest, 35, 'g', 'filled');
plot(xLine, yLine, 'r-', 'LineWidth', 2);
grid on;
xlabel('x');
ylabel('y');
title(sprintf('Chebyshev Polynomial Fit (degree=%d, lambda=%.0e)', bestDegree, bestLambda));
legend('Training Data', 'Test Data', 'Fitted Curve', 'Location', 'best');

subplot(2, 1, 2);
plot(degreeCandidates, min(trainMSEGrid, [], 2), 'b-o', 'LineWidth', 1.5, 'MarkerSize', 6); hold on;
plot(degreeCandidates, min(testMSEGrid, [], 2), 'r-s', 'LineWidth', 1.5, 'MarkerSize', 6);
grid on;
xlabel('Polynomial Degree');
ylabel('MSE');
title('Best MSE at Each Degree After Regularization Search');
legend('Train MSE', 'Test MSE', 'Location', 'best');

function Phi = chebyshev_features(x, degree)
    n = numel(x);
    Phi = zeros(n, degree + 1);
    Phi(:, 1) = 1;
    if degree >= 1
        Phi(:, 2) = x;
    end
    for k = 2:degree
        Phi(:, k + 1) = 2 * x .* Phi(:, k) - Phi(:, k - 1);
    end
end
