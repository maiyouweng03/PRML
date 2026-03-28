clear; clc; close all;

% 牛顿法线性拟合
fileName = 'Data4Regression.xlsx';
if ~isfile(fileName)
    fileName = 'Data4Regression (1).xlsx';
end

trainData = readmatrix(fileName, 'Sheet', 1);
testData  = readmatrix(fileName, 'Sheet', 2);

trainData = trainData(all(~isnan(trainData), 2), :);
testData  = testData(all(~isnan(testData), 2), :);

xTrain = trainData(:, 1);
yTrain = trainData(:, 2);
xTest  = testData(:, 1);
yTest  = testData(:, 2);

XTrain = [ones(size(xTrain)), xTrain];
XTest  = [ones(size(xTest)),  xTest];

[n, d] = size(XTrain);
w = zeros(d, 1);

maxIter = 20;
tol = 1e-10;
lambda = 1e-10;
lossHistory = zeros(maxIter, 1);

H = (XTrain' * XTrain) / n + lambda * eye(d);

for iter = 1:maxIter
    grad = (XTrain' * (XTrain * w - yTrain)) / n;
    delta = H \ grad;
    wNew = w - delta;
    lossHistory(iter) = mean((XTrain * wNew - yTrain).^2) / 2;

    if norm(delta, 2) < tol
        w = wNew;
        lossHistory = lossHistory(1:iter);
        break;
    end

    w = wNew;
end

if iter == maxIter
    lossHistory = lossHistory(1:maxIter);
end

yTrainPred = XTrain * w;
yTestPred  = XTest * w;

trainMSE = mean((yTrainPred - yTrain).^2);
testMSE  = mean((yTestPred - yTest).^2);

fprintf('===== 牛顿法线性拟合 =====\n');
fprintf('模型: y = %.6f + %.6f * x\n', w(1), w(2));
fprintf('迭代次数: %d\n', numel(lossHistory));
fprintf('训练误差 MSE: %.6f\n', trainMSE);
fprintf('测试误差 MSE: %.6f\n', testMSE);

xLine = linspace(min([xTrain; xTest]), max([xTrain; xTest]), 400)';
yLine = [ones(size(xLine)), xLine] * w;

figure('Name', 'Newton Linear Regression');
subplot(1, 2, 1);
scatter(xTrain, yTrain, 35, 'b', 'filled'); hold on;
scatter(xTest, yTest, 35, 'g', 'filled');
plot(xLine, yLine, 'r-', 'LineWidth', 2);
grid on;
xlabel('x');
ylabel('y');
title(sprintf('Newton Linear Fit (Train MSE = %.4f, Test MSE = %.4f)', trainMSE, testMSE));
legend('Training Data', 'Test Data', 'Linear Fit', 'Location', 'best');

subplot(1, 2, 2);
plot(lossHistory, 'k-o', 'LineWidth', 1.5, 'MarkerSize', 5);
grid on;
xlabel('Iteration');
ylabel('Training Loss');
title('Newton Method Convergence');
