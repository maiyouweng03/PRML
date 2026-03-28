clear; clc; close all;

% 最小二乘法线性拟合
fileName = 'Data4Regression.xlsx';


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

% 闭式解
w = pinv(XTrain) * yTrain;

yTrainPred = XTrain * w;
yTestPred  = XTest * w;

trainMSE = mean((yTrainPred - yTrain).^2);
testMSE  = mean((yTestPred - yTest).^2);

fprintf('===== 最小二乘法线性拟合 =====\n');
fprintf('模型: y = %.6f + %.6f * x\n', w(1), w(2));
fprintf('训练误差 MSE: %.6f\n', trainMSE);
fprintf('测试误差 MSE: %.6f\n', testMSE);

xLine = linspace(min([xTrain; xTest]), max([xTrain; xTest]), 400)';
yLine = [ones(size(xLine)), xLine] * w;

figure('Name', 'Least Squares Linear Regression');
scatter(xTrain, yTrain, 35, 'b', 'filled'); hold on;
scatter(xTest, yTest, 35, 'g', 'filled');
plot(xLine, yLine, 'r-', 'LineWidth', 2);
grid on;
xlabel('x');
ylabel('y');
title(sprintf('Least Squares Linear Fit (Train MSE = %.4f, Test MSE = %.4f)', trainMSE, testMSE));
legend('Training Data', 'Test Data', 'Linear Fit', 'Location', 'best');
