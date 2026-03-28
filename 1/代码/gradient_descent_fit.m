clear; clc; close all;

% 梯度下降法线性拟合
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

[n, d] = size(XTrain);
w = zeros(d, 1);

H = (XTrain' * XTrain) / n;
learningRate = 1 / (max(eig(H)) + 1e-8);
maxIter = 5000;
tol = 1e-8;

lossHistory = zeros(maxIter, 1);

for iter = 1:maxIter
    grad = (XTrain' * (XTrain * w - yTrain)) / n;
    wNew = w - learningRate * grad;
    lossHistory(iter) = mean((XTrain * wNew - yTrain).^2) / 2;

    if norm(wNew - w, 2) < tol
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

fprintf('===== 梯度下降法线性拟合 =====\n');
fprintf('模型: y = %.6f + %.6f * x\n', w(1), w(2));
fprintf('学习率: %.6f\n', learningRate);
fprintf('迭代次数: %d\n', numel(lossHistory));
fprintf('训练误差 MSE: %.6f\n', trainMSE);
fprintf('测试误差 MSE: %.6f\n', testMSE);

xLine = linspace(min([xTrain; xTest]), max([xTrain; xTest]), 400)';
yLine = [ones(size(xLine)), xLine] * w;

figure('Name', 'Gradient Descent Linear Regression');
subplot(1, 2, 1);
scatter(xTrain, yTrain, 35, 'b', 'filled'); hold on;
scatter(xTest, yTest, 35, 'g', 'filled');
plot(xLine, yLine, 'r-', 'LineWidth', 2);
grid on;
xlabel('x');
ylabel('y');
title(sprintf('GD Linear Fit (Train MSE = %.4f, Test MSE = %.4f)', trainMSE, testMSE));
legend('Training Data', 'Test Data', 'Linear Fit', 'Location', 'best');

subplot(1, 2, 2);
plot(lossHistory, 'm-', 'LineWidth', 1.8);
grid on;
xlabel('Iteration');
ylabel('Training Loss');
title('Gradient Descent Convergence');
