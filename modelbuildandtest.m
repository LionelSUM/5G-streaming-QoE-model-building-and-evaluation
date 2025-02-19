clc;
clear;

 % 输入文件夹地址，然后逗号之后拼接上文件名称
temp = dir(['D:\Course_Material\Undergraduate\毕设\test\waterloo_sqoe3_full\testcode\','*.csv']);
N = length(temp);  % 文件夹内文件的个数
Data_all = [];  % 用来盛放所有数据

% 循环读取文件，按列存放循环读取的数据
for i = 1:N
    Data = readtable(temp(i).name);
    Data_all(1,i) = mean(Data.rebuffering_duration);
    Data_all(2,i) = mean(Data.video_bitrate);
    Data_all(3,i) = mean(Data.chunk_duration);
    Data_all(4,i) = max(Data.qp);
    Data_all(5,i) = min(Data.framerate);
    Data_all(6,i) = min(Data.width);
    Data_all(7,i) = min(Data.height);
    Data_all(8,i) = mean(Data.is_best);
    Data_all(9,i) = mean(Data.mvm);
    Data_all(10,i) = mean(Data.psnr);
    Data_all(11,i) = mean(Data.ssim);
    Data_all(12,i) = mean(Data.msssim);
    Data_all(13,i) = mean(Data.vqm);
    Data_all(14,i) = mean(Data.strred);
    Data_all(15,i) = mean(Data.vmaf);
    Data_all(16,i) = mean(Data.ssimplus);
end
MOStable = readtable('D:\Course_Material\Undergraduate\毕设\test\waterloo_sqoe3_full\data.csv');
MOS = MOStable.mos;% 拟合因变量存放在MOS数组中

data_all = array2table(Data_all);
data_all.Properties.RowNames = ...
{'RBuf', 'VBr', 'CDu', 'QP', 'FR', 'Wid', 'He', 'IsB', 'Mv', 'Ps', 'Ss', 'Mss', 'Vqm', 'Str', 'Vma', 'Ss+'};
% 给每个行变量命名

% 几乎无关的指标：CDu, IsB 小相关指标：FR
% 选择想要的指标值
features = data_all({'RBuf', 'VBr', 'QP', 'Wid', 'He', 'Mv', 'Ps', 'Ss', 'Mss', 'Vqm', 'Str', 'Vma', 'Ss+'}, :);
features = table2array(features)';% 进行行列转置并转换为数组数据
a = 0.12;
cv = cvpartition(N, 'Holdout', a);% 划分训练集和测试集
trainingData = [features(training(cv), :) MOS(training(cv), :)];
testData = [features(test(cv), :) MOS(test(cv), :)];

% model = glmfit(trainingData(:, 2:end), trainingData(:, 1), 'normal');

% 用fitnlm非线性拟合———————————————————————————————
modelfun = @(b, X) b(1)*X(:,1) + b(2)*X(:,1).^-1 + b(3)*X(:,1).^2 +...
                                b(4)*X(:,2) + b(5)*X(:,2).^-1 + b(6)*X(:,2).^2 +...
                                b(7)*X(:,3) + b(8)*X(:,3).^-1 + b(9)*X(:,3).^2 +...
                                b(10)*X(:,4) + b(11)*X(:,4).^-1 + b(12)*X(:,4).^2 +...
                                b(13)*X(:,5) + b(14)*X(:,5).^-1 + b(15)*X(:,5).^2 +...
                                b(16)*X(:,6) + b(17)*X(:,6).^-1 + b(18)*X(:,6).^2 +...
                                b(19)*X(:,7) + b(20)*X(:,7).^-1 + b(21)*X(:,7).^2 +...
                                b(22)*X(:,8) + b(23)*X(:,8).^-1 + b(24)*X(:,8).^2 +...
                                b(25)*X(:,9) + b(26)*X(:,9).^-1 + b(27)*X(:,9).^2 +...
                                b(28)*X(:,10) + b(29)*X(:,10).^-1 + b(30)*X(:,10).^2 +...
                                b(31)*X(:,11) + b(32)*X(:,11).^-1 + b(33)*X(:,11).^2 +...
                                b(34)*X(:,12) + b(35)*X(:,12).^-1 + b(36)*X(:,12).^2 +...
                                b(37)*X(:,13) + b(38)*X(:,13).^-1 + b(39)*X(:,13).^2 + b(40);
beta0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
% modelfun = @(b, X) b(1)*X(:,1) + b(2)*X(:,1).^-1 +...
%                                 b(3)*X(:,2) + b(4)*X(:,2).^2 +...
%                                 b(5)*X(:,3) + b(6)*X(:,3).^-1 +...
%                                 b(7)*X(:,4) + b(8)*X(:,4).^2 +...
%                                 b(9)*X(:,5) + b(10)*X(:,5).^2 +...
%                                 b(11)*X(:,6) + b(12)*X(:,6).^-1 + b(13)*X(:,6).^2 +...
%                                 b(14)*X(:,7) + b(15)*X(:,7).^-1 + b(16)*X(:,7).^2 +...
%                                 b(17)*X(:,8) + b(18)*X(:,8).^-1 + b(19)*X(:,8).^2 +...
%                                 b(20)*X(:,9) + b(21)*X(:,9).^-1 +...
%                                 b(22)*X(:,10) + b(23)*X(:,10).^-1 + b(24)*X(:,10).^2 +...
%                                 b(25)*X(:,11) + b(26)*X(:,11).^-1 + b(27)*X(:,11).^2 +...
%                                 b(28)*X(:,12) + b(29)*X(:,12).^2 +...
%                                 b(30)*X(:,13) + b(31)*X(:,13).^2 + b(32);
% beta0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...
%                0, 0];
model = fitnlm(trainingData(:, 1:end-1), trainingData(:, end), modelfun, beta0);
model = model;
disp(model);

coeff_rsm = model.Coefficients{:,{'Estimate'}};%获取拟合系数
y_estimated = modelfun(coeff_rsm, testData);%计算预测值
compare = [y_estimated, testData(:, end)];

figure(1);
plot(testData(:, end), y_estimated,'bo');%作散点图
hold on;
xlabel('MOS value (true)'); % 设置横轴名称
ylabel('MOS value (true & estimated)'); % 设置纵轴名称
title('Trend map');
plot(testData(:, end), testData(:, end),'r-');hold off;

figure(2);
plot(1:N*a, y_estimated,'b-');hold on;%作折线图
plot(1:N*a, y_estimated,'bo');hold on;
xlabel('No. of video stream'); % 设置横轴名称
ylabel('MOS value (true & estimated)'); % 设置纵轴名称
title('Deviation degree visual map (red is true, blue is the estimation by QoE model)');
plot(1:N*a, testData(:, end),'ro');hold on;
plot(1:N*a, testData(:, end),'r-');hold off;
%————————————————————————————————————————
