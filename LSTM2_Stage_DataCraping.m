%% A multistage deep neural network model for blood pressure estimation using photoplethysmogram signals

% This program trains and uses two seperate Long Short-Term Networks (LSTMs) to estimate Diastolic & systolic Blood Pressures.

% Trained LSTM Models: SystolicNet_Stage2_64Units_crp & DiastolicNet_Stage2_64Units_crp

% J. Esmaelpoor- Sep 2019

%% Load Data and Networks

clc, close all, clear all

load TrainSeq;
load ValidationSeq;
load TestSeq;

load SystolicNet_Stage1
load DiastolicNet_Stage1

%% Stage One: Convolutional Network

% Train Data
disp('Stage One Output Calculation ...')
for i = 1:length(TrainSeq)
    clear SeqData2 TarData2;
    
    SeqData1 = TrainSeq{1,i};
    
    for j = 1:size(SeqData1,1)
        SeqData2(j,1) = SeqData1(j,254); %predict(DiastolicNet_Stage1,SeqData1(j,3:252));
        SeqData2(j,2:61) = activations(SystolicNet_Stage1,SeqData1(j,3:252),'conv_4','OutputAs','rows');
        SeqData2(j,62) = SeqData1(j,253); %predict(SystolicNet_Stage1,SeqData1(j,3:252));%
        SeqData2(j,63:122) = activations(DiastolicNet_Stage1,SeqData1(j,3:252),'conv_4','OutputAs','rows');
        TarData2(j,1:2) = SeqData1(j,253:254);        
    end
    STrainStage2{i,1} = SeqData2(:,1:61)';
    STrainTarStage2{i,1} = TarData2(:,1)';
    DTrainStage2{i,1} = SeqData2(:,62:122)';
    DTrainTarStage2{i,1} = TarData2(:,2)';
end

% Training Data Croping
cnt_seg = 0;
for i = 1:length(STrainStage2)
    s2 = STrainStage2{i,1};
    ts2 = STrainTarStage2{i,1};
    d2 = DTrainStage2{i,1};
    td2 = DTrainTarStage2{i,1};
    ns=6;
    Lims = fix(linspace(1,size(STrainStage2{i,1},2),ns));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(1):Lims(3));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(1):Lims(3));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(1):Lims(3));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(1):Lims(3));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(1):Lims(4));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(1):Lims(4));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(1):Lims(4));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(1):Lims(4));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(1):Lims(5));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(1):Lims(5));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(1):Lims(5));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(1):Lims(5));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(1):Lims(6));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(1):Lims(6));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(1):Lims(6));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(1):Lims(6));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(2):Lims(6));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(2):Lims(6));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(2):Lims(6));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(2):Lims(6));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(3):Lims(6));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(3):Lims(6));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(3):Lims(6));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(3):Lims(6));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(4):Lims(6));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(4):Lims(6));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(4):Lims(6));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(4):Lims(6));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(5):Lims(6));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(5):Lims(6));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(5):Lims(6));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(5):Lims(6));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(2):Lims(5));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(2):Lims(5));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(2):Lims(5));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(2):Lims(5));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(3):Lims(5));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(3):Lims(5));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(3):Lims(5));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(3):Lims(5));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(1):Lims(2));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(1):Lims(2));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(1):Lims(2));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(1):Lims(2));
    
    cnt_seg = cnt_seg + 1;
    STrainStage2_seg{cnt_seg,1} = s2(:,Lims(2):Lims(3));
    STrainTarStage2_seg{cnt_seg,1} = ts2(:,Lims(2):Lims(3));
    DTrainStage2_seg{cnt_seg,1} = d2(:,Lims(2):Lims(3));
    DTrainTarStage2_seg{cnt_seg,1} = td2(:,Lims(2):Lims(3));
    
end    
cnt_seg   
% Validation Data
for i = 1:length(ValidationSeq)
    clear SeqData2 TarData2;
    SeqData1 = ValidationSeq{1,i};
    for j = 1:size(SeqData1,1)
        SeqData2(j,1) = predict(DiastolicNet_Stage1,SeqData1(j,3:252));
        SeqData2(j,2:61) = activations(SystolicNet_Stage1,SeqData1(j,3:252),'conv_4','OutputAs','rows');
        SeqData2(j,62) = predict(SystolicNet_Stage1,SeqData1(j,3:252));
        SeqData2(j,63:122) = activations(DiastolicNet_Stage1,SeqData1(j,3:252),'conv_4','OutputAs','rows');
        TarData2(j,1:2) = SeqData1(j,253:254);
    end
    SValidationStage2{i,1} = SeqData2(:,1:61)';
    SValidationTarStage2{i,1} = TarData2(:,1)';
    DValidationStage2{i,1} = SeqData2(:,62:122)';
    DValidationTarStage2{i,1} = TarData2(:,2)';
end

% Test Data
for i = 1:length(TestSeq)
    clear SeqData2 TarData2;
    SeqData1 = TestSeq{1,i};
    for j = 1:size(SeqData1,1)
        SeqData2(j,1) = predict(DiastolicNet_Stage1,SeqData1(j,3:252));
        SeqData2(j,2:61) = activations(SystolicNet_Stage1,SeqData1(j,3:252),'conv_4','OutputAs','rows');
        SeqData2(j,62) = predict(SystolicNet_Stage1,SeqData1(j,3:252));
        SeqData2(j,63:122) = activations(DiastolicNet_Stage1,SeqData1(j,3:252),'conv_4','OutputAs','rows');
        TarData2(j,1:2) = SeqData1(j,253:254);
    end
    STestStage2{i,1} = SeqData2(:,1:61)';
    STestTarStage2{i,1} = TarData2(:,1)';
    DTestStage2{i,1} = SeqData2(:,62:122)';
    DTestTarStage2{i,1} = TarData2(:,2)';
end

%% Stage Two: Systolic LSTM Network

clc, disp('Train LSTM Network ...')
numResponses = 1;
featureDimension = 61;
numHiddenUnits = 64;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    lstmLayer(numHiddenUnits/2,'OutputMode','sequence')
    fullyConnectedLayer(16)
    dropoutLayer(0.3)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 350;
miniBatchSize = 64;

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',50,...
    'Shuffle','every-epoch', ...
    'GradientThreshold',1, ...
    'ValidationData',{SValidationStage2,SValidationTarStage2}, ...
    'Plots','training-progress',...
    'Verbose',0);

SystolicNet_Stage2_64Units_crp = trainNetwork(STrainStage2_seg,STrainTarStage2_seg,layers,options);

save('SystolicNet_Stage2_64Units_crp.mat', 'SystolicNet_Stage2_64Units_crp')

%% Stage Two: Diastolic LSTM Network

clc, disp('Train LSTM Network ...')
numResponses = 1;
featureDimension = 61;
numHiddenUnits = 64;

layers = [ ...
    sequenceInputLayer(featureDimension)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    lstmLayer(numHiddenUnits/2,'OutputMode','sequence')
    fullyConnectedLayer(16)
    dropoutLayer(0.3)
    fullyConnectedLayer(numResponses)
    regressionLayer];

maxEpochs = 350;
miniBatchSize = 64;

options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'InitialLearnRate',0.01, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',50,...
    'Shuffle','every-epoch', ...
    'GradientThreshold',1, ...
    'ValidationData',{DValidationStage2,DValidationTarStage2}, ...
    'Plots','training-progress',...
    'Verbose',0);

DiastolicNet_Stage2_64Units_crp = trainNetwork(DTrainStage2_seg,DTrainTarStage2_seg,layers,options);

save('DiastolicNet_Stage2_64Units_crp.mat', 'DiastolicNet_Stage2_64Units_crp')


%% Evaluate Performance

clc, disp('Evaluate Performance...')
SYPredicted = predict(SystolicNet_Stage2_64Units_crp,STestStage2);
DYPredicted = predict(DiastolicNet_Stage2_64Units_crp,DTestStage2);

cnt = 0;
for i = 1:length(STestStage2)
    STarCell(:,cnt+1 : cnt+size(STestTarStage2{i,1},2)) = STestTarStage2{i,1};
    SPredCell(:,cnt+1 : cnt+size(STestTarStage2{i,1},2)) = SYPredicted{i,1};
    SpredictionError(:,cnt+1 : cnt+size(STestTarStage2{i,1},2)) = STestTarStage2{i,1}-SYPredicted{i,1};
    
    DTarCell(:,cnt+1 : cnt+size(DTestTarStage2{i,1},2)) = DTestTarStage2{i,1};
    DPredCell(:,cnt+1 : cnt+size(DTestTarStage2{i,1},2)) = DYPredicted{i,1};
    DpredictionError(:,cnt+1 : cnt+size(DTestTarStage2{i,1},2)) = DTestTarStage2{i,1}-DYPredicted{i,1};
    
    cnt = cnt+size(STestTarStage2{i,1},2);
end
    
STD_S = std(SpredictionError')
squares_S = SpredictionError.^2;
rmse_S = sqrt(mean(squares_S'))
MAE_S = mean(abs(SpredictionError'))
ME_S = mean(SpredictionError')

STD_D = std(DpredictionError')
squares_D = DpredictionError.^2;
rmse_D = sqrt(mean(squares_D'))
MAE_D = mean(abs(DpredictionError'))
ME_D = mean(DpredictionError')

figure
plot(STarCell(1,:)), hold on, plot(SPredCell(1,:))
figure
plot(DTarCell(1,:)), hold on, plot(DPredCell(1,:))

