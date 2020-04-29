# A-multistage-deep-neural-network-model-for-blood-pressure-estimation-using-photoplethysmogram-signal
* Processed Data Cells for Training, Validation, and Testing:
	TrainSeq.mat
	ValidationSeq.mat
	TestSeq.mat
The length of each row is 254. The first two elements are the primary values of systolic and diastolic BPs in each Invasive BP recording. They can be used to evaluate the effect of calibration on the algorithm. We have not assessed the issue in the paper. The PPG segment itself is elements from 3 to 252.
The last two elements (253th and 254th) are Systolic and Diastolic target values for each segment.

* Convolution_DBP.m: The m.file to train the CNN for diastolic BP estimation and feature extraction in the first stage.
* Convolution_SBP.m: The m.file to train the CNN for systolic BP estimation and feature extraction in the first stage.
* LSTM2_Stage_DataCraping.m The m.file to train both LSTM networks for systolic and diastolic blood pressure estimations in the second stage.

* Traned Networks:
	DiastolicNet_Stage1.mat
	SystolicNet_Stage1.mat
	DiastolicNet_Stage2_64Units_crp.mat
	SystolicNet_Stage2_64Units_crp.mat

