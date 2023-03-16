# cs5228
to run the models,please follow the steps below.

step1:
due to the file size,we cannot upload those data sets to this repository,
please paste sample-submission.csv,train.csv,test.csv to the data folder,
and also decompress the auxiliary_data.zip to form a new folder named auxiliary_data under the data folder,
i.e.\data\auxiliary-data.

step2:
run the data manupulation scripts in data folder, 
data_analyse.py and data_analyse.py  are to provide some insights into the data,
while data_fusion.py is to fuse some new features.

step3:
run the models,
dl-lstm used the deep learning method BiLSTM,
while other models used traditional machine learning models,like RandomForest,Xgboost and LinearRegression.
