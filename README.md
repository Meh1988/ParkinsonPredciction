# ParkinsonPredciction
Parkinson's Disease Prediction (Traditional Machine Learning vs Deep Learning)


Overview
This repository contains code to predict the onset of Parkinson's Disease using various machine learning and deep learning models. The models are trained on a dataset that includes several voice measure features from individuals, some of whom are diagnosed with Parkinson's Disease.

About Dataset
Parkinson's Data Set
This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to the "status" column which is set to 0 for healthy and 1 for PD.

The data is in ASCII CSV format. The rows of the CSV file contain an instance corresponding to one voice recording. There are around six recordings per patient, the name of the patient is identified in the first column.For further information or to pass on comments, please contact Max Little (little '@' robots.ox.ac.uk).

Further details are contained in the following reference -- if you use this dataset, please cite:
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig (2008), 'Suitability of dysphonia measurements for telemonitoring of Parkinson's disease', IEEE Transactions on Biomedical Engineering (to appear).

Attribute Information:
Matrix column entries (attributes):
name - ASCII subject name and recording number
MDVP:Fo(Hz) - Average vocal fundamental frequency
MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
MDVP:Flo(Hz) - Minimum vocal fundamental frequency
MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP - Several measures of variation in fundamental frequency
MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
NHR, HNR - Two measures of the ratio of noise to tonal components in the voice
status - The health status of the subject (one) - Parkinson's, (zero) - healthy
RPDE, D2 - Two nonlinear dynamical complexity measures
DFA - Signal fractal scaling exponent
spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

The dataset can be downloaded from here:
https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set


Models
The following machine learning and deep learning models are implemented:

Support Vector Machine (SVM)
Naive Bayes (NB)
Decision Tree (DT)
Simple Neural Network
Convolutional Neural Network (CNN)
Recurrent Neural Network (RNN)
Setup
To run the code, you will need the following libraries:

pandas
numpy
matplotlib
scikit-learn
tensorflow



Outputs
The code performs the following tasks:

Loads the dataset and prepares the features and labels.
Computes and plots feature importance using SelectKBest.
Splits the data into training and test sets and scales the features.
Trains three traditional machine learning models and evaluates them.
Trains three deep learning models and evaluates them.
Plots the accuracy scores of all the models for comparison.
Visualization

Two visualizations are generated:
- Feature Importance Plot
- Model Comparison Plot
