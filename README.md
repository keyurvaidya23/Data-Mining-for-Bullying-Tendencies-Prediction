# Data-Mining-for-Bullying-Tendencies-Prediction
This project utilizes R and several machine learning packages to process data, build predictive models, and evaluate their performance. It focuses on addressing issues like missing values, duplicate data, and feature selection, and implements various algorithms including Naive Bayes, Decision Trees (J48), C5.0, KNN, and Neural Networks to predict outcomes based on a dataset concerning instances of bullying.

Project Structure
project_dataset.csv - The main dataset file used for the analysis.
machine_learning_project.R - Main R script file that contains all preprocessing, model building, and evaluation code.
requirements.R - An R script to install all required packages.
Setup and Installation
Prerequisites
R and RStudio (Recommended)
Required R packages: caret, rsample, RWeka, rpart, rpart.plot, MASS, kernlab, pROC, dplyr, C50, e1071, ROSE, FSelector, Boruta
Installation
Install R and RStudio:
Download and install R from The Comprehensive R Archive Network (CRAN).
Download and install RStudio from RStudio Download page.
Install Required Packages:
You can install the required packages using RStudio or R console. Open R and execute:
install.packages(c("caret", "rsample", "RWeka", "rpart", "rpart.plot", "MASS", "kernlab", "pROC", "dplyr", "C50", "e1071", "ROSE", "FSelector", "Boruta"))
Running the Script
Load the Script:
Open machine_learning_project.R in RStudio or your preferred R environment.
Execute the Script:
Run the script in RStudio by pressing Ctrl+A to select all and then Ctrl+Enter to execute, or use the source command in the R console:
source("path_to_your_script/machine_learning_project.R")
View Results:
The script will output various metrics and performance evaluations for each model directly in your R console or RStudio environment.
Features and Models
Data Preprocessing: Handling missing values, removing duplicates, and feature selection.
Predictive Models: Naive Bayes, J48 Decision Tree, C5.0, Logistic Regression, KNN, and Neural Networks.
Model Evaluation: Confusion Matrix, ROC Curves, and other performance metrics.
Contributing
Contributions to this project are welcome. Please ensure any pull requests are validated locally for performance and correctness.

