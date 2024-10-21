# Loan-Approval-Prediction
Leveraging and tuning a LightGBM model to predict whether or not a loan will be approved

![image](https://github.com/user-attachments/assets/84e69eb4-ad76-42ac-ab29-c797bbc11c8d)

---

### Objective
The overarching goal of this project is to develop a model to predict whether or not an applicant is approved for a loan. This project has been completed as part of Kaggle's Playground Series competitions (Season 4, Episode 10), and as part of my continued learning of various data science concepts.

In this project, I will first use adversarial validation to determine whether or not the original data (discussed in the 'About the Data' section) follows the same distribution as Kaggle's synthetic data. This step will likely be a key factor in the final model's performance, as incorporating data from a different distribution can sometimes negatively impact a model's performance on the withheld testing data.

Next, I will briefly explore the datasets by viewing basic summary statistics, visualizing the distributions of the target variable and the categorical and numerical features, and also by using a heatmap to identify any potential correlations amongst features.

Following this, I will prepare the data for modeling. This will be accomplished by encoding all object type features as integers. I will also employ some feature engineering, by creating new features which are essentially ratios of existing features.

The final model will be a LightGBM model. LightGBM (LGBM) is a fast, distributed, high performance gradient boosting framework that is based on decision tree algorithms. This type of model is typically effective in classification tasks. LGBM is developed by Microsoft and is publicly available for usage.

In order to identify the best model, I will use optuna, which is a hyperparameter optimization framework. Optuna will create and run through 100 trials to determine the ideal hyperparameters for the final LGBM model.

It should also be noted that the target variable 'loan_status' is not evenly distributed (shown in the EDA section). Because of this, I will be using Stratified K-Fold Cross Validation for stratified sampling, instead of random sampling.

The model will be evaluated using the area under the curve (AUC) of the receiver operating characteristic curve (ROC curve). The ROC curve is esentially the plot of the true positive rate versus the false positive rate. In the case of this model and competition, a 'perfect' model would yield an AUC of 1.0, while a terrible model would likely result in an AUC of 0.5 (basically random guess). However, obtaining an AUC of 1 is typically impossible, so the objective will be to get the AUC as close to 1.0 as possible.

---

### Methods
Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- catboost (Pool, CatBoostClassifier, eval_metric)
- sklearn (train_test_split, StratifiedKFold, roc_curve, roc_auc_score)
- sklearn (log_loss, confusion_matrix, classification_report)
- lightgbm (LGBMClassifier, plot_importance)
- optuna (TPESampler)
- scipy
- sweetviz

Adversarial Validation
- Determining whether or not the synthetic training data and original data are from the same distribution
- Combining both datasets, labeling all train data as 0, all original data as 1
- Leveraging a simple Catboost classifier to distinguish between the two datasets
- Metric: AUC-ROC, if close to 0.5, the datasets likely come from the same distribution

Exploratory Data Analysis
- Creating a function to show summary statistics of training and testing datasets
- Visualizing the distributions of target variable and categorical variables with countplots and donutplots
- Visualizing the distributions of numerical variables with KDE plots
- Visualizing potential correlations amongst numerical variables with correlation heatmap
- Automating EDA of training and testing datasets with Sweetviz

Data Preprocessing
- Encoding object type features as int type
- Feature engineering, creating new features from ratios of exisiting features
- Splitting data into training and testing sets

Model Building/Tuning
- Predetermined parameters:
    - Objective: binary
    - Device: cpu
    - Boosting_type: gbdt
    - Random_state: 42
- Parameters to be determined:
    - Num_leaves
    - Learning_rate
    - N_estimators
    - Subsample_for_bin
    - Reg_alpha
    - Reg_lambda
    - Max_depth
    - Colsample_bytree
    - Subsample
    - Min_child_samples
    - Feature_fraction
    - Bagging_fraction
- 100 Optuna trials
- Metric: AUC ROC score

Final Model
- Parameters:
    - Num_leaves: 301
    - Learning_rate: 0.030842070949044974
    - N_estimators: 921
    - Subsample_for_bin: 128087
    - Reg_alpha: 0.08543291154990179
    - Reg_lambda: 3.9292955013618704
    - Max_depth: 6
    - Colsample_bytree: 0.6857619348039488
    - Subsample: 0.7774078680670069
    - Min_child_samples: 44
    - Feature_fraction: 0.5288157389802886
    - Bagging_fraction: 0.6226401151972283

---

### General Results
![image](https://github.com/user-attachments/assets/419d8022-d0ec-4095-827f-2622f961fd8b)
![image](https://github.com/user-attachments/assets/53302b95-c057-4f08-b1ba-f38de6e24db0)
![image](https://github.com/user-attachments/assets/f94fb9e9-a81b-4115-898e-c9b8d2d7dcae)

- Final Model Performance --> 0.96 AUC ROC
