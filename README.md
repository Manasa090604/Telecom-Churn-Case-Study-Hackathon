# Telecom Churn Prediction

General Information :-

Overview:
This project aims to predict telecom customer churn using a RandomForestClassifier, focusing on key features that influence churn.

Problem statement

In the telecom industry, customers frequently switch between service providers, resulting in an annual churn rate of 15-25%. 

Acquiring a new customer costs 5-10 times more than retaining an existing one, making customer retention crucial. 

For many established operators, retaining highly profitable customers is the top priority.

Technologies used :

Python: Used for data preprocessing, analysis, and modeling.

Pandas: Used for data manipulation and exploration.

Scikit-learn: Used for building and evaluating regression models.

Matplotlib and Seaborn: Used for data visualization.

klearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, precision_score, recall_score

#Loading dependencies & datasets

Addded any necessary imports to this cell block as I write more code

# Create X, y and then Train test split

Created X and y datasets and skip "circle_id" since it has only 1 unique value

# Handling Missing Data

Analysing the missing data. by using  missingno library for quick visualization

# Exploratory Data Analysis & Preprocessing¶

Lets start by analysing the univariate distributions of each feature.

#  Feature engineering and selection

Understanding feature importances for raw features as well as components to decide top features for modelling.

# Model building

##Building a quick model with logistic regression and the first 2 PCA components

#  Creating submission file

#Data Preprocessing:
1. Feature Selection: Selected relevant features based on domain knowledge.
2. Missing Values Filled missing values using `SimpleImputer`.
3. Scaling : Normalized data with `StandardScaler`.

# Model Training:
1. PCA: Reduced data dimensionality to 10 components.
2. Classifier Trained a RandomForestClassifier on PCA-transformed data.
3. Pipeline: Created a pipeline for imputation, scaling, PCA, and classification.

# Evaluation:
1. Confusion Matrix: Evaluated performance, noting high type 2 error due to class imbalance.
2. Metrics: Calculated precision and recall.

# Addressing Class Imbalance:
Class imbalance led to the model predicting non-churn more often. Techniques like SMOTE or adjusting class weights are recommended.

# Key Insights:
1. Feature Importance: Identified top features for churn prediction.
2. Class Imbalance: Skewed predictions, suggesting advanced handling techniques.

# Recommendations:
1. Feature Engineering: Create features based on business insights.
2. Class Imbalance: Use resampling techniques or adjust class weights.
3. Model Improvement: Explore different models and hyperparameters.

# Conclusion:
Our model offers a baseline for predicting customer churn. By addressing class imbalance and refining features, we can improve accuracy, aiding customer retention strategies.
Technologies Used:-

# Collaberators:- Mayank and Preethi for supporting



