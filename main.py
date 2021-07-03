from os import write
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import csv

# Scikit learn utilities
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Get the actual data
student_data = pd.read_csv('student_data/student-mat.csv')

# In order to see the correlation between the weekday, and weekend alcohol use
def graph_comparaison_weekend_weekdays(students=30):
    width = 0.33333
    
    x = np.arange(students)
    y1 = student_data['Dalc'].values[:students]
    y2 = student_data['Walc'].values[:students]

    plt.bar(x - width, y1, label='Daily', width=width)
    plt.bar(x + width, y2, label='Weekends', width=width)

    plt.legend()
    plt.show()

# See the correlation of each column with the weekend consumption
correlation_weekend_columns = student_data.corr()['Walc'].sort_values(ascending=False)

# Remove the daily consumption column and the specific school
del student_data['Dalc'], student_data['school'], student_data['reason']

# Separate the test set and the training set (Stratified Sampling, so there's no bias)
# Given that going out is highly correlated to the consumption, that attribute will be used to stratify data
strat_splitting = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=10)

for train_index, test_index in strat_splitting.split(student_data, student_data['goout']):
    training_data = student_data.iloc[train_index]
    testing_data = student_data.iloc[test_index]

# Remove and copy the column we want to predict ("Weekend alcohol consumption")
consumption_before = training_data.drop('Walc', axis=1, inplace=False)
consumption_labels = training_data['Walc'].copy()

numerical_columns = [
    'age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'health', 'absences',
    'G1', 'G2', 'G3'
]

categorical_columns = [
    'sex', 'address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 
    'activities', 'nursery', 'higher', 'internet', 'romantic', 'Mjob', 'Fjob', 'guardian'
]

# Actions that will be performed on the numerical columns
numerical_columns_pipeline = Pipeline(steps=[
    ('standard_scaler', StandardScaler())
])

# Actions that will be performed on the categorical columns (those that contain strings)
categorical_columns_pipeline = Pipeline(steps=[
    ('1_hot_encoding', OneHotEncoder())
])

# Create the full pipeline with both types of categories
full_pipeline = ColumnTransformer([
    ("numerical_columns_transformation", numerical_columns_pipeline, numerical_columns),
    ("categorical_columns_transformation", categorical_columns_pipeline, categorical_columns)
])

# Call the pipeline with our data
consumption_prepared = full_pipeline.fit_transform(consumption_before)

# Remove the solutions before training the model
del student_data["Walc"]

def linear_regression_testing():
    # Select a model (Linear regression)
    lin_reg = LinearRegression()
    lin_reg.fit(consumption_prepared, consumption_labels)

    ln_predictions = lin_reg.predict(consumption_prepared)
    lin_mse = mean_squared_error(consumption_labels, ln_predictions)
    scores = cross_val_score(lin_reg, consumption_prepared, consumption_labels, scoring="neg_mean_squared_error", cv=10)

    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

def decision_tree():
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(consumption_prepared, consumption_labels)

    tree_predictions = tree_reg.predict(consumption_prepared)
    tree_mse = mean_squared_error(consumption_labels, tree_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)

    scores = cross_val_score(tree_reg, consumption_prepared, consumption_labels, scoring="neg_mean_squared_error", cv=10)

    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

forest_reg = RandomForestRegressor()
forest_reg.fit(consumption_prepared, consumption_labels)

forest_predictions = forest_reg.predict(consumption_prepared)
forest_mse = mean_squared_error(consumption_labels, forest_predictions)
forest_rmse = np.sqrt(forest_mse)


# Save final results to file for clear representation
predictions_np = np.round(forest_predictions)
results_np = np.array(consumption_labels)

predictions_pd = pd.DataFrame(np.round(predictions_np)).astype(int)
results_pd = pd.DataFrame(results_np)

df = pd.DataFrame()
df[['predictions']] = predictions_pd
df[['results']] = results_pd
df.to_csv('save.csv')

# Calculate accuracy
predictions_arr = list(predictions_np)
predictions_arr = [int(x) for x in predictions_arr]
results_arr = list(results_np)

correct_predictions = 0
for pred, res in zip(predictions_arr, results_arr):
    if pred == res: correct_predictions += 1 

print("Precision: ", correct_predictions / len(predictions_arr) * 100)