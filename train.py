import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the Titanic dataset directly from seaborn
data = sns.load_dataset('titanic')

# Explanation: Replacing missing values ensures the dataset is complete and suitable for analysis.

# Explanation: Replacing missing values ensures the dataset is complete and suitable for analysis.

# Write code for - 'Age': Replace missing values with the mean age
data['age'].fillna(data['age'].mean(), inplace = True)

# Write code for - 'Embarked': Replace missing values with the mode (most frequent value)
data['embarked'].fillna(data['embarked'].mode()[0], inplace = True)

# Write code for - 'Fare': Replace missing values with zero
data['fare'].fillna(0, inplace = True)

# Write code for - 'Deck': Add 'Unknown' as a new category and replace missing values with 'Unknown'
data['deck'] = data['deck'].cat.add_categories('Unknown')
data['deck'] = data['deck'].fillna('Unknown')

# Write code for - 'Embark_town': Replace missing values with the mode (most frequent value)
data['embark_town'].fillna(data['embark_town'].mode()[0], inplace = True)

# Write code for - Display missing values count
print("\nMissing Values Count After Handling:\n", data.isnull().sum())

# Write code for - Percentage after imputation
missing_percentage_after = (data.isnull().sum()/len(data))*100
print("\nMissing Values Percentage After Handling:\n", missing_percentage_after)


# Explanation: Ensuring correct data types helps in proper analysis and computation.

# Write code for - converting 'age' and 'fare' to float
data['age'] = data['age'].astype(float)
data['fare'] = data['fare'].astype(float)
# Write code for - convert 'sex' and 'embarked' to categorical
data['sex'] = data['sex'].astype('category')
data['embarked'] = data['embarked'].astype('category')
data = data.copy()  # Create a copy to avoid SettingWithCopyWarning

# Explanation: Standardizing numerical features ensures they are on the same scale.

# Write code for - standardize the 'age' and 'fare' columns
scaler = StandardScaler()
data[['age', 'fare']] = scaler.fit_transform(data[['age', 'fare']])

# Explanation: Feature engineering can provide additional insights and improve model performance.

# Write code for - create a new feature 'family_size' which is the sum of 'sibsp' and 'parch' plus 1
data['family_size'] = data['sibsp'] + data['parch'] + 1


data = data[data['fare'] < 200]
# Explanation: Separating variables helps in organizing the data for more targeted analysis.

numerical_features = data.select_dtypes(include=['float64', 'int64'])
categorical_features = data.select_dtypes(include=['object', 'category', 'bool'])


# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
# Drop irrelevant columns based on the dataset and correlation analysis
# Write code for dropping Columns such as 'survived', 'who', 'alive', and 'alone' are considered irrelevant for predicting 'fare'

data_cleaned = data.drop(['survived', 'who', 'alive', 'alone'], axis=1)
# Define numerical and categorical features
# Numerical features include 'age', 'sibsp', 'parch', and 'family_size'
# Categorical features include 'sex', 'embarked', 'class', 'adult_male', 'deck', 'embark_town', and 'pclass'

# write your code here
numerical_features = ['age', 'sibsp', 'parch', 'family_size']
categorical_features = ['sex', 'embarked', 'class', 'adult_male', 'deck', 'embark_town', 'pclass']


# Separate features and target variable
# 'X' contains all the features (both numerical and categorical) except 'fare'
# 'y' is the target variable which is 'fare'

# write your code here
X = data_cleaned.drop('fare', axis=1)
y = data_cleaned['fare']

# Handle categorical variables by one-hot encoding
# This will convert categorical features into numerical format using one-hot encoding
X = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Standardize the numerical features
# StandardScaler will normalize numerical features to have a mean of 0 and a standard deviation of 1
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])
# Split the dataset into training and testing sets
# 80% of the data will be used for training, and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import mlflow
import mlflow.sklearn
mlflow.set_experiment("my_experiment")
# Define and train the model of RandomForest Regression
def train_and_log_model():
    with mlflow.start_run() as run:
    # Initialize the RandomForest Regression model
        model = RandomForestRegressor()

    # Fit the model on the training data
        model.fit(X_train, y_train)
    # Predict the fare on the test data
        y_pred = model.predict(X_test)

    # Evaluate the model performance
    # Calculate Mean Squared Error, R-squared, and Mean Absolute Error
        mse1 = mean_squared_error(y_test, y_pred)
        r21 = r2_score(y_test, y_pred)
        mae1 = mean_absolute_error(y_test, y_pred)

        print(f"Mean Squared Error: {mse1}")
        print(f"R-squared: {r21}")
        print(f"Mean Absolute Error: {mae1}")
    # Log parameters, metrics, and model
        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_metric("mse", mse1)
        mlflow.log_metric("r2", r21)
        mlflow.sklearn.log_model(model, "model")
        mlflow.register_model(f"runs:/{run.info.run_id}/random_forest_model", "RandomForestModel")
        print(f"Run URL: {mlflow.active_run().info.artifact_uri}")

mlflow.end_run()

# Define and train the model of LinearRegression
def train_and_log_model2():
    with mlflow.start_run() as run:
    # Initialize the Linear Regression model
        model = RandomForestRegressor()

    # Fit the model on the training data
        model.fit(X_train, y_train)
    # Predict the fare on the test data
        y_pred = model.predict(X_test)

    # Evaluate the model performance
    # Calculate Mean Squared Error, R-squared, and Mean Absolute Error
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R-squared: {r2}")
        print(f"Mean Absolute Error: {mae}")
    # Log parameters, metrics, and model
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
        mlflow.register_model(f"runs:/{run.info.run_id}/linear_regression_model", "LinearRegressionModel")
        print(f"Run URL: {mlflow.active_run().info.artifact_uri}")

train_and_log_model()

train_and_log_model2()

