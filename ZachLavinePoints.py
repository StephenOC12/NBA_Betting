import os
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


directory = "/Users/stephenoconnellclare/Downloads"

file_name = "zachlavinebettingtest2.csv"


file_path = os.path.join(directory, file_name)


df = pd.read_csv(file_path)

df.head()
print(list(df.columns))


directory2 = "/Users/stephenoconnellclare/Downloads"

file_name2 = "NBA_defence_vs_SG2.csv"

file_path2 = os.path.join(directory2, file_name2)

df2 = pd.read_csv(file_path2)

df2.head()

print(list(df2.columns))


# List of columns to compare
columns_to_compare = ['3PM', 'BLK', 'STL', 'TO', 'AST', 'REB']

# For each column it finds the 5 highest
for column in columns_to_compare:
    df2[column] = df2[column].astype(float)
    top_5 = df2.nlargest(5, column)  # Gets the top 5 highest values
    print(f"Top 5 teams for {column}:\n{top_5[['Team', column]]}\n")

# List of columns to compare
columns_to_compare = ['3PM', 'BLK', 'STL', 'TO', 'AST', 'REB']

# Version for 5 lowest
for column in columns_to_compare:
    df2[column] = df2[column].astype(float)
    bottom_5 = df2.nsmallest(5, column)  # Gets the 5 lowest
    print(f"Bottom 5 teams for {column}:\n{bottom_5[['Team', column]]}\n")



features = ['MIN','FG%','3P%','FT%', 'REB', 'AST', 'BLK', 'STL', 'PF', 'TO']

df_without_missing_values = df.dropna()

target = df_without_missing_values['PTS']
target.head()


X_train, X_test, y_train, y_test = train_test_split(df_without_missing_values[features],target, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and fit the Lasso regression model
alpha = 0.01  # You can adjust the regularization strength
lasso_model = Lasso(alpha=alpha)
lasso_model.fit(X_train_scaled, y_train)

# Get the coefficients and corresponding feature names
coefficients = pd.DataFrame({'Feature': features, 'Coefficient': lasso_model.coef_})

# Sort the coefficients by absolute value to see the most influential features
coefficients['Absolute_Coefficient'] = abs(coefficients['Coefficient'])
coefficients = coefficients.sort_values(by='Absolute_Coefficient', ascending=False)

print(coefficients)

# Predict on the test set
prediction = lasso_model.predict(X_test_scaled)



