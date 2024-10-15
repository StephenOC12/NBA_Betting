import os
import pandas as pd

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

directory = "/Users/stephenoconnellclare/Downloads"

file_name = "CobyWhiteFile.csv"


file_path = os.path.join(directory, file_name)


df = pd.read_csv(file_path)

df.head()
print(list(df.columns))

directory2 = "/Users/stephenoconnellclare/Downloads"

file_name2 = "NBA_defence_vs_PG.csv"

file_path2 = os.path.join(directory2, file_name2)

df2 = pd.read_csv(file_path2)

df2.head()

print(df2)