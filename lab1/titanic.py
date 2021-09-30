# # Dependencies
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

# # Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

# print("***** Train_Set *****")
# print(train.head())
# print("\n")
# print("***** Test_Set *****")
# print(test.head())


# Fill missing values with mean column values in the train set
train.fillna(train.mean(), inplace=True)
# Fill missing values with mean column values in the test set test.fillna(test.mean(), inplace=True)


print(train.columns.values)
# print(train.describe())
# print("*****In the train set*****")
# print(train.isna().sum())
# print("\n")
# print("*****In the test set*****") 
# print(test.isna().sum())
print(train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print(train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))

g = sns.FacetGrid(train, col='Survived') 
g.map(plt.hist, 'Age', bins=20) 
plt.show()
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6) 
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()