import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_csv("/Simple Dataset/Laptop-Users.csv")

print(df.head())

# Select independent and dependent variable
X = df[["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]]
y = df["has-laptop"]

X = df.iloc[:,1:]
y = df.iloc[:,0:1]

regressor = LinearRegression()
regressor.fit(X,y)

# Make predictions using the testing set
bp_predict = regressor.predict(X)


# Make pickle file of our model
pickle.dump(sv, open('model.pkl','wb'))
