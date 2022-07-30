from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


data = pd.read_csv(
    'https://raw.githubusercontent.com/harshlangade19/Data_Analysis_Engage/main/auto-mpg.csv')
data.head()

data.info()

data.describe()
data.isnull().sum()

data = data.drop(['car name', 'origin', 'model year'], axis=1)
data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

value = data[data['horsepower'].isnull()].index.tolist()


hp_avg = data['horsepower'].mean()
disp_avg = data['displacement'].mean()
factor = hp_avg / disp_avg

for i in value:
    data['horsepower'].fillna(value=(data.iloc[i][2])*factor, inplace=True)

X = data.drop('mpg', axis=1)
y = data['mpg']
X = np.asarray(X).astype(np.float32)
y = np.asarray(y).astype(np.float32)

X_train, X_cv, y_train, y_cv = train_test_split(
    X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.fit_transform(X_cv)

regressor = LinearRegression()

regressor.fit(X_train, y_train)

predictions = regressor.predict(X_cv)
predictions

mae = mean_absolute_error(y_cv, predictions)
mse = mean_squared_error(y_cv, predictions)
rmse = np.sqrt(mse)
print("Mean Absolute Error = ", mae)
print("Mean Squared Error = ", mse)
print("Root Mean Squared Error = ", rmse)


x = np.array([[8, 340, 145, 3600, 12, ]])
x = x.astype(float)
x

y = regressor.predict(x)
y

# Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[8, 340, 145, 3600, 12]]))
