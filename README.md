# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
Developed by: Monisha D
RegisterNumber: 25007487
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Sample data: Hours studied vs Scores
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
Y = np.array([10, 20, 25, 35, 45, 50, 60, 65, 75, 80])

print("Hours:", *X.flatten())
print("Scores:", *Y)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Make predictions
Y_pred = regressor.predict(X_test)
print("Predicted Scores:", *Y_pred)
print("Actual Scores:", *Y_test)

# Plot Training set
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()

# Plot Testing set
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, Y_pred, color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours Studied")
plt.ylabel("Scores")
plt.show()

# Evaluate the model
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
```

## Output:
Hours: 1 2 3 4 5 6 7 8 9 10
Scores: 10 20 25 35 45 50 60 65 75 80
Predicted Scores: 73.73762376237624 18.811881188118804 50.1980198019802
Actual Scores: 75 20 50
<img width="735" height="588" alt="{18201718-1D3C-4787-835F-ED9FF56D6AB8}" src="https://github.com/user-attachments/assets/a5c738a1-54db-40d5-b8f1-dbb731245723" />
<img width="785" height="590" alt="{C7E30A14-D2FA-4E9E-9F94-CA6A00573EEA}" src="https://github.com/user-attachments/assets/b9509107-12e0-4907-a9d0-1f13d393dee3" />
Mean Absolute Error: 0.8828382838283844
Mean Squared Error: 1.0148106394797904
Root Mean Squared Error: 1.007378101548664
​




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
