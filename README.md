# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
Developed By: A Ahil Santo

Register Number: 212224040018

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

#obtaining x and y value from csv file
df=pd.read_csv(r"E:\Desktop\CSE\Introduction To Machine Learning\Ex_2\student_scores.csv")
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
print("x values:",x)
print("y values:",y)

#spliting dataset for testing and training
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#training the model by linear regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print("Y Prediction:",y_pred)
print("Actual Value:",y_test)
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,regressor.predict(x_train),color="orange")
plt.title("Hours vs Score (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color="red")
plt.plot(x_test,regressor.predict(x_test),color="green")
plt.title("House vs Score (Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#measuring the error percentage for the model
mse=mean_squared_error(y_test,y_pred)
print("MSE =",mse)
mae=mean_absolute_error(y_test,y_pred)
print("MAE =",mae)
rmse=np.sqrt(mse)
print("RMSE =",rmse)

```

## Output:
## Head and Tail of CSV File

![1](https://github.com/user-attachments/assets/fd61c673-c569-40d7-80b1-84c3cc50541e)

## Value of X And Y

![2](https://github.com/user-attachments/assets/a7173309-412d-44d5-a983-898c7b14150f)

## Y Prediction And Actual Value

![3](https://github.com/user-attachments/assets/44aff23d-ee54-416f-a901-285205f34c64)

## Training Best-Fit-Line

![4](https://github.com/user-attachments/assets/4fd63747-faa9-4ef3-97c9-85721196f7ef)

## Testing Best-Fit-Line

![5](https://github.com/user-attachments/assets/2402076f-f595-496c-a0e7-e888d40059db)

## Error Percentage

![6](https://github.com/user-attachments/assets/b940a44b-62b3-46df-94ab-2ab2c005a69b)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
