# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the Employee.csv dataset and display the first few rows.

2.Check dataset structure and find any missing values.

3.Display the count of employees who left vs stayed.

4.Encode the "salary" column using LabelEncoder to convert it into numeric values.

5.Define features x with selected columns and target y as the "left" column.

6.Split the data into training and testing sets (80% train, 20% test).

7.Create and train a DecisionTreeClassifier model using the training data.

8.Predict the target values using the test data.

9.Evaluate the model’s accuracy using accuracy score.

10.Predict whether a new employee with specific features will leave or not.
## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Hashwatha M
RegisterNumber: 212223240051
*/
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data

data.head()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]
y.head()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier (criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
print("Name:Hashwatha M")
print("Reg no:212223240051")

```
## Output:
## Data:
![image](https://github.com/user-attachments/assets/62e3607f-a01d-4d1e-8c97-b14f83f4efe4)

## Data Head:

![image](https://github.com/user-attachments/assets/b3b4577a-b0f7-4ea7-a486-d0abd2e1b341)

## Null Dataset:

![image](https://github.com/user-attachments/assets/9d7c3f38-cc62-418e-9d93-e9df46f3679d)

## Values count in left column:

![image](https://github.com/user-attachments/assets/df8aa6a6-69c8-4a9a-85d0-0e701a375d78)

## Dataset transformed head:

![image](https://github.com/user-attachments/assets/574b4063-5e49-4026-8d89-25dd1d6ae106)

## x.head():

![image](https://github.com/user-attachments/assets/1bc0f440-ab84-4fcb-b22c-36ebef345f19)

## y.head():

![image](https://github.com/user-attachments/assets/e082f996-855c-437d-b9dc-171605c6ceff)

## Accuracy:

![image](https://github.com/user-attachments/assets/1be9e773-abd7-49bf-b2ea-f9b7a28f419a)

## Data prediction:

![image](https://github.com/user-attachments/assets/3170e7e1-30b9-4fcc-b0c2-f883464a6ae8)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
