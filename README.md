# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import dataset

2.Assign x and y values

3.Calculate Logistic sigmoid function and plot the graph

4.Calculate the cos function

5.Calculate x train and y train grad value

6.Calculate and Plot decision boundry

7.Calculate the probability value and predict the mean value

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: A.Harini Shamlin
RegisterNumber:  212220040040

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]
x[:5]
y[:5]
plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("exam 1 score")
plt.ylabel("exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()
def cf(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=cf(theta,x_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=cf(theta,x_train,y)
print(j)
print(grad)
def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j
def gradient(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def decbou(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,0].min()-1,x[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.01))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  decbou(res.x,x,y)
  prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
 np.mean(predict(res.x,x)==y)
*/
```

## Output:
# ARRAY VALUE OF X:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/579c4d6a-6e5d-464f-8372-058086d148f3)
# ARRAY VALUE OF Y:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/5b8f07c6-875e-46dc-ab6b-807cae948ca1)
# VISUALIZING OF DATA:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/8f8ee2c2-277b-41df-918a-02492af86764)
# SIGMOID FUNCTION:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/e7835b35-b17a-40de-9b9c-2a0ff3cff7a3)
# COST FUNCTION:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/eeef03f6-9240-46a4-aaa5-3d1517af3aed)
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/cb58d3e7-3548-4a69-80a8-c0101d9c6601)
# DECISION BOUNDARY-GRAPH:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/5e6ed937-7f09-4e22-883a-e787316035f9)
# PROBABILITY VALUE :
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/400bf928-112d-4862-9d18-526de362d282)
# PREDICTION VALUE OF MEAN:
![image](https://github.com/ATHDY005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/84709944/855de139-e74d-47db-8785-879982d73208)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

