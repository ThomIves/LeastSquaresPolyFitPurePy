import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random
import sys

# Fake data preparation
X = [[2,3,4,5,6,7,8],[1,1.5,2,2.5,3,3.5,4]]
chk = ml.Poly_Features_Pure_Py(order=2)
chk.fit(la.transpose(X))
print('Features Names: ', chk.get_feature_names())
print()

def y_of_x(xa):
    return 0.2*xa[0]**2 + 0.3*xa[0]*xa[1] + 0.7*xa[0] + 0.4*xa[1]**2 \
            + 0.1*xa[1] + 2.0 #+ random.uniform(-0.2,0.2)

Y = []
for xa in la.transpose(X):
    Y.append(y_of_x(xa))

# Plotting Prep
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(X[0], X[1], Y)
X = la.transpose(X)
Y = la.transpose(Y)
print(X)
print(Y)
print()
# sys.exit()

# Pure Py Fit
poly_pp = ml.Poly_Features_Pure_Py(order=2) # print(X) sys.exit()
Xp = poly_pp.fit_transform(X)
ls_pp = ml.Least_Squares(tol=2)
print(Xp)
print(Y)
print()
ls_pp.fit(Xp, Y)

# SKLearn Fit
poly_sk = PolynomialFeatures(degree = 2)
Xp = poly_sk.fit_transform(X)
ls_sk = LinearRegression()
ls_sk.fit(Xp, Y)

# Coefficients Comparison
temp_lsp_coefs = sorted(ls_pp.coefs)
rounded_lsp_coefs = [round(x,8)+0 for x in la.transpose(temp_lsp_coefs)[0]]
print('PurePy  LS coefficients:', rounded_lsp_coefs)
print('SKLearn LS coefficients:', ls_sk.coef_, '\n')

# Predictions 
XLS = [[2.0,2.50,3.0,3.50,4.0,4.50,5.0,5.50,6.0,6.50,7.0,7.5,8.0,8.5],
       [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.,4.25]]

XLSpp = poly_pp.transform(la.transpose(XLS))       
YLSpp = ls_pp.predict(XLSpp)
YLSpp = la.transpose(YLSpp)[0]

XLSsk = poly_sk.transform(la.transpose(XLS))
YLSsk = ls_sk.predict(XLSsk)
YLSsk = YLSsk.tolist()
YLSsk = la.transpose(YLSsk)[0]

# Prediction Differences 
YLSpp.sort()
YLSsk.sort()
deltas = [ YLSpp[i] - YLSsk[i] for i in range(len(YLSpp)) ]
print( 'Prediction Deltas:', deltas , '\n')

# Plotting
ax.plot3D(XLS[0], XLS[1], YLSpp)
ax.plot3D(XLS[0], XLS[1], YLSsk)
ax.set_xlabel('X0 Values')
ax.set_ylabel('X1 Values')
ax.set_zlabel('Y Values')
ax.set_title('Pure Python Least Squares Fit with Two 2nd Order Inputs')
plt.show()
