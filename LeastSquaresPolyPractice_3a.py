import LinearAlgebraPurePython as la 
import MachineLearningPurePy as ml

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import random
import sys

# Section 1: Fake data preparation 
#   and visualization
X = [[x for x in range(2,9)],
     [2**(x/1.0)  for x in range(0,7)]]

def y_of_x(xa):
    return 0.2*xa[0]**2 + 0.03*xa[0]*xa[1] + 0.7*xa[0] + 0.01*xa[1]**2 \
            + 0.04*xa[1] + 2.0 #+ random.uniform(-2.0,2.0)
Y = []
for xa in la.transpose(X):
    Y.append(y_of_x(xa))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[0], X[1], Y)

# Section 2: Get fake data in correct format
X = la.transpose(X)
Y = la.transpose([Y])

# Section 3: Pure Python Tools Fit
poly_pp = ml.Poly_Features_Pure_Py(order=2) #, include_bias=False) # print(X) sys.exit()
Xpp = poly_pp.fit_transform(X)
ls_pp = ml.Least_Squares(add_ones_column=False)
ls_pp.fit(Xpp, Y)

# Section 4: SciKit Learn Fit
poly_sk = PolynomialFeatures(degree = 2)
Xps = poly_sk.fit_transform(X)
ls_sk = LinearRegression()
ls_sk.fit(Xps, Y)

# Section 5: Coefficients Comparison
tmp_ls_pp_coefs = sorted(ls_pp.coefs) # ls_pp.coefs # sorted(ls_pp.coefs)
rounded_ls_pp_coefs = [round(x,8)+0 for x in la.transpose(tmp_ls_pp_coefs)[0]] # 
print('PurePy  LS coefficients:', rounded_ls_pp_coefs)

tmp_ls_sk_coefs = ls_sk.intercept_.tolist() + ls_sk.coef_[0][1:].tolist()
tmp_ls_sk_coefs = sorted(tmp_ls_sk_coefs)
rounded_ls_sk_coefs = [round(x,8)+0 for x in tmp_ls_sk_coefs] # 
print('SKLearn LS coefficients:', rounded_ls_sk_coefs)

print('Coef Deltas:', [rounded_ls_pp_coefs[i] - rounded_ls_sk_coefs[i] 
        for i in range(len(rounded_ls_pp_coefs))])

# Section 6: Create Fake Test Data
XLS = [[x/2.0 for x in range(4,18)],
       [2**(x/2.0)  for x in range(0,14)]]

# Section 6.1: Predict with Fake Test Data
#   Using Pure Python Tools
XLSpp = poly_pp.transform(la.transpose(XLS))       
YLSpp = ls_pp.predict(XLSpp)
YLSpp = la.transpose(YLSpp)[0]

# Section 6.2: Predict with Fake Test Data
#   Using SciKit Learn Tools
XLSsk = poly_sk.transform(la.transpose(XLS))
YLSsk = ls_sk.predict(XLSsk)
YLSsk = YLSsk.tolist()
YLSsk = la.transpose(YLSsk)[0]

# Section 7: Calculate Prediction Differences 
YLSpp.sort()
YLSsk.sort()
deltas = [ YLSpp[i] - YLSsk[i] for i in range(len(YLSpp)) ]
print( '\nPrediction Deltas:', deltas , '\n')

# Section 8: Plot Both Methods
ax.plot3D(XLS[0], XLS[1], YLSpp)
ax.plot3D(XLS[0], XLS[1], YLSsk)
ax.set_xlabel('X0 Values')
ax.set_ylabel('X1 Values')
ax.set_zlabel('Y Values')
ax.set_title('Pure Python Least Squares Fit with Two 2nd Order Inputs')
plt.show()
