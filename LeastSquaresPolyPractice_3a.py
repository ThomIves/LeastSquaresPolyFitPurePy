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
X = [[2,3,4,5,6,7,8],[1,1.5,2,2.5,3,3.5,4]]
def y_of_x(xa):
    return 0.2*xa[0]**2 + 0.3*xa[0]*xa[1] + 0.7*xa[0] + 0.4*xa[1]**2 \
            + 0.1*xa[1] + 2.0 #+ random.uniform(-0.2,0.2)
Y = []
for xa in la.transpose(X):
    Y.append(y_of_x(xa))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[0], X[1], Y)

# Section 2: Get fake data in correct format
X = la.transpose(X)
Y = la.transpose(Y)

# Section 3: Pure Python Tools Fit
poly_pp = ml.Poly_Features_Pure_Py(order=2) # print(X) sys.exit()
Xp = poly_pp.fit_transform(X)
ls_pp = ml.Least_Squares(tol=2)
ls_pp.fit(Xp, Y)

# Section 4: SciKit Learn Fit
poly_sk = PolynomialFeatures(degree = 2)
Xp = poly_sk.fit_transform(X)
ls_sk = LinearRegression()
ls_sk.fit(Xp, Y)

# Section 5: Coefficients Comparison
temp_lsp_coefs = sorted(ls_pp.coefs)
rounded_lsp_coefs = [round(x,8)+0 for x in la.transpose(temp_lsp_coefs)[0]]
print('PurePy  LS coefficients:', rounded_lsp_coefs)
print('SKLearn LS coefficients:', ls_sk.coef_, '\n')

# Section 6: Create Fake Test Data
XLS = [[2.0,2.50,3.0,3.50,4.0,4.50,5.0,5.50,6.0,6.50,7.0,7.5,8.0,8.5],
       [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.,4.25]]

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
print( 'Prediction Deltas:', deltas , '\n')

# Section 8: Plot Both Methods
ax.plot3D(XLS[0], XLS[1], YLSpp)
ax.plot3D(XLS[0], XLS[1], YLSsk)
ax.set_xlabel('X0 Values')
ax.set_ylabel('X1 Values')
ax.set_zlabel('Y Values')
ax.set_title('Pure Python Least Squares Fit with Two 2nd Order Inputs')
plt.show()
