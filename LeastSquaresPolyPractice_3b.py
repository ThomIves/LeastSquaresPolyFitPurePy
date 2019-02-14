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
X = [[float(x)/25.0 for x in range(401)],[float(x)/50.0 for x in range(401)]]
def y_of_x(xa):
    return 0.2*xa[0]**2 + 0.3*xa[0]*xa[1] + 0.7*xa[0] + 0.4*xa[1]**2 \
            + 0.1*xa[1] + 2.0 #+ random.uniform(-0.2,0.2)
Y = []
for xa in la.transpose(X):
    Y.append(y_of_x(xa))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(X[0], X[1], Y)
ax.set_xlabel('X1 Values')
ax.set_ylabel('X2 Values')
ax.set_zlabel('Y Values')
ax.set_title('Pure Python Least Squares Two Inputs Data Fit')

# Section 2: Putting fake data in correct format 
#   done with inplace transpose statements

# Section 3: Pure Python Tools Fit
poly_pp = ml.Poly_Features_Pure_Py(order=2)
Xp = poly_pp.fit_transform(la.transpose(X))
ls_pp = ml.Least_Squares(tol=2)
ls_pp.fit(Xp, Y)

# Section 4: SciKit Learn Fit NOT USED

# Section 5: Coefficients for Pure Method
print('LS coefficients:', ls_pp.coefs)
print()

# Section 6: Create Fake Test Data
XLS = [[2.0,2.50,3.0,3.50,4.0,4.50,5.0,5.50,6.0,6.50],
       [1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25]]

# Section 6.1: Predict with Fake Test Data
#   Using Pure Python Tools
XLSpp = poly_pp.transform(la.transpose(XLS))
YLSpp = ls_pp.predict(XLSpp)

# Section 6.2: Predict with Fake Test Data
#   Using SciKit Learn Tools NOT USED

# Section 7: NO Prediction Differences THIS SAME

# Section 8: Plot Pure Output
ax.plot3D(XLS[0], XLS[1], la.transpose(YLSpp)[0])
plt.show()
